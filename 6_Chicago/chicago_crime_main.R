setwd("~/R-VGAL/6_Chicago")
## Structure:
# 1. Generate data
# 2. Run R-VGAL algorithm
# 3. Run HMC
# 4. Plot results

rm(list=ls())

library("tensorflow")
tfp <- import("tensorflow_probability")
tfd <- tfp$distributions
library(keras)
library("dplyr")
library("mvtnorm")
library("rstan")
library("gridExtra")
library("grid")
library("gtable")
library(Matrix)
library(coda)
library(matrixcalc)

# List physical devices
gpus <- tf$config$experimental$list_physical_devices('GPU')

if (length(gpus) > 0) {
  tryCatch({
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    tf$config$experimental$set_virtual_device_configuration(
      gpus[[1]],
      list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=2*4096))
    )

    logical_gpus <- tf$config$experimental$list_logical_devices('GPU')

    print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
  }, error = function(e) {
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  })
}
tfp <- import("tensorflow_probability")
tfd <- tfp$distributions

source("./source/run_rvgal.R")
source("./source/run_stan_poisson.R")
# source("./source/generate_data.R")
# source("./source/compute_joint_llh_tf.R")
source("./source/compute_grad_hessian_all.R")
# source("./source/compute_grad_hessian_theoretical.R")

## Flags
date <- "20231018" #"20231018" has 2 fixed effects, ""20231030" has 4    
reload_data <- F
rerun_rvga <- T
rerun_stan <- F
save_data <- F
save_rvgal_results <- F
save_hmc_results <- F
plot_prior <- T
save_plots <- F
reorder_data <- F
use_tempering <- T
use_tensorflow <- T

plot_trace <- F

if (use_tempering) {
  n_obs_to_temper <- 20
  K <- 10
  a_vals_temper <- rep(1/K, K)
}

n_post_samples <- 10000
n_random_effects <- 2

## Read data
if (reload_data) {
  load(file = "./data/chicagocrimedf.rda")
  head(df)
  
  population_data <- read.csv(file = "./data/chicago_population.csv")
  pop_data_sub <- population_data[c("GEOID", "GEOG", "X2000_POP", "X2010_POP", 
                                    "X2020_POP", "TOT_POP")]
  
  crimedata_df <- df %>% filter(primary_type == "BATTERY" | primary_type == "ASSAULT",
                                !is.na(community_area))
  
  crimedata_grouped_df <- crimedata_df %>% group_by(community_area, year) %>% 
    summarise(ncases = n()) %>% 
    filter(community_area != 0, year >= 2002)
  
  # year_df <- crimedata_grouped_df %>% select(community_area, year) %>% 
  # group_by(community_area) %>% summarise(n = n())
  
  all_df <- left_join(crimedata_grouped_df, pop_data_sub, 
                      by = join_by(community_area == GEOID))
  
  
  # all_df$year_centred <- all_df$year - 2010 # centre the year variable
  # all_df$pop_thousands <- all_df$X2010_POP/1000
  all_df <- all_df %>% mutate(year_centred = year - 2011, 
                              pop_thousands = X2010_POP/10000,
                              log_pop = log(X2010_POP))
  
  standard_df <- all_df %>% ungroup() %>% 
    mutate(pop_std = (log_pop - mean(log_pop))/sd(log_pop), 
           year_std = (year_centred - mean(year_centred))/sd(year_centred)
    )
  # saveRDS(standard_df, file = "chicago_crime_standardised.rds")
} else {
  standard_df <- readRDS(file = "./data/chicago_crime_standardised.rds")
}

##### Try max likelihood first?? ,#####
# library(lme4)
# glm_fit <- glmer(ncases ~ 1 + year_std + pop_std + (1 + year_std | community_area),
#       family = "poisson", data = standard_df)
# fixef(glm_fit) ## returns fixed effects
# ranef(glm_fit) ## returns random effects
# summary(glm_fit)
# browser()
# ## Prediction to check that glmer is running properly
# test_comm_area = 40
# new_df <- data.frame(year_centred = -9:9, community_area = test_comm_area, 
#                      log_pop = filter(all_df, community_area == test_comm_area)$log_pop[1])
# prediction <- predict(glm_fit, new_df, type = "response")
# plot(-9:9, prediction)
# lines(-9:9, filter(all_df, community_area == test_comm_area)$ncases)
# browser()
#############################################################################

## Fixed covariate matrix
X_long <- cbind(rep(1, nrow(standard_df)), standard_df[c("community_area", "year_std", 
                                               #"X2010_POP")])
                                               # "pop_thousands")])
                                                  "pop_std")])
X <- X_long %>% group_split(community_area) # split observations by ID
X <- lapply(X, function(x) { x["community_area"] <- NULL; data.matrix(x) }) # get rid of the CA column then convert from df to matrix

## Random covariate matrix
Z_long <- cbind(rep(1, nrow(standard_df)), standard_df[c("community_area", "year_std")])
Z <- Z_long %>% group_split(community_area) # split observations by ID
Z <- lapply(Z, function(x) { x["community_area"] <- NULL; data.matrix(x) }) # get rid of the CA column then convert from df to matrix

## Response variable
y_long <- standard_df[c("community_area", "ncases")]
y <- y_long %>% group_split(community_area) # split observations by ID
y <- lapply(y, function(x) { x["community_area"] <- NULL; as.vector(data.matrix(x)) }) # get rid of the CA column then convert from df to matrix
  
N <- length(y)
n <- length(y[[1]])

## Reorder data if needed
if (reorder_data) {
  reorder_seed <- 2024
  set.seed(reorder_seed) 
  reordered_ind <- sample(1:length(y))
  print(head(reordered_ind))
  reordered_y <- lapply(reordered_ind, function(i) y[[i]])
  reordered_X <- lapply(reordered_ind, function(i) X[[i]])
  reordered_Z <- lapply(reordered_ind, function(i) Z[[i]])
  
  y <- reordered_y
  X <- reordered_X
  Z <- reordered_Z
}

hist(unlist(y))

###################
##     R-VGA     ##
###################
S <- 20L
S_alpha <- 20L

## Set up result directory
if (use_tempering) {
  temper_info <- paste0("_temper", n_obs_to_temper)
} else {
  temper_info <- ""
}

if (reorder_data) {
  reorder_info <- paste0("_seed", reorder_seed)
} else {
  reorder_info <- ""
}
result_directory <- paste0("./results/")
results_file <- paste0("poisson_mm_rvga", temper_info, reorder_info, 
                       "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, ".rds")


## Initialise variational parameters
n_fixed_effects <- as.integer(ncol(X[[1]]))
n_random_effects <- as.integer(ncol(Z[[1]]))
n_elements_L <- n_random_effects + n_random_effects * (n_random_effects - 1)/2
param_dim <- n_fixed_effects + n_elements_L

beta_0 <- rep(0, n_fixed_effects)
l_vec_0 <- c(rep(0, n_random_effects), rep(0, n_random_effects * (n_random_effects - 1)/2))
mu_0 <- c(beta_0, l_vec_0)
P_0 <- diag(c(1, 1, 1, rep(0.1, n_elements_L)))

## Plot prior samples first
if (n_random_effects == 2) {
  param_names <- c(sapply(1:n_fixed_effects, function(x) paste0("beta[", x, "]")), 
                   "Sigma_alpha[1,1]", "Sigma_alpha[2,1]",
                   "Sigma_alpha[2,2]")
  
} else {
  param_names <- c(sapply(1:n_fixed_effects, function(x) paste0("beta[", x, "]")), 
                   "Sigma_alpha[1,1]", 
                   "Sigma_alpha[2,1]",
                   "Sigma_alpha[2,2]", "Sigma_alpha[3,1]",
                   "Sigma_alpha[3,2]", "Sigma_alpha[3,3]")
}

if (plot_prior) {
  
  n_prior_samples <- 1000
  prior_samples <- rmvnorm(n_prior_samples, mu_0, P_0)
  
  rvgal.Sigma_prior_samples <- list()
  for (k in 1:n_prior_samples) {
    rvgal.Sigma_prior_samples[[k]] <- construct_Sigma(prior_samples[k, -(1:n_fixed_effects)], 
                                                      n_random_effects)
  }
  
  rvgal.prior_samples <- matrix(NA, nrow = n_prior_samples, ncol = param_dim)
  rvgal.prior_samples[, 1:n_fixed_effects] <- prior_samples[, 1:n_fixed_effects]
  
  nlower <- n_random_effects * (n_random_effects-1)/2 + n_random_effects
  lower_ind <- lapply(1:nlower, index_to_i_j_rowwise_diag)
  for (d in 1:(param_dim - n_fixed_effects)) {
    inds <- lower_ind[[d]]
    rvgal.prior_samples[, n_fixed_effects+d] <- unlist(lapply(rvgal.Sigma_prior_samples, function(Sigma) Sigma[inds[1], inds[2]]))
  }
  
  
  par(mfrow = c(2,3))
  # true_vals <- c(beta, c(Sigma_alpha[t(lower.tri(Sigma_alpha, diag = T))]))
  for (p in 1:param_dim) {
    plot(density(rvgal.prior_samples[, p]), main = param_names[p])
    # abline(v = true_vals[p], lty = 2)
  }
}

#################
##     R-VGA   ##
#################

if (rerun_rvga) {
  rvgal_results <- run_rvgal(y, X, Z, mu_0, P_0, 
                             S = S, S_alpha = S_alpha,
                             n_post_samples = n_post_samples,
                             use_tempering = use_tempering, 
                             n_temper = n_obs_to_temper, 
                             temper_schedule = a_vals_temper,
                             use_tf = use_tensorflow)
  
  if (save_rvgal_results) {
    saveRDS(rvgal_results, file = paste0(result_directory, results_file))
  }
  
} else {
  rvgal_results <- readRDS(file = paste0(result_directory, results_file))
}


rvgal.post_samples <- rvgal_results$post_samples


# rvgal.Sigma_post_samples <- list()
# for (k in 1:n_post_samples) {
#   rvgal.Sigma_post_samples[[k]] <- construct_Sigma(rvgal.post_samples[k, -(1:n_fixed_effects)],
#                                                    n_random_effects)
# }

# nlower <- n_random_effects * (n_random_effects-1)/2 + n_random_effects
# lower_ind <- lapply(1:nlower, index_to_i_j_rowwise_diag)
# for (d in 1:(param_dim - n_fixed_effects)) {
#   inds <- lower_ind[[d]]
#   rvgal.post_samples[, n_fixed_effects+d] <- unlist(lapply(rvgal.Sigma_post_samples, function(Sigma) Sigma[inds[1], inds[2]]))
# }


# ########################
# ##        STAN        ##
# ########################
burn_in <- 5000
n_chains <- 1
hmc.iters <- n_post_samples/n_chains + burn_in

if (rerun_stan) {
  
  ## Data manipulation ##
  y_long <- unlist(y) #as.vector(t(y))
  X_long <- do.call("rbind", X)
  Z_long <- do.call("rbind", Z)
  
  hmc_results <- run_stan_poisson(iters = hmc.iters, burn_in = burn_in,
                           n_chains = n_chains, data = y_long,
                           grouping = rep(1:N, each = n), n_groups = N,
                           fixed_covariates = X_long,
                           rand_covariates = Z_long,
                           prior_mean = mu_0,
                           prior_var = P_0)
  
  if (save_hmc_results) {
    saveRDS(hmc_results, file = paste0(result_directory, "poisson_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
} else {
  hmc_results <- readRDS(file = paste0(result_directory, "poisson_mm_hmc_N", N, "_n", n, "_", date, ".rds")) # for the experiements on starting points
  
}

# param_names <- c("beta[1]","beta[2]", "Sigma_alpha[1,1]", 
#                  "Sigma_alpha[2,1]",
#                  "Sigma_alpha[2,2]", "Sigma_alpha[3,1]",
#                  "Sigma_alpha[3,2]", "Sigma_alpha[3,3]")
# 
# hmc.fit <- extract(hfit, pars = param_names,
#                    permuted = F, inc_warmup = F)

hmc.fit <- hmc_results$post_samples[-(1:burn_in),,]
hmc.summ <- hmc_results$summary
hmc.n_eff <- hmc_results$n_eff
hmc.Rhat <- hmc_results$Rhat

######################## Results #########################

# rvgal.post_samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
hmc.samples <- matrix(NA, n_post_samples, param_dim)

for (p in 1:param_dim) {
  if (length(dim(hmc.fit)) < 3) {
    hmc.samples[, p] <- rbind(hmc.fit[, p])
  } else {
    hmc.samples[, p] <- rbind(hmc.fit[, , p]) 
  }
}

par(mfrow = c(n_random_effects, ceiling(param_dim/n_random_effects)))
# true_vals <- c(beta, c(Sigma_alpha[t(lower.tri(Sigma_alpha, diag = T))]))
for (p in 1:param_dim) {
  plot(density(hmc.samples[, p]), main = param_names[p])
  lines(density(rvgal.post_samples[, p]), col = "red")
  # abline(v = true_vals[p], lty = 2)
}

# 
# 
# # rvgal.post_mean <- as.vector(apply(rvgal.post_samples, 2, mean))
# # rvgal.post_sd <- as.vector(apply(rvgal.post_samples, 2, sd))
# # hmc.post_mean <- as.vector(apply(hmc.samples, 2, mean))
# # hmc.post_sd <- as.vector(apply(hmc.samples, 2, sd))
# 
# # dens <- stan_dens(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]")) #, "sigma"))
# # dens <- dens + ggtitle ("HMC posteriors") 
# # print(dens)
# # 
# # traceplot(hfit, c("beta[1]","beta[2]","beta[3]","beta[4]"),
# #           ncol=1,nrow=5,inc_warmup=F)
# 
# ## Combine all results into dataframe
# # results <- data.frame(true_vals = c(beta, c(Sigma_alpha[lower.tri(Sigma_alpha, diag = T)])), 
# #                       rvga_mean = rvgal.post_mean, rvga_sd = rvgal.post_sd,
# #                       hmc_mean = hmc.post_mean, hmc_sd = hmc.post_sd)
# # print(results)
# 
# ## Posterior plots
# param_names <- c("beta1", "beta2", "sigma_11", "sigma_21", "sigma_22",
#                  "sigma_31", "sigma_32", "sigma_33")  
# # rvga.df <- data.frame(rvgal.post_samples) 
# hmc.df <- data.frame(hmc.samples)
# # colnames(rvga.df) <- param_names
# colnames(hmc.df) <- param_names
# 
# true_vals.df <- data.frame(beta1 = beta[1], beta2 = beta[2], beta3 = beta[3],
#                            beta4 = beta[4], sigma11 = Sigma_alpha[1,1],
#                            sigma21 = Sigma_alpha[2,1], sigma22 = Sigma_alpha[2,2],
#                            sigma31 = Sigma_alpha[3,1], sigma32 = Sigma_alpha[3,2],
#                            sigma33 = Sigma_alpha[3,3])
# 
# plots <- list()
# 
# for (p in 1:(param_dim-1)) {
#   plot <- ggplot(rvga.df, aes(x=.data[[param_names[p]]])) + 
#     geom_density(col = "red", lwd = 1) +
#     geom_density(data = hmc.df, col = "blue", lwd = 1) +
#     geom_vline(data = true_vals.df, aes(xintercept=.data[[param_names[p]]]),
#                color="black", linetype="dashed", linewidth = 0.75) +
#     labs(x = bquote(beta[.(p)])) +
#     theme_bw() +
#     theme(axis.title = element_blank(), text = element_text(size = 18)) +                               # Assign pretty axis ticks
#     scale_x_continuous(breaks = scales::pretty_breaks(n = 4)) 
#   
#   plots[[p]] <- plot  
# }
# 
# tau_plot <- ggplot(rvga.df, aes(x=tau)) + 
#   geom_density(col = "red", lwd = 1) +
#   geom_density(data = hmc.df, col = "blue", lwd = 1) +
#   geom_vline(data = true_vals.df, aes(xintercept=tau),
#              color="black", linetype="dashed", linewidth = 0.75) +
#   labs(x = expression(tau)) +
#   theme_bw() +
#   theme(axis.title = element_blank(), text = element_text(size = 18)) +                               # Assign pretty axis ticks
#   scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) 
# 
# plots[[param_dim]] <- tau_plot
# 
# ## Posterior covariance plot
# 
# n_lower_tri <- (param_dim^2 - param_dim)/2
# 
# index_to_i_j_colwise_nodiag <- function(k, n) {
#   kp <- n * (n - 1) / 2 - k
#   p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
#   i  <- n - (kp - p * (p + 1) / 2)
#   j  <- n - 1 - p
#   c(i, j)
# }
# 
# param_values <- c(beta, tau)
# cov_plots <- list()
# for (ind in 1:n_lower_tri) {
#   mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
#   p <- mat_ind[1]
#   q <- mat_ind[2]
#   
#   param_df <- data.frame(x = param_values[q], y = param_values[p])
#   
#   cov_plot <- ggplot(rvga.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
#     stat_ellipse(col = "goldenrod", type = "norm", lwd = 1) +
#     stat_ellipse(data = rvga.df, col = "red", type = "norm", lwd = 1) +
#     stat_ellipse(data = hmc.df, col = "blue", type = "norm", lwd = 1) +
#     geom_point(data = param_df, aes(x = x, y = y),
#                shape = 4, color = "black", size = 4) +
#     theme_bw() +
#     theme(axis.title = element_blank(), text = element_text(size = 18)) +                               # Assign pretty axis ticks
#     scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) 
#   
#   cov_plots[[ind]] <- cov_plot
# }
# 
# m <- matrix(NA, param_dim, param_dim)
# n_cov_plots <- param_dim * (param_dim-1)/2
# m[lower.tri(m, diag = F)] <- 1:n_cov_plots
# gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
# gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
# gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:5, l = 1:5)
# 
# # A list of text grobs - the labels
# vars <- list(textGrob(bquote(beta[1])), textGrob(bquote(beta[2])), textGrob(bquote(beta[3])), 
#              textGrob(bquote(beta[4])), textGrob(bquote(tau)))
# vars <- lapply(vars, editGrob, gp = gpar(col = "black", fontsize = 20))
# 
# # So that there is space for the labels,
# # add a row to the top of the gtable,
# # and a column to the left of the gtable.
# gp <- gtable_add_cols(gr3, unit(1.5, "lines"), 0)
# gp <- gtable_add_rows(gp, unit(1.5, "lines"), -1) #0 adds on the top
# 
# # gtable_show_layout(gp)
# 
# # Add the label grobs.
# # The labels on the left should be rotated; hence the edit.
# # t and l refer to cells in the gtable layout.
# # gtable_show_layout(gp) shows the layout.
# gp <- gtable_add_grob(gp, lapply(vars[1:5], editGrob, rot = 90), t = 1:5, l = 1) # add column names to column 1, rows 2:5
# gp <- gtable_add_grob(gp, vars[1:5], t = 6, l = 2:6) # add row names to row 6, columns 1:5
# 
# grid.newpageS()
# grid.draw(gp)
# 
# if (save_plots) {
#   plot_file <- paste0("poisson_posterior", temper_info, reorder_info,
#                       "_S", S, "_Sa", S_alpha, "_", date, ".png")
#   filepath = paste0("./plots/", plot_file)
#   png(filepath, width = 1000, height = 700)
#   grid.newpage()
#   grid.draw(gp)
#   dev.off()
# } 
# 
## Time benchmark
hmc.time <- hmc_results$time
rvga.time <- rvgal_results$time_elapsed
print(hmc.time)
print(rvga.time)

## Trajectory for R-VGA
if (plot_trace) {
  par(mfrow = c(2, ceiling(param_dim/2)))
  trajectories <- list()
  for (p in 1:param_dim) {
    if (p > n_fixed_effects && p <= (n_fixed_effects + n_random_effects)) {
      trajectories[[p]] <- sapply(1:N, function(x) exp(rvgal_results$mu[[x]][p])^2)
    } else {
      trajectories[[p]] <- sapply(1:N, function(x) rvgal_results$mu[[x]][p])
    }
    plot(trajectories[[p]], main = param_names[p], type = "l")
    # abline(h = true_vals[p], lty = 2, col = "red")
  }
  
  ## HMC traceplots
  par(mfrow = c(ceiling(param_dim/2), 2))
  for (p in 1:param_dim) {
    plot(hmc.samples[,p], type = "l", main = param_names[p])
    # abline(h = true_vals[p], lty = 2, col = "red")
  }
  
}


# gc()
# tf$keras$backend$clear_session()
