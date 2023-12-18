# setwd("~/R-VGAL/6_Epilepsy")

rm(list=ls())

## Flags
date <- "20231018" #"20231018_interact" # has 2 fixed effects, ""20231030" has 4    
rerun_rvgal <- T
rerun_stan <- F
save_rvgal_results <- F
save_hmc_results <- F
reorder_data <- F
use_tempering <- T

plot_trace <- F
plot_prior <- F
save_plots <- F

use_tensorflow <- T

## Load packages
if (use_tensorflow) {
  library("tensorflow")
  tfp <- import("tensorflow_probability")
  tfd <- tfp$distributions
  library(keras)

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
}

library(HSAUR3) # for the dataset
library(dplyr)
library(mvtnorm)
library(rstan)
library(gridExtra)
library(grid)
library(gtable)
library(Matrix)
library(coda)
library(matrixcalc)

source("./source/run_rvgal.R")
source("./source/run_stan_poisson.R")
# source("./source/compute_grad_hessian_all.R")
source("./source/compute_grad_hessian_theoretical.R")

if (use_tempering) {
  n_obs_to_temper <- 10
  K <- 4
  a_vals_temper <- rep(1/K, K)
}

########################
##    1. Read data    ##
########################

data(epilepsy)
head(epilepsy)
epilepsy$treatment <- ifelse(epilepsy$treatment == "Progabide", 1, 0)

epilepsy2 <- epilepsy %>% mutate(log_age = log(age),
                    log_base = log(0.25*base),
                    visit = case_when(period == 1 ~ -0.3,
                                      period == 2 ~ -0.1,
                                      period == 3 ~ 0.1,
                                      period == 4 ~ 0.3))
epilepsy2 <- epilepsy2 %>% mutate(log_age_centred = log_age - mean(log_age),
                                  base_treat = log_base * treatment)

library(lme4)
if (grepl("interact", date)) {
  glm_fit <- glmer(seizure.rate ~ 1 + treatment + log_base + base_treat + 
                                  log_age_centred + visit + 
                                  (1 + visit | subject),
                                  family = "poisson", data = epilepsy2)
} else {
  glm_fit <- glmer(seizure.rate ~ 1 + treatment + log_base +
                                  log_age_centred + visit + 
                                  (1 + visit | subject),
                                  family = "poisson", data = epilepsy2)
}

fixef(glm_fit) ## returns fixed effects
# ranef(glm_fit) ## returns random effects

lower_L <- glm_fit@theta 
L <- matrix(c(lower_L[1:2], 0, lower_L[3]), 2, 2)
Sigma_glmfit <- tcrossprod(L)

summary(glm_fit)

## Fixed covariate matrix
fixed_efs <- c()
if (grepl("interact", date)) {
  fixed_efs <- c("treatment", "log_base", "base_treat",
                "log_age_centred", "visit", "subject")
} else {
  fixed_efs <- c("treatment", "log_base", 
                "log_age_centred", "visit", "subject")
}
X_long <- cbind(rep(1, nrow(epilepsy2)), epilepsy2[fixed_efs])
X <- X_long %>% group_split(subject) # split observations by ID
X <- lapply(X, function(x) { x["subject"] <- NULL; data.matrix(x) }) # get rid of the CA column then convert from df to matrix

## Random covariate matrix
Z_long <- cbind(rep(1, nrow(epilepsy2)), epilepsy2[c("visit", "subject")])
Z <- Z_long %>% group_split(subject) # split observations by ID
Z <- lapply(Z, function(x) { x["subject"] <- NULL; data.matrix(x) }) # get rid of the CA column then convert from df to matrix

## Response variable
y_long <- epilepsy2[c("seizure.rate", "subject")]
y <- y_long %>% group_split(subject) # split observations by ID
y <- lapply(y, function(x) { x["subject"] <- NULL; as.vector(data.matrix(x)) }) # get rid of the CA column then convert from df to matrix

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

######################
##     2. R-VGA     ##
######################
n_post_samples <- 20000
S <- 200L
S_alpha <- 200L

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
results_file <- paste0("epilepsy_rvgal", temper_info, reorder_info, 
                       "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, ".rds")


## Initialise variational parameters
n_fixed_effects <- as.integer(ncol(X[[1]]))
n_random_effects <- as.integer(ncol(Z[[1]]))
n_elements_L <- n_random_effects + n_random_effects * (n_random_effects - 1)/2
param_dim <- n_fixed_effects + n_elements_L

beta_0 <- rep(0, n_fixed_effects)
l_vec_0 <- c(rep(0, n_random_effects), rep(0, n_random_effects * (n_random_effects - 1)/2))
mu_0 <- c(beta_0, l_vec_0)
P_0 <- diag(c(rep(1, n_fixed_effects), rep(0.1, n_elements_L)))

if (rerun_rvgal) {
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

##########################
##        3. HMC        ##
##########################
burn_in <- 5000
n_chains <- 2
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
    saveRDS(hmc_results, file = paste0(result_directory, "epilepsy_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
} else {
  hmc_results <- readRDS(file = paste0(result_directory, "epilepsy_mm_hmc_N", N, "_n", n, "_", date, ".rds")) # for the experiements on starting points
  
}

hmc.fit <- hmc_results$post_samples[-(1:burn_in),,]
hmc.summ <- hmc_results$summary
hmc.n_eff <- hmc_results$n_eff
hmc.Rhat <- hmc_results$Rhat

###########################
##        Results        ##
###########################

# rvgal.post_samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
hmc.samples <- matrix(NA, n_post_samples, param_dim)

for (p in 1:param_dim) {
  if (length(dim(hmc.fit)) < 3) {
    hmc.samples[, p] <- rbind(hmc.fit[, p])
  } else {
    hmc.samples[, p] <- rbind(hmc.fit[, , p]) 
  }
}

if (grepl("interact", date)) {
  fixed_ef_names <- c("1", "treatment", "base", "base_treat", "age", "visit")
} else {
  fixed_ef_names <- c("1", "treatment", "base", "age", "visit")
}

fixed_ef_labels <- sapply(1:n_fixed_effects, function(x) paste0("beta[", fixed_ef_names[x], "]"))
param_names <- c(fixed_ef_labels, "sigma_alpha[11]", "sigma_alpha[21]", "sigma_alpha[22]")  

## ggplot 

rvgal.df <- data.frame(beta = rvgal.post_samples) 
hmc.df <- data.frame(beta = hmc.samples)
colnames(rvgal.df) <- param_names
colnames(hmc.df) <- param_names
glm_params <- c(glm_fit@beta, Sigma_glmfit[lower.tri(Sigma_glmfit, diag = T)])
plots <- list()

for (p in 1:param_dim) {
  glm.df <- data.frame(param = param_names[p], val = glm_params[p])

  plot <- ggplot(rvgal.df, aes(x = .data[[param_names[p]]])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmc.df, col = "blue", lwd = 1) +
    # labs(x = bquote(beta[.(p)])) +
    geom_vline(data = glm.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth = 0.75) +
    theme_bw() + 
    theme(axis.title = element_blank(), axis.text = element_text(size = 18)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 2)) 
  # theme(legend.position="bottom") + 
  # scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))
  
  plots[[p]] <- plot  
}

## Posterior covariance plot

## Function to map indices of plots inside a list to matrix (i,j) indices
## for arranging the plots in a lower triangular formation
n_lower_tri <- (param_dim^2 - param_dim)/2

index_to_i_j_colwise_nodiag <- function(k, n) {
  kp <- n * (n - 1) / 2 - k
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- n - (kp - p * (p + 1) / 2)
  j  <- n - 1 - p
  c(i, j)
}

cov_plots <- list()
for (ind in 1:n_lower_tri) {
  mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
  p <- mat_ind[1]
  q <- mat_ind[2]
  glm.df <- data.frame(x = glm_params[q], y = glm_params[p])

  # cov_plot <- ggplot(rvgal.df, aes_string(x = param_names[p], y = param_names[q])) +
  cov_plot <- ggplot(rvgal.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = rvgal.df, col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "blue", type = "norm", lwd = 1) +
    geom_point(data = glm.df, aes(x = x, y = y),
               shape = 4, color = "black", size = 4) +
    theme_bw() +
    theme(axis.title = element_blank(), axis.text = element_text(size = 18)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 2)) 
  
  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
n_cov_plots <- param_dim * (param_dim-1)/2
m[lower.tri(m, diag = F)] <- 1:n_cov_plots
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:param_dim, l = 1:param_dim)
# gtable_show_layout(gr3)

# A list of text grobs - the labels
fixed_vars <- lapply(1:n_fixed_effects, function(p) textGrob(bquote(beta[.(fixed_ef_names[p])])))
random_vars <- list(textGrob(bquote(Sigma[alpha[11]])), textGrob(bquote(Sigma[alpha[21]])),
             textGrob(bquote(Sigma[alpha[22]])))
vars <- c(fixed_vars, random_vars)
vars <- lapply(vars, editGrob, gp = gpar(col = "black", fontsize = 20))

# So that there is space for the labels,
# add a row to the top of the gtable,
# and a column to the left of the gtable.
gp <- gtable_add_cols(gr3, unit(1.5, "lines"), 0)
gp <- gtable_add_rows(gp, unit(1.5, "lines"), -1) #0 adds on the top

# gtable_show_layout(gp)

# Add the label grobs.
# The labels on the left should be rotated; hence the edit.
# t and l refer to cells in the gtable layout.
# gtable_show_layout(gp) shows the layout.
gp <- gtable_add_grob(gp, lapply(vars[1:param_dim], editGrob, rot = 90), t = 1:param_dim, l = 1) # add column names to column 1, rows 2:9
gp <- gtable_add_grob(gp, vars[1:param_dim], t = param_dim+1, l = 2:(param_dim+1)) # add row names to row 6, columns 1:9

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("epilepsy_posterior", temper_info, reorder_info,
                      "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1000, height = 700)
  grid.newpage()
  grid.draw(gp)
  dev.off()
} 

## Time benchmark
hmc.time <- sum(colSums(hmc_results$time)) # sum over all chains
rvga.time <- rvgal_results$time_elapsed
cat("HMC time:", hmc.time, ", R-VGAL time:", rvga.time[3], "\n")

## Trajectory for R-VGA
if (plot_trace) {

  ## R-VGAL trajectories
  trajectories <- list()
  for (p in 1:n_fixed_effects) {
    # if (p > n_fixed_effects && p <= (n_fixed_effects + n_random_effects)) {
    #   trajectories[[p]] <- sapply(1:N, function(x) exp(rvgal_results$mu[[x]][p])^2)
    # } else {
      trajectories[[p]] <- sapply(1:N, function(x) rvgal_results$mu[[x]][p])
    # }
  }

  zeta_trajectories <- lapply(1:N, function(x) rvgal_results$mu[[x]][-(1:n_fixed_effects)])
  Sigma_trajectories <- lapply(zeta_trajectories, construct_Sigma,
                            d = n_random_effects)
  nlower <- n_random_effects * (n_random_effects-1)/2 + n_random_effects
  lower_ind <- lapply(1:nlower, index_to_i_j_rowwise_diag)
  for (p in 1:nlower) {
    inds <- lower_ind[[p]]
    trajectories[[n_fixed_effects+p]] <- unlist(lapply(Sigma_trajectories, function(Sigma) Sigma[inds[1], inds[2]]))
  }                          

  traceplot_file <- paste0("rvgal_trace", temper_info, reorder_info,
                      "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath <- paste0("./plots/", traceplot_file)
  png(filepath, width = 1000, height = 600)
  par(mfrow = c(2, ceiling(param_dim/2)))
  
  margins <- c(-0.8, 0.8)
  for (p in 1:param_dim) {
    plot(trajectories[[p]], main = param_names[p], 
    ylim = glm_params[p] + margins, type = "l")
    abline(h = glm_params[p], lty = 2, col = "red")
  }
  dev.off()

  ## HMC traceplots
  hmc_traceplot_file <- paste0("hmc_trace_", date, ".png")
  hmc_trace_filepath <- paste0("./plots/", hmc_traceplot_file)
  png(hmc_trace_filepath, width = 1000, height = 600)
  
  par(mfrow = c(ceiling(param_dim/2), 2))
  for (p in 1:param_dim) {
    plot(hmc.samples[,p], type = "l", main = param_names[p])
    # abline(h = true_vals[p], lty = 2, col = "red")
  }
  dev.off()
}


# gc()
# tf$keras$backend$clear_session()