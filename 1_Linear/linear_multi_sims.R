setwd("~/R-VGAL/1_Linear/")

## Structure of the code:
## 1. Regenerate data
## 2. Run R-VGAL with estimated gradients/Hessians
## 3. Run R-VGAL with theoretical gradients/Hessians
## 4. Run HMC 

rm(list = ls())

# library("tensorflow")
# 
# # List physical devices
# gpus <- tf$config$experimental$list_physical_devices('GPU')
# 
# if (length(gpus) > 0) {
#   tryCatch({
#     # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
#     tf$config$experimental$set_virtual_device_configuration(
#       gpus[[1]],
#       list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=4096))
#     )
#     
#     logical_gpus <- tf$config$experimental$list_logical_devices('GPU')
#     
#     print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
#   }, error = function(e) {
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
#   })
# }

library("mvtnorm")
library("Matrix")
library("rstan")
rstan_options(auto_write = TRUE)
library("ggplot2")
library("grid")
library("gtable")
library("gridExtra")
library("reshape2")

source("./source/generate_data.R")
source("./source/run_est_rvgal.R")
source("./source/run_exact_rvgal.R")
# source("./source/run_finite_difference.R") # to calculate numerical gradients/Hessians for comparison with theoretical ones
source("./source/run_stan_lmm.R")

date <- "20230329"
regenerate_data <- F
rerun_rvgal_sims <- F
rerun_hmc_sims <- T
reorder_data <- F
use_tempering <- T

save_datasets <- F
save_rvgal_sim_results <- F
save_hmc_sim_results <- T
save_plots <- F

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

nsims <- 100
N <- 200L
n <- 10L
S <- 100L
S_alpha <- 100L
n_post_samples <- 20000

datasets <- list()

## 1. Generate data
### True parameters
sigma_a <- 0.9
sigma_e <- 0.7
beta <- c(-1.5, 1.5, 0.5, 0.25) 

if (regenerate_data) {
  
  for (sim in 1:nsims) {
    datasets[[sim]] <- generate_data(beta, sigma_a, sigma_e)
    
    if (save_datasets) {
      saveRDS(datasets[[sim]], file = paste0("./data/multi_sims/linear_data_N", N, "_n", n, "_", date, "_",
                                      formatC(sim, width=3, flag="0"), ".rds"))
    }
  }
  
} else {
  for (sim in 1:nsims) {
    datasets[[sim]] <- readRDS(file = paste0("./data/multi_sims/linear_data_N", N, "_n", n, "_", date, "_",
                                      formatC(sim, width=3, flag="0"), ".rds"))
  }
}

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
result_directory <- "./results/multi_sims/"

# result_file <- paste0("linear_rvgal_multi", temper_info, reorder_info, 
#                       "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, ".rds")

## Initialise the variational mean and covariance
param_dim <- as.integer(length(beta) + 2) # theta = (beta, log(sigma_a), log(sigma_e))'
beta_0 <- rep(0, length(beta))  
sigma_a_0 <- 0.5
sigma_e_0 <- 0.5
phi_0 <- log(sigma_a_0^2)
psi_0 <- log(sigma_e_0^2)
mu_0 <- c(beta_0, phi_0, psi_0)
var_beta_0 <- rep(10, length(beta))
var_phi_0 <- 1
var_psi_0 <- 1
P_0 <- diag(c(var_beta_0, var_phi_0, var_psi_0), param_dim)

rvgal_sim_results <- list()

if (rerun_rvgal_sims) {
  
  for (sim in 1:nsims) {
    cat("Sim", sim, "in progress... \n")
    
    rvgal.result_file <- paste0("linear_rvgal", temper_info, reorder_info,
                          "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, "_",
                          formatC(sim, width=3, flag="0"), ".rds")

    
    linear_data <- datasets[[sim]]
    
    y <- linear_data$y
    X <- linear_data$X
    Z <- linear_data$Z
    beta <- linear_data$beta
    sigma_a <- linear_data$sigma_a
    sigma_e <- linear_data$sigma_e
    
    ## Reorder data if needed
    if (reorder_data) {
      set.seed(reorder_seed) 
      reordered_ind <- sample(1:length(y))
      print(head(reordered_ind))
      reordered_y <- lapply(reordered_ind, function(i) y[[i]])
      reordered_X <- lapply(reordered_ind, function(i) X[[i]])
      
      y <- reordered_y
      X <- reordered_X
    }
    
    rvgal_sim_results[[sim]] <- run_est_rvgal(y = y, X = X, Z = Z, 
                                 mu_0 = mu_0, P_0 = P_0,
                                 S = S, S_alpha = S_alpha,
                                 n_temper = n_obs_to_temper,
                                 n_post_samples = n_post_samples,
                                 use_tempering = use_tempering)
    
    if (save_rvgal_sim_results) {
      saveRDS(rvgal_sim_results[[sim]], file = paste0(result_directory, rvgal.result_file))
    }
    
  }
  
} else {
  
  for (sim in 1:nsims) {
    rvgal.result_file <- paste0("linear_rvgal", temper_info, reorder_info,
                                "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, "_",
                                formatC(sim, width=3, flag="0"), ".rds")
    
    rvgal_sim_results[[sim]] <- readRDS(file = paste0(result_directory, rvgal.result_file))
  }
  
}

# ## Sample to see if prior mean and variance are reasonable
# samples <- rmvnorm(10000, mu_0, P_0)
# sigma_a_samples <- sqrt(exp(samples[, 5]))
# sigma_e_samples <- sqrt(exp(samples[, 6]))
# par(mfrow = c(1, 2))
# plot(density(sigma_a_samples), main = "Prior density of sigma_a")
# plot(density(sigma_e_samples), main = "Prior density of sigma_e")

# browser()


## 4. Run HMC 
burn_in <- 5000
n_chains <- 2
hmc.iters <- n_post_samples/n_chains + burn_in

hmc_sim_results <- list()
if (rerun_hmc_sims) {
  
  for (sim in 8:nsims) {
    cat("Sim", sim, "in progress... \n")
    
    linear_data <- datasets[[sim]]
    
    y <- linear_data$y
    X <- linear_data$X
    Z <- linear_data$Z
    beta <- linear_data$beta
    sigma_a <- linear_data$sigma_a
    sigma_e <- linear_data$sigma_e
    
    hmc.result_file <- paste0(result_directory, 
           "linear_hmc_N", N, "_n", n, "_", date, "_",
           formatC(sim, width=3, flag="0"), ".rds")
    
    hfit <- run_stan_lmm(data = y, fixed_covariates = X, 
                         random_covariates = Z,
                         iters = hmc.iters, burn_in = burn_in,
                         nchains = n_chains,
                         prior_mean = mu_0,
                         prior_var = P_0)
    
    # hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]", "phi", "psi"), 
    #                    permuted = F)
    
    hmc_sim_results[[sim]] <- hfit #hmc.fit
    
    if (save_hmc_sim_results) {
      saveRDS(hmc_sim_results[[sim]], file = hmc.result_file)
    }
  }
  
  
} else {
  
  for (sim in 1:nsims) {
    hmc.result_file <- paste0(result_directory, 
           "linear_hmc_N", N, "_n", n, "_", date, "_",
           formatC(sim, width=3, flag="0"), ".rds")
    
    hmc_sim_results[[sim]] <- readRDS(file = hmc.result_file)
  }
  
}

######################################
##              Results             ##
######################################

## R-VGA results
rvgal.sim_results_list <- list()
hmc.sim_results_list <- list()
for (sim in 1:nsims) {
  
  ## R-VGAL posterior samples
  # post_mu <- rvgal_sim_results[[sim]]$mu[[N+1]]
  # post_var <- chol2inv(chol(rvgal_sim_results[[sim]]$prec[[N+1]]))
  rvgal.sim_results_list[[sim]] <- rvgal_sim_results[[sim]]$post_samples  #rmvnorm(n_post_samples, post_mu, post_var)
  
  hmc.samples <- matrix(NA, n_post_samples, param_dim)
  for (p in 1:param_dim) {
    # if (p == (param_dim - 1) || p == param_dim) { # if the parameters are variance parameters
    #   hmc.samples[, p] <- sqrt(exp(hmc.fit[, , p]))
    # } else {
      hmc.samples[, p] <- hmc_sim_results[[sim]]$post_samples[-(1:burn_in), , p]
    # }
  }
  hmc.sim_results_list[[sim]] <- hmc.samples
}

subscripts <- c("1", "2", "3", "4")
param_values <- c(beta, sigma_a, sigma_e)

param_plots <- list()
for (p in 1:param_dim) {
  
  # hmc.df <- data.frame(samples = hmc.samples[, p])
  param_df <- data.frame(x = param_values[p])
  rvgal.post_samples_p <- lapply(rvgal.sim_results_list, function(x) x[, p])
  hmc.post_samples_p <- lapply(hmc.sim_results_list, function(x) x[, p])
  
  rvgal.post_samples_df <- as.data.frame(rvgal.post_samples_p, 
                                   col.names = 1:length(rvgal.sim_results_list))
  rvgal.post_samples_df$id <- 1:n_post_samples
  rvgal.post_samples_df_long <- melt(rvgal.post_samples_df, id.vars = 'id', variable.name = 'run')
  rvgal.post_samples_df_long$method <- rep("R-VGAL", nrow(rvgal.post_samples_df_long))
  
  hmc.post_samples_df <- as.data.frame(hmc.post_samples_p, 
                                        col.names = nsims+1:length(hmc.sim_results_list))
  hmc.post_samples_df$id <- 1:n_post_samples
  hmc.post_samples_df_long <- melt(hmc.post_samples_df, id.vars = 'id', variable.name = 'run')
  hmc.post_samples_df_long$method <- rep("HMC", nrow(hmc.post_samples_df_long))
  
  # if (p == param_dim - 1) { ## if the parameter is sigma_alpha
  #   plot <- ggplot(rvgal.post_samples_df_long, aes(x = value, group = run)) + #geom_line(aes(colour = series))
  #     # geom_density(aes(col = method)) +
  #     geom_density(colour = "salmon") +
  #     # scale_color_brewer(palette="Reds") +
  #     geom_density(data = hmc.post_samples_df_long, aes(x = value, group = run)) +
  #     # scale_color_brewer(palette="Blues") +
  #     geom_vline(data = param_df, aes(xintercept=x),
  #                color="black", linetype="dashed", linewidth=0.75) +
  #     theme_bw() +
  #     theme(legend.position="none") +
  #     labs(x = expression(sigma[alpha]))
  # } else if (p == param_dim) { # the parameter is sigma_epsilon
  #   plot <- ggplot(rvgal.post_samples_df_long, aes(x = value, group = run)) + #geom_line(aes(colour = series))
  #     # geom_density(aes(col = method)) +
  #     geom_density(colour = "salmon") +
  #     # scale_color_brewer(palette="Reds") +
  #     geom_density(data = hmc.post_samples_df_long, aes(x = value, group = run)) +
  #     # scale_color_brewer(palette="Blues") +
  #     geom_vline(data = param_df, aes(xintercept=x),
  #                color="black", linetype="dashed", linewidth=0.75) +
  #     theme_bw() +
  #     theme(legend.position="none") +
  #     labs(x = expression(sigma[epsilon]))
  # } else {
  #   plot <- ggplot(rvgal.post_samples_df_long, aes(x = value, group = run)) + #geom_line(aes(colour = series))
  #     # geom_density(aes(col = method)) +
  #     geom_density(colour = "salmon") + 
  #     scale_color_brewer(palette="Reds") +
  #     geom_density(data = hmc.post_samples_df_long, aes(x = value, group = run)) +
  #     # scale_color_brewer(palette="Blues") +
  #     geom_vline(data = param_df, aes(xintercept=x),
  #                color="black", linetype="dashed", linewidth=0.75) +
  #     theme_bw() +
  #     theme(legend.position="none") +
  #     labs(x = bquote(beta[.(subscripts[p])]))
  # }

  plot <- ggplot(rvgal.post_samples_df_long, aes(x = value, group = run)) + #geom_line(aes(colour = series))
    # geom_density(aes(col = method)) +
    geom_density(colour = "salmon") + 
    scale_color_brewer(palette="Reds") +
    geom_density(data = hmc.post_samples_df_long, aes(x = value, group = run)) +
    # scale_color_brewer(palette="Blues") +
    geom_vline(data = param_df, aes(xintercept=x),
               color="black", linetype="dashed", linewidth=0.75) +
    theme_bw() +
    theme(legend.position="none") +
    labs(x = bquote(beta[.(subscripts[p])]))
  
  param_plots[[p]] <- plot
}

## Saving the plots
if (save_plots) {
  
  filename = paste0("multi_sim", temper_info, reorder_info,
                    "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0(result_directory, filename)
  
  png(filepath, width = 700, height = 400)
  grid.arrange(grobs = param_plots, nrow = 2, ncol = 3)
  dev.off()
  
} else {
  grid.arrange(grobs = param_plots, nrow = 2, ncol = 3)
}

# my_col_scheme <- c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#7B8A7B",
# "#0B29D6", "#f781bf", "#999999", "black")

#################

## Obtain the means and standard deviations for each simulation
rvgal.means <- matrix(nrow = nsims, ncol = param_dim)
rvgal.sds <- matrix(nrow = nsims, ncol = param_dim)
hmc.means <- matrix(nrow = nsims, ncol = param_dim)
hmc.sds <- matrix(nrow = nsims, ncol = param_dim)

hmc.means_allparams <- lapply(hmc.sim_results_list, colMeans) # parameter means for each sim
hmc.sds_allparams <- lapply(hmc.sim_results_list, function(x) apply(x, 2, sd)) # parameter means for each sim

rvgal.means_allparams <- lapply(rvgal.sim_results_list, colMeans) # parameter means for each sim
rvgal.sds_allparams <- lapply(rvgal.sim_results_list, function(x) apply(x, 2, sd)) # parameter means for each sim


par(mfrow = c(2, 3))
for (p in 1:param_dim) {
  
  # if (p == param_dim || p == (param_dim - 1)) {
  #   rvgal.means[, p] <- sapply(1:nsims, function(sim) sqrt(exp(rvgal_sim_results[[sim]]$mu[[N+1]][p]))) 
  # } else {
  #   rvgal.means[, p] <- sapply(1:nsims, function(sim) rvgal_sim_results[[sim]]$mu[[N+1]][p]) 
  #   
  # }
  # rvgal.means[, p] <- sapply(1:nsims, function(sim) rvgal_sim_results[[sim]]$mu[[N+1]][p])
  rvgal.means[, p] <- sapply(rvgal.means_allparams, function(x) x[p])
  hmc.means[, p] <- sapply(hmc.means_allparams, function(x) x[p])
  
  ## Need to transform parameters back to their original scale here before doing sd()
  rvgal.sds[, p] <- sapply(rvgal.sds_allparams, function(x) x[p])
  hmc.sds[, p] <- sapply(hmc.sds_allparams, function(x) x[p])
  
  # plot(hmc.means[, p], rvgal.means[, p])
  # abline(0, 1, lty = 2)
}

param_names <- c("beta_1", "beta_2", "beta_3", "beta_4", "sigma_a", "sigma_e")
means_df <- data.frame(rvgal_mean = c(rvgal.means), hmc_mean = c(hmc.means))
means_df$param <- rep(param_names, each = nsims)

sds_df <- data.frame(rvgal_sd = c(rvgal.means), hmc_sd = c(hmc.means))
sds_df$ratio <- sds_df$rvgal_sd / sds_df$hmc_sd
sds_df$param <- rep(param_names, each = nsims)
sds_df$sim <- rep(1:nsims, param_dim)

## Plot of R-VGAL means against HMC means
plot_means <- ggplot(means_df, aes(hmc_mean, rvgal_mean)) + geom_point() +
  geom_abline(linetype = 2) +
  labs(x = "HMC means", y = "R-VGAL means") +
  theme_bw() +
  facet_wrap(~param, scales = "free")
print(plot_means)

## Plot of the variance ratio between R-VGAL and HMC
plot_sds <- ggplot(sds_df, aes(x = sim, y = ratio)) + 
  geom_point() +
  geom_abline(slope = 0, intercept = 1, linetype = 2) +
  labs(x = "Simulation", y = "Variance ratio") +
  theme_bw() +
  facet_wrap(~param, scales = "free")
print(plot_sds)

