setwd("~/R-VGAL/2_Logistic/")

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
library("ggplot2")
library("grid")
library("gtable")
library("gridExtra")
library("reshape2")

source("./source/generate_data.R")
# source("./source/run_rvgal.R")
source("./source/run_stan_logmm.R")

date <- "20231017"
regenerate_data <- F
rerun_rvgal_sims <- F
rerun_hmc_sims <- F
reorder_data <- F
use_tempering <- T

save_datasets <- F
save_rvgal_sim_results <- F
save_hmc_sim_results <- F
save_plots <- F

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

nsims <- 100
N <- 500L #number of individuals
n <- 10L # number of responses per individual
S <- 100L
S_alpha <- 100L
n_post_samples <- 20000

datasets <- list()

## 1. Generate data
### True parameters
beta <- c(-1.5, 1.5, 0.5, 0.25) 
tau <- 0.9

if (regenerate_data) {
  
  for (sim in 1:nsims) {
    datasets[[sim]] <- generate_data(N = N, n = n, beta = beta, tau = tau,
                                    date = date)
    
    if (save_datasets) {
      saveRDS(datasets[[sim]], file = paste0("./data/multi_sims/logistic_data_N", N, 
                                             "_n", n, "_", date, "_",
                                             formatC(sim, width=3, flag="0"), ".rds"))
    }
  }
  
} else {
  for (sim in 1:nsims) {
    datasets[[sim]] <- readRDS(file = paste0("./data/multi_sims/logistic_data_N", N, 
                                             "_n", n, "_", date, "_",
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

# result_file <- paste0("logistic_rvgal_multi", temper_info, reorder_info, 
#                       "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, ".rds")

## Initialise the variational mean and covariance
# param_dim <- as.integer(length(beta) + 2) # theta = (beta, log(sigma_a), log(sigma_e))'
# beta_0 <- rep(0, length(beta))  
# sigma_a_0 <- 0.5
# sigma_e_0 <- 0.5
# phi_0 <- log(sigma_a_0^2)
# psi_0 <- log(sigma_e_0^2)
# mu_0 <- c(beta_0, phi_0, psi_0)
# var_beta_0 <- rep(10, length(beta))
# var_phi_0 <- 1
# var_psi_0 <- 1
# P_0 <- diag(c(var_beta_0, var_phi_0, var_psi_0), param_dim)

## Initialise variational parameters
n_fixed_effects <- as.integer(length(beta))
param_dim <- n_fixed_effects + 1L

beta_0 <- rep(0, param_dim - 1L)
omega_0 <- log(0.5^2) 

mu_0 <- c(beta_0, omega_0)
P_0 <- diag(c(rep(10, n_fixed_effects), 1))

rvgal_sim_results <- list()

if (rerun_rvgal_sims) {
  
  for (sim in 1:nsims) {
    rvgal.result_file <- paste0("logistic_rvgal", temper_info, reorder_info,
                                "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, "_",
                                formatC(sim, width=3, flag="0"), ".rds")
    
    
    logistic_data <- datasets[[sim]]
    
    y <- logistic_data$y
    X <- logistic_data$X
    beta <- logistic_data$beta
    tau <- logistic_data$tau
    
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
    
    rvgal_sim_results[[sim]] <- run_rvgal(y, X, mu_0, P_0, S = S, S_alpha = S_alpha,
                                          n_post_samples = n_post_samples,
                                          use_tempering = use_tempering, 
                                          n_temper = n_obs_to_temper, 
                                          temper_schedule = a_vals_temper)
    
    if (save_rvgal_sim_results) {
      saveRDS(rvgal_sim_results[[sim]], file = paste0(result_directory, rvgal.result_file))
    }
    
  }
  
} else {
  
  for (sim in 1:nsims) {
    rvgal.result_file <- paste0("logistic_rvgal", temper_info, reorder_info,
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

## 4. Run HMC 
burn_in <- 5000
n_chains <- 2
hmc.iters <- n_post_samples/n_chains + burn_in

hmc_sim_results <- list()
if (rerun_hmc_sims) {
  
  for (sim in 1:nsims) {
    cat("Sim", sim, "in progress... \n")
    
    logistic_data <- datasets[[sim]]
    
    y_long <- unlist(logistic_data$y) #as.vector(t(y))
    X_long <- do.call("rbind", logistic_data$X)
    beta <- logistic_data$beta
    tau <- logistic_data$tau
    
    hmc.result_file <- paste0(result_directory, 
                              "logistic_hmc_N", N, "_n", n, "_", date, "_",
                              formatC(sim, width=3, flag="0"), ".rds")
    
    hmc_results <- run_stan_logmm(iters = hmc.iters, burn_in = burn_in, 
                           n_chains = n_chains, data = y_long, 
                           grouping = rep(1:N, each = n), n_groups = N,
                           fixed_covariates = X_long)
    
    hmc.fit <- hmc_results$post_samples[-(1:burn_in),,]
    
    hmc_sim_results[[sim]] <- hmc.fit
    
    if (save_hmc_sim_results) {
      saveRDS(hmc_sim_results[[sim]], file = hmc.result_file)
    }
  }
  
  
} else {
  
  for (sim in 1:nsims) {
    hmc.result_file <- paste0(result_directory, 
                              "logistic_hmc_N", N, "_n", n, "_", date, "_",
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
  
  rvgal.samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
  hmc.samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
  for (p in 1:param_dim) {
    
    ## Extract R-VGAL samples
    if (p == param_dim) { # if the parameters are variance parameters
      rvgal.samples[, p] <- sqrt(exp(rvgal_sim_results[[sim]]$post_samples[, p]))
      hmc.samples[, p] <- sqrt(exp(hmc_sim_results[[sim]][, , p]))
    } else {
      rvgal.samples[, p] <- rvgal_sim_results[[sim]]$post_samples[, p]
      hmc.samples[, p] <- hmc_sim_results[[sim]][, , p]
    }
      
    ## Extract HMC samples
    # if (length(dim(hmc_sim_results[[1]])) < 3) {
    #   hmc.samples[, p] <- hmc_sim_results[[sim]][, p]
    # } else {
    #   hmc.samples[, p] <- hmc_sim_results[[sim]][, , p]
    # }
  }
  
  rvgal.sim_results_list[[sim]] <- rvgal.samples
  hmc.sim_results_list[[sim]] <- hmc.samples
  
}

subscripts <- c("1", "2", "3", "4")
param_values <- c(beta, tau)

param_plots <- list()
for (p in 1:param_dim) {

  # hmc.df <- data.frame(samples = hmc.samples[, p])
  param_df <- data.frame(x = param_values[p])

  # if (p == param_dim) { # if the parameter is sigma_a or sigma_e
    rvgal.post_samples_p <- lapply(rvgal.sim_results_list, function(x) x[, p])
    hmc.post_samples_p <- lapply(hmc.sim_results_list, function(x) x[, p])
    # hmc.samples[, p] <- hmc.fit[, , p]
  # }

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

  if (p == param_dim) { # the parameter is sigma_epsilon
    plot <- ggplot(rvgal.post_samples_df_long, aes(x = value, group = run)) + #geom_line(aes(colour = series))
      # geom_density(aes(col = method)) +
      geom_density(colour = "salmon") +
      # scale_color_brewer(palette="Reds") +
      geom_density(data = hmc.post_samples_df_long, aes(x = value, group = run)) +
      # scale_color_brewer(palette="Blues") +
      geom_vline(data = param_df, aes(xintercept=x),
                 color="black", linetype="dashed", linewidth=0.75) +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = expression(sigma[epsilon]))
  } else {
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
  }

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

## Obtain the means for each simulation
rvgal.means <- matrix(nrow = nsims, ncol = param_dim)
rvgal.sds <- matrix(nrow = nsims, ncol = param_dim)
hmc.means <- matrix(nrow = nsims, ncol = param_dim)
hmc.sds <- matrix(nrow = nsims, ncol = param_dim)

rvgal.means_allparams <- lapply(rvgal.sim_results_list, colMeans) # parameter means for each sim
rvgal.sds_allparams <- lapply(rvgal.sim_results_list, function(x) apply(x, 2, sd)) # parameter means for each sim

hmc.means_allparams <- lapply(hmc.sim_results_list, colMeans) # parameter means for each sim
hmc.sds_allparams <- lapply(hmc.sim_results_list, function(x) apply(x, 2, sd)) # parameter means for each sim

par(mfrow = c(2, 3))
for (p in 1:param_dim) {
  # rvgal.means[, p] <- sapply(1:nsims, function(sim) rvgal_sim_results[[sim]]$mu[[N+1]][p]) 
  
  rvgal.means[, p] <- sapply(rvgal.means_allparams, function(x) x[p])
  hmc.means[, p] <- sapply(hmc.means_allparams, function(x) x[p])
  
  # rvgal.sds[, p] <- sqrt(1/rvgal.precs)
  rvgal.sds[, p] <- sapply(rvgal.sds_allparams, function(x) x[p])
  hmc.sds[, p] <- sapply(hmc.sds_allparams, function(x) x[p])
  
  # plot(hmc.means[, p], rvgal.means[, p])
  # abline(0, 1, lty = 2)
}
 
param_names <- c("beta_1", "beta_2", "beta_3", "beta_4", "tau")
means_df <- data.frame(rvgal_mean = c(rvgal.means), hmc_mean = c(hmc.means))
means_df$param <- rep(param_names, each = nsims)
means_df$true_vals <- rep(c(beta, tau), each = nsims)
means_df$rvgal_diff <- means_df$rvgal_mean - means_df$true_vals
means_df$hmc_diff <- means_df$hmc_mean - means_df$true_vals


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
              labs(x = "Simulation", y = "SD ratio") +
              theme_bw() +
              facet_wrap(~param, scales = "free")
print(plot_sds)

## Do a plot here of R-VGAL mean - true parameter and HMC mean - true param
plot_means_diff <- ggplot(means_df, aes(hmc_diff, rvgal_diff)) + geom_point() +
  geom_abline(linetype = 2) +
  labs(x = "Difference between HMC mean and true parameter", 
       y = "Difference between R-VGAL mean and true parameter") +
  theme_bw() +
  facet_wrap(~param, scales = "free")
print(plot_means_diff)
