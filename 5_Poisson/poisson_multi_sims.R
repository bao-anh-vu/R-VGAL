setwd("/home/babv971/R-VGAL/5_Poisson/")
## Structure:
# 1. Generate data
# 2. Run R-VGAL algorithm
# 3. Run HMC
# 4. Plot results

rm(list=ls())

# reticulate::use_condaenv("tf2.11", required = TRUE)
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
# tfp <- import("tensorflow_probability")
# tfd <- tfp$distributions

source("./source/run_rvgal.R")
source("./source/run_stan_poisson.R")
source("./source/generate_data.R")
# source("./source/compute_joint_llh_tf.R")
# source("./source/compute_grad_hessian_all.R")
source("./source/compute_grad_hessian_theoretical.R")
source("./source/run_multi_sims_hmc.R")

## Flags
date <- "20231018" #"20231018" has 2 fixed effects, ""20231030" has 4    
regenerate_data <- T
rerun_rvgal_sims <- T
rerun_hmc_sims <- T
save_datasets <- T
save_rvgal_sim_results <- T
save_hmc_sim_results <- T
plot_prior <- F
save_plots <- F
reorder_data <- F
use_tempering <- T

plot_trace <- F

if (use_tempering) {
  n_obs_to_temper <- 10
  K <- 4
  a_vals_temper <- rep(1/K, K)
}

n_post_samples <- 2000#0
n_random_effects <- 2

datasets <- list()
nsims <- 100

## Generate data
N <- 200L #number of individuals
n <- 10L # number of responses per individual
beta <- c(-0.75, 1.25) #c(-0.75, -0.25, 0.25, 0.5) 
n_fixed_effects <- length(beta)
param_dim <- n_fixed_effects + n_random_effects * (n_random_effects+1)/2

Sigma_alpha <- 0
nlower <- n_random_effects + n_random_effects * (n_random_effects-1)/2

if (grepl("_0", date)) {
  Sigma_alpha <- 0.1*diag(1:n_random_effects)
} else {
  L <- matrix(0, n_random_effects, n_random_effects)
  L[lower.tri(L, diag = T)] <- runif(nlower, 0, 1)
  Sigma_alpha <- tcrossprod(L)
  Sigma_alpha <- Sigma_alpha + 0.1*diag(1:n_random_effects)
}

if (regenerate_data) {
  for (sim in 1:nsims) {
    datasets[[sim]] <- generate_data(N = N, n = n, beta = beta, 
                                     Sigma_alpha = Sigma_alpha)
    if (save_datasets) {
      saveRDS(datasets[[sim]], file = paste0("./multi_sims/data/poisson_data_N", N, "_n", n, "_", date, "_",
                                             formatC(sim, width=3, flag="0"), ".rds"))
    }
  }
} else {
  for (sim in 1:nsims) {
    datasets[[sim]] <- readRDS(file = paste0("./multi_sims/data/poisson_data_N", N, "_n", n, "_", date, "_",
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
result_directory <- "./multi_sims/results/"

####################
##     R-VGAL     ##
####################

S <- 200L
S_alpha <- 200L

rvgal_sim_results <- list()

for (sim in 1:nsims) {
  
  rvgal.result_file <- paste0("poisson_rvgal", temper_info, reorder_info,
                              "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, "_",
                              formatC(sim, width=3, flag="0"), ".rds")
  
  if (rerun_rvgal_sims) {
    
      cat("Sim", sim, "in progress... \n")
      
      poisson_data <- datasets[[sim]]
      
      y <- poisson_data$y
      X <- poisson_data$X
      Z <- poisson_data$Z
      
      beta <- poisson_data$beta
      Sigma_alpha <- poisson_data$Sigma_alpha
      
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
      
      ## Initialise variational parameters
      n_fixed_effects <- as.integer(ncol(X[[1]]))
      n_random_effects <- as.integer(ncol(Z[[1]]))
      n_elements_L <- n_random_effects + n_random_effects * (n_random_effects - 1)/2
      param_dim <- n_fixed_effects + n_elements_L
      
      beta_0 <- rep(0, n_fixed_effects)
      l_vec_0 <- c(rep(0, n_random_effects), rep(0, n_random_effects * (n_random_effects - 1)/2))
      mu_0 <- c(beta_0, l_vec_0)
      P_0 <- diag(c(rep(1, n_fixed_effects), rep(0.1, n_elements_L)))
      
      
      rvgal_sim_results[[sim]] <- run_rvgal(y, X, Z, mu_0, P_0, 
                                            S = S, S_alpha = S_alpha,
                                            n_post_samples = n_post_samples,
                                            use_tempering = use_tempering, 
                                            n_temper = n_obs_to_temper, 
                                            temper_schedule = a_vals_temper,
                                            use_tf = T)
      
      if (save_rvgal_sim_results) {
        saveRDS(rvgal_sim_results[[sim]], file = paste0(result_directory, rvgal.result_file))
      } 
      
  } else {
      rvgal_sim_results[[sim]] <- readRDS(file = paste0(result_directory, rvgal.result_file))
  }
}


## Run HMC simulations
hmc_sim_results <- list()
burn_in <- 500#0
n_chains <- 2

if (rerun_hmc_sims) {
  
  sims <- 1:nsims
  parallel::mclapply(sims, run_multi_sims_hmc, mc.cores = 10L,
                     n_post_samples = n_post_samples,
                     burn_in = burn_in)
  # lapply(sims, run_multi_sims_hmc, n_post_samples = n_post_samples,
  #        burn_in = burn_in) #, mc.cores = 10L)
  
} else {
  
  for (sim in 1:nsims) {
    hmc.result_file <- paste0(result_directory, 
                              "poisson_hmc_N", N, "_n", n, "_", date, "_",
                              formatC(sim, width=3, flag="0"), ".rds")
    
    hmc_sim_results[[sim]] <- readRDS(file = hmc.result_file)
  }
  
}

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

param_names <- c("beta[1]", "beta[2]", "sigma_alpha[11]", "sigma_alpha[21]", "sigma_alpha[22]")  
true_vals <- c(beta, c(Sigma_alpha[t(lower.tri(Sigma_alpha, diag = T))]))

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

means_df <- data.frame(rvgal_mean = c(rvgal.means), hmc_mean = c(hmc.means))
means_df$param <- rep(param_names, each = nsims)
means_df$true_vals <- rep(true_vals, each = nsims)
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
  facet_wrap(~param, scales = "free", labeller=label_parsed)
print(plot_means)

if (save_plots) {
  filename = paste0("poisson_multi_sim_means_", date, ".png")
  filepath = paste0("./multi_sims/plots/", filename)
  
  png(filepath, width = 1000, height = 500)
  print(plot_means)
  dev.off()
}