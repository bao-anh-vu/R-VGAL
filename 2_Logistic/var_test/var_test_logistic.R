## Test the variance of R-VGAL results on simulated logistic mixed model data ##

## Will take a while to run depending on how many R-VGAL runs are specified
## (change the "runs" parameter as needed)
## If only looking for saved results, set rerun_test <- FALSE

rm(list=ls())

# reticulate::use_condaenv("tf2.11", required = TRUE)
library("readxl") # part of tidyverse
library("dplyr")
library("tensorflow")
library("mvtnorm")
library("rstan")
library("gridExtra")
library("grid")
library("reshape2")

source("./source/run_rvgal.R")
source("./source/run_stan_logmm.R")

## Flags
date <- "20230329" 
regenerate_data <- F
save_data <- F
rerun_test <- T
rerun_stan <- F
save_results <- F
save_hmc_results <- F
reorder_data <- F
use_tempering <- T
save_images <- F

runs <- 10 # number of R-VGAL runs
S <- 100L
S_alpha <- 100L

if (reorder_data) {
  reorder_seed <- 2023
}

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

## Generate data
N <- 500L #number of individuals
n <- 10L # number of responses per individual
beta <- c(-1.5, 1.5, 0.5, 0.25) # c(-2, 1, param_dim, -4)  #
n_fixed_effects <- length(beta)
tau <- 0.9
X <- list()
p <- matrix(NA, nrow = N, ncol = n)
y <- list()

if (regenerate_data) {
  # set.seed(2023)
  for (i in 1:N) {
    X[[i]] <- matrix(rnorm(n * n_fixed_effects), nrow = n, ncol = n_fixed_effects)
    alpha_i <- rnorm(1, 0, tau)
    
    p_i <- exp(X[[i]] %*% beta + alpha_i) / (1 + exp(X[[i]] %*% beta + alpha_i))
    y[[i]] <- rbinom(n, 1, p_i)
  }
  
  logistic_data <- list(y = y, X = X, beta = beta, tau = tau, N = N, n = n)
  
  if (save_data) {
    saveRDS(logistic_data, file = paste0("./data/logistic_data_N", N, "_n", n, "_", date, ".rds"))
  }
  
} else {
  logistic_data <- readRDS(file = paste0("./data/logistic_data_N", N, "_n", n, "_", date, ".rds"))
  
  y <- logistic_data$y
  X <- logistic_data$X
  beta <- logistic_data$beta
  tau <- logistic_data$tau
}

###################
##     R-VGA     ##
###################

n_fixed_effects <- as.integer(ncol(X[[1]]))
param_dim <- n_fixed_effects + 1L

## File name and path for saving results
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

result_directory <- paste0("./var_test/results/")
results_file <- paste0("var_test", temper_info, reorder_info,
                       "_S", S, "_Sa", S_alpha, "_", date, ".rds")
results_filepath <- paste0(result_directory, results_file)

if (rerun_test) {
  
  r <- 1
  results <- list()
  
  while (r <= runs) {
    
    if (reorder_data) { # only reorder once!
      set.seed(reorder_seed) 
      reordered_ind <- sample(1:length(y))
      print(head(reordered_ind))
      reordered_y <- lapply(reordered_ind, function(i) y[[i]])
      reordered_X <- lapply(reordered_ind, function(i) X[[i]])
      
      y <- reordered_y
      X <- reordered_X
    }
    
    ## 1. Initialise the variational mean and covariance
    beta_0 <- rep(0, param_dim - 1L)
    tau_0 <- 1#0.5 #0.7
    omega_0 <- log(tau_0^2) ## Maybe I was starting with too "good" initial values before?
    
    mu_0 <- c(beta_0, omega_0)
    P_0 <- diag(c(rep(10, n_fixed_effects), 1))
    
    ## Sample from the "prior"
    par(mfrow = c(1, 1))
    test_omega <- rnorm(10000, mu_0[param_dim], sqrt(P_0[param_dim, param_dim]))
    plot(density(exp(test_omega)), main = "RVGA: Prior of tau^2")
     
    # plot(density(sqrt(exp(test_omega))), xlim = c(0,2), main = "RVGA: Prior of tau")
    # vals <- sort((exp(test_omega))) # tau^2
    # q025 <- vals[2501]
    # q975 <- vals[97501]
    
    # tau2 <- rlnorm(10000, mu_0[param_dim], sqrt(P_0[param_dim, param_dim]))
    # lines(density(tau2), col = "red")
    # print(qlnorm(c(0.025, 0.975), mu_0[param_dim], sqrt(P_0[param_dim, param_dim])))
    
    mu_vals <- lapply(1:N, function(x) mu_0)
    prec <- lapply(1:N, function(x) solve(P_0))
    
    try({
      
      rvga_results <- run_rvgal(y, X, mu_0, P_0, S = S, S_alpha = S_alpha,
                                use_tempering = use_tempering, 
                                n_temper = n_obs_to_temper, 
                                temper_schedule = a_vals_temper)
      
      mu_vals <- rvga_results$mu
      prec <- rvga_results$prec
      
      ## Save posterior mean and variance 
      result <- list(mu = mu_vals[[N+1]], prec = prec[[N+1]])
      results[[r]] <- result
      
      r <- r + 1
      
    })
    
  }
  
  if (save_results) {
    saveRDS(results, file = results_filepath)
  }
  
} else {
  results <- readRDS(file = results_filepath)
}


##########
## STAN ##
##########
n_post_samples <- 10000
burn_in <- 5000

if (rerun_stan) {
  hmc.t1 <- proc.time()
  
  ## Data manipulation ##
  y_long <- unlist(y) #as.vector(t(y))
  X_long <- do.call("rbind", X)
  
  hfit <- run_stan_logmm(iters = n_post_samples + burn_in, burn_in = burn_in,
                         data = y_long, grouping = rep(1:N, each = n), 
                         n_groups = N, fixed_covariates = X_long)
  
  if (save_hmc_results) {
    saveRDS(hfit, file = paste0("./results/logistic_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
} else {
  hfit <- readRDS(file = paste0("./results/logistic_mm_hmc_N", N, "_n", n, "_", date, ".rds")) # for the experiements on starting points
}

######################## Results #########################

## HMC results for comparison
hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]","omega"),
                   permuted = F)
hmc.samples <- matrix(NA, dim(hmc.fit)[1], param_dim)
for (p in 1:(param_dim-1)) {
  hmc.samples[, p] <- hmc.fit[, , p]
}
hmc.samples[, (param_dim-1)+1] <- sqrt(exp(hmc.fit[, , param_dim])) # transform omega samples to tau samples
# hmc.df <- data.frame(hmc.samples)

## R-VGA results
n_post_samples <- 10000
post_samples_list <- list()
for (r in 1:length(results)) {
  post_mu <- results[[r]]$mu
  post_var <- chol2inv(chol(results[[r]]$prec))
  
  post_samples_list[[r]] <- rmvnorm(n_post_samples, post_mu, post_var)
}

subscripts <- c("1", "2", "3", "4")
param_values <- c(beta, tau)

param_plots <- list()
for (p in 1:param_dim) {
  
  hmc.df <- data.frame(samples = hmc.samples[, p])
  param_df <- data.frame(x = param_values[p])
  
  if (p == param_dim) { # if the parameter is tau
    post_samples_p <- lapply(post_samples_list, function(x) sqrt(exp(x[, p])))
  } else {
    post_samples_p <- lapply(post_samples_list, function(x) x[, p])
    
  }
  
  post_samples_df <- as.data.frame(post_samples_p, 
                                   col.names = 1:length(post_samples_list))
  post_samples_df$id <- 1:n_post_samples
  post_samples_df_long <- melt(post_samples_df, id.vars = 'id', variable.name = 'run')
  
  if (p == param_dim) { ## if the parameter is tau
    plot <- ggplot(post_samples_df_long, aes(x = value)) + #geom_line(aes(colour = series))
      geom_density(aes(col = run), linewidth = 1) +
      geom_density(data = hmc.df, aes(x = samples), col = "black", linewidth = 1) +
      geom_vline(data = param_df, aes(xintercept=x),
                 color="black", linetype="dashed", linewidth=0.75) +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = bquote(tau)) + 
      theme(text = element_text(size = 20)) 
  } else {
    plot <- ggplot(post_samples_df_long, aes(x = value)) + #geom_line(aes(colour = series))
      geom_density(aes(col = run), linewidth = 1) +
      geom_density(data = hmc.df, aes(x = samples), col = "black", linewidth = 1) +
      geom_vline(data = param_df, aes(xintercept=x),
                 color="black", linetype="dashed", linewidth=0.75) +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = bquote(beta[.(subscripts[p])])) +
      theme(text = element_text(size = 20)) 
  }
  
  param_plots[[p]] <- plot
}

grid.arrange(grobs = param_plots, nrow = 1, ncol = 5)

## Saving the plots
if (save_images) {
  plot_directory <- paste0("./plots/var_test_", date, "/")
  plot_file = paste0("logistic_var_test", temper_info, reorder_info,
                    "_S", S, "_Sa", S_alpha, "_", date, ".png")
  
  png(paste0(plot_directory, plot_file), width = 1500, height = 250)
  grid.arrange(grobs = param_plots, nrow = 1, ncol = 5)
  
  dev.off()
  
} 

