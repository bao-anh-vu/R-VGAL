## R-VGA on a linear mixed model ##

## IMPORTANT: VARIABLE NAMES ##
## phi = phi_alpha in the paper
## psi = phi_epsilon in the paper

rm(list = ls())

reticulate::use_condaenv("tf2.11", required = TRUE)
library("tensorflow")
library("mvtnorm")
library("Matrix")
library("rstan")
library("car")
library("ggplot2")
library("grid")
library("gtable")
library("gridExtra")
library("reshape2")

source("./scripts/run_stan_lmm.R")
source("./scripts/run_est_rvgal.R")

regenerate_data <- F
rerun_test <- F
rerun_stan <- F
save_hmc_results <- F
save_results <- F # save variance test results
use_tempering <- T
reorder_data <- F
save_plots <- F

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

if (reorder_data) {
  reorder_seed <- 2024
}

S <- 100L
S_alpha <- 100L

runs <- 10 # number of R-VGAL runs

date <- "20230329" #"20230127"
N <- 200L
n <- 10L

## 1. Generate data
if (regenerate_data) {
  ## True parameters
  sigma_a <- 0.9
  sigma_e <- 0.7
  beta <- c(-1.5, 1.5, 0.5, 0.25) # runif(4, -3, 3)
  
  linear_data <- generate_data(beta, sigma_a, sigma_e, save_data = save_data, date)
} else {
  linear_data <- readRDS(file = paste0("./data/linear_data_N", N, "_n", n, "_", date, ".rds"))
}

y <- linear_data$y
X <- linear_data$X
Z <- linear_data$Z
beta <- linear_data$beta
sigma_a <- linear_data$sigma_a
sigma_e <- linear_data$sigma_e

if (reorder_data) { # only reorder once!
  set.seed(reorder_seed) 
  reordered_ind <- sample(1:length(y))
  print(head(reordered_ind))
  reordered_y <- lapply(reordered_ind, function(i) y[[i]])
  reordered_X <- lapply(reordered_ind, function(i) X[[i]])
  
  y <- reordered_y
  X <- reordered_X
}

##############################
##          R-VGAL          ##
##############################

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

result_directory <- paste0(getwd(), "/var_test/results/")
results_file <- paste0("var_test", temper_info, reorder_info,
                       "_S", S, "_Sa", S_alpha, "_", date, ".rds")
results_filepath <- paste0(result_directory, results_file)

r <- 1
results <- list()

param_dim <- as.integer(length(beta) + 2)

if (rerun_test) {
  
  ## 1. Initialise the variational mean and covariance
   # theta = (beta, log(sigma_a), log(sigma_e))'
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

  while (r <= runs) {
    
    rvgal_results <- run_est_rvgal(y = y, X = X, Z = Z, mu_0 = mu_0, P_0 = P_0,
                                   S = S, S_alpha = S_alpha,
                                   n_temper = n_obs_to_temper,
                                   n_post_samples = n_post_samples,
                                   use_tempering = use_tempering)
    
    result <- list(mu = rvgal_results$mu[[N+1]], prec = rvgal_results$prec[[N+1]])
    results[[r]] <- result

    r <- r+1
    
    if (save_results) {
      saveRDS(results, file = results_filepath)
    }
  } 
  
} else {
  results <- readRDS(file = results_filepath)
}

##########
## STAN ##
##########
hmc_iters <- 15000
n_post_samples <- 10000

if (rerun_stan) {
  
  hfit <- run_stan_lmm(data = y, fixed_covariates = X, 
                       random_covariates = Z,
                       iters = hmc_iters, burn_in = hmc_iters - n_post_samples)
  
  if (save_hmc_results) {
    saveRDS(hfit, file = paste0(result_directory, "linear_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
  
} else {
  hfit <- readRDS(file = paste0(result_directory, "linear_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
}

print(hfit, pars=c("beta[1]","beta[2]","beta[3]","beta[4]","phi", "psi"),
      probs = c(0.025, 0.50, 0.975), digits_summary = 3)

hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]", "phi", "psi"), 
                   permuted = F)

## Check trace plots
traceplot(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]","phi", "psi"), 
          inc_warmup = FALSE)

## Extract posterior samples
hmc.samples <- matrix(NA, n_post_samples, param_dim)
for (p in 1:(param_dim - 2)) {
  hmc.samples[, p] <- hmc.fit[, , p]
}
hmc.samples[, (param_dim - 2)+1] <- sqrt(exp(hmc.fit[, , (param_dim - 2)+1]))
hmc.samples[, (param_dim - 2)+2] <- sqrt(exp(hmc.fit[, , (param_dim - 2)+2]))

# ## Plot results ##
## R-VGA results
n_post_samples <- 10000
post_samples_list <- list()
for (r in 1:length(results)) {
  post_mu <- results[[r]]$mu
  post_var <- chol2inv(chol(results[[r]]$prec))
  
  post_samples_list[[r]] <- rmvnorm(n_post_samples, post_mu, post_var)
}

subscripts <- c("1", "2", "3", "4")
param_values <- c(beta, sigma_a, sigma_e)
 
param_plots <- list()
for (p in 1:param_dim) {
  
  hmc.df <- data.frame(samples = hmc.samples[, p])
  param_df <- data.frame(x = param_values[p])
  
  if (p == param_dim || p == (param_dim - 1)) { # if the parameter is sigma_a or sigma_e
    post_samples_p <- lapply(post_samples_list, function(x) sqrt(exp(x[, p])))
  } else {
    post_samples_p <- lapply(post_samples_list, function(x) x[, p])
    
  }
  
  post_samples_df <- as.data.frame(post_samples_p, 
                                   col.names = 1:length(post_samples_list))
  post_samples_df$id <- 1:n_post_samples
  post_samples_df_long <- melt(post_samples_df, id.vars = 'id', variable.name = 'run')
  
  if (p == param_dim - 1) { ## if the parameter is sigma_alpha
    plot <- ggplot(post_samples_df_long, aes(x = value)) + #geom_line(aes(colour = series))
      geom_density(aes(col = run)) +
      geom_density(data = hmc.df, aes(x = samples), col = "black") +
      geom_vline(data = param_df, aes(xintercept=x),
                 color="black", linetype="dashed", linewidth=0.75) +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = expression(sigma[alpha]))
  } else if (p == param_dim) { # the parameter is sigma_epsilon
    plot <- ggplot(post_samples_df_long, aes(x = value)) + #geom_line(aes(colour = series))
      geom_density(aes(col = run)) +
      geom_density(data = hmc.df, aes(x = samples), col = "black") +
      geom_vline(data = param_df, aes(xintercept=x),
                 color="black", linetype="dashed", linewidth=0.75) +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = expression(sigma[epsilon]))
  } else {
    plot <- ggplot(post_samples_df_long, aes(x = value)) + #geom_line(aes(colour = series))
      geom_density(aes(col = run)) +
      geom_density(data = hmc.df, aes(x = samples), col = "black") +
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
  
  filename = paste0("var_test", temper_info, reorder_info,
                    "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0(result_directory, filename)
  
  png(filepath, width = 700, height = 400)
  grid.arrange(grobs = param_plots, nrow = 2, ncol = 3)
  dev.off()
  
} else {
  grid.arrange(grobs = param_plots, nrow = 2, ncol = 3)
}
