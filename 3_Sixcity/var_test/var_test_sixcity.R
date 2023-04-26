## Test the variance of R-VGAL results on the Six city dataset ##

## Will take a while to run depending on how many R-VGAL runs are specified
## (change the "runs" parameter as needed)
## If only looking for saved results, set rerun_test <- FALSE

setwd("~/R-VGAL/3_Sixcity/")

rm(list=ls())

reticulate::use_condaenv("tf2.11", required = TRUE)
library("readxl") # part of tidyverse
library("dplyr")
library("tensorflow")
library("mvtnorm")
library("rstan")
library("gridExtra")
library("grid")
library("reshape2")

source("./source/run_rvgal.R")

## Flags
date <- "20230327" #"20230125" #"20230117"
rerun_test <- T
rerun_stan <- F
save_results <- F
save_hmc_results <- F
use_tempering <- T
reorder_data <- T
save_images <- F

runs <- 10 # number of R-VGAL runs
S <- 100L
S_alpha <- 100L

if (reorder_data) {
  reorder_seed <- 2024
}

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

############################
##        Read data       ##
############################

data <- read.table("./data/sixcity.txt", row.names = 1)
colnames(data) <- c("wheezing", "subject", "age", "smoke")
head(data)

param_names <- c("beta_0", "beta_age", "beta_smoke", "tau")

y_long <- data[, c("wheezing", "subject")]
y <- y_long %>% group_split(subject)
y <- lapply(y, function(x) { x["subject"] <- NULL; as.vector(data.matrix(x)) }) # get rid of the subject column then convert from df to matrix

intercept <- rep(1, length(y)) ## intercept term
fixed_effects <- c("age", "smoke")
X_long = cbind(data[ , c("subject")], intercept, data[, fixed_effects])
colnames(X_long) <- c("subject", "intercept", fixed_effects)
X <- X_long %>% group_split(subject) # split observations by subject (child)
X <- lapply(X, function(x) { x["subject"] <- NULL; data.matrix(x) }) # get rid of the subject column then convert from df to matrix

if (reorder_data) {
  print("Reordering data...")
  set.seed(reorder_seed)
  reordered_ind <- sample(1:length(y))
  print(head(reordered_ind))
  reordered_y <- lapply(reordered_ind, function(i) y[[i]])
  reordered_X <- lapply(reordered_ind, function(i) X[[i]])
  
  y <- reordered_y
  X <- reordered_X
} else {
  y <- y
  X <- X
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

result_directory <- paste0("./var_test/results/")
results_file <- paste0("var_test", temper_info, reorder_info,
                       "_S", S, "_Sa", S_alpha, "_", date, ".rds")
results_filepath <- paste0(result_directory, results_file)

###################
##     R-VGA     ##
###################

n_fixed_effects <- as.integer(ncol(X[[1]]))
param_dim <- n_fixed_effects + 1L

if (rerun_test) {
  
  r <- 1
  results <- list()
  
  N <- length(y)
  n <- nrow(X[[1]])
  
  param_dim <- as.integer(ncol(X[[1]]) + 1)
  beta_ini <- rep(0, param_dim-1)
  omega_ini <- 1 #rnorm(1, 0, 1)
  
  while (r <= runs) {
    
    ## 1. Initialise the variaftional mean and covariance
    mu_0 <- c(beta_ini, omega_ini)
    P_0 <- diag(c(rep(10, length(beta_ini)), 1))
    
    mu_vals <- lapply(1:N, function(x) mu_0)
    prec <- lapply(1:N, function(x) solve(P_0))
    
    ## Sample from the "prior"
    par(mfrow = c(1,1))
    test_omega <- rnorm(10000, mu_0[param_dim], sqrt(P_0[param_dim, param_dim]))
    plot(density(sqrt(exp(test_omega))), main = "RVGA: Prior of tau")
    
    try({
      rvgal_results <- run_rvgal(y, X, mu_0, P_0, S = S, S_alpha = S_alpha,
                                 use_tempering = use_tempering, 
                                 n_temper = n_obs_to_temper, 
                                 temper_schedule = a_vals_temper) 
      
      mu_vals <- rvgal_results$mu
      prec <- rvgal_results$prec
      
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
hmc.iters <- 15000
burn_in <- 5000

if (rerun_stan) {
  hmc.t1 <- proc.time()
  
  ## Data manipulation ##
  y <- unlist(y)
  X <- cbind(intercept, data[, fixed_effects])
  # test_y <- as.vector(t(Y))
  # test_x <- do.call("rbind", X)
  # put these in a dataframe later for easier retrieval
  # df <- data.frame(y = test_y, g = rep(1:N, each = n), x = test_x)
  
  logistic_code <- '
  data {
      int N; // number of obs (total)
      int M; // number of groups (children)
      int K; // number of covariates
      
      int y[N]; // outcome
      row_vector[K] x[N]; // covariates
      int g[N];    // map obs to groups (this is probably like 11111 22222 33333 etc in my model)
  }
  parameters {
      real a[M]; 
      vector[K] beta;
      real<lower=0> omega;  
  }
  model {
    omega ~ normal(1, 1);
    a ~ normal(0, sqrt(exp(omega)));
    beta ~ normal(0, sqrt(10));
    for(n in 1:N) {
      y[n] ~ bernoulli(inv_logit(a[g[n]] + x[n]*beta));
    }
  }
  '
  logistic_data <- list(N = N * n, M = N, K = length(fixed_effects)+1, y = y, 
                        x = X, g = rep(1:N, each = n))
  
  hfit <- stan(model_code = logistic_code, 
               model_name="logistic_mm", data = logistic_data, 
               iter = hmc.iters, warmup = burn_in, chains=1)
  
  hmc.t2 <- proc.time()
  
  if (save_hmc_results) {
    saveRDS(hfit, file = paste0("./results/sixcity_mm_hmc_", date, ".rds"))
  }
  
} else {
  hfit <- readRDS(file = paste0("./results/sixcity_mm_hmc_", date, ".rds")) 
}
######################## Results #########################

## Now plot the densities

## HMC results for comparison
hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]", "omega"),
                   permuted = F)
hmc.samples <- matrix(NA, dim(hmc.fit)[1], param_dim)
for (p in 1:(param_dim-1)) {
  hmc.samples[, p] <- hmc.fit[, , p]
}
hmc.samples[, (param_dim-1)+1] <- sqrt(exp(hmc.fit[, , param_dim])) # transform omega samples to tau samples
hmc.df <- data.frame(hmc.samples)

## R-VGA results
n_post_samples <- 10000
post_samples_list <- list()
for (r in 1:length(results)) {
  post_mu <- results[[r]]$mu
  post_var <- chol2inv(chol(results[[r]]$prec))
  
  post_samples_list[[r]] <- rmvnorm(n_post_samples, post_mu, post_var)
}

subscripts <- c("1", "2", "3", "4")

param_plots <- list()
for (p in 1:param_dim) {
  
  hmc.df <- data.frame(samples = hmc.samples[, p])
  
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
      geom_density(aes(col = run)) +
      geom_density(data = hmc.df, aes(x = samples), col = "black") +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = bquote(tau))
  } else {
    plot <- ggplot(post_samples_df_long, aes(x = value)) + #geom_line(aes(colour = series))
      geom_density(aes(col = run)) +
      geom_density(data = hmc.df, aes(x = samples), col = "black") +
      theme_bw() +
      theme(legend.position="none") +
      labs(x = bquote(beta[.(subscripts[p])]))
  }
  
  param_plots[[p]] <- plot
  # plot(density(post_samples_p[[1]]), xlab = param_names[p], main = "")
  # for (r in 2:length(post_samples_list)) {
  #   lines(density(post_samples_p[[r]]))
  # }
}

grid.arrange(grobs = param_plots, nrow = 2, ncol = 2)

## Saving the plots
if (save_images) {
  plot_directory <- paste0("./plots/var_test_", date, "/")
  plot_file = paste0("var_test", temper_info, reorder_info,
                    "_S", S, "_Sa", S_alpha, "_", date, ".png")
  
  png(paste0(plot_directory, plot_file), width = 600, height = 500)
  grid.arrange(grobs = param_plots, nrow = 2, ncol = 2)
  dev.off()
  
}

