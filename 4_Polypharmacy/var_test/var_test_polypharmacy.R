## Test the variance of R-VGAL results on the POLYPHARMACY dataset ##

## Will take a while to run depending on how many R-VGAL runs are specified
## (change the "runs" parameter as needed)
## If only looking for saved results, set rerun_test <- FALSE

rm(list = ls())

reticulate::use_condaenv("tf2.11", required = TRUE)
library("readxl") # part of tidyverse
library("dplyr")
library("tensorflow")
library("mvtnorm")
library("rstan")
# library("car")
library("gridExtra")
library("ggplot2")
library("reshape2")
# library("grid")
# library("gtable")

source("./source/run_rvgal.R")

## Flags
date <- "20230327_1"
rerun_test <- T
save_results <- F
save_images <- F
reorder_data <- F #this is only for reordering the data ONCE, then that order is fixed for all runs
use_tempering <- T

if (reorder_data) {
  reorder_seed <- 2023
}

if (use_tempering) {
  n_obs_to_temper <- 10
  # temps <- 10:1
  # a_vals_temper <- 1/temps 
  a_vals_temper <- rep(1/4, 4)
}

runs <- 10 # number of R-VGAL runs
S <- 100L
S_alpha <- 100L

## Read data
data <- read_excel("./data/polypharm.xls")

# data <- full_data[1:(7*50), ]
head(data)

## need to split the variable MHV4 into MHV1, MHV2, MHV3
data$MHV1 <- ifelse(data$MHV4 == "1", 1, 0)
data$MHV2 <- ifelse(data$MHV4 == "2", 1, 0)
data$MHV3 <- ifelse(data$MHV4 == "3", 1, 0)

data$RACE_transf <- ifelse(data$RACE == "0", 0, 1)
data$INPTMHV <- ifelse(data$INPTMHV3 == "0", 0, 1)

## Logistic model: 
## y_ij = logit(beta_0 + beta_gender * Gender_i + beta_race * Race_i + beta_age * Age_ij
## + beta_M1 * MHV1_ij + beta_M2 * MHV2_ij + beta_M3 * MHV3_ij + beta_IM * INPTMHV_ij + u_i
## u_i ~ N(0, exp(2*tau)) is a subject level random intercept

param_names <- c("beta_0", "beta_gender", "beta_race", "beta_age",
                 "beta_M1", "beta_M2", "beta_M3", "beta_IM", "tau")

param_dim <- length(param_names)

## Split data by subjects (ID)
y_long = data[, c("ID", "POLYPHARMACY")]
y <- y_long %>% group_split(ID)
y <- lapply(y, function(x) { x["ID"] <- NULL; as.vector(data.matrix(x)) }) # get rid of the ID column then convert from df to matrix

intercept <- rep(1, length(y)) ## intercept term
fixed_effects <- c("GENDER", "RACE_transf", "AGE", "MHV1", "MHV2", "MHV3", "INPTMHV")
X_long = cbind(data[, "ID"], intercept, data[, fixed_effects])

## Also need a mapping of observations to subject... which in this case is the ID column
X <- X_long %>% group_split(ID) # split observations by ID
X <- lapply(X, function(x) { x["ID"] <- NULL; data.matrix(x) }) # get rid of the ID column then convert from df to matrix

###################
##     R-VGA     ##
###################

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
  
  if (reorder_data) { # only reorder once!
    set.seed(reorder_seed) 
    reordered_ind <- sample(1:length(y))
    print(head(reordered_ind))
    reordered_y <- lapply(reordered_ind, function(i) y[[i]])
    reordered_X <- lapply(reordered_ind, function(i) X[[i]])
    
    y <- reordered_y
    X <- reordered_X
  }
  
  N <- length(y)
  n <- nrow(X[[1]])
  
  param_dim <- as.integer(ncol(X[[1]]) + 1)
  beta_0 <- rep(0, param_dim - 1) # initial values of beta
  omega_0 <- 1 #log(0.5^h2)
  
  results <- list()
  r <- 1
  t1 <- proc.time()
  
  while (r <= runs) { 
    
    ## 1. Initialise the variational mean and covariance
    mu_0 <- c(beta_0, omega_0)
    P_0 <- diag(c(rep(10, length(beta_0)), 1))
    
    ## Sample from the "prior"
    # par(mfrow = c(1, 1))
    # test_omega <- rnorm(10000, mu_0[param_dim], sqrt(P_0[param_dim, param_dim]))
    # plot(density(sqrt(exp(test_omega))), main = "RVGA: Prior of tau")
    
    mu_vals <- lapply(1:N, function(x) mu_0)
    prec <- lapply(1:N, function(x) solve(P_0))
    
    try({
      rvgal_results <- run_rvgal(y, X, mu_0, P_0, S = S, S_alpha = S_alpha,
                                use_tempering = use_tempering, 
                                n_temper = n_obs_to_temper, 
                                temper_schedule = a_vals_temper) 
      
      mu_vals <- rvgal_results$mu
      prec <- rvgal_results$prec
      
      ## Save trajectory of mean
      par(mfrow = c(3,3))
      trajectories <- list()
      for (p in 1:param_dim) {
        trajectories[[p]] <- sapply(mu_vals, function(e) e[p])
        plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = param_names[p], main = "")
      }
      
      ## Save posterior mean and variance 
      result <- list(mu = mu_vals[[N+1]], prec = prec[[N+1]], 
                     mean_trajectories = trajectories)
      results[[r]] <- result
      
      r <- r+1
    })
    
  }
  t2 <- proc.time()
  # print(t2 - t1)
  
  if (save_results) {
    saveRDS(results, file = results_filepath)
  }
  
} else {
  results <- readRDS(file = results_filepath)
}

###########################################
##              Plot results            ##
###########################################

## Plot parameter trajectories
subscripts <- c("0", "gender", "race", "age", "M1", "M2", "M3", "IM")

param_trajectories <- list()
for (p in 1:param_dim) {
  param_trajectories[[p]] <- sapply(results, function(r) r$mean_trajectories[[p]])
}

if (save_images) {
  filename2 = paste0("trajectories", temper_info, reorder_info,
                     "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath2 = paste0("./var_test/plots/", filename2)
  
  png(filepath2, width = 700, height = 500)
}

par(mfrow = c(3, 3))
for (p in 1:param_dim) {
  if (p == param_dim) { ## if the parameter is tau
    matplot(sqrt(exp(param_trajectories[[p]])), type = "l", xlab = "Iterations",
            ylab = expression(tau), main = "")
  } else {
    matplot(param_trajectories[[p]], type = "l", xlab = "Iterations",
            ylab = bquote(beta[.(subscripts[p])]), main = "")
  }
  
}

if (save_images) {
  dev.off()  
}

## Plot posterior densities

## HMC posterior for comparison
hfit <- readRDS(file = paste0(result_directory, "polypharmacy_mm_hmc_", date, ".rds"))
hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]", 
                                  "beta[5]","beta[6]","beta[7]","beta[8]", "omega"),
                   permuted = F)
hmc.samples <- matrix(NA, dim(hmc.fit)[1], param_dim)
for (p in 1:(param_dim-1)) {
  hmc.samples[, p] <- hmc.fit[, , p]
}
hmc.samples[, (param_dim-1)+1] <- sqrt(exp(hmc.fit[, , param_dim])) # transform omega samples to tau samples
# hmc.df <- data.frame(hmc.samples)

## R-VGAL posteriors
n_post_samples <- 10000
post_samples_list <- list()
for (r in 1:length(results)) {
  post_mu <- results[[r]]$mu
  post_var <- chol2inv(chol(results[[r]]$prec))
  
  post_samples_list[[r]] <- rmvnorm(n_post_samples, post_mu, post_var)
}

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
}

## Saving the plots
grid.arrange(grobs = param_plots, nrow = 3, ncol = 3)

if (save_images) {
  plot_directory <- paste0("./var_test/plots/")
  plot_file = paste0("var_test", temper_info, reorder_info,
                    "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0(plot_directory, plot_file)
    
  png(filepath, width = 800, height = 500)
  grid.arrange(grobs = param_plots, nrow = 3, ncol = 3)
  dev.off()
} 
  
