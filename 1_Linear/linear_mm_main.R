## Structure of the code:
## 1. Regenerate data
## 2. Run R-VGAL with estimated gradients/Hessians
## 3. Run R-VGAL with theoretical gradients/Hessians
## 4. Run HMC 

rm(list = ls())

# reticulate::use_condaenv("tf2.11", required = TRUE)
library("tensorflow")
library("mvtnorm")
library("Matrix")
library("rstan")
library("ggplot2")
library("grid")
library("gtable")
library("gridExtra")

source("./source/generate_data.R")
source("./source/run_est_rvgal.R")
source("./source/run_exact_rvgal.R")
# source("./source/run_finite_difference.R") # to calculate numerical gradients/Hessians for comparison with theoretical ones
source("./source/run_stan_lmm.R")

date <- "20230329"
regenerate_data <- F
rerun_est_rvgal <- T
rerun_exact_rvgal <- T
rerun_hmc <- T
reorder_data <- F
use_tempering <- T

save_data <- F
save_est_rvgal_results <- F
save_exact_rvgal_results <- F
save_hmc_results <- F
save_plots <- F

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

N <- 200L
n <- 10L
S <- 100L
S_alpha <- 100L
n_post_samples <- 10000

## 1. Generate data
if (regenerate_data) {
  ## True parameters
  sigma_a <- 0.9
  sigma_e <- 0.7
  beta <- c(-1.5, 1.5, 0.5, 0.25) 
  
  linear_data <- generate_data(beta, sigma_a, sigma_e, save_data = save_data, date)
  
  if (save_data) {
    saveRDS(linear_data, file = paste0("./data/linear_data_N", N, "_n", n, "_", date, ".rds"))
  }
} else {
  linear_data <- readRDS(file = paste0("./data/linear_data_N", N, "_n", n, "_", date, ".rds"))
}

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

## Sample to see if prior mean and variance are reasonable
# samples <- rmvnorm(10000, mu_0, P_0)
# sigma_a_samples <- sqrt(exp(samples[, 5]))
# sigma_e_samples <- sqrt(exp(samples[, 6]))
# par(mfrow = c(1, 2))
# plot(density(sigma_a_samples), main = "Prior density of sigma_a")
# plot(density(sigma_e_samples), main = "Prior density of sigma_e")

## 2. Run R-VGAL with estimated gradients/Hessians

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
result_directory <- "./results/"
est_results_file <- paste0("linear_mm_rvga_fisher", temper_info, reorder_info, 
                           "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, ".rds")

if (rerun_est_rvgal) {
  est_rvgal_results <- run_est_rvgal(y = y, X = X, Z = Z, mu_0 = mu_0, P_0 = P_0,
                                     S = S, S_alpha = S_alpha,
                                     n_temper = n_obs_to_temper,
                                     n_post_samples = n_post_samples,
                                     use_tempering = use_tempering)
  
  if (save_est_rvgal_results) {
    saveRDS(est_rvgal_results, file = paste0(result_directory, est_results_file))
  }
  
} else {
  est_rvgal_results <- readRDS(file = paste0(result_directory, est_results_file))
}


# ## 3. Run R-VGAL with theoretical gradients/Hessians
exact_results_file <- paste0("linear_mm_rvga", temper_info, reorder_info,
                             "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, ".rds")

if (rerun_exact_rvgal) {
  exact_rvgal_results <- run_exact_rvgal(y = y, X = X, Z = Z, 
                                         mu_0 = mu_0, P_0 = P_0, S = S,
                                         n_temper = n_obs_to_temper,
                                         n_post_samples = n_post_samples,
                                         # save_results = save_exact_rvgal_results,
                                         use_tempering = use_tempering)
  if (save_exact_rvgal_results) {
    saveRDS(exact_rvgal_results, file = paste0(result_directory, exact_results_file))
  }
  
  
} else {
  exact_rvgal_results <- readRDS(file = paste0(result_directory, exact_results_file))
}

## 4. Run HMC 
hmc_iters <- 15000
if (rerun_hmc) {
  hfit <- run_stan_lmm(data = y, fixed_covariates = X, 
                       random_covariates = Z,
                       iters = hmc_iters, burn_in = hmc_iters - n_post_samples)
  
  if (save_hmc_results) {
    saveRDS(hfit, file = paste0(result_directory, "linear_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
} else {
  hfit <- readRDS(file = paste0(result_directory, "linear_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
}
hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]", "phi", "psi"), 
                   permuted = F)

######################################
##              Results             ##
######################################

## Extract posterior samples
est_rvgal.post_samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
exact_rvgal.post_samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
hmc.samples <- matrix(NA, n_post_samples, param_dim)

for (p in 1:param_dim) {
  
  if (p == (param_dim - 1) || p == param_dim) { # if the parameters are variance parameters
    est_rvgal.post_samples[, p] <- sqrt(exp(est_rvgal_results$post_samples[, p])) # then back-transform
    exact_rvgal.post_samples[, p] <- sqrt(exp(exact_rvgal_results$post_samples[, p]))
    hmc.samples[, p] <- sqrt(exp(hmc.fit[, , p]))
  } else {
    est_rvgal.post_samples[, p] <- est_rvgal_results$post_samples[, p]
    exact_rvgal.post_samples[, p] <- exact_rvgal_results$post_samples[, p]
    hmc.samples[, p] <- hmc.fit[, , p]
  }
}

## Parameter trajectories
# par(mfrow = c(2,3))
# trajectories <- list()
# for (p in 1:param_dim) {
#   trajectories[[p]] <- sapply(est_rvgal_results$mu, function(e) e[p])
#   plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = param_names[p], main = "")
# }

## Posterior density plots (ggplot version) 

param_names <- c("beta1", "beta2", "beta3", "beta4", "sigma_a", "sigma_e")
est_rvgal.df <- data.frame(est_rvgal.post_samples)
exact_rvgal.df <- data.frame(exact_rvgal.post_samples)
hmc.df <- data.frame(beta = hmc.samples)
colnames(est_rvgal.df) <- param_names
colnames(exact_rvgal.df) <- param_names
colnames(hmc.df) <- param_names

true_vals.df <- data.frame(beta1 = beta[1], beta2 = beta[2], beta3 = beta[3],
                           beta4 = beta[4], sigma_a = sigma_a, sigma_e = sigma_e)
param_values <- c(beta, sigma_a, sigma_e)

plots <- list()

for (p in 1:(param_dim-2)) {
  
  true_vals.df <- data.frame(name = param_names[p], val = param_values[p])
  
  plot <- ggplot(exact_rvgal.df, aes(x=.data[[param_names[p]]])) +
  # plot <- ggplot(exact_rvgal.df, aes(x=colnames(exact_rvgal.df)[p])) + 
    geom_density(col = "goldenrod") +
    geom_density(data = est_rvgal.df, col = "red") +
    geom_density(data = hmc.df, col = "blue") +
    geom_vline(data = true_vals.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth=0.5) +
    labs(x = bquote(beta[.(p)])) +
    theme_bw() +
    theme(axis.title = element_blank()) 
  # theme(legend.position="bottom") + 
  # scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))
  
  plots[[p]] <- plot  
}

sigma_a_plot <- ggplot(exact_rvgal.df, aes(x=sigma_a)) + 
  geom_density(col = "goldenrod") +
  geom_density(data = est_rvgal.df, col = "red") +
  geom_density(data = hmc.df, col = "blue") +
  geom_vline(data = true_vals.df, aes(xintercept=sigma_a),
             color="black", linetype="dashed", linewidth=0.5) +
  labs(x = expression(sigma[alpha])) +
  theme_bw() +
  theme(axis.title = element_blank())
# theme(legend.position="bottom") +
# scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))

sigma_e_plot <- ggplot(exact_rvgal.df, aes(x=sigma_e)) + 
  geom_density(col = "goldenrod") +
  geom_density(data = est_rvgal.df, col = "red") +
  geom_density(data = hmc.df, col = "blue") +
  geom_vline(data = true_vals.df, aes(xintercept=sigma_e),
             color="black", linetype="dashed", linewidth=0.5) + 
  labs(x = expression(sigma[epsilon])) +
  theme_bw() +
  theme(axis.title = element_blank())

plots[[param_dim-1]] <- sigma_a_plot
plots[[param_dim]] <- sigma_e_plot

## Arrange bivariate plots in lower off-diagonals
n_lower_tri <- (param_dim^2 - param_dim)/2 # number of lower triangular elements

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
  
  param_df <- data.frame(x = param_values[p], y = param_values[q])
  
  cov_plot <- ggplot(exact_rvgal.df, aes(x = .data[[param_names[p]]], y = .data[[param_names[q]]])) +
    stat_ellipse(col = "goldenrod", type = "norm") +
    stat_ellipse(data = est_rvgal.df, col = "red", type = "norm") +
    stat_ellipse(data = hmc.df, col = "blue", type = "norm") +
    geom_point(data = param_df, aes(x = x, y = y),
               shape = 4, color = "black", size = 2) +
    theme_bw() +
    theme(axis.title = element_blank())
  
  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
m[lower.tri(m, diag = F)] <- 1:n_lower_tri 
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:6, l = 1:6)
# grid.draw(gr3)

# A list of text grobs - the labels
vars <- list(textGrob(bquote(beta[1])), textGrob(bquote(beta[2])), textGrob(bquote(beta[3])), 
             textGrob(bquote(beta[4])), textGrob(bquote(sigma[alpha])), textGrob(bquote(sigma[epsilon])))

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
gp <- gtable_add_grob(gp, lapply(vars[1:6], editGrob, rot = 90), t = 1:6, l = 1)
gp <- gtable_add_grob(gp, vars[1:6], t = 7, l = 2:7)

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("linear_posterior", temper_info, reorder_info,
                      "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 800, height = 600)
  grid.newpage()
  grid.draw(gp)
  dev.off()
} 
