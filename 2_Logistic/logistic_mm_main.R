setwd("~/R-VGAL/2_Logistic/")

# Structure:
# 1. Generate data
# 2. Run R-VGAL algorithm
# 3. Run HMC
# 4. Plot results

rm(list=ls())

# reticulate::use_condaenv("tf2.11", required = TRUE)
library("dplyr")
library("tensorflow")
library("mvtnorm")
library("rstan")
library("gridExtra")
library("grid")
library("gtable")
library(coda)

source("./source/run_rvgal.R")
source("./source/run_stan_logmm.R")
source("./source/generate_data.R")

## Flags
date <- "20230329"  
regenerate_data <- F
rerun_rvga <- F
rerun_stan <- F
save_data <- F
save_rvga_results <- F
save_hmc_results <- F
save_plots <- F
reorder_data <- F
use_tempering <- T

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

n_post_samples <- 20000

## Generate data
N <- 500L #number of individuals
n <- 10L # number of responses per individual
beta <- c(-1.5, 1.5, 0.5, 0.25) 
tau <- 0.9

if (regenerate_data) {
  logistic_data <- generate_data(N = N, n = n, beta = beta, tau = tau,
                                 save_data = save_data, date = date)
} else {
  logistic_data <- readRDS(file = paste0("./data/logistic_data_N", N, "_n", n, "_", "20230329", ".rds"))
}

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

###################
##     R-VGA     ##
###################
S <- 100L
S_alpha <- 100L

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
results_file <- paste0("logistic_mm_rvga", temper_info, reorder_info, 
                       "_N", N, "_n", n, "_S", S, "_Sa", S_alpha, "_", date, ".rds")


## Initialise variational parameters
n_fixed_effects <- as.integer(ncol(X[[1]]))
param_dim <- n_fixed_effects + 1L

beta_0 <- rep(0, param_dim - 1L)
omega_0 <- log(0.5^2) 

mu_0 <- c(beta_0, omega_0)
P_0 <- diag(c(rep(10, n_fixed_effects), 1))

if (rerun_rvga) {
  rvga_results <- run_rvgal(y, X, mu_0, P_0, S = S, S_alpha = S_alpha,
                            n_post_samples = n_post_samples,
                            use_tempering = use_tempering, 
                            n_temper = n_obs_to_temper, 
                            temper_schedule = a_vals_temper)
  
  if (save_rvga_results) {
    saveRDS(rvga_results, file = paste0(result_directory, results_file))
  }
  
} else {
  rvga_results <- readRDS(file = paste0(result_directory, results_file))
}

########################
##        STAN        ##
########################
burn_in <- 5000
n_chains <- 2
hmc.iters <- n_post_samples/n_chains + burn_in

if (rerun_stan) {
  
  ## Data manipulation ##
  y_long <- unlist(y) #as.vector(t(y))
  X_long <- do.call("rbind", X)
  
  hmc_results <- run_stan_logmm(iters = hmc.iters, burn_in = burn_in, 
                         n_chains = n_chains, data = y_long, 
                         grouping = rep(1:N, each = n), n_groups = N,
                         fixed_covariates = X_long)
  
  if (save_hmc_results) {
    saveRDS(hmc_results, file = paste0(result_directory, "logistic_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
} else {
  hmc_results <- readRDS(file = paste0(result_directory, "logistic_mm_hmc_N", N, "_n", n, "_", date, ".rds")) # for the experiements on starting points
  
}

# hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]", "omega"),
#                    permuted = F, inc_warmup = F)

hmc.fit <- hmc_results$post_samples[-(1:burn_in),,]
hmc.n_eff <- hmc_results$n_eff
hmc.Rhat <- hmc_results$Rhat
######################## Results #########################

rvgal.post_samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
hmc.samples <- matrix(NA, n_post_samples, param_dim)

for (p in 1:param_dim) {
  
  if (p == param_dim) { # if the parameter is tau
    rvgal.post_samples[, p] <- sqrt(exp(rvga_results$post_samples[, p])) # then back-transform
    hmc.samples[, p] <- sqrt(exp(hmc.fit[, , p]))
  } else {
    rvgal.post_samples[, p] <- rvga_results$post_samples[, p]
    hmc.samples[, p] <- hmc.fit[, , p]
  }
}

rvgal.post_mean <- as.vector(apply(rvgal.post_samples, 2, mean))
rvgal.post_sd <- as.vector(apply(rvgal.post_samples, 2, sd))
hmc.post_mean <- as.vector(apply(hmc.samples, 2, mean))
hmc.post_sd <- as.vector(apply(hmc.samples, 2, sd))
# dens <- stan_dens(hfit, pars = c("beta[1]","beta[2]","beta[3]","beta[4]")) #, "sigma"))
# dens <- dens + ggtitle ("HMC posteriors") 
# print(dens)
# 
# traceplot(hfit, c("beta[1]","beta[2]","beta[3]","beta[4]"),
#           ncol=1,nrow=5,inc_warmup=F)

## Combine all results into dataframe
results <- data.frame(true_vals = c(beta, tau), 
                      rvga_mean = rvgal.post_mean, rvga_sd = rvgal.post_sd,
                      hmc_mean = hmc.post_mean, hmc_sd = hmc.post_sd)
print(results)

## Posterior plots
param_names <- c("beta1", "beta2", "beta3", "beta4", "tau")  
rvga.df <- data.frame(rvgal.post_samples) 
hmc.df <- data.frame(hmc.samples)
colnames(rvga.df) <- param_names
colnames(hmc.df) <- param_names

true_vals.df <- data.frame(beta1 = beta[1], beta2 = beta[2], beta3 = beta[3],
                           beta4 = beta[4], tau = tau)

plots <- list()

for (p in 1:(param_dim-1)) {
  plot <- ggplot(rvga.df, aes(x=.data[[param_names[p]]])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmc.df, col = "blue", lwd = 1) +
    geom_vline(data = true_vals.df, aes(xintercept=.data[[param_names[p]]]),
               color="black", linetype="dashed", linewidth = 0.75) +
    labs(x = bquote(beta[.(p)])) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 18)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4)) 
  
  plots[[p]] <- plot  
}

tau_plot <- ggplot(rvga.df, aes(x=tau)) + 
  geom_density(col = "red", lwd = 1) +
  geom_density(data = hmc.df, col = "blue", lwd = 1) +
  geom_vline(data = true_vals.df, aes(xintercept=tau),
             color="black", linetype="dashed", linewidth = 0.75) +
  labs(x = expression(tau)) +
  theme_bw() +
  theme(axis.title = element_blank(), text = element_text(size = 18)) +                               # Assign pretty axis ticks
  scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) 

plots[[param_dim]] <- tau_plot

## Posterior covariance plot

n_lower_tri <- (param_dim^2 - param_dim)/2

index_to_i_j_colwise_nodiag <- function(k, n) {
  kp <- n * (n - 1) / 2 - k
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- n - (kp - p * (p + 1) / 2)
  j  <- n - 1 - p
  c(i, j)
}

param_values <- c(beta, tau)
cov_plots <- list()
for (ind in 1:n_lower_tri) {
  mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
  p <- mat_ind[1]
  q <- mat_ind[2]
  
  param_df <- data.frame(x = param_values[q], y = param_values[p])
  
  cov_plot <- ggplot(rvga.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = rvga.df, col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "blue", type = "norm", lwd = 1) +
    geom_point(data = param_df, aes(x = x, y = y),
               shape = 4, color = "black", size = 4) +
    theme_bw() +
    theme(axis.title = element_blank(), text = element_text(size = 18)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) 
  
  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
n_cov_plots <- param_dim * (param_dim-1)/2
m[lower.tri(m, diag = F)] <- 1:n_cov_plots
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:5, l = 1:5)

# A list of text grobs - the labels
vars <- list(textGrob(bquote(beta[1])), textGrob(bquote(beta[2])), textGrob(bquote(beta[3])), 
             textGrob(bquote(beta[4])), textGrob(bquote(tau)))
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
gp <- gtable_add_grob(gp, lapply(vars[1:5], editGrob, rot = 90), t = 1:5, l = 1) # add column names to column 1, rows 2:5
gp <- gtable_add_grob(gp, vars[1:5], t = 6, l = 2:6) # add row names to row 6, columns 1:5

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("logistic_posterior", temper_info, reorder_info,
                      "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 1000, height = 700)
  grid.newpage()
  grid.draw(gp)
  dev.off()
} 

## Time benchmark
hmc.time <- sum(colSums(hmc_results$time)) # sum over all chains
rvga.time <- rvga_results$time_elapsed
cat("HMC time:", hmc.time, ", R-VGAL time:", rvga.time[3], "\n")
