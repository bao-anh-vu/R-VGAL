## R-VGAL with six city data ##

reticulate::use_condaenv("tf2.11", required = TRUE)
library("tensorflow")
library("dplyr")
library("mvtnorm")
library("rstan")
library("ggplot2")
library("gridExtra")
library("grid")
library("gtable")

source("./source/run_rvgal.R")

rerun_rvga <- T
save_rvga_results <- F
rerun_stan <- T
save_hmc_results <- F
date <- "20230327" #"20230217g"
reorder_data <- T
use_tempering <- T
save_plots <- F

if (reorder_data) {
  reorder_seed <- 2024
}

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

S <- 200L
S_alpha <- 200L

## Read data
data <- read.table("./data/sixcity.txt", row.names = 1)
colnames(data) <- c("wheezing", "subject", "age", "smoke")
# as.factor(data$subject)
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

## RVGA ##

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
results_file <- paste0("sixcity_mm_rvga", temper_info, reorder_info,
                       "_S", S, "_Sa", S_alpha, "_", date, ".rds")
results_filepath <- paste0(result_directory, results_file)

param_dim <- length(param_names)
N <- length(unique(data$subject))
n <- nrow(X[[1]])

beta_ini <- rep(0, param_dim-1)
omega_ini <- 1 #rnorm(1, 0, 1)

## 1. Initialise the variational mean and covariance
mu_0 <- c(beta_ini, omega_ini)
P_0 <- diag(c(rep(10, length(beta_ini)), 1))

mu_vals <- lapply(1:N, function(x) mu_0)
prec <- lapply(1:N, function(x) solve(P_0))

## Sample from the prior to check that the prior is reasonable
# par(mfrow = c(1,1))
# test_omega <- rnorm(10000, mu_0[param_dim], sqrt(P_0[param_dim, param_dim]))
# plot(density(sqrt(exp(test_omega))), main = "RVGA: Prior of tau")

if (rerun_rvga) {
  
  if (reorder_data) {
    print("Reordering data...")
    set.seed(reorder_seed)
    reordered_ind <- sample(1:length(y))
    # print(head(reordered_ind))
    reordered_y <- lapply(reordered_ind, function(i) y[[i]])
    reordered_X <- lapply(reordered_ind, function(i) X[[i]])
    
    y <- reordered_y
    X <- reordered_X
  } else {
    y <- y
    X <- X
  }
  
  rvga_results <- run_rvgal(y, X, mu_0, P_0, S = S, S_alpha = S_alpha,
                            use_tempering = use_tempering, 
                            n_temper = n_obs_to_temper, 
                            temper_schedule = a_vals_temper)
  
  if (save_rvga_results) {
    saveRDS(rvga_results, file = results_filepath)
  }
  
} else {
  rvga_results <- readRDS(file = results_filepath)
}

rvga.post_samples <- rvga_results$post_samples
mu_vals <- rvga_results$mu

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
  
  logistic_code <- '
  data {
      int N; // number of obs (total)
      int M; // number of groups (children)
      int K; // number of covariates
      
      int y[N]; // outcome
      row_vector[K] x[N]; // covariates
      int g[N];    // map obs to groups (this is e.g. 1111 2222 3333 etc in my model)
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
    saveRDS(hfit, file = paste0(result_directory, "sixcity_mm_hmc_", date, ".rds"))
  }
  
} else {
  hfit <- readRDS(file = paste0(result_directory, "sixcity_mm_hmc_", date, ".rds")) # for the experiements on starting points
}

## Extract samples from STAN
hmc.fit <- extract(hfit, pars = c("beta[1]","beta[2]","beta[3]", "omega"),
                   permuted = F)
hmc.samples <- matrix(NA, hmc.iters - burn_in, param_dim)
for (p in 1:(param_dim-1)) {
  hmc.samples[, p] <- hmc.fit[, , p]
}
hmc.samples[, (param_dim-1)+1] <- sqrt(exp(hmc.fit[, , (param_dim-1)+1])) # transform omega samples to tau samples

################################# Results ######################################

## Parameter trajectories
par(mfrow = c(2,2))
trajectories <- list()
for (p in 1:param_dim) {
  trajectories[[p]] <- sapply(mu_vals, function(e) e[p])
  plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = param_names[p], main = "")
}

## Posterior plots: ggplot version
rvga.post_samples_beta <- rvga.post_samples[, 1:(param_dim-1)]
rvga.post_samples_tau <- sqrt(exp(rvga.post_samples[, param_dim]))
rvga.df <- data.frame(beta = rvga.post_samples_beta, tau = rvga.post_samples_tau) 
hmc.df <- data.frame(beta = hmc.samples)
colnames(rvga.df) <- param_names
colnames(hmc.df) <- param_names

plots <- list()

for (p in 1:(param_dim-1)) {
  plot <- ggplot(rvga.df, aes(x = .data[[param_names[p]]])) + 
    geom_density(col = "red") +
    geom_density(data = hmc.df, col = "blue") +
    labs(x = bquote(beta[.(p)])) +
    theme_bw() + 
    theme(axis.title = element_blank())
  # theme(legend.position="bottom") + 
  # scale_color_manual(values = c('RVGA' = 'red', 'HMC' = 'blue'))
  
  plots[[p]] <- plot  
}

tau_plot <- ggplot(rvga.df, aes(x=tau)) + 
  geom_density(col = "red") +
  geom_density(data = hmc.df, col = "blue") +
  theme_bw() +
  labs(x = expression(tau)) +
  theme(axis.title = element_blank())

plots[[param_dim]] <- tau_plot

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
  
  cov_plot <- ggplot(rvga.df, aes(x = .data[[param_names[p]]], y = .data[[param_names[q]]])) +
    stat_ellipse(col = "red", type = "norm") +
    stat_ellipse(data = hmc.df, col = "blue", type = "norm") +
    theme_bw() +
    theme(axis.title = element_blank())
  
  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
n_cov_plots <- param_dim * (param_dim-1)/2
m[lower.tri(m, diag = F)] <- 1:n_cov_plots
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:param_dim, l = 1:param_dim)


# A list of text grobs - the labels
vars <- list(textGrob(bquote(beta[0])), textGrob(bquote(beta[age])), 
             textGrob(bquote(beta[smoke])), textGrob(bquote(tau)))

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
gp <- gtable_add_grob(gp, lapply(vars[1:4], editGrob, rot = 90), t = 1:param_dim, l = 1) # add column names to column 1, rows 2:9
gp <- gtable_add_grob(gp, vars[1:param_dim], t = 5, l = 2:(param_dim+1)) # add row names to row 6, columns 1:9

grid.newpage()
grid.draw(gp)

if (save_plots) {
  plot_file <- paste0("sixcity_posterior", temper_info, reorder_info,
                      "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath = paste0("./plots/", plot_file)
  png(filepath, width = 800, height = 550)
  grid.newpage()
  grid.draw(gp)
  dev.off()
} 

