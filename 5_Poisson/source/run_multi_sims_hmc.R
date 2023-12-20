run_multi_sims_hmc <- function(nsims, date, n_post_samples = 2000, 
                               burn_in = 5000, n_chains = 2) {
  # date <- "20231018"
  
  hmc.iters <- n_post_samples/n_chains + burn_in
  
  result_directory <- paste0("./multi_sims/results/", date, "/")
  
  # Read data
  for (sim in 1:nsims) {
    datasets[[sim]] <- readRDS(file = paste0("./multi_sims/data/", date, 
                                             "/poisson_data_N", N, "_n", n, "_", date, "_",
                                             formatC(sim, width=3, flag="0"), ".rds"))
  }
  
  poisson_data <- datasets[[sim]]
  
  y <- poisson_data$y
  X <- poisson_data$X
  Z <- poisson_data$Z
  
  N <- length(y)
  n <- length(y[[1]])
  
  beta <- poisson_data$beta
  Sigma_alpha <- poisson_data$Sigma_alpha
  
  ## Data manipulation ##
  y_long <- unlist(y) #as.vector(t(y))
  X_long <- do.call("rbind", X)
  Z_long <- do.call("rbind", Z)
  
  ## Initialise variational parameters
  n_fixed_effects <- as.integer(ncol(X[[1]]))
  n_random_effects <- as.integer(ncol(Z[[1]]))
  n_elements_L <- n_random_effects + n_random_effects * (n_random_effects - 1)/2
  param_dim <- n_fixed_effects + n_elements_L
  
  beta_0 <- rep(0, n_fixed_effects)
  l_vec_0 <- c(rep(0, n_random_effects), rep(0, n_random_effects * (n_random_effects - 1)/2))
  mu_0 <- c(beta_0, l_vec_0)
  P_0 <- diag(c(rep(1, n_fixed_effects), rep(0.1, n_elements_L)))
  
  
  ## Run HMC
  hmc.result_file <- paste0(result_directory, 
                            "poisson_hmc_N", N, "_n", n, "_", date, "_",
                            formatC(sim, width=3, flag="0"), ".rds")
  
  cat("Sim", sim, "in progress... \n")
  
  hmc_sim_result <- run_stan_poisson(iters = hmc.iters, burn_in = burn_in,
                                     n_chains = n_chains, data = y_long,
                                     grouping = rep(1:N, each = n), n_groups = N,
                                     fixed_covariates = X_long,
                                     rand_covariates = Z_long,
                                     prior_mean = mu_0,
                                     prior_var = P_0)
  
  ## Save results
  saveRDS(hmc_sim_result, file = hmc.result_file)
}