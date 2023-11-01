run_multi_sims_hmc <- function(sim) {
  
  date <- "20230329"
  N <- 200L #number of individuals
  n <- 10L # number of responses per individual
  n_post_samples <- 20000
  
  dataset <- readRDS(file = paste0("./data/multi_sims/linear_data_N", N, 
                                   "_n", n, "_", date, "_",
                                   formatC(sim, width=3, flag="0"), ".rds"))
  
  ## Result directory
  result_directory <- "./multi_sims/results/"
  
  ## Initialise the variational mean and covariance
  n_fixed_effects <- ncol(dataset$X[[1]])
  param_dim <- as.integer(n_fixed_effects + 2) # theta = (beta, log(sigma_a), log(sigma_e))'
  beta_0 <- rep(0, n_fixed_effects)  
  sigma_a_0 <- 0.5
  sigma_e_0 <- 0.5
  phi_0 <- log(sigma_a_0^2)
  psi_0 <- log(sigma_e_0^2)
  mu_0 <- c(beta_0, phi_0, psi_0)
  var_beta_0 <- rep(10, n_fixed_effects)
  var_phi_0 <- 1
  var_psi_0 <- 1
  P_0 <- diag(c(var_beta_0, var_phi_0, var_psi_0), param_dim)
  
  ## Run HMC 
  burn_in <- 5000
  n_chains <- 2
  hmc.iters <- n_post_samples/n_chains + burn_in
  
  hmc_sim_results <- list()
  
  cat("Sim", sim, "in progress... \n")
  
  linear_data <- dataset#s[[sim]]
  
  y <- linear_data$y
  X <- linear_data$X
  Z <- linear_data$Z
  beta <- linear_data$beta
  sigma_a <- linear_data$sigma_a
  sigma_e <- linear_data$sigma_e
  
  hmc.result_file <- paste0(result_directory, 
                            "linear_hmc_N", N, "_n", n, "_", date, "_",
                            formatC(sim, width=3, flag="0"), ".rds")
  
  hmc_sim_result <- run_stan_lmm(data = y, fixed_covariates = X, 
                                 random_covariates = Z,
                                 iters = hmc.iters, burn_in = burn_in,
                                 nchains = n_chains,
                                 prior_mean = mu_0,
                                 prior_var = P_0)
  
  ## Save results
  saveRDS(hmc_sim_result, file = hmc.result_file)
  
}