## Run STAN on linear MM ##

run_stan_lmm <- function(data, fixed_covariates, 
                         random_covariates,
                         prior_mean, prior_var,
                         iters = 10000, burn_in = 5000,
                         nchains = 1) {
  
  ## Stan ##
  
  linearmm_code <- "./source/linear_mm.stan"
  
  N <- length(data)
  n <- length(data[[1]])
  y_long <- unlist(data)
  X_long <- do.call("rbind", fixed_covariates)
  Z_long <- unlist(random_covariates)
  grouping <- rep(1:N, each = n)
  
  p <- ncol(fixed_covariates[[1]])
  
  linearmm_data <- list(N = N * n, M = N, K = p, 
                        y = y_long, x = X_long, z = Z_long, g = grouping,
                        prior_mean_beta = prior_mean[1:p], 
                        prior_mean_sigma_a = prior_mean[p+1],
                        prior_mean_sigma_e = prior_mean[p+2],
                        prior_sd_beta = sqrt(diag(prior_var)[1:p]), 
                        prior_sd_sigma_a = sqrt(diag(prior_var)[p+1]),
                        prior_sd_sigma_e = sqrt(diag(prior_var)[p+2])
                        )
  
  hfit <- stan(file = linearmm_code, 
               model_name = "linear_mm", data = linearmm_data, 
               iter = iters, warmup = burn_in, chains=nchains)
  
  param_names <- c("beta[1]","beta[2]","beta[3]","beta[4]", 
                   "sigma_a", "sigma_e")
  hmc.fit <- extract(hfit, pars = param_names,
                     permuted = F, inc_warmup = T)
  
  hmc.summ <- summary(hfit, pars = param_names)$summary
  hmc.n_eff <- hmc.summ[, "n_eff"]
  hmc.Rhat <- hmc.summ[, "Rhat"]
  
  hmc_results <- list(post_samples = hmc.fit,
                      summary = hmc.summ,
                      n_eff = hmc.n_eff,
                      Rhat = hmc.Rhat, 
                      time = get_elapsed_time(hfit))
  
  # if (save_hmc_results) {
  #   saveRDS(hfit, file = paste0("linear_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  # }
  
  return(hmc_results)
}