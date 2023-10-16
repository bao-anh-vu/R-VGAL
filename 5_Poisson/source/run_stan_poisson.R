##########
## STAN ##
##########

run_stan_poisson <- function(iters, burn_in, n_chains, data, grouping, n_groups, 
                           fixed_covariates, rand_covariates, 
                           prior_mean, prior_var, save_results = T) {
  
  poisson_code <- "./source/poisson_mm.stan"
    
  p <- ncol(fixed_covariates)
  r <- ncol(rand_covariates)
  nlower <- r * (r - 1)/2 + r# number of lower triangular elements of Sigma_alpha
  
  poisson_data <- list(N = length(data), M = n_groups, 
                       y = data, 
                       x = fixed_covariates, 
                       z = rand_covariates, 
                       p = ncol(fixed_covariates), 
                       r = ncol(rand_covariates), 
                       g = grouping, 
                       nlower = nlower,
                       prior_mean_beta = prior_mean[1:p],
                       diag_prior_var_beta = diag(prior_var)[1:p],
                       prior_mean_gamma = prior_mean[-(1:p)],
                       diag_prior_var_gamma = diag(prior_var)[-(1:p)]
  )
  
  hfit <- stan(file = poisson_code, 
               model_name="poisson_mm", data = poisson_data, 
               iter = iters, warmup = burn_in, chains=n_chains)
  
  if (save_results) {
    saveRDS(hfit, file = paste0("poisson_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
  return(hfit)
}


