##########
## STAN ##
##########

run_stan_poisson <- function(iters, burn_in, n_chains, data, grouping, n_groups, 
                           fixed_covariates, rand_covariates, 
                           prior_mean, prior_var) {
  
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
                       diag_prior_var_beta = sqrt(diag(prior_var)[1:p]),
                       prior_mean_gamma = prior_mean[-(1:p)],
                       diag_prior_var_gamma = sqrt(diag(prior_var)[-(1:p)])
  )
  
  hfit <- stan(file = poisson_code, 
               model_name="poisson_mm", data = poisson_data, 
               iter = iters, warmup = burn_in, chains=n_chains)
  
  
  if (r == 2) {
    param_names <- c(sapply(1:p, function(x) paste0("beta[", x, "]")), 
                     "Sigma_alpha[1,1]", 
                     "Sigma_alpha[2,1]", "Sigma_alpha[2,2]")
    
  } else {
    param_names <- c(sapply(1:3, function(x) paste0("beta[", x, "]")), 
                     "Sigma_alpha[1,1]", 
                     "Sigma_alpha[2,1]",
                     "Sigma_alpha[2,2]", "Sigma_alpha[3,1]",
                     "Sigma_alpha[3,2]", "Sigma_alpha[3,3]")
  }
  
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
  
  return(hmc_results)
}


