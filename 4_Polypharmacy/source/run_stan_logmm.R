##########
## STAN ##
##########

run_stan_logmm <- function(iters, burn_in, n_chains, data, grouping, n_groups, 
                           fixed_covariates, prior_mean, prior_var) {
  
  logistic_code <- "./source/logistic_mm.stan"
  
  p <- ncol(fixed_covariates)
  logistic_data <- list(N = length(data), M = n_groups, 
                        K = p, y = data, 
                        x = fixed_covariates, g = grouping
  )
  
  hfit <- stan(file = logistic_code, 
               model_name="logistic_mm", data = logistic_data, 
               iter = iters, warmup = burn_in, chains=n_chains)
  
  param_names <- c("beta[1]","beta[2]","beta[3]","beta[4]",
                    "beta[5]","beta[6]","beta[7]","beta[8]", "omega")
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


