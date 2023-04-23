##########
## STAN ##
##########

run_stan_logmm <- function(iters, burn_in, data, grouping, n_groups, 
                           fixed_covariates, save_results = T) {
  
    logistic_code <- '
    data {
        int N; // number of obs (pregnancies)
        int M; // number of groups (women)
        int K; // number of predictors
        
        int y[N]; // outcome
        row_vector[K] x[N]; // predictors
        int g[N];    // map obs to groups (this is probably like 11111 22222 33333 etc in my model)
    }
    parameters {
        real a[M]; 
        vector[K] beta;
        real omega;
    }
    model {
      
      //omega ~ normal(0, 1);
      omega ~ normal(log(0.5^2), sqrt(1));
      a ~ normal(0, sqrt(exp(omega)));
      beta ~ normal(0, sqrt(10));
      for(n in 1:N) {
        y[n] ~ bernoulli(inv_logit(a[g[n]] + x[n]*beta));
      }
    }
    '
    logistic_data <- list(N = length(data), M = n_groups, 
                          K = ncol(fixed_covariates), y = data, 
                          x = fixed_covariates, g = grouping)
    
    hfit <- stan(model_code = logistic_code, 
                 model_name="logistic_mm", data = logistic_data, 
                 iter = iters, warmup = burn_in, chains=1)
    
    if (save_results) {
      saveRDS(hfit, file = paste0("logistic_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
    }
    
  return(hfit)
}


