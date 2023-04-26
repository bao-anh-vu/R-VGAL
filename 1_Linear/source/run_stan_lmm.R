## Run STAN on linear MM ##

run_stan_lmm <- function(data, fixed_covariates, 
                         random_covariates,
                         iters = 10000, burn_in = 5000) {
  
  ## Stan ##
  
  linearmm_code <- '
  data {
      int N; // number of obs 
      int M; // number of groups 
      int K; // number of predictors
      
      real y[N]; // outcome
      row_vector[K] x[N]; // predictors
      vector[N] z;
      int g[N];    // map obs to groups (this is probably like 11111 22222 33333 etc in my model)
  }
  parameters {
      real a[M]; 
      vector[K] beta;
      real phi;
      real psi;
  }
  model {
    
    phi ~ normal(log(0.5^2), 1);
    psi ~ normal(log(0.5^2), 1);
    a ~ normal(0, sqrt(exp(phi)));
    beta ~ normal(0, sqrt(10));
    for(n in 1:N) {
      y[n] ~ normal(x[n] * beta + z[n] * a[g[n]], sqrt(exp(psi)));
      
    }
  }
  '
  N <- length(data)
  n <- length(data[[1]])
  y_long <- unlist(data)
  X_long <- do.call("rbind", fixed_covariates)
  Z_long <- unlist(random_covariates)
  grouping <- rep(1:N, each = n)
  
  linearmm_data <- list(N = N * n, M = N, K = ncol(fixed_covariates[[1]]), 
                        y = y_long, x = X_long, z = Z_long, g = grouping)
  
  hfit <- stan(model_code = linearmm_code, 
               model_name = "linear_mm", data = linearmm_data, 
               iter = hmc_iters, warmup = burn_in, chains=1)
  
  if (save_hmc_results) {
    saveRDS(hfit, file = paste0("linear_mm_hmc_N", N, "_n", n, "_", date, ".rds"))
  }
  
  return(hfit)
}