functions {
  matrix to_lowertri(vector x, int d) { // map vector elements to lower triangular matrix
    int pos = 1;
    matrix[d, d] L;
    vector[num_elements(x) - d] lower_entries;
    L = diag_matrix(exp(x[1:d])); // the first d elements form the diagonal entries
    lower_entries = x[(d+1):num_elements(x)];
    
    for (i in 2:d) {
      for (j in 1:(i-1)) {
        L[i, j] = lower_entries[pos];
        pos += 1;
      }
    }
    return L;
  }
}

data {
  int N; // number of obs 
  int M; // number of groups 
  int p; // number of fixed effects
  int r; // number of random effects
  
  int y[N]; // vector of response variable
  row_vector[p] x[N]; // fixed effect covariates
  row_vector[r] z[N]; // random effect covariates
  
  int g[N];    // map obs to groups (this is like 11111 22222 33333 etc in my model)
  int nlower;  // number of lower triangular elements of the random effect covariance matrix;
  
  vector[p] prior_mean_beta;
  vector[p] diag_prior_var_beta;
  vector[nlower] prior_mean_gamma;
  vector[nlower] diag_prior_var_gamma;
}

parameters {
  // real a[M]; 
  vector[p] beta;
  vector[r] alpha[M];
  vector[nlower] gamma;
}

transformed parameters {
  
  cov_matrix[r] Sigma_alpha;
  matrix[r,r] L;
  
  L = to_lowertri(gamma, r);
  Sigma_alpha = L*L';
}
    
model {
  gamma ~ normal(prior_mean_gamma, diag_prior_var_gamma);
  alpha ~ multi_normal(rep_vector(0, r), Sigma_alpha);
  beta ~ normal(prior_mean_beta, diag_prior_var_beta);
  for(i in 1:N) {
    // y[n] ~ bernoulli(inv_logit(a[g[n]] + x[n]*beta));
    y[i] ~ poisson(exp(z[i]*alpha[g[i]] + x[i]*beta));
  }
}