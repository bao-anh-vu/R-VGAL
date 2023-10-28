data {
  int N; // number of obs 
  int M; // number of groups 
  int K; // number of predictors
  
  real y[N]; // outcome
  row_vector[K] x[N]; // predictors
  vector[N] z;
  int g[N];    // map obs to groups (this is like 11111 22222 33333 etc in my model)
  vector[K] prior_mean_beta;
  real prior_mean_sigma_a;
  real prior_mean_sigma_e;
  
  vector[K] prior_sd_beta;
  real prior_sd_sigma_a;
  real prior_sd_sigma_e;
}
parameters {
  real a[M]; 
  vector[K] beta;
  real phi; // = theta_sigma_a
  real psi; // = theta_sigma_e
}
transformed parameters {
  real sigma_a;
  real sigma_e;
  
  sigma_a = sqrt(exp(phi));
  sigma_e = sqrt(exp(psi));
}
model {
  
  phi ~ normal(prior_mean_sigma_a, prior_sd_sigma_a);
  psi ~ normal(prior_mean_sigma_e, prior_sd_sigma_e);
  a ~ normal(0, sigma_a);
  beta ~ normal(prior_mean_beta, prior_sd_beta);
  for(n in 1:N) {
    y[n] ~ normal(x[n] * beta + z[n] * a[g[n]], sigma_e);
    
  }
}