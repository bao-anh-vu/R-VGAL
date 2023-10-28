data {
  int N; // number of obs 
  int M; // number of groups
  int K; // number of predictors
  
  int y[N]; // outcome
  row_vector[K] x[N]; // predictors
  int g[N];    // map obs to groups (this is e.g. 11111 22222 33333 etc)
}
parameters {
  real a[M]; 
  vector[K] beta;
  real<lower=0> omega;  
}
model {
  omega ~ normal(1, 1);
  a ~ normal(0, sqrt(exp(omega)));
  beta ~ normal(0, sqrt(10));
  for(n in 1:N) {
    y[n] ~ bernoulli(inv_logit(a[g[n]] + x[n]*beta));
  }
}