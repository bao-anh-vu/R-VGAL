data {
  int N; // number of obs in total (obs per groups * number of groups)
  int M; // number of groups 
  int K; // number of predictors
  
  int y[N]; // outcome
  row_vector[K] x[N]; // predictors
  int g[N];    // map obs to groups (this is probably like 11111 22222 33333 etc in my model)
}
parameters {
  real a[M]; 
  vector[K] beta;
  real tau;
}
model {
  
  //tau ~ normal(0, 1);
  tau ~ normal(log(0.5^2), sqrt(1));
  a ~ normal(0, sqrt(exp(tau)));
  beta ~ normal(0, sqrt(10));
  for(n in 1:N) {
    y[n] ~ bernoulli(inv_logit(a[g[n]] + x[n]*beta));
  }
}