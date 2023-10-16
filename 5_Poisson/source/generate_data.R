generate_data <- function(N, n, beta, Sigma_alpha, date) {
  
  X <- list() # covariates for fixed effects
  Z <- list() # covariates for random effects
  r <- nrow(Sigma_alpha)
  y <- list()
  alpha <- list()
  
  set.seed(2023)
  
  for (i in 1:N) {
    X[[i]] <- matrix(rnorm(n * length(beta)), nrow = n, ncol = length(beta))
    alpha[[i]] <- t(rmvnorm(1, rep(0, r), Sigma_alpha))
    
    Z[[i]] <- matrix(rnorm(n * r), nrow = n, ncol = r)
    
    lambda_i <- exp(X[[i]] %*% beta + Z[[i]] %*% alpha[[i]])
    
    # p_i <- exp(X[[i]] %*% beta + alpha_i) / (1 + exp(X[[i]] %*% beta + alpha_i))
    y[[i]] <- rpois(n, lambda_i)
    
    # if (any(y[[i]] > 100)) {
    #   browser()
    # }
  }
  
  poisson_data <- list(y = y, X = X, beta = beta, Z = Z, alpha = alpha, 
                       Sigma_alpha = Sigma_alpha,
                       N = N, n = n)
  
  return(poisson_data)  
  
}