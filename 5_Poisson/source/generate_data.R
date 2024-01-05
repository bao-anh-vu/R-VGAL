generate_data <- function(N, n, beta, Sigma_alpha, date, seed = NULL,
                          use_intercept = F) {
  
  X <- list() # covariates for fixed effects
  Z <- list() # covariates for random effects
  r <- nrow(Sigma_alpha)
  p <- length(beta)
  y <- list()
  alpha <- list()
  lambda <- list()
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  for (i in 1:N) {
    alpha[[i]] <- t(rmvnorm(1, rep(0, r), Sigma_alpha))
    
    if (use_intercept) {
      X[[i]] <- cbind(rep(1, n), matrix(rnorm(n * (p-1)), nrow = n, ncol = p-1))
    
      Z[[i]] <- cbind(rep(1, n), matrix(rnorm(n * (r-1)), nrow = n, ncol = r-1))
    } else {
      X[[i]] <- matrix(rnorm(n * p), nrow = n, ncol = p)
      Z[[i]] <- matrix(rnorm(n * r), nrow = n, ncol = r)
    }
    
    lambda[[i]] <- exp(X[[i]] %*% beta + Z[[i]] %*% alpha[[i]])
    
    # p_i <- exp(X[[i]] %*% beta + alpha_i) / (1 + exp(X[[i]] %*% beta + alpha_i))
    y[[i]] <- rpois(n, lambda[[i]])
    
    # if (any(y[[i]] > 100)) {
    #   browser()
    # }
  }
  
  poisson_data <- list(y = y, X = X, beta = beta, Z = Z, alpha = alpha, 
                       Sigma_alpha = Sigma_alpha,
                       lambda = lambda,
                       N = N, n = n)
  
  return(poisson_data)  
  
}