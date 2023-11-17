generate_data2 <- function(N, n, beta, Sigma_alpha, date, seed = NULL) {
  
  X <- list() # covariates for fixed effects
  Z <- list() # covariates for random effects
  r <- nrow(Sigma_alpha)
  y <- list()
  alpha <- list()
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  for (i in 1:N) {
    ones <- rep(1, n)
    X[[i]] <- cbind(ones, matrix(rnorm(n * (length(beta)-1)), nrow = n, ncol = length(beta)-1))
    alpha[[i]] <- t(rmvnorm(1, rep(0, r), Sigma_alpha))
    
    Z[[i]] <- cbind(ones, matrix(rnorm(n * (r-1)), nrow = n, ncol = r-1))
    
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