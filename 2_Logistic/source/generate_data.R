generate_data <- function(N, n, beta, tau, save_data = T, seed = NULL) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  X <- list()
  p <- matrix(NA, nrow = N, ncol = n)
  y <- list()
  
  # set.seed(2023)
  for (i in 1:N) {
    X[[i]] <- matrix(rnorm(n * length(beta)), nrow = n, ncol = length(beta))
    alpha_i <- rnorm(1, 0, tau)
    
    p_i <- exp(X[[i]] %*% beta + alpha_i) / (1 + exp(X[[i]] %*% beta + alpha_i))
    y[[i]] <- rbinom(n, 1, p_i)
    # for (j in 1:n) {
    #   p[i, j] <- exp(X[[i]][j, ] %*% beta + alpha_i) / (1 + exp(X[[i]][j, ] %*% beta + alpha_i))
    #   y[[i]] <- rbinom(1, 1, p[i, j])
    # }
  }
  
  logistic_data <- list(y = y, X = X, beta = beta, tau = tau, N = N, n = n)
  
  # if (save_data) {
  #   saveRDS(logistic_data, file = paste0("logistic_data_N", N, "_n", n, "_", date, ".rds"))
  # }
    
  return(logistic_data)  
  
}