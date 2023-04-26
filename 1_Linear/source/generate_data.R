generate_data <- function(beta, sigma_a, sigma_e, save_data = T, date) {
  ## True parameters
  # sigma_a <- 0.9
  # sigma_e <- 0.7
  # beta <- c(-1.5, 1.5, 0.5, 0.25) # runif(4, -3, 3) 
  
  X <- list()
  Z <- list()
  y <- list()
  # y <- matrix(NA, nrow = N, ncol = n)
  for (i in 1:N) {
    u <- rnorm(1, 0, sigma_a)
    eps <- rnorm(n, 0, sigma_e)
    X_i <- matrix(rnorm(n * length(beta), 0, 1), n, length(beta))
    Z_i <- rnorm(n, 0, 1)
    y[[i]] <- X_i %*% beta + Z_i * u + eps ## I should be generating multiple independent y's?
    X[[i]] <- X_i
    Z[[i]] <- Z_i
  }
  
  linear_data <- list(y = y, X = X, Z = Z, beta = beta, 
                      sigma_a = sigma_a, sigma_e = sigma_e, N = N, n = n)
  
  if (save_data) {
    saveRDS(linear_data, file = paste0("linear_data_N", N, "_n", n, "_", date, ".rds"))
  }
  
  return(linear_data)
}