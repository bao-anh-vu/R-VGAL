run_finite_difference <- function(y_i, X_i, z_i, beta_s, phi_s, psi_s, incr = 1e-07) {
  # incr <- 1e-06
  
  n <- length(y_i)
  
  Sigma_y <- exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_s), n)
  Sigma_y_inv <- solve(Sigma_y)
  grad_beta_fd <- c()
  grad_beta_2_fd <- c()
  for (p in 1:length(beta)) {
    incr_vec <- rep(0, length(beta)) 
    incr_vec[p] <- incr
    beta_add <- beta_s + incr_vec
    beta_sub <- beta_s - incr_vec
    f_beta_s <- dmvnorm(y_i, X_i %*% beta_s, Sigma_y, log = TRUE)
    f_beta_add <- dmvnorm(y_i, X_i %*% beta_add, Sigma_y, log = TRUE)
    f_beta_sub <- dmvnorm(y_i, X_i %*% beta_sub, Sigma_y, log = TRUE)
    grad_beta_fd[p] <- (f_beta_add - f_beta_s) / incr
    grad_beta_2_fd[p] <- (f_beta_add - 2 * f_beta_s + f_beta_sub) / incr^2
  }
  
  # (f(x + h) - f(x))/h
  
  phi_add <- phi_s + incr
  phi_sub <- phi_s - incr
  f_phi_s <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
  f_phi_add <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_add) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
  f_phi_sub <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_sub) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
  
  grad_phi_fd <- (f_phi_add - f_phi_s) / incr
  grad_phi_2_fd <- (f_phi_add - 2 * f_phi_s + f_phi_sub) / incr^2
  
  psi_add <- psi_s + incr
  psi_sub <- psi_s - incr
  f_psi_s <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
  f_psi_add <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_add), n), log = TRUE)
  f_psi_sub <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_sub), n), log = TRUE)
  
  grad_psi_fd <- (f_psi_add - f_psi_s) / incr
  grad_psi_2_fd <- (f_psi_add - 2 * f_psi_s + f_psi_sub) / incr^2
  
  grad_fd <- c(grad_beta_fd, grad_phi_fd, grad_psi_fd)
  
  ## Constructing the Hessian 
  hessian_fd <- matrix(NA, nrow = param_dim, ncol = param_dim)
  
  ## First fill in the "beta" block
  for (r in 1:(length(beta)-1)) {
    for (c in (r+1):length(beta)) {
      beta_incr_both <- beta_s
      beta_incr_both[r] <- beta_s[r] + incr
      beta_incr_both[c] <- beta_s[c] + incr
      
      beta_incr_r <- beta_s
      beta_incr_r[r] <- beta_s[r] + incr
      
      beta_incr_c <- beta_s
      beta_incr_c[c] <- beta_s[c] + incr
      
      f_beta_incr_both <- dmvnorm(y_i, X_i %*% beta_incr_both, Sigma_y, log = TRUE)
      f_beta_incr_r <- dmvnorm(y_i, X_i %*% beta_incr_r, Sigma_y, log = TRUE)
      f_beta_incr_c <- dmvnorm(y_i, X_i %*% beta_incr_c, Sigma_y, log = TRUE)
      f_beta_incr_none <- dmvnorm(y_i, X_i %*% beta_s, Sigma_y, log = TRUE)
      
      hessian_fd[r, c] <- 1/(incr^2) * (f_beta_incr_both - f_beta_incr_r - f_beta_incr_c + f_beta_incr_none)
      
    }
  }
  
  # diag(hessian_fd) <- grad_beta_2_fd
  
  ## Mixed derivatives between beta, phi and psi
  f_phi_add_psi_add <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_add) * z_i %*% t(z_i) + diag(exp(psi_add), n), log = TRUE)
  f_phi_sub_psi_add <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_sub) * z_i %*% t(z_i) + diag(exp(psi_add), n), log = TRUE)
  f_phi_add_psi_sub <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_add) * z_i %*% t(z_i) + diag(exp(psi_sub), n), log = TRUE)
  f_phi_sub_psi_sub <- dmvnorm(y_i, X_i %*% beta_s, exp(phi_sub) * z_i %*% t(z_i) + diag(exp(psi_sub), n), log = TRUE)
  
  grad_phi_psi_fd <- 1/(4*incr^2) * (f_phi_add_psi_add - f_phi_sub_psi_add - f_phi_add_psi_sub + f_phi_sub_psi_sub)
  
  grad_beta_psi_fd <- c()
  grad_beta_phi_fd <- c()
  
  for (p in 1:length(beta)) {
    incr_vec <- rep(0, length(beta)) 
    incr_vec[p] <- incr
    beta_add <- beta_s + incr_vec
    beta_sub <- beta_s - incr_vec
    
    f_beta_add_psi_add <- dmvnorm(y_i, X_i %*% beta_add, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_add), n), log = TRUE)
    f_beta_sub_psi_add <- dmvnorm(y_i, X_i %*% beta_sub, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_add), n), log = TRUE)
    f_beta_add_psi_sub <- dmvnorm(y_i, X_i %*% beta_add, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_sub), n), log = TRUE)
    f_beta_sub_psi_sub <- dmvnorm(y_i, X_i %*% beta_sub, exp(phi_s) * z_i %*% t(z_i) + diag(exp(psi_sub), n), log = TRUE)
    
    f_beta_add_phi_add <- dmvnorm(y_i, X_i %*% beta_add, exp(phi_add) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
    f_beta_sub_phi_add <- dmvnorm(y_i, X_i %*% beta_sub, exp(phi_add) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
    f_beta_add_phi_sub <- dmvnorm(y_i, X_i %*% beta_add, exp(phi_sub) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
    f_beta_sub_phi_sub <- dmvnorm(y_i, X_i %*% beta_sub, exp(phi_sub) * z_i %*% t(z_i) + diag(exp(psi_s), n), log = TRUE)
    
    grad_beta_phi_fd[p] <- 1/(4*incr^2) * (f_beta_add_phi_add - f_beta_sub_phi_add - f_beta_add_phi_sub + f_beta_sub_phi_sub)
    grad_beta_psi_fd[p] <- 1/(4*incr^2) * (f_beta_add_psi_add - f_beta_sub_psi_add - f_beta_add_psi_sub + f_beta_sub_psi_sub)
  }
  
  ## Construct the full Hessian from finite difference
  hessian_fd[1:4, 5] <- grad_beta_phi_fd
  hessian_fd[1:4, 6] <- grad_beta_psi_fd
  hessian_fd[5, 6] <- grad_phi_psi_fd
  diag(hessian_fd) <- c(grad_beta_2_fd, grad_phi_2_fd, grad_psi_2_fd)
  hessian_fd[lower.tri(hessian_fd)] <- t(hessian_fd)[lower.tri(hessian_fd)]
  
  return(list(grad_fd = grad_fd, hessian_fd = hessian_fd))
  
}