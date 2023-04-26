run_exact_rvgal <- function(y, X, Z, mu_0, P_0, S = 100L, 
                            use_tempering = T, n_temper = 10, 
                            a_vals_temper = rep(0.25, 4),
                            n_post_samples = 10000) {
  
  print("Starting exact R-VGAL...")
  
  mu_vals <- list()
  mu_vals[[1]] <- mu_0
  
  prec <- list()
  prec[[1]] <- chol2inv(chol(P_0))
  a_vals <- 1 # for tempering
  
  # param_dim <- length(mu_0)
  N <- length(y)
  n <- length(y[[1]])
  
  rvga.t1 <- proc.time()
  
  for (i in 1:N) {
    
    a_vals <- 1
    if (use_tempering) {
      
      if (i <= n_temper) { # only temper the first n_temper observations
        a_vals <- a_vals_temper
      } 
    } 
    
    mu_temp <- mu_vals[[i]]
    prec_temp <- prec[[i]] 
    
    for (v in 1:length(a_vals)) {
      
      a <- a_vals[v]
      
      P <- chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
      
      score <- list()
      hessian <- list()
      
      for (s in 1:S) {
        beta_s <- samples[s, 1:length(beta)]
        phi_s <- samples[s, length(beta) + 1]
        psi_s <- samples[s, length(beta) + 2]
        
        Sigma_y <- exp(phi_s) * Z[[i]] %*% t(Z[[i]]) + diag(exp(psi_s), n)
        Sigma_y_inv <- solve(Sigma_y)
        
        d_Sigma_d_phi <- exp(phi_s) * tcrossprod(Z[[i]])
        d_Sigma_d_phi_2 <- exp(phi_s) * tcrossprod(Z[[i]])
        d_Sigma_d_psi <- diag(exp(psi_s), n)
        d_Sigma_d_psi_2 <- diag(exp(psi_s), n)
        
        # First derivative
        grad_beta <- t(X[[i]]) %*% Sigma_y_inv %*% (y[[i]] - X[[i]] %*% beta_s)
        
        grad_phi <- -1/2 * sum(diag(Sigma_y_inv %*% d_Sigma_d_phi)) +
          1/2 * t(y[[i]] - X[[i]] %*% beta_s) %*% Sigma_y_inv %*%
          d_Sigma_d_phi %*% Sigma_y_inv %*% (y[[i]] - X[[i]] %*% beta_s)
        
        grad_psi <- -1/2 * sum(diag(Sigma_y_inv %*% d_Sigma_d_psi)) +
          1/2 * t(y[[i]] - X[[i]] %*% beta_s) %*% Sigma_y_inv %*%
          d_Sigma_d_psi %*% Sigma_y_inv %*% (y[[i]] - X[[i]] %*% beta_s)
        
        score[[s]] <- c(grad_beta, grad_phi, grad_psi)
        
        ## Second derivative
        grad2_beta <- - t(X[[i]]) %*% Sigma_y_inv %*% X[[i]]
        
        G_phi <- - Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv %*% d_Sigma_d_phi + Sigma_y_inv %*% d_Sigma_d_phi_2
        H_phi <- -2 * Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv +
          Sigma_y_inv %*% d_Sigma_d_phi_2 %*% Sigma_y_inv
        
        grad2_phi <- -1/2 * sum(diag(G_phi)) + 1/2 * t(y[[i]] - X[[i]] %*% beta_s) %*%
          H_phi %*% (y[[i]] - X[[i]] %*% beta_s)
        
        G_psi <- - Sigma_y_inv %*% d_Sigma_d_psi %*% Sigma_y_inv %*% d_Sigma_d_psi + Sigma_y_inv %*% d_Sigma_d_psi_2
        H_psi <- -2 * Sigma_y_inv %*% d_Sigma_d_psi %*% Sigma_y_inv %*% d_Sigma_d_psi %*% Sigma_y_inv +
          Sigma_y_inv %*% d_Sigma_d_psi_2 %*% Sigma_y_inv
        
        grad2_psi <- -1/2 * sum(diag(G_psi)) + 1/2 * t(y[[i]] - X[[i]] %*% beta_s) %*%
          H_psi %*% (y[[i]] - X[[i]] %*% beta_s)
        
        ## Mixed derivatives
        grad2_beta_phi <- - t(X[[i]]) %*% Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv %*% (y[[i]] - X[[i]] %*% beta_s)
        grad2_beta_psi <- - t(X[[i]]) %*% Sigma_y_inv %*% d_Sigma_d_psi %*% Sigma_y_inv %*% (y[[i]] - X[[i]] %*% beta_s)
        
        G_phi_psi <- - Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv %*% d_Sigma_d_psi
        H_phi_psi <- - Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv %*% d_Sigma_d_psi %*% Sigma_y_inv -
          Sigma_y_inv %*% d_Sigma_d_psi %*% Sigma_y_inv %*% d_Sigma_d_phi %*% Sigma_y_inv
        grad2_phi_psi <- -1/2 * sum(diag(G_phi_psi)) + 1/2 * t(y[[i]] - X[[i]] %*% beta_s) %*% H_phi_psi %*% (y[[i]] - X[[i]] %*% beta_s)
        
        # Construct Hessian matrix
        grad_theta_2 <- as.matrix(bdiag(grad2_beta, grad2_phi, grad2_psi))
        grad_theta_2[1:(param_dim-2), 5] <- grad2_beta_phi
        grad_theta_2[1:(param_dim-2), 6] <- grad2_beta_psi
        grad_theta_2[5, 6] <- grad2_phi_psi
        grad_theta_2[lower.tri(grad_theta_2)] <- t(grad_theta_2)[lower.tri(grad_theta_2)] # reflect upper triangular part to lower triangular part
        hessian[[s]] <- grad_theta_2
        
        ## Use finite difference to check the gradient here
        # if (run_finite_diff) {
        #   
        #   fd_result <- run_finite_difference(y_i = y[[i]], X_i = X[[i]], 
        #                                      z_i = Z[[i]], beta_s, phi_s, psi_s)
        #   
        #   score[[s]] <- fd_result$grad_fd
        #   hessian[[s]] <- fd_result$hessian_fd
        #   
        # }
        
      }
      
      E_score <- Reduce("+", score)/ length(score)
      E_hessian <- Reduce("+", hessian)/ length(hessian)
      
      prec_temp <- prec_temp - a * as.matrix(E_hessian)
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_score))  
      
    }  
    
    prec[[i+1]] <- prec_temp
    mu_vals[[i+1]] <- mu_temp
    
    if (i %% (N/10) == 0) {
      cat(i/N * 100, "% complete \n")
    }
    
  }
  
  rvga.t2 <- proc.time()
  
  # ## Plot posterior
  post_var <- solve(prec[[N+1]])
  rvga.post_samples <- rmvnorm(10000, mu_vals[[N+1]], post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  ## Save results
  rvga_results <- list(mu = mu_vals,
                       prec = prec,
                       post_samples = rvga.post_samples,
                       S = S,
                       # S_alpha = S_alpha, 
                       N = N, 
                       n = n,
                       time_elapsed = rvga.t2 - rvga.t1)
  
  return(rvga_results)
}