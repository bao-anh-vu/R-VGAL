run_est_rvgal <- function(y, X, Z, mu_0, P_0, S = 1000L, S_alpha = 1000L,
                          use_tempering = T, n_temper = 10, 
                          a_vals_temper = rep(0.25, 4),
                          n_post_samples = 10000) {
  
  print("Starting R-VGAL...")
  
  mu_vals <- list()
  mu_vals[[1]] <- mu_0
  
  prec <- list()
  prec[[1]] <- chol2inv(chol(P_0))
  a_vals <- 1 # for tempering
  
  param_dim <- length(mu_0)
  N <- length(y)
  n <- length(y[[1]])
  
  rvgal.t1 <- proc.time()
  
  for (i in 1:N) {
    
    if (i %% (N/10) == 0) {
      cat(i/N * 100, "% complete \n")
    }
    
    a_vals <- 1
    if (use_tempering) {
      # n_temper <- n_obs_to_temper+1 # correcting since indices in R start from 1
      
      if (i <= n_obs_to_temper) { # only temper the first n_temper observations
        a_vals <- a_vals_temper
      } 
    }
    
    mu_temp <- mu_vals[[i]]
    prec_temp <- prec[[i]] 
    
    for (v in 1:length(a_vals)) {
      
      a <- a_vals[v]
      
      P <- chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
      
      ## Reshape and tile X so that it's in dimension (S, S_alpha, ~, ~))
      X_i <- X[[i]]
      X_i_tf <- tf$constant(X_i, dtype = "float64")
      X_i_tf2 <- tf$reshape(X_i_tf, c(1L, n, param_dim - 2L))
      X_i_tf3 <- tf$tile(X_i_tf2, c(S, 1L, 1L))
      # X_i_tf4 <- tf$reshape(X_i_tf3, c(1L, dim(X_i_tf3)))
      # X_i_tf5 <- tf$tile(X_i_tf4, c(S, 1L, 1L, 1L))
      
      ## Reshape and tile Z so that it's in dimension (S, S_alpha, ~, ~))
      Z_i <- Z[[i]]
      Z_i_tf <- tf$constant(Z_i, dtype = "float64")
      Z_i_tf2 <- tf$reshape(Z_i_tf, c(1L, n, 1L))
      Z_i_tf3 <- tf$tile(Z_i_tf2, c(S, 1L, 1L))
      # Z_i_tf4 <- tf$reshape(Z_i_tf3, c(1L, dim(Z_i_tf3)))
      # Z_i_tf5 <- tf$tile(Z_i_tf4, c(S, 1L, 1L, 1L))
      
      ## Reshape and tile beta, exp(phi), exp(psi) so that 
      ## they're all in dimensions (S, S_alpha, ~, ~))
      beta_s_all_tf <- tf$constant(samples[, 1:(param_dim-2L)],
                                   dtype = "float64")
      beta_tf_2 <- tf$reshape(beta_s_all_tf, c(S, param_dim - 2L, 1L))
      
      phi_s_all_tf <- tf$constant(samples[, param_dim-1, drop = FALSE],
                                  dtype = "float64")
      exp_phi_s_all_tf <- tf$exp(phi_s_all_tf)
      exp_phi_s_all_tf2 <- tf$reshape(exp_phi_s_all_tf, c(S, 1L, 1L))
      # exp_phi_s_all_tf2 <- tf$reshape(exp_phi_s_all_tf, c(S, 1L, 1L, 1L))
      # exp_phi_s_all_tf3 <- tf$tile(exp_phi_s_all_tf2, c(1L, S_alpha, 1L, 1L))
      
      psi_s_all_tf <- tf$constant(samples[, param_dim, drop = FALSE],
                                  dtype = "float64")
      exp_psi_s_all_tf <- tf$exp(psi_s_all_tf)
      exp_psi_s_all_tf2 <- tf$reshape(exp_psi_s_all_tf, c(S, 1L, 1L))
      # exp_psi_s_all_tf2 <- tf$reshape(exp_psi_s_all_tf, c(S, 1L, 1L, 1L))
      # exp_psi_s_all_tf3 <- tf$tile(exp_psi_s_all_tf2, c(1L, S_alpha, 1L, 1L))
      
      ## Tile a bunch of identity matrices
      eye_tf <- tf$constant(diag(n), dtype = "float64")
      eye_tf2 <- tf$reshape(eye_tf, c(1L, n, n))
      eye_list_tf <- tf$tile(eye_tf2, c(S, 1L, 1L))
      # eye_list_tf2 <- tf$reshape(eye_list_tf, c(1L, dim(eye_list_tf)))
      # eye_list_tf3 <- tf$tile(eye_list_tf2, c(S, 1L, 1L, 1L))
      
      ## Then compute Sigma_y
      
      # cp_test <- Z[[i-1]] %*% t(Z[[i-1]])
      # cp_test_tf <- tf$constant(cp_test, dtype = "float64")
      # cp_test_tf2 <- tf$reshape(cp_test_tf, c(1L, dim(cp_test_tf)))
      # cp_test_tf3 <- tf$tile(cp_test_tf2, c(S, 1L, 1L))
      # Sigma_y_tf <- tf$multiply(cp_test_tf3, exp_phi_s_all_tf2) + diag_add
      # Sigma_y_inv_tf <- tf$linalg$inv(Sigma_y_tf)
      
      # Sigma_y <- exp(phi_s) * Z[[i-1]] %*% t(Z[[i-1]]) + diag(exp(psi_s), n)
      Sigma_y_tf <- tf$multiply(exp_phi_s_all_tf2,
                                tf$linalg$matmul(Z_i_tf3, 
                                                 tf$linalg$matrix_transpose(Z_i_tf3)
                                )
      ) + 
        tf$multiply(exp_psi_s_all_tf2, eye_list_tf)
      
      Sigma_y_inv_tf <- tf$linalg$inv(Sigma_y_tf)
      
      ## Compute the mean and variance of alpha_i for all the S samples at once
      
      y_tf <- tf$constant(t(outer(as.vector(y[[i]]), rep(1, S))), dtype = "float64")
      y_tf2 <- tf$reshape(y_tf, c(dim(y_tf), 1L))
      
      #y_tf <- tf$matmul(ones_S_alpha, tf$linalg$matrix_transpose(y_array_tf[i,,]))
      # y_tf2 <- tf$reshape(y_tf, c(1L, dim(y_tf), 1L))
      # y_tf3 <- tf$tile(y_tf2, c(S, 1L, 1L, 1L))
      
      TempMat <- tf$linalg$matmul(X_i_tf3, beta_tf_2)
      
      alpha_i_mean_tf <- tf$multiply(exp_phi_s_all_tf2, 
                                     tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matrix_transpose(Z_i_tf3),
                                                                       Sigma_y_inv_tf), y_tf2 - TempMat))
      
      alpha_i_var_tf <- exp_phi_s_all_tf2 - tf$multiply(tf$square(exp_phi_s_all_tf2),
                                                        tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matrix_transpose(Z_i_tf3), 
                                                                                          Sigma_y_inv_tf), 
                                                                         Z_i_tf3)
      )
      
      ## Draw S_alpha samples from each pair of mean-var
      alpha_i_samples_tf <- tf$random$normal(c(S_alpha, 1L), alpha_i_mean_tf, tf$math$sqrt(alpha_i_var_tf),
                                             dtype = "float64")
      alpha_i_samples_tf2 <- tf$reshape(alpha_i_samples_tf, c(dim(alpha_i_samples_tf), 1L))
      
      ## Now compute the gradient
      
      ## TIME TO RESHAPE A BUNCH OF STUFF to dim (S, S_alpha, ~, ~)
      X_i_tf4 <- tf$reshape(X_i_tf3, c(S, 1L, n, param_dim - 2L))
      X_i_tf5 <- tf$tile(X_i_tf4, c(1L, S_alpha, 1L, 1L))
      
      Z_i_tf4 <- tf$reshape(Z_i_tf3, c(S, 1L, n, 1L))
      Z_i_tf5 <- tf$tile(Z_i_tf4, c(1L, S_alpha, 1L, 1L))
      
      y_tf3 <- tf$reshape(y_tf2, c(S, 1L, n, 1L))
      y_tf4 <- tf$tile(y_tf3, c(1L, S_alpha, 1L, 1L))
      
      beta_tf_3 <- tf$reshape(beta_tf_2, c(S, 1L, param_dim - 2L, 1L))
      beta_tf_4 <- tf$tile(beta_tf_3, c(1L, S_alpha, 1L, 1L))
      
      y_minus_mean_tf <- y_tf4 - tf$linalg$matmul(X_i_tf5, beta_tf_4) - 
        tf$math$multiply(Z_i_tf5, alpha_i_samples_tf2)
      
      exp_phi_s_all_tf3 <- tf$reshape(exp_phi_s_all_tf2, c(S, 1L, 1L, 1L))
      exp_phi_s_all_tf4 <- tf$tile(exp_phi_s_all_tf3, c(1L, S_alpha, 1L, 1L))
      
      
      exp_psi_s_all_tf3 <- tf$reshape(exp_psi_s_all_tf2, c(S, 1L, 1L, 1L))
      exp_psi_s_all_tf4 <- tf$tile(exp_psi_s_all_tf3, c(1L, S_alpha, 1L, 1L))
      
      ## Compute gradients reeeeeeeeeeeee
      grad_beta_tf <- tf$multiply(tf$math$reciprocal(exp_psi_s_all_tf4),
                                  tf$linalg$matmul(tf$linalg$matrix_transpose(X_i_tf5),
                                                   y_minus_mean_tf))
      
      grad_phi_tf <- -1/2 + tf$multiply(1/(2*exp_phi_s_all_tf4), tf$square(alpha_i_samples_tf2)) 
      
      grad_psi_tf <- -n/2 + tf$multiply(tf$math$reciprocal(2 * exp_psi_s_all_tf4), 
                                        tf$linalg$matmul(tf$linalg$matrix_transpose(y_minus_mean_tf), 
                                                         y_minus_mean_tf)
      )
      
      grad_theta_tf <- tf$concat(list(grad_beta_tf, grad_phi_tf, grad_psi_tf),
                                 2L)
      score_all_tf <- tf$reduce_mean(grad_theta_tf, 1L)
      
      ## Now the Hessian
      A_all_tf <- tf$linalg$matmul(score_all_tf, tf$linalg$matrix_transpose(score_all_tf))
      B1_terms_tf <- tf$linalg$matmul(grad_theta_tf, tf$linalg$matrix_transpose(grad_theta_tf))
      B1_tf <- tf$reduce_mean(B1_terms_tf, 1L)
      
      grad2_beta_tf <- - tf$multiply(tf$math$reciprocal(exp_psi_s_all_tf4), 
                                     tf$linalg$matmul(tf$linalg$matrix_transpose(X_i_tf5), X_i_tf5)) 
      grad2_phi_tf <- - tf$multiply(tf$math$reciprocal(2 * exp_phi_s_all_tf4), tf$square(alpha_i_samples_tf2))
      grad2_psi_tf <- - tf$multiply(tf$math$reciprocal(2 * exp_psi_s_all_tf4), 
                                    tf$linalg$matmul(tf$linalg$matrix_transpose(y_minus_mean_tf), y_minus_mean_tf))
      grad2_beta_psi_tf <- - tf$multiply(tf$math$reciprocal(exp_psi_s_all_tf4), 
                                         tf$linalg$matmul(tf$linalg$matrix_transpose(X_i_tf5),
                                                          y_minus_mean_tf)
      )
      ## Some padding so that we can arrange the components of the Hessian into a 6x6 matrix
      grad2_theta_tf <- tf$pad(grad2_beta_tf, matrix(c(0L, 0L, 0L, 0L, 0L, 2L, 0L, 2L), 4, 2, byrow = TRUE)) +
        tf$pad(grad2_phi_tf, matrix(c(0L, 0L, 0L, 0L, param_dim - 2L, 1L, param_dim - 2L, 1L), 4, 2, byrow = TRUE)) +
        tf$pad(grad2_psi_tf, matrix(c(0L, 0L, 0L, 0L, param_dim - 1L, 0L, param_dim - 1L, 0L), 4, 2, byrow = TRUE)) +
        tf$pad(grad2_beta_psi_tf, matrix(c(0L, 0L, 0L, 0L, 0L, 2L, param_dim - 1L, 0L), 4, 2, byrow = TRUE)) +
        tf$pad(tf$linalg$matrix_transpose(grad2_beta_psi_tf), 
               matrix(c(0L, 0L, 0L, 0L, param_dim - 1L, 0L, 0L, 2L), 4, 2, byrow = TRUE))
      
      B2_tf <- tf$reduce_mean(grad2_theta_tf, 1L)
      
      hessian_all_tf <- B1_tf + B2_tf - A_all_tf
      
      E_score_tf <- tf$math$reduce_mean(score_all_tf, 0L)
      E_hessian_tf <- tf$math$reduce_mean(hessian_all_tf, 0L)
      
      # cat("a =", a, "\n")
      
      prec_temp <- prec_temp - a * as.matrix(E_hessian_tf)
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_score_tf))   
    }
    
    prec[[i+1]] <- prec_temp #prec[[i-1]] - as.matrix(E_hessian_tf)
    mu_vals[[i+1]] <- mu_temp #mu_vals[[i-1]] + chol2inv(chol(prec[[i]])) %*% as.matrix(E_score_tf)        
    
  }
  
  rvgal.t2 <- proc.time()
  print(rvgal.t2 - rvgal.t1)
  
  ## Posterior samples
  rvgal.post_var <- solve(prec[[N+1]])
  rvgal.post_samples <- rmvnorm(10000, mu_vals[[N+1]], rvgal.post_var) # these are samples of beta, log(sigma_a^2), log(sigma_e^2)
  
  ## Save results
  rvgal_results <- list(mu = mu_vals,
                        prec = prec,
                        post_samples = rvgal.post_samples,
                        S = S,
                        S_alpha = S_alpha, 
                        N = N, 
                        n = n,
                        time_elapsed = rvgal.t2 - rvgal.t1)
  
  # if (save_results) {
  #   if (use_tempering) {
  #     filename <- paste0("linear_mm_rvga_fisher_temper", n_obs_to_temper, 
  #                        "_N", N, "_n", n, 
  #                        "_S", S, "_Sa", S_alpha,
  #                        "_", date, ".rds")
  #   } else {
  #     filename <- paste0("linear_mm_rvga_fisher_N", N, "_n", n, 
  #                        "_S", S, "_Sa", S_alpha,
  #                        "_", date, ".rds")
  #   }
  #   saveRDS(rvgal_results, file = filename)
  # }
  
  return(rvgal_results)
  
}