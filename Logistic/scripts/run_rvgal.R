run_rvgal <- function(y, X, mu_0, P_0, S = 100L, S_alpha = 100L,
                      use_tempering = T, n_temper = 10, 
                      temper_schedule = rep(0.25, 4),
                      n_post_samples = 10000,
                      save_results = F) {
  
  if (!use_tempering) {
    temper_schedule <- 1
  }
  
  print("Starting R-VGA...")
  t1 <- proc.time()
  
  # n_fixed_effects <- as.integer(ncol(X[[1]]))
  param_dim <- as.integer(length(mu_0))
  
  ## Sample from the "prior"
  ## par(mfrow = c(1, 1))
  ## test_omega <- rnorm(10000, mu_0[param_dim], P_0[param_dim, param_dim])
  ## plot(density(sqrt(exp(test_omega))), main = "RVGA: Prior of tau")
  
  mu_vals <- lapply(1:N, function(x) mu_0)
  prec <- lapply(1:N, function(x) solve(P_0))
  
  for (i in 1:N) {
    
    if (S >= 500 && S_alpha >= 500) {
      gc()
    }
    
    a_vals <- 1 # for tempering
    if (use_tempering) {
      if (i <= n_temper) { # only temper the first n_temper observations
        a_vals <- temper_schedule
      }  
    }
    
    mu_temp <- mu_vals[[i]]
    prec_temp <- prec[[i]] 
    
    for (v in 1:length(a_vals)) {
      
      a <- a_vals[v]
      
      P <- chol2inv(chol(prec_temp))
      samples <- rmvnorm(S, mu_temp, P)
      
      X_i <- X[[i]]
      X_i_tf <- tf$constant(X_i, dtype = "float64")
      #X_i_tf <- X_array_tf[i,,]        
      X_i_tf2 <- tf$reshape(X_i_tf, c(1L, n, param_dim - 1L))
      X_i_tf3 <- tf$tile(X_i_tf2, c(S_alpha, 1L, 1L))
      X_i_tf4 <- tf$reshape(X_i_tf3, c(1L, dim(X_i_tf3)))
      X_i_tf5 <- tf$tile(X_i_tf4, c(S, 1L, 1L, 1L))
      
      beta_s_all_tf <- tf$constant(samples[, 1:(param_dim-1)],
                                   dtype = "float64")
      
      omega_s_all_tf <- tf$constant(samples[, param_dim, drop = FALSE],
                                    dtype = "float64")
      
      tau_s_all_tf <- tf$sqrt(tf$exp(omega_s_all_tf))
      tau_s_all <- sqrt(exp(samples[, param_dim, drop = FALSE]))   
      alpha_all <- t(sapply(tau_s_all, function(x) rnorm(S_alpha, 0, x)))
      
      beta_tf_2 <- tf$reshape(beta_s_all_tf, c(S, 1L, param_dim - 1L, 1L))
      beta_tf_3 <- tf$tile(beta_tf_2, c(1L, S_alpha, 1L, 1L))
      
      omega_tf_2 <- tf$reshape(omega_s_all_tf, c(S, 1L, 1L, 1L))
      omega_tf_3 <- tf$tile(omega_tf_2, c(1L, S_alpha, 1L, 1L))
      
      
      alpha_tf <- tf$constant(alpha_all, dtype = "float64")
      alpha_tf_2 <- tf$reshape(alpha_tf, c(dim(alpha_all), 1L, 1L))
      alpha_tf_3 <- tf$tile(alpha_tf_2, c(1L, 1L, n, 1L))
      
      y_tf <- tf$constant(t(outer(y[[i]], rep(1, S_alpha))), dtype = "float64")
      #y_tf <- tf$matmul(ones_S_alpha, tf$linalg$matrix_transpose(y_array_tf[i,,]))
      y_tf2 <- tf$reshape(y_tf, c(1L, dim(y_tf), 1L))
      y_tf3 <- tf$tile(y_tf2, c(S, 1L, 1L, 1L))
      
      TempMat <- tf$linalg$matmul(X_i_tf5, beta_tf_3) + alpha_tf_3
      
      log_pi <- -tf$math$log(1 + tf$exp(-TempMat))
      log_1_minus_pi <- -TempMat - tf$math$log(1 + tf$exp(-TempMat))
      log_likelihood <- y_tf3 * log_pi + (1 - y_tf3) * log_1_minus_pi
      log_weights_all <- tf$math$reduce_sum(log_likelihood, c(2L, 3L))
      
      shifted_weights_all_tf <-log_weights_all - tf$math$reduce_max(log_weights_all, 1L, keepdims = TRUE)
      normalised_weights_all_tf <- tf$exp(shifted_weights_all_tf) /
        tf$math$reduce_sum(tf$exp(shifted_weights_all_tf), 1L, keepdims = TRUE)
      
      scalars_tf <- tf$squeeze(y_tf3 - tf$math$reciprocal(1 + tf$exp(-TempMat)))
      scalars_tf2 <- tf$linalg$diag(scalars_tf)
      
      
      grad_j_tf <- tf$linalg$matmul(
        tf$linalg$matrix_transpose(X_i_tf3),
        scalars_tf2)
      
      grad_beta_tf <- tf$math$reduce_sum(grad_j_tf, axis = 3L, keepdims = TRUE)
      
      grad_omega_tf <- -1/2 + 1/2 * tf$math$divide(tf$expand_dims(tf$math$square(alpha_tf_3[,,1L,]), 3L),
                                                   tf$exp(omega_tf_3))
      
      grad_theta_tf <- tf$concat(list(grad_beta_tf, grad_omega_tf),
                                 2L)
      
      probs_tf <-   tf$squeeze(tf$math$reciprocal(1 + tf$exp(-TempMat)))
      rho_tf <- tf$linalg$diag(probs_tf * (1 - probs_tf))
      
      grad2_beta_tf <- -tf$linalg$matmul(
        tf$linalg$matmul(
          tf$linalg$matrix_transpose(X_i_tf5),
          rho_tf),
        X_i_tf5)
      grad2_omega_tf <- -1/2 * tf$math$divide(tf$expand_dims(tf$math$square(alpha_tf_3[,,1L,]), 3L),
                                              tf$exp(omega_tf_3))
      
      
      ## Put some padding around grad2_beta (4x4) and grad2_omega(1x1) so that 
      ## they both become 5x5, then add them to form grad2_theta (5x5)
      grad2_theta_tf <-   tf$pad(grad2_beta_tf, matrix(c(0L,0L, 0L, 0L, 0L, 1L, 0L, 1L), 4, 2, byrow = TRUE)) +
        tf$pad(grad2_omega_tf, matrix(c(0L, 0L, 0L, 0L, param_dim - 1L, 0L, param_dim - 1L, 0L), 4, 2, byrow = TRUE))
      
      
      normalised_weights_all_diag <- tf$linalg$diag(normalised_weights_all_tf)
      weighted_grad_all_tf <-tf$matmul(normalised_weights_all_diag,
                                       tf$squeeze(grad_theta_tf))
      
      
      XXt_all <- tf$linalg$matmul(grad_theta_tf, tf$linalg$matrix_transpose(grad_theta_tf))
      NW_all <- tf$expand_dims(normalised_weights_all_tf, 2L)
      NW2_all <- tf$tile(NW_all, c(1L, 1L, param_dim))
      NW3_all <- tf$linalg$diag(NW2_all, param_dim)
      B1_terms_all_tf <- tf$linalg$matmul(XXt_all, NW3_all)
      B2_terms_all_tf <- tf$linalg$matmul(grad2_theta_tf, NW3_all)
      
      score_all_tf <- tf$expand_dims(tf$reduce_sum(weighted_grad_all_tf, 1L), 2L)
      
      
      #        stopifnot(!is.na(score[[s]]))
      
      ## Approximation of the Hessian
      A_all_tf <- tf$linalg$matmul(score_all_tf, tf$linalg$matrix_transpose(score_all_tf))
      
      B1_all_tf <- tf$reduce_sum(B1_terms_all_tf, 1L)
      B2_all_tf <- tf$reduce_sum(B2_terms_all_tf, 1L)
      
      Hessian_all_tf <- B1_all_tf + B2_all_tf - A_all_tf
      
      E_score_tf <- tf$math$reduce_mean(score_all_tf, 0L)
      E_hessian_tf <- tf$math$reduce_mean(Hessian_all_tf, 0L)
      
      # cat("a =", a, "\n")
      prec_temp <- prec_temp - a * as.matrix(E_hessian_tf)
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_score_tf))       
      
    }
    
    prec[[i+1]] <- prec_temp #prec[[i]] - as.matrix(E_hessian_tf)
    mu_vals[[i+1]] <- mu_temp #mu_vals[[i]] + chol2inv(chol(prec[[i+1]])) %*% as.matrix(E_score_tf)        
    
    if (i %% (N/10) == 0) {
      cat(i/N * 100, "% complete \n")
    }
    
  }
  
  t2 <- proc.time()
  print(t2 - t1)
  
  post_var <- solve(prec[[N+1]])
  rvga.post_samples <- rmvnorm(n_post_samples, mu_vals[[N+1]], post_var)  
  
  rvga_results <- list(mu = mu_vals,
                       prec = prec,
                       post_samples = rvga.post_samples,
                       S = S,
                       S_alpha = S_alpha,
                       N = N, 
                       n = n,
                       use_tempering = use_tempering,
                       temper_schedule = temper_schedule,
                       time_elapsed = t2 - t1)
  
  return(rvga_results)
}