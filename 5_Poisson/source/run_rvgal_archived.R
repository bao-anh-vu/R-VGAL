run_rvgal <- function(y, X, Z, mu_0, P_0, S = 100L, S_alpha = 100L,
                      use_tempering = T, n_temper = 10, 
                      temper_schedule = rep(0.25, 4),
                      n_post_samples = 10000,
                      use_tf = T,
                      save_results = F) {
  
  tf64 <- function(x) tf$constant(x, dtype = "float64")
  
  if (!use_tempering) {
    temper_schedule <- 1
  }
  
  print("Starting R-VGA...")
  t1 <- proc.time()
  
  n_fixed_effects <- as.integer(ncol(X[[1]]))
  n_random_effects <- as.integer(ncol(Z[[1]]))
  
  param_dim <- as.integer(length(mu_0))
  N <- length(y)
  n <- length(y[[1]])
  ## Sample from the "prior"
  ## par(mfrow = c(1, 1))
  ## test_zeta <- rnorm(10000, mu_0[param_dim], P_0[param_dim, param_dim])
  ## plot(density(sqrt(exp(test_zeta))), main = "RVGA: Prior of tau")
  
  mu_vals <- lapply(1:N, function(x) mu_0)
  prec <- lapply(1:N, function(x) solve(P_0))
  
  for (i in 1:N) {
    
    cat("i = ", i, "\n")
    
    # if (S >= 500 && S_alpha >= 500) {
    gc()
    # }
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
      
      samples_list <- lapply(1:S, function(r) samples[r, -(1:n_fixed_effects)])
      
      log_likelihood <- list()
      grads <- list()
      hessians <- list()
      
      y_i_tf <- tf$Variable(y[[i]], dtype = "float64")
      X_i_tf <- tf$Variable(X[[i]], dtype = "float64")
      Z_i_tf <- tf$Variable(Z[[i]], dtype = "float64")
      
      ####### Development ###########
      beta_all <- samples[, 1:n_fixed_effects]
      
      Sigma_alpha_all <- lapply(samples_list, construct_Sigma,
                                d = n_random_effects)
      
      if (n_random_effects == 1) {
        alpha_all <- lapply(Sigma_alpha_all,
                            function(Sigma) rnorm(S_alpha, 0, sqrt(Sigma)))
      } else {
        alpha_all <- lapply(Sigma_alpha_all,
                            function(Sigma) rmvnorm(S_alpha, rep(0, n_random_effects), Sigma))
      } 
      
      ########### TF ###########
      
      if (use_tf) {
        
        alpha_all_tf <- tf$Variable(alpha_all, dtype = "float64")
        theta_tf <- tf$Variable(samples, dtype = "float64")
        # 
        # # # #
        tf_out <- compute_grad_hessian2(y_i_tf, X_i_tf, Z_i_tf,
                                        alpha_all_tf, theta_tf, S_alpha)
        # # 
        # tf_out <- compute_grad_hessian(y_i_tf, X_i_tf, Z_i_tf,
        #                                 alpha_all_tf, theta_tf, S_alpha)
        
        # browser()
        E_score_tf <- tf$math$reduce_mean(tf_out$grad, 0L)
        E_hessian_tf <- tf$math$reduce_mean(tf_out$hessian, 0L)
        prec_temp <- prec_temp - a * as.matrix(E_hessian_tf)
        mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_score_tf))
        
      } else {
        ############# Theoretical gradient ############
        prior_var_beta <- diag(P_0)[1:n_fixed_effects]
        
        I_d <- diag(n_random_effects)
        
        log_likelihood <- list()
        log_likelihood_y <- list()
        log_likelihood_alpha <- list()
        
        llh_test <- list()
        grad_beta <- list()
        grad_zeta <- list()
        grad_joint <- list()
        hess_joint <- list()
        grad_is <- list()
        hess_is <- list()
        for (l in 1:S) {
          theta_l <- samples[l, ]
          
          beta_l <- theta_l[1:n_fixed_effects]
          
          Sigma_alpha_l <- construct_Sigma(theta_l[-(1:n_fixed_effects)],
                                           d = n_random_effects)
          alpha_l <- alpha_all[[l]]
          
          llh_tan <- list() # log likelihood from Tan and Nott
          llh_y_tan <- list()
          llh_alpha_tan <- list()
          
          log_weights <- c()
          grad_beta_l <- list()
          grad_zeta_l <- list()
          grad_joint_l <- list()
          hess_beta_l <- list()
          hess_zeta_l <- list()
          hess_joint_l <- list()
          
          L <- t(chol(Sigma_alpha_l))
          L_inv <- solve(L)
          
          for (s in 1:S_alpha) {
            
            if (n_random_effects == 1) {
              alpha_l_s <- alpha_l[s]
            } else {
              alpha_l_s <- alpha_l[s, ]
            }
            
            ## Log likelihood from Tan and Nott -- to compare with TF llh
            llh_y_tan[[s]] <- y[[i]] %*% (X[[i]] %*% beta_l + Z[[i]] %*% alpha_l_s) -
              sum(exp(X[[i]] %*% beta_l + Z[[i]] %*% alpha_l_s)) - sum(lfactorial(y[[i]]))
            llh_alpha_tan[[s]] <- - 1/2 * n_random_effects * log(2*pi) - sum(log(diag(L))) - 1/2 * t(alpha_l_s) %*% t(L_inv) %*% L_inv %*% alpha_l_s
            llh_tan[[s]] <- llh_y_tan[[s]] + llh_alpha_tan[[s]]
            log_weights[s] <- llh_y_tan[[s]]
            
            ## Gradients of p(y_i, alpha_i^(s) | beta, zeta^(s)) from Tan and Nott
            grad_beta_l[[s]] <- t(y[[i]] - exp(X[[i]] %*% beta_l + Z[[i]] %*% alpha_l_s)) %*% X[[i]] #- 1/prior_var_beta * beta_l
            
            # ## gradient of zeta from Tan and Nott
            # I_zeta <- L #diag(alpha_l_s)
            # I_zeta[lower.tri(I_zeta)] <- 1
            #
            # A <- t(L_inv) %*% L_inv %*% alpha_l_s %*% t(alpha_l_s) %*% t(L_inv)
            #
            # grad_zeta_tan <- - I_d[lower.tri(I_d, diag = T)] +
            #   I_zeta[lower.tri(I_zeta, diag = T)] * A[lower.tri(A, diag = T)]
            
            
            ## My own derivations
            ind_df <- data.frame(r = c(1,2,2), c = c(1,2,1))
            grad_zeta_test <- matrix(0, n_random_effects, n_random_effects)
            grad_L_zeta_list <- list()
            grad2_L_zeta_list <- list()
            
            grad_p_L <- - t(L_inv) + t(L_inv) %*% L_inv %*% alpha_l_s %*% t(alpha_l_s) %*% t(L_inv)
            
            for (row in 1:nrow(ind_df)) {
              j <- ind_df[row, 1]
              k <- ind_df[row, 2]
              grad_L_zeta_jk <- matrix(0, nrow = n_random_effects, ncol = n_random_effects) # since the diagonals of L are exp(zeta_ii)
              grad2_L_zeta_jk <- matrix(0, nrow = n_random_effects, ncol = n_random_effects) # since the diagonals of L are exp(zeta_ii)
              
              if (j == k) {
                grad_L_zeta_jk[j,k] <- L[j,k]
                grad2_L_zeta_jk[j,k] <- L[j,k]
              } else {
                grad_L_zeta_jk[j,k] <- 1
              }
              grad_L_zeta_list[[row]] <- grad_L_zeta_jk
              grad2_L_zeta_list[[row]] <- grad2_L_zeta_jk
              
              grad_zeta_test[j,k] <- sum(diag(t(grad_p_L) %*% grad_L_zeta_jk))
            }
            grad_zeta_l[[s]] <- c(diag(grad_zeta_test), grad_zeta_test[lower.tri(grad_zeta_test)])
            
            grad_joint_l[[s]] <- c(grad_beta_l[[s]], grad_zeta_l[[s]])
            
            ## Hessian of beta
            hess_beta_l[[s]] <- - t(X[[i]]) %*% diag(as.vector(exp(X[[i]] %*% beta_l + Z[[i]] %*% alpha_l_s))) %*% X[[i]]
            
            ## Hessian of zeta
            grad2_zeta <- c()
            Amat <- - t(L_inv) + t(L_inv) %*% L_inv %*% alpha_l_s %*% t(alpha_l_s) %*% t(L_inv)
            Bmat <- L_inv %*% alpha_l_s %*% t(alpha_l_s) %*% t(L_inv)
            grad_Linv_zeta_list <- list()
            grad_A_zeta_list <- list()
            grad_B_zeta_list <- list()
            
            ## the diagonal entries
            for (row in 1:nrow(ind_df)) {
              j <- ind_df[row, 1]
              k <- ind_df[row, 2]
              grad_Linv_zeta_jk <- matrix(0, n_random_effects, n_random_effects)
              if (j == k) {
                grad_Linv_zeta_jk[j,k] <- -L_inv[j,k]
                grad_Linv_zeta_jk[2,1] <- L[2,1]/(L[1,1] * L[2,2])
              } else {
                grad_Linv_zeta_jk[2,1] <- -1/(L[1,1] * L[2,2])
              }
              grad_Linv_zeta_list[[row]] <- grad_Linv_zeta_jk
              grad_B_zeta_jk <- grad_Linv_zeta_jk %*% tcrossprod(alpha_l_s) %*% t(L_inv) +
                L_inv %*% tcrossprod(alpha_l_s) %*% t(grad_Linv_zeta_jk)
              
              grad_A_zeta_jk <- - t(grad_Linv_zeta_jk) + t(grad_Linv_zeta_jk) %*% Bmat +
                t(L_inv) %*% grad_B_zeta_jk
              
              grad2_zeta_jk <- sum(diag(t(grad_A_zeta_jk) %*% grad_L_zeta_list[[row]] +
                                          t(Amat) %*% grad2_L_zeta_list[[row]]))
              grad2_zeta[row] <- grad2_zeta_jk
              
              grad_A_zeta_list[[row]] <- grad_A_zeta_jk
              grad_B_zeta_list[[row]] <- grad_B_zeta_jk
              
            }
            
            ## Now the cross entries
            grad_A_zeta_11 <- grad_A_zeta_list[[1]]
            grad_A_zeta_22 <- grad_A_zeta_list[[2]]
            grad_A_zeta_21 <- grad_A_zeta_list[[3]]
            
            grad_L_zeta_11 <- grad_L_zeta_list[[1]]
            grad_L_zeta_22 <- grad_L_zeta_list[[2]]
            grad_L_zeta_21 <- grad_L_zeta_list[[3]]
            
            grad2_zeta_22_11 <- sum(diag(t(grad_A_zeta_22) %*% grad_L_zeta_11))
            grad2_zeta_21_11 <- sum(diag(t(grad_A_zeta_21) %*% grad_L_zeta_11))
            grad2_zeta_21_22 <- sum(diag(t(grad_A_zeta_21) %*% grad_L_zeta_22))
            
            grad2_zeta_mat <- diag(grad2_zeta)
            grad2_zeta_mat[2,1] <- grad2_zeta_22_11
            grad2_zeta_mat[3,1] <- grad2_zeta_21_11
            grad2_zeta_mat[3,2] <- grad2_zeta_21_22
            grad2_zeta_mat[upper.tri(grad2_zeta_mat)] = t(grad2_zeta_mat[lower.tri(grad2_zeta_mat)])
            
            hess_zeta_l[[s]] <- grad2_zeta_mat
            hess_joint_l[[s]] <- bdiag(hess_beta_l[[s]], hess_zeta_l[[s]])
            
          }
          
          ## likelihood seems ok -- same between TF and for loop
          log_likelihood[[l]] <- unlist(llh_tan)
          log_likelihood_y[[l]] <- unlist(llh_y_tan)
          log_likelihood_alpha[[l]] <- unlist(llh_alpha_tan)
          
          # grad_beta[[l]] <- grad_beta_l
          # grad_zeta[[l]] <- grad_zeta_l
          #
          grad_joint[[l]] <- grad_joint_l
          hess_joint[[l]] <- hess_joint_l
          # test <- poisson_joint_likelihood(y[[i]], X[[i]], Z[[i]],
          #                                  alpha_l, theta_l, S_alpha)
          # llh_test[[l]] <- test$llh
          # llh_test_y <- test$llh_y
          # llh_test_alpha <- test$llh_alpha
          
          ## Now importance sampling
          max_weights <- max(log_weights)
          log_w_shifted <- log_weights - max_weights
          norm_weights <- as.list(exp(log_w_shifted) / sum(exp(log_w_shifted)))
          weighted_grad <- Map("*", norm_weights, grad_joint_l)
          grad_is[[l]] <- Reduce("+", weighted_grad)
          
          hess_part1 <- tcrossprod(grad_is[[l]])
          joint_grad_crossprod <- lapply(grad_joint_l, "tcrossprod")
          unweighted_hess <- Map("+", joint_grad_crossprod, hess_joint_l)
          weighted_hess <- Map("*", norm_weights, unweighted_hess)
          hess_part2 <- Reduce("+", weighted_hess)
          
          hess_is[[l]] <- as.matrix(hess_part2) - hess_part1
          
        }
        
        E_score_tf2 <- Reduce("+", grad_is)/length(grad_is)
        E_hessian_tf2 <- Reduce("+", hess_is)/length(grad_is)
        
        prec_temp <- prec_temp - a * as.matrix(E_hessian_tf2)
        mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_score_tf2))
      }
      
      ############ TF #############
      
      # 
      # tf_grad_beta <- tf_grad[,, 1:n_fixed_effects]
      # tf_grad_zeta <- tf_grad[,, (n_fixed_effects+1):param_dim]
      # 
      # tf_out <- compute_joint_llh_tf2(y_i_tf, X_i_tf, Z_i_tf,
      #                                 alpha_all_tf, theta_tf, S_alpha)
      # tf_grad <- tf_out$grad
      # tf_hessian <- tf_out$hessian
      # tf_hessian <- tf_out$hessian
      # browser()
      
      #### TF ####
      
      ###### end dev ###############
      ## Check against gradient from Tan and Nott
      
      
      #   
      #   # Now we need a for loop from s = 1, ..., S_alpha
      #   # and do importance sampling
      #   
      #   # alpha_i <- rmvnorm(S_alpha, rep(0, n_random_effects), Sigma_alpha_l)
      #   alpha_i <- alpha_all[[1]]
      #   alpha_i_tf <- tf$Variable(alpha_i)
      #   
      #   # plot(density(rmvnorm(1000L, rep(0, n_random_effects), Sigma_alpha_l)[, 1]))
      #   # browser()
      #   # L <- t(chol(Sigma_alpha_l))
      #   
      #   # norm <- tfd$MultivariateNormalTriL(loc = 0, scale_tril = L)
      #   # alpha_i_test <- norm$sample(S_alpha)
      #   
      #   theta_l_tf <- tf$Variable(theta_l, dtype = "float64")
      #   
      #   ## test compute_grad_hessian
      #   tf_out <- compute_grad_hessian(y_i_tf, X_i_tf, Z_i_tf,
      #                                  alpha_i_tf, theta_l_tf)
      #   
      #   tf_grad <- tf_out$grad
      #   tf_hessian <- tf_out$hessian
      #   
      #   grads[[l]] <- as.vector(tf_grad)
      #   hessians[[l]] <- as.matrix(tf_hessian)
      #   ########################################################################
      #   # 
      #   # tf_out <- compute_joint_llh_tf(y_i_tf, X_i_tf, Z_i_tf,
      #   #                                           alpha_i_tf, theta_l_tf)
      #   # 
      #   # log_likelihood_tf <- tf_out$llh
      #   # joint_grads <- tf_out$grad # this is only the gradient of the joint llh
      #   # joint_hessians <- tf_out$hessian
      #   # 
      #   # joint_grads <- lapply(1:S_alpha, function(r) as.vector(joint_grads[r,]))
      #   # joint_hessians <- lapply(1:S_alpha, function(r) as.matrix(joint_hessians[r,,]))
      #   # 
      #   # # log_weights_tf <- tf_out$log_weights
      #   # weights <- as.list(tf_out$weights)
      #   # 
      ############ Test with the usual for loop ##############
      #   test <- poisson_joint_likelihood(y[[i]], X[[i]], Z[[i]],
      #                                              alpha_i, theta_l, S_alpha)
      #   log_likelihood <- test$llh
      #   llh_y <- test$llh_y
      # 
      #   lambda_i <- exp(X_i %*% beta + Z_i %*% alpha_i_s)
      #   llh_y_i_s <- dpois(y_i, lambda_i, log = T)
      #   llh_y_i_s <- sum(llh_y_i_s)
      # 
      #   ## Leights for importance sampling
      #   log_weights <- c()
      #   for (s in 1:S_alpha) { # parallelise later
      #     lambda_i_s <- exp(X[[i]] %*% beta + Z[[i]] %*% alpha_i[, s])
      #     llh_y_i_s <- dpois(y[[i]], lambda_i_s, log = T)
      #     log_weights[s] <- sum(llh_y_i_s)
      #   }
      #   log_w_shifted <- log_weights - max(log_weights)
      #   weights <- as.list(exp(log_w_shifted)/sum(exp(log_w_shifted))) # normalised weights
      #   
      # }
      #   browser()
      #   # ################ end for loop ##################
      #   # 
      #   # weighted_grads <- Map('*', weights, joint_grads)
      #   # # log_likelihood[[l]] <- poisson_joint_likelihood(y_i = y[[i]], X_i = X[[i]], 
      #   # #                                              Z_i = Z[[i]], theta = theta_l,
      #   # #                                              alpha_i = alpha_i,
      #   # #                                              S_alpha = S_alpha)$llh
      #   # grads[[l]] <- Reduce("+", weighted_grads)
      #   # 
      #   # 
      #   # 
      #   # ## Now the Hessian
      #   # hess_part1 <- tcrossprod(grads[[l]])
      #   # 
      #   # joint_grad_crossprod <- lapply(joint_grads, tcrossprod)
      #   # unweighted_hess <- Map('+', joint_grad_crossprod, joint_hessians)
      #   # weighted_hess <- Map('*', weights, unweighted_hess)
      #   # hess_part2 <- Reduce("+", weighted_hess)
      #   # 
      #   # hessians[[l]] <- hess_part2 - hess_part1
      #   # 
      # }
      # 
      # E_grad <- Reduce("+", grads)/ length(grads)
      # E_hessian <- Reduce("+", hessians)/ length(hessians)
      # 
      # prec_temp <- prec_temp - a * E_hessian
      # 
      # if (any(eigen(prec_temp)$values < 0)) {
      #   browser()
      # }
      # mu_temp <- mu_temp + a * chol2inv(chol(prec_temp)) %*% E_grad    
      
    }
    
    prec[[i+1]] <- prec_temp #prec[[i]] - as.matrix(E_hessian_tf)
    mu_vals[[i+1]] <- mu_temp #mu_vals[[i]] + chol2inv(chol(prec[[i+1]])) %*% as.matrix(E_score_tf)        
    
    if (i %% (N/10) == 0) {
      cat(i/N * 100, "% complete \n")
    }
    
  }
  
  t2 <- proc.time()
  # print(t2 - t1)
  
  post_var <- chol2inv(chol(prec[[N+1]]))
  rvgal.post_samples <- rmvnorm(n_post_samples, mu_vals[[N+1]], post_var)  
  
  post_samples_list <- lapply(1:n_post_samples, function(r) rvgal.post_samples[r, -(1:n_fixed_effects)])
  post_samples_Sigma <- lapply(post_samples_list, construct_Sigma,
                               d = n_random_effects)
  
  nlower <- n_random_effects * (n_random_effects-1)/2 + n_random_effects
  lower_ind <- lapply(1:nlower, index_to_i_j_rowwise_diag)
  for (d in 1:(param_dim - n_fixed_effects)) {
    inds <- lower_ind[[d]]
    rvgal.post_samples[, n_fixed_effects+d] <- unlist(lapply(post_samples_Sigma, function(Sigma) Sigma[inds[1], inds[2]]))
  }
  
  rvgal_results <- list(mu = mu_vals,
                        prec = prec,
                        post_samples = rvgal.post_samples,
                        S = S,
                        S_alpha = S_alpha,
                        N = N, 
                        n = n,
                        use_tempering = use_tempering,
                        temper_schedule = temper_schedule,
                        time_elapsed = t2 - t1)
  
  return(rvgal_results)
}

index_to_i_j_rowwise_nodiag <- function(k) { # maps vector entries to lower triangular indices
  kp <- k - 1
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- p + 2
  j  <- kp - p * (p + 1) / 2 + 1
  c(i, j)
}

index_to_i_j_rowwise_diag <- function(k) {
  p  <- (sqrt(1 + 8 * k) - 1) / 2
  i0 <- floor(p)
  if (i0 == p) {
    return(c(i0, i0)) # (i, j)
  } else {
    i <- i0 + 1
    j <- k - i0 * (i0 + 1) / 2
    c(i, j)
  }
}

construct_Sigma <- function(theta, d, use_chol = T) { #d is the dimension of Sigma_eta
  
  if (d == 1) {
    Sigma <- exp(theta[1])^2
  } else {
    nlower <- d*(d-1)/2
    L <- diag(exp(theta[1:d]))
    offdiags <- theta[-(1:d)] # off diagonal elements are those after the first 2*d elements
    if (use_chol) {
      for (k in 1:nlower) {
        ind <- index_to_i_j_rowwise_nodiag(k)
        L[ind[1], ind[2]] <- offdiags[k]
      }
      Sigma <- L %*% t(L)
    } else {
      Sigma <- L
    }
    
  }
  
  return(Sigma)
}

to_triangular <- function (x, d) { ## for stan
  # // could check rows(y) = K * (K + 1) / 2
  # matrix[K, K] y;    
  K = d-1
  pos = 1
  nlower = d*(d-1)/2
  # L = matrix(NA, d, d)
  L = exp(diag(x[1:d]))
  for (i in 2:d) {
    for (j in 1:(i-1)) {
      L[i,j] = x[pos]
      pos = pos + 1
    }
  }
  
  return (L)
  # for (int i = 1; i < K; ++i) {
  #   for (int j = 1; j <= i; ++j) {
  #     y[i, j] = y_basis[pos];
  #     pos += 1;
  #   }
  # }
  # return y;
}

poisson_joint_likelihood <- function(y_i, X_i, Z_i, alpha_i, theta, S_alpha) { 
  # beta, alpha_i, Sigma_alpha) {
  ## Construct parameters on the unstransformed scale
  n_fixed_effects <- ncol(X_i)
  n_random_effects <- ncol(Z_i)
  
  beta <- theta[1:n_fixed_effects]
  
  Sigma_alpha <- construct_Sigma(theta[-(1:n_fixed_effects)], 
                                 d = n_random_effects)
  
  ## Now we need a for loop from s = 1, ..., S_alpha
  ## and do importance sampling
  
  # alpha_i <- t(rmvnorm(S_alpha, rep(0, n_random_effects), Sigma_alpha))
  
  llh <- c()  
  llh_y <- c()
  llh_alpha <- c()
  
  for (s in 1:S_alpha) {
    alpha_i_s <- alpha_i[s, ]
    
    # llh_y_i_s <- c()
    # for (j in 1:length(y_i)) {
    #   lambda_ij <- exp(X_i[j, ] %*% beta + Z_i[j, ] %*% alpha_i_s) #vectorise this!
    #   # llh_y_i[j] <- y_i[j] * log(lambda_ij) - lambda_ij - log(factorial(y_i[j]))
    #   llh_y_i_s[j] <- dpois(y_i[j], lambda_ij, log = T)
    # }
    lambda_i <- exp(X_i %*% beta + Z_i %*% alpha_i_s)
    llh_y_i_s <- dpois(y_i, lambda_i, log = T)
    llh_y_i_s <- sum(llh_y_i_s)
    llh_y[s] <- llh_y_i_s
    
    llh_alpha_i_s <- dmvnorm(t(alpha_i_s), rep(0, length(alpha_i_s)), Sigma_alpha, log = T)
    # llh_alpha_i_s2 <- -1/2 * log(det(Sigma_alpha)) - n_random_effects/2 * log(2*pi) - 
    #   1/2 * t(alpha_i_s) %*% solve(Sigma_alpha) %*% alpha_i_s
    llh_alpha[s] <- llh_alpha_i_s
    
    llh[s] <- llh_y_i_s + llh_alpha_i_s
    # llh[s] <- llh_alpha_i_s
    
    
  }
  # llh <- sum(llh)
  
  return(list(llh = llh, llh_y = llh_y, llh_alpha = llh_alpha,
              beta = beta, Sigma_alpha = Sigma_alpha))
}