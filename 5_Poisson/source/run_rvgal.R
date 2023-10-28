run_rvgal <- function(y, X, Z, mu_0, P_0, S = 100L, S_alpha = 100L,
                      use_tempering = T, n_temper = 10, 
                      temper_schedule = rep(0.25, 4),
                      n_post_samples = 10000,
                      save_results = F) {
  
  tf64 <- function(x) tf$constant(x, dtype = "float64")
  
  if (!use_tempering) {
    temper_schedule <- 1
  }
  
  print("Starting R-VGA...")
  t1 <- proc.time()
  
  # n_fixed_effects <- as.integer(ncol(X[[1]]))
  # n_random_effects <- as.integer(ncol(Z[[1]]))
  
  param_dim <- as.integer(length(mu_0))
  N <- length(y)
  n <- length(y[[1]])
  ## Sample from the "prior"
  ## par(mfrow = c(1, 1))
  ## test_omega <- rnorm(10000, mu_0[param_dim], P_0[param_dim, param_dim])
  ## plot(density(sqrt(exp(test_omega))), main = "RVGA: Prior of tau")
  
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
      
      n_fixed_effects <- ncol(X[[i]])
      n_random_effects <- ncol(Z[[i]])
      
      y_i_tf <- tf$Variable(y[[i]], dtype = "float64")
      X_i_tf <- tf$Variable(X[[i]], dtype = "float64")
      Z_i_tf <- tf$Variable(Z[[i]], dtype = "float64")
      
      
      ####### Development ###########
      beta_all <- samples[, 1:n_fixed_effects]

      Sigma_alpha_all <- lapply(samples_list, construct_Sigma,
                                d = n_random_effects)
      alpha_all <- lapply(Sigma_alpha_all,
                          function(Sigma) rmvnorm(S_alpha, rep(0, n_random_effects), Sigma))

      prior_var_beta <- diag(P_0)[1:n_fixed_effects]
      
      I_d <- diag(n_random_effects)
        
      log_likelihood <- list()
      for (l in 1:S) {
        theta_l <- samples[l, ]
        
        beta_l <- theta_l[1:n_fixed_effects]
        
        Sigma_alpha_l <- construct_Sigma(theta_l[-(1:n_fixed_effects)],
                                         d = n_random_effects)
        alpha_l <- alpha_all[[l]]
        
        grad_beta <- list()
        grad_zeta <- list()
        
        W <- t(chol(Sigma_alpha_l))
        W_inv <- solve(W)
        
        for (s in 1:S_alpha) {
          alpha_l_s <- alpha_l[s, ]
          
          grad_beta[[s]] <- t(y[[i]] - exp(X[[i]] %*% beta_l + Z[[i]] %*% alpha_l_s)) %*% X[[i]] #- 1/prior_var_beta * beta_l
          
          I_zeta <- diag(alpha_l_s)
          I_zeta[lower.tri(I_zeta)] <- 1
          A <- t(W_inv) %*% W_inv %*% alpha_l_s %*% t(alpha_l_s) %*% t(W_inv)
          grad_zeta[[s]] <- I_d[lower.tri(I_d, diag = T)] + 
            I_zeta[lower.tri(I_zeta, diag = T)] * A[lower.tri(A, diag = T)]
        }
        
        ## likelihood seems ok -- same between TF and for loop
        # test <- poisson_joint_likelihood(y[[i]], X[[i]], Z[[i]],
        #                                  alpha_all[[l]], theta_l, S_alpha)
        # log_likelihood[[l]] <- test$llh
        # llh_y <- test$llh_y
      }
      
      ############ TF #############
      
      alpha_all_tf <- tf$Variable(alpha_all, dtype = "float64")
      theta_tf <- tf$Variable(samples, dtype = "float64")

      ##  
      tf_out <- compute_joint_llh_tf2(y_i_tf, X_i_tf, Z_i_tf,
                                      alpha_all_tf, theta_tf)
      tf_grad <- tf_out$grad
      tf_grad_1 <- tf_grad[1,,]
      tf_grad_beta <- tf_grad_1[, 1:n_fixed_effects]
      tf_grad_zeta <- tf_grad_1[, (n_fixed_effects+1):param_dim]
      
      browser()
      ##
      
      tf_out2 <- compute_grad_hessian2(y_i_tf, X_i_tf, Z_i_tf,
                                     alpha_all_tf, theta_tf)
      
      E_score_tf <- tf$math$reduce_mean(tf_out2$grad, 0L)
      E_hessian_tf <- tf$math$reduce_mean(tf_out2$hessian, 0L)
      prec_temp <- prec_temp - a * as.matrix(E_hessian_tf)
      mu_temp <- mu_temp + chol2inv(chol(prec_temp)) %*% (a * as.matrix(E_score_tf))
      # 
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
      #   ## Weights for importance sampling
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
  print(t2 - t1)
  
  post_var <- chol2inv(chol(prec[[N+1]]))
  rvgal.post_samples <- rmvnorm(n_post_samples, mu_vals[[N+1]], post_var)  
  
  # rvgal.post_samples_beta <- rvgal.post_samples[, 1:n_fixed_effects]
  
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
    llh[s] <- llh_y_i_s + llh_alpha_i_s
    # llh[s] <- llh_alpha_i_s
    
    
  }
  # llh <- sum(llh)
  
  return(list(llh = llh, llh_y = llh_y, beta = beta, Sigma_alpha = Sigma_alpha))
}