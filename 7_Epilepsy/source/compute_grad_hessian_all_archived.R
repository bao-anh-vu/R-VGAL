compute_grad_hessian2 <- tf_function(
  compute_grad_hessian2 <- function(y_i, X_i, Z_i, alpha_i, theta, S_alpha) {
    
    # n_fixed_effects <- as.integer(ncol(X_i))#tf$cast(ncol(X_i), dtype = "int64")
    # n_random_effects <- as.integer(ncol(Z_i)) #tf$cast(ncol(Z_i), dtype = "int64")
    # param_dim <- as.integer(length(theta))
    # # n <- tf$cast(length(y_i), dtype = "int64")
    # S_alpha <- as.integer(ncol(alpha_i))
    
    autodiff_out <- compute_joint_llh_tf2(y_i, X_i, Z_i, alpha_i, theta, S_alpha)
    
    joint_grads <- autodiff_out$grad
    joint_hessians <- autodiff_out$hessian
    weights <- autodiff_out$weights
    
    joint_grads_reshape <- tf$reshape(joint_grads, c(dim(joint_grads), 1L))
    weights_reshape <- tf$reshape(weights, c(dim(weights), 1L, 1L))
    
    weighted_grads <- tf$multiply(weights_reshape, joint_grads_reshape)
    grad <- tf$reduce_sum(weighted_grads, 1L)
    
    # weighted_hess <- tf$multiply(weights_reshape, joint_hessians)
    # hess <- tf$reduce_sum(weighted_hess, 0L)
    
    ## Now the Hessian
    # hess_part1 <- tcrossprod(grads[[l]])
    hess_part1 <- tf$linalg$matmul(grad, tf$transpose(grad, perm = c(0L, 2L, 1L)))
    
    # joint_grad_crossprod <- lapply(joint_grads, tcrossprod)
    # unweighted_hess <- Map('+', joint_grad_crossprod, joint_hessians)
    # weighted_hess <- Map('*', weights, unweighted_hess)
    # hess_part2 <- Reduce("+", weighted_hess)
    
    joint_grad_crossprod <- tf$linalg$matmul(joint_grads_reshape, 
                                             tf$transpose(joint_grads_reshape, 
                                                          perm = c(0L, 1L, 3L, 2L)))
    unweighted_hess <- joint_grad_crossprod + joint_hessians
    
    weighted_hess <- tf$multiply(weights_reshape, unweighted_hess)
    hess_part2 <- tf$reduce_sum(weighted_hess, 1L)
    
    hessian <- hess_part2 - hess_part1
    
    return(list(grad = grad, hessian = hessian))    
  }, 
reduce_retracing = T)


# compute_joint_llh_tf2 <- tf_function(
  compute_joint_llh_tf2 <- function(y_i, X_i, Z_i, alpha_i, theta, S_alpha) {
  
  with (tf$GradientTape() %as% tape2, {
    with (tf$GradientTape(persistent = TRUE) %as% tape1, {
      ## Construct parameters on the untransformed scale
      n_fixed_effects <- as.integer(ncol(X_i))#tf$cast(ncol(X_i), dtype = "int64")
      n_random_effects <- as.integer(ncol(Z_i)) #tf$cast(ncol(Z_i), dtype = "int64")
      param_dim <- as.integer(ncol(theta))
      # n <- tf$cast(length(y_i), dtype = "int64")
      # S_alpha <- as.integer(dim(alpha_i)[2])
      beta_tf <- theta[, 1:n_fixed_effects]
      
      Lelements_tf <-  theta[, (n_fixed_effects+1):param_dim]
      Lsamples_tf <- fill_lower_tri2(n_random_effects, Lelements_tf)
      # Sigma_alpha_tf <- tf$linalg$matmul(Lsamples_tf, tf$transpose(Lsamples_tf))
      
      ## Sample alpha_i here
      # alpha_i_og <- alpha_i
      # 
      # alpha_i <- norm$sample(S_alpha)
      
      # lines(density(as.matrix(alpha_i_test)[, 1]), col = "red")
      
      # Need to replicate and reshape beta S_alpha times here
      # so that we then have S_alpha "samples" of lambda_i
      beta_tf_reshape <- tf$reshape(beta_tf, c(dim(beta_tf)[1], 1L, dim(beta_tf)[2], 1L))
      beta_tf_tiled <- tf$tile(beta_tf_reshape, c(1L, S_alpha, 1L, 1L))
      
      ## and tile L samples S_alpha times 
      L_tf_reshape <- tf$reshape(Lsamples_tf, c(dim(Lsamples_tf)[1], 1L, dim(Lsamples_tf)[2:3]))
      L_tf_tiled <- tf$tile(L_tf_reshape, c(1L, S_alpha, 1L, 1L)) 
      
      ## Tile X_i again too
      X_i_reshaped <- tf$reshape(X_i, c(1L, dim(X_i)))
      X_i_tf_tiled <- tf$tile(X_i_reshaped, c(S_alpha, 1L, 1L))
      
      X_i_reshaped2 <- tf$reshape(X_i_tf_tiled, c(1L, dim(X_i_tf_tiled)))
      X_i_tf_tiled2 <- tf$tile(X_i_reshaped2, c(S, 1L, 1L, 1L))
      
      Z_i_reshaped <- tf$reshape(Z_i, c(1L, dim(Z_i)))
      Z_i_tf_tiled <- tf$tile(Z_i_reshaped, c(S_alpha, 1L, 1L))
      
      Z_i_reshaped2 <- tf$reshape(Z_i_tf_tiled, c(1L, dim(Z_i_tf_tiled)))
      Z_i_tf_tiled2 <- tf$tile(Z_i_reshaped2, c(S, 1L, 1L, 1L))
      
      if (n_random_effects == 1) {
        alpha_i_reshaped <- tf$reshape(alpha_i, c(dim(alpha_i), 1L, 1L))
      } else {
        alpha_i_reshaped <- tf$reshape(alpha_i, c(dim(alpha_i), 1L))
      }
      
      lambda_i_tf <- tf$exp(tf$linalg$matmul(X_i_tf_tiled2, beta_tf_tiled) + 
                              tf$linalg$matmul(Z_i_tf_tiled2, alpha_i_reshaped))
      
      ## Now compute the likelihood here
      # y_i <- tf$cast(y_i, dtype = "float64")
      # y_i_reshape <- tf$reshape(y_i, c(1L, 1L, dim(y_i)))
      # y_i_tiled <- tf$tile(y_i_reshape, c(S_alpha, 1L, 1L))
      
      # llh_y_i_tf <- tf$squeeze(tf$linalg$matmul(y_i_tiled, tf$math$log(lambda_i_tf_reshape))) -
      #   tf$reduce_sum(lambda_i_tf, 0L) -
      #   tf$squeeze(tf$reduce_sum(tf$math$lgamma(y_i_tiled + 1), 2L))
      ## these can be used as the log weights too
      
      y_i_reshape <- tf$reshape(y_i, c(1L, dim(y_i), 1L))
      y_i_tiled <- tf$tile(y_i_reshape, c(S_alpha, 1L, 1L))
      
      y_i_reshape2 <- tf$reshape(y_i_tiled, c(1L, dim(y_i_tiled)))
      y_i_tiled2 <- tf$tile(y_i_reshape2, c(S, 1L, 1L, 1L))
      
      pois <- tfd$Poisson(rate = lambda_i_tf)
      
      llh_y_i_tf_s <- pois$log_prob(y_i_tiled2)
      llh_y_i_tf <- tf$reduce_sum(llh_y_i_tf_s, 2L) # should be size S x S_alpha
      llh_y_i_tf <- tf$reshape(llh_y_i_tf, c(dim(llh_y_i_tf)[1], dim(llh_y_i_tf)[2]))
      # Lsamples_tf_reshape <- tf$reshape(Lsamples_tf, c(1L, dim(Lsamples_tf)))
      # Lsamples_tf_tiled <- tf$tile(Lsamples_tf_reshape, c(S_alpha, 1L, 1L))
      # alpha_i_reshape <- tf$reshape(alpha_i, c(dim(alpha_i), 1L))
      # 
      # Amat <- tf$linalg$matmul(tf$linalg$inv(Lsamples_tf_tiled), alpha_i_reshape)
      # 
      # llh_alpha_i_tf <- - tf$reduce_sum(tf$math$log(tf$linalg$diag_part(Lsamples_tf_tiled)), 1L) -
      #   tf$cast(n_random_effects/2 * tf$math$log(2*pi), dtype = "float64") -
      #   1/2 * tf$squeeze(tf$linalg$matmul(tf$transpose(Amat, perm = c(0L, 2L, 1L)), Amat))
      
      if (n_random_effects == 1) {
        norm <- tfd$Normal(loc = 0, scale = L_tf_tiled)
        llh_alpha_i_tf <- tf$squeeze(norm$log_prob(alpha_i_reshaped))
      } else {
        norm <- tfd$MultivariateNormalTriL(loc = 0, scale_tril = L_tf_tiled)
        llh_alpha_i_tf <- norm$log_prob(alpha_i)
      }
      # llh_alpha_i_og <- norm$log_prob(alpha_i_og)
      
      log_likelihood_tf <- llh_y_i_tf + llh_alpha_i_tf
      
      
      log_likelihood_tf_reshape <- tf$reshape(log_likelihood_tf, c(dim(log_likelihood_tf), 1L, 1L))
      theta_reshape <- tf$reshape(theta, c(dim(theta)[1], 1L, dim(theta)[2]))
      theta_tiled <- tf$tile(theta_reshape, c(1L, S_alpha, 1L))
    })
    # browser()
    # grad_tf %<-% tape1$batch_jacobian(log_likelihood_tf_reshape, theta_tiled)
    grad_tf %<-% tape1$batch_jacobian(log_likelihood_tf, theta)
    
    # grad_tf_test %<-% tape1$batch_jacobian(log_likelihood_tf_reshape, theta_tiled)
    browser()
  })
  # grad2_tf %<-% tape2$batch_jacobian(grad_tf, theta)
  grad2_tf %<-% tape2$batch_jacobian(grad_tf, theta)
  
  
  # ## Computing the weights in importance sampling
  log_weights <- llh_y_i_tf
  max_weight <- tf$math$reduce_max(log_weights, 1L)
  max_weight_reshape <- tf$reshape(max_weight, c(dim(max_weight), 1L))
  log_w_shifted <- log_weights - tf$tile(max_weight_reshape, c(1L, S_alpha))
  sum_weights <- tf$math$reduce_sum(tf$exp(log_w_shifted), 1L)
  sum_weights_reshape <- tf$reshape(sum_weights, c(dim(sum_weights), 1L))
  weights <- tf$divide(tf$math$exp(log_w_shifted), 
                       tf$tile(sum_weights_reshape, c(1L, S_alpha))) # normalised weights
  
  return(list(llh = log_likelihood_tf,
              grad = grad_tf,
              hessian = grad2_tf,
              log_weights = log_weights,
              weights = weights))
}#,
# reduce_retracing = T)

# fill_lower_tri_tf <- tf_function(
fill_lower_tri2 <- function(dim, vals) {
  
  
  d <- as.integer(dim)
  S <- as.integer(nrow(vals))
  # vals_tf <- tf$constant(vals, dtype = "float64")
  
  diag_mat <- tf$linalg$diag(tf$exp(vals[, 1:d]))
  # diag_mat_tiled <- tf$tile(diag_mat, c(S, 1L, 1L))
  
  nlower <- as.integer(d*(d-1)/2)
  numlower = vals[, (d+1):(d+nlower)]
  # if (S != 1L) {
    numlower = tf$reshape(numlower, c(S*nlower, 1L))
    numlower = tf$squeeze(numlower)
  # }
  
  ones = tf$ones(c(d, d), dtype="int64")
  mask_a = tf$linalg$band_part(ones, -1L, 0L)  # Upper triangular matrix of 0s and 1s
  mask_b = tf$linalg$band_part(ones, 0L, 0L)  # Diagonal matrix of 0s and 1s
  mask = tf$subtract(mask_a, mask_b) # Mask of upper triangle above diagonal
  
  zero = tf$constant(0L, dtype="int64")
  non_zero = tf$not_equal(mask, zero) #Conversion of mask to Boolean matrix
  non_zero_tile <- tf$tile(non_zero, c(S, 1L))
  indices = tf$where(non_zero_tile) # Extracting the indices of upper triangular elements
  
  ## need to reshape indices here
  # shape <- tf$cast(c(d, d), dtype="int64")
  # out = tf$SparseTensor(indices, numlower, 
  #                       dense_shape = tf$cast(c(d, d), dtype="int64"))
  # lower_tri = tf$sparse$to_dense(out)
  shape <- tf$cast(c(S*d, d), dtype="int64")
  # shape_test <- tf$reshape(shape, c(dim(shape), 1L))
  # batch_shapes <- tf$tile(shape, c(S, 1L))
  if (S == 1L) {
    out = tf$SparseTensor(indices, as.numeric(numlower), 
                          dense_shape = shape)
  } else {
    out = tf$SparseTensor(indices, numlower, 
                          dense_shape = shape)
  }
  
  lower_tri = tf$sparse$to_dense(out)
  lower_tri_reshaped = tf$reshape(lower_tri, c(S, d, d))
  
  L = diag_mat + lower_tri_reshaped
  
  return(L)
}
# )