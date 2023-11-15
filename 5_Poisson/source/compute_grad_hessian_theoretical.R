# compute_grad_hessian2 <- tf_function(
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
    
    # joint_grads_reshape <- tf$reshape(joint_grads, c(dim(joint_grads), 1L))
    weights_reshape <- tf$reshape(weights, c(dim(weights), 1L, 1L))
    
    weighted_grads <- tf$multiply(weights_reshape, joint_grads)
    grad <- tf$reduce_sum(weighted_grads, 1L)
    
    # weighted_hess <- tf$multiply(weights_reshape, joint_hessians)
    # hess <- tf$reduce_sum(weighted_hess, 0L)
    
    ## Now the Hessian
    # hess_part1 <- tcrossprod(grads[[l]])
    # hess_part1 <- tf$linalg$matmul(grad, tf$transpose(grad, perm = c(0L, 2L, 1L)))
    hess_part1 <- tf$linalg$matmul(grad, tf$linalg$matrix_transpose(grad))
    
    # joint_grad_crossprod <- lapply(joint_grads, tcrossprod)
    # unweighted_hess <- Map('+', joint_grad_crossprod, joint_hessians)
    # weighted_hess <- Map('*', weights, unweighted_hess)
    # hess_part2 <- Reduce("+", weighted_hess)
    
    # joint_grad_crossprod <- tf$linalg$matmul(joint_grads_reshape, 
    #                                          tf$transpose(joint_grads_reshape, 
    #                                                       perm = c(0L, 1L, 3L, 2L)))
    joint_grad_crossprod <- tf$linalg$matmul(joint_grads, 
                                             tf$linalg$matrix_transpose(joint_grads))
    
    unweighted_hess <- joint_grad_crossprod + joint_hessians
    
    weighted_hess <- tf$multiply(weights_reshape, unweighted_hess)
    hess_part2 <- tf$reduce_sum(weighted_hess, 1L)
    
    hessian <- hess_part2 - hess_part1
    
    return(list(grad = grad, hessian = hessian))    
  }#, 
  # reduce_retracing = T)


# compute_joint_llh_tf2 <- tf_function(
  compute_joint_llh_tf2 <- function(y_i, X_i, Z_i, alpha_i, theta, S_alpha) {
    
    # with (tf$GradientTape() %as% tape2, {
    #   with (tf$GradientTape(persistent = TRUE) %as% tape1, {
        ## Construct parameters on the untransformed scale
        n_fixed_effects <- as.integer(ncol(X_i))#tf$cast(ncol(X_i), dtype = "int64")
        n_random_effects <- as.integer(ncol(Z_i)) #tf$cast(ncol(Z_i), dtype = "int64")
        param_dim <- as.integer(ncol(theta))
        # n <- tf$cast(length(y_i), dtype = "int64")
        
        # S_alpha <- as.integer(dim(alpha_i)[2])
        beta_tf <- theta[, 1:n_fixed_effects]
        zeta_tf <- theta[, (n_fixed_effects+1):param_dim]
        
        Lelements_tf <-  theta[, (n_fixed_effects+1):param_dim]
        Lsamples_tf <- fill_lower_tri2(n_random_effects, Lelements_tf)
        
        # Lelements_diag <-  tf$exp(theta[, (n_fixed_effects+1):(n_fixed_effects+n_random_effects)])
        # Lelements_offdiag <- theta[, (n_fixed_effects+n_random_effects+1):param_dim]
        # Lelements_tf <- tf$concat(list(Lelements_diag, Lelements_offdiag), 1L)
        # Lsamples_tf <- fill_lower_tri(n_random_effects, Lelements_tf)
        # Sigma_alpha_tf <- tf$linalg$matmul(Lsamples_tf, tf$transpose(Lsamples_tf))
        
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
        y_i_reshape <- tf$reshape(y_i, c(1L, dim(y_i), 1L))
        y_i_tiled <- tf$tile(y_i_reshape, c(S_alpha, 1L, 1L))
        
        y_i_reshape2 <- tf$reshape(y_i_tiled, c(1L, dim(y_i_tiled)))
        y_i_tiled2 <- tf$tile(y_i_reshape2, c(S, 1L, 1L, 1L))
        
        pois <- tfd$Poisson(rate = lambda_i_tf)
        
        llh_y_i_tf_s <- pois$log_prob(y_i_tiled2)
        llh_y_i_tf <- tf$squeeze(tf$reduce_sum(llh_y_i_tf_s, 2L)) # should be size S x S_alpha
        
        if (n_random_effects == 1) {
          norm <- tfd$Normal(loc = 0, scale = L_tf_tiled)
          llh_alpha_i_tf <- tf$squeeze(norm$log_prob(alpha_i_reshaped))
        } else {
          norm <- tfd$MultivariateNormalTriL(loc = 0, scale_tril = L_tf_tiled)
          llh_alpha_i_tf <- norm$log_prob(alpha_i)
        }
        # llh_alpha_i_og <- norm$log_prob(alpha_i_og)
        
        log_likelihood_tf <- llh_y_i_tf + llh_alpha_i_tf
        # log_likelihood_tf <- llh_y_i_tf + tf$reshape(llh_alpha_i_tf, c(dim(llh_alpha_i_tf), 1L))
        
        # theta_reshape <- tf$reshape(theta, c(dim(theta)[1], 1L, dim(theta)[2]))
        # theta_tiled <- tf$tile(theta_reshape, c(1L, S_alpha, 1L))
        
        # ## Testing theoretical gradient
        tempMat <- y_i_tiled2 - lambda_i_tf
        # tempMat <- tf$reshape(tempMat, c(dim(tempMat), 1L))
        grad_beta_tf <- tf$linalg$matmul(tf$linalg$matrix_transpose(X_i_tf_tiled2), 
                                         tempMat)
        
        grad2_beta_tf <- -tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matrix_transpose(X_i_tf_tiled2),
                                                            tf$linalg$diag(tf$squeeze(lambda_i_tf))),
                                          X_i_tf_tiled2)
        
        ## Gradient of zeta
        zeta_11 <- zeta_tf[, 1]
        shape <- tf$cast(c(S*n_random_effects, n_random_effects), dtype="int64")
        ones = tf$cast(matrix(c(1, 0, 0, 0), 2, 2), dtype="int64")
        zero = tf$constant(0L, dtype="int64")
        non_zero = tf$not_equal(ones, zero)
        non_zero_tile <- tf$tile(non_zero, c(S, 1L))
        indices = tf$where(non_zero_tile) # Extracting the indices of upper triangular elements
        # shape_test <- tf$reshape(shape, c(dim(shape), 1L))
        # batch_shapes <- tf$tile(shape, c(S, 1L))
        grad_L_zeta_11 = tf$SparseTensor(indices, tf$exp(zeta_11), 
                                         dense_shape = shape)
        grad_L_zeta_11 = tf$sparse$to_dense(grad_L_zeta_11)
        
        grad_L_zeta_11 = tf$reshape(grad_L_zeta_11, c(S, n_random_effects, n_random_effects))
        
        zeta_22 <- zeta_tf[, 2]
        ones = tf$cast(matrix(c(0, 0, 0, 1), 2, 2), dtype="int64")
        non_zero = tf$not_equal(ones, zero)
        non_zero_tile <- tf$tile(non_zero, c(S, 1L))
        indices = tf$where(non_zero_tile)
        grad_L_zeta_22 = tf$SparseTensor(indices, tf$exp(zeta_22), 
                                         dense_shape = shape)
        grad_L_zeta_22 = tf$sparse$to_dense(grad_L_zeta_22)
        grad_L_zeta_22 = tf$reshape(grad_L_zeta_22, c(S, n_random_effects, n_random_effects))
        
        zeta_21 <- zeta_tf[, 3]
        ones = tf$cast(matrix(c(0, 1, 0, 0), 2, 2), dtype="int64")
        non_zero = tf$not_equal(ones, zero)
        non_zero_tile <- tf$tile(non_zero, c(S, 1L))
        indices = tf$where(non_zero_tile)
        grad_L_zeta_21 = tf$SparseTensor(indices, tf$Variable(rep(1, S), dtype = "float64"),
                                         dense_shape = shape)
        grad_L_zeta_21 = tf$sparse$to_dense(grad_L_zeta_21)
        grad_L_zeta_21 = tf$reshape(grad_L_zeta_21, c(S, n_random_effects, n_random_effects))
        
        L_inv = tf$linalg$inv(L_tf_tiled)
        L_inv_trans <- tf$linalg$matrix_transpose(L_inv)
        Amat = - L_inv_trans + tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(L_inv_trans, L_inv),
                                                                 tf$linalg$matmul(alpha_i_reshaped, 
                                                                                  tf$linalg$matrix_transpose(alpha_i_reshaped))), 
                                                L_inv_trans)  
        grad_L_zeta_11_reshape <- tf$reshape(grad_L_zeta_11, c(S, 1L, n_random_effects, n_random_effects))
        grad_L_zeta_22_reshape <- tf$reshape(grad_L_zeta_22, c(S, 1L, n_random_effects, n_random_effects))
        grad_L_zeta_21_reshape <- tf$reshape(grad_L_zeta_21, c(S, 1L, n_random_effects, n_random_effects))
        
        grad_L_zeta_11_tiled <- tf$tile(grad_L_zeta_11_reshape, c(1L, S_alpha, 1L, 1L))
        grad_L_zeta_22_tiled <- tf$tile(grad_L_zeta_22_reshape, c(1L, S_alpha, 1L, 1L))
        grad_L_zeta_21_tiled <- tf$tile(grad_L_zeta_21_reshape, c(1L, S_alpha, 1L, 1L))
        
        ## grad2_L_zeta_ij
        ##
        # grad2_L_zeta_11 = grad_L_zeta_11_tiled
        # grad2_L_zeta_22 = grad_L_zeta_22_tiled
        # grad2_L_zeta_21 = tf$cast(matrix(0, S*n_random_effects, n_random_effects), dtype = "float64")
        # grad2_L_zeta_21 = tf$reshape(grad2_L_zeta_21, c(S, n_random_effects, n_random_effects))
        # grad2_L_zeta_21 <- tf$reshape(grad2_L_zeta_21, c(S, 1L, n_random_effects, n_random_effects))
        
        
        grad_zeta_11 = tf$reduce_sum(tf$linalg$diag_part(tf$linalg$matmul(tf$linalg$matrix_transpose(Amat),
                                                            grad_L_zeta_11_tiled)), 2L)
        grad_zeta_22 = tf$reduce_sum(tf$linalg$diag_part(tf$linalg$matmul(tf$linalg$matrix_transpose(Amat),
                                                                          grad_L_zeta_22_tiled)), 2L)
        grad_zeta_21 = tf$reduce_sum(tf$linalg$diag_part(tf$linalg$matmul(tf$linalg$matrix_transpose(Amat),
                                                                          grad_L_zeta_21_tiled)), 2L)
        grad_zeta_11_reshape <- tf$reshape(grad_zeta_11, c(dim(grad_zeta_11), 1L, 1L))
        grad_zeta_22_reshape <- tf$reshape(grad_zeta_22, c(dim(grad_zeta_22), 1L, 1L))
        grad_zeta_21_reshape <- tf$reshape(grad_zeta_21, c(dim(grad_zeta_21), 1L, 1L))
        grad_zeta_tf <- tf$concat(list(grad_zeta_11_reshape, grad_zeta_22_reshape, grad_zeta_21_reshape), 2L) 
        
        grad_tf <- tf$concat(list(grad_beta_tf, grad_zeta_tf), 2L)
        
        ## Then the Hessian of zeta
        
        ### First, the gradients of Linv wrt each zeta_ij
        ones = tf$cast(matrix(c(1, 1, 0, 0), 2, 2), dtype="int64")
        # zero = tf$constant(0L, dtype="int64")
        non_zero = tf$not_equal(ones, zero)
        non_zero_tile <- tf$tile(non_zero, c(S, 1L))
        indices = tf$where(non_zero_tile)
        
        vals = tf$stack(list(-tf$exp(-zeta_11), 
                             tf$multiply(tf$cast(zeta_21, dtype = "float64"), 
                                         tf$exp(-(zeta_11 + zeta_22)))), 1L)
        grad_Linv_zeta_11 = tf$SparseTensor(indices, tf$reshape(vals, S*n_random_effects), 
                                         dense_shape = shape)
        grad_Linv_zeta_11 = tf$sparse$to_dense(grad_Linv_zeta_11)
        grad_Linv_zeta_11 = tf$reshape(grad_Linv_zeta_11, c(S, n_random_effects, n_random_effects))
        
        ## grad_Linv_zeta_22
        ones = tf$cast(matrix(c(0, 1, 0, 1), 2, 2), dtype="int64")
        # zero = tf$constant(0L, dtype="int64")
        non_zero = tf$not_equal(ones, zero)
        non_zero_tile <- tf$tile(non_zero, c(S, 1L))
        indices = tf$where(non_zero_tile)
        vals = tf$stack(list(tf$multiply(tf$cast(zeta_21, dtype = "float64"), 
                                         tf$exp(-(zeta_11 + zeta_22))),
                             -tf$exp(-zeta_22)), 1L)
        
        grad_Linv_zeta_22 = tf$SparseTensor(indices, tf$reshape(vals, S*n_random_effects), 
                                            dense_shape = shape)
        grad_Linv_zeta_22 = tf$sparse$to_dense(grad_Linv_zeta_22)
        grad_Linv_zeta_22 = tf$reshape(grad_Linv_zeta_22, c(S, n_random_effects, n_random_effects))
        
        ## grad_Linv_zeta_21
        vals = -tf$exp(-(zeta_11 + zeta_22))
        ones = tf$cast(matrix(c(0, 1, 0, 0), 2, 2), dtype="int64")
        # zero = tf$constant(0L, dtype="int64")
        non_zero = tf$not_equal(ones, zero)
        non_zero_tile <- tf$tile(non_zero, c(S, 1L))
        indices = tf$where(non_zero_tile)
        
        grad_Linv_zeta_21 = tf$SparseTensor(indices, vals, 
                                            dense_shape = shape)
        grad_Linv_zeta_21 = tf$sparse$to_dense(grad_Linv_zeta_21)
        grad_Linv_zeta_21 = tf$reshape(grad_Linv_zeta_21, c(S, n_random_effects, n_random_effects))
        
        grad_Linv_zeta_11 <- tf$reshape(grad_Linv_zeta_11, c(S, 1L, n_random_effects, n_random_effects))
        grad_Linv_zeta_22 <- tf$reshape(grad_Linv_zeta_22, c(S, 1L, n_random_effects, n_random_effects))
        grad_Linv_zeta_21 <- tf$reshape(grad_Linv_zeta_21, c(S, 1L, n_random_effects, n_random_effects))
        
        grad_Linv_zeta_11_tiled <- tf$tile(grad_Linv_zeta_11, c(1L, S_alpha, 1L, 1L))
        grad_Linv_zeta_22_tiled <- tf$tile(grad_Linv_zeta_22, c(1L, S_alpha, 1L, 1L))
        grad_Linv_zeta_21_tiled <- tf$tile(grad_Linv_zeta_21, c(1L, S_alpha, 1L, 1L))
        
        ## Nooooow the Hessian of zeta
        tcrossp_alpha = tf$linalg$matmul(alpha_i_reshaped, 
                                         tf$linalg$matrix_transpose(alpha_i_reshaped))
        A = - L_inv_trans + tf$linalg$matmul(tf$linalg$matmul(tf$linalg$matmul(L_inv_trans, L_inv),
                                                              tcrossp_alpha),
                                             L_inv_trans)
        B = tf$linalg$matmul(tf$linalg$matmul(L_inv, tcrossp_alpha), 
                             L_inv_trans) 
        # A = tf$reshape(A, c(S, 1L, n_random_effects, n_random_effects))
        
        grad_B_zeta_11 = tf$linalg$matmul(tf$linalg$matmul(grad_Linv_zeta_11_tiled, 
                                                           tcrossp_alpha), 
                                          L_inv_trans) + 
          tf$linalg$matmul(tf$linalg$matmul(L_inv, tcrossp_alpha), 
                           tf$linalg$matrix_transpose(grad_Linv_zeta_11_tiled))
        
        grad_A_zeta_11 = - tf$linalg$matrix_transpose(grad_Linv_zeta_11_tiled) + 
          tf$linalg$matmul(tf$linalg$matrix_transpose(grad_Linv_zeta_11_tiled), B) + 
          tf$linalg$matmul(L_inv_trans, grad_B_zeta_11)
        
        grad_B_zeta_22 = tf$linalg$matmul(tf$linalg$matmul(grad_Linv_zeta_22_tiled, 
                                                           tcrossp_alpha), 
                                          L_inv_trans) + 
          tf$linalg$matmul(tf$linalg$matmul(L_inv, tcrossp_alpha), 
                           tf$linalg$matrix_transpose(grad_Linv_zeta_22_tiled))
        
        grad_A_zeta_22 = - tf$linalg$matrix_transpose(grad_Linv_zeta_22_tiled) + 
          tf$linalg$matmul(tf$linalg$matrix_transpose(grad_Linv_zeta_22_tiled), B) + 
          tf$linalg$matmul(L_inv_trans, grad_B_zeta_22)
        
        grad_B_zeta_21 = tf$linalg$matmul(tf$linalg$matmul(grad_Linv_zeta_21_tiled, 
                                                           tcrossp_alpha), 
                                          L_inv_trans) + 
          tf$linalg$matmul(tf$linalg$matmul(L_inv, tcrossp_alpha), 
                           tf$linalg$matrix_transpose(grad_Linv_zeta_21_tiled))
        
        grad_A_zeta_21 = - tf$linalg$matrix_transpose(grad_Linv_zeta_21_tiled) + 
          tf$linalg$matmul(tf$linalg$matrix_transpose(grad_Linv_zeta_21_tiled), B) + 
          tf$linalg$matmul(L_inv_trans, grad_B_zeta_21)
        
        grad2_zeta_11 = tf$linalg$matmul(tf$linalg$matrix_transpose(grad_A_zeta_11), grad_L_zeta_11_tiled) +
          tf$linalg$matmul(tf$linalg$matrix_transpose(A), grad_L_zeta_11_tiled)
        grad2_zeta_11 = tf$reduce_sum(tf$linalg$diag_part(grad2_zeta_11), 2L)
        grad2_zeta_11 = tf$reshape(grad2_zeta_11, c(dim(grad2_zeta_11), 1L))
        
        grad2_zeta_22 = tf$linalg$matmul(tf$linalg$matrix_transpose(grad_A_zeta_22), grad_L_zeta_22_tiled) +
          tf$linalg$matmul(tf$linalg$matrix_transpose(A), grad_L_zeta_22_tiled)
        grad2_zeta_22 = tf$reduce_sum(tf$linalg$diag_part(grad2_zeta_22), 2L)
        grad2_zeta_22 = tf$reshape(grad2_zeta_22, c(dim(grad2_zeta_22), 1L))
        
        grad2_zeta_21 = tf$linalg$matmul(tf$linalg$matrix_transpose(grad_A_zeta_21), grad_L_zeta_21_tiled) # grad2_L_zeta_21 = 0 in this case so no need for the second term
        grad2_zeta_21 = tf$reduce_sum(tf$linalg$diag_part(grad2_zeta_21), 2L)
        grad2_zeta_21 = tf$reshape(grad2_zeta_21, c(dim(grad2_zeta_21), 1L))
        
        grad2_zeta_22_11 = tf$linalg$matmul(tf$linalg$matrix_transpose(grad_A_zeta_22), grad_L_zeta_11_tiled)
        grad2_zeta_22_11 = tf$reduce_sum(tf$linalg$diag_part(grad2_zeta_22_11), 2L)
        grad2_zeta_22_11 = tf$reshape(grad2_zeta_22_11, c(dim(grad2_zeta_22_11), 1L))
        
        grad2_zeta_21_11 = tf$linalg$matmul(tf$linalg$matrix_transpose(grad_A_zeta_21), grad_L_zeta_11_tiled)
        grad2_zeta_21_11 = tf$reduce_sum(tf$linalg$diag_part(grad2_zeta_21_11), 2L)
        grad2_zeta_21_11 = tf$reshape(grad2_zeta_21_11, c(dim(grad2_zeta_21_11), 1L))
        
        grad2_zeta_21_22 = tf$linalg$matmul(tf$linalg$matrix_transpose(grad_A_zeta_21), grad_L_zeta_22_tiled)
        grad2_zeta_21_22 = tf$reduce_sum(tf$linalg$diag_part(grad2_zeta_21_22), 2L)
        grad2_zeta_21_22 = tf$reshape(grad2_zeta_21_22, c(dim(grad2_zeta_21_22), 1L))
        
        grad2_zeta_diag = tf$linalg$diag(tf$concat(list(grad2_zeta_11, grad2_zeta_22, grad2_zeta_21), 2L))
        grad2_zeta_offdiag = tf$concat(list(grad2_zeta_22_11, grad2_zeta_21_11, grad2_zeta_21_22), 2L)
        
        nlower <- as.integer(n_random_effects*(n_random_effects+1)/2)
        # numlower = vals[, (d+1):(d+nlower)]
        # numlower = tf$reshape(numlower, c(S*nlower, 1L))
        # numlower = tf$squeeze(numlower)
        
        ones = tf$ones(c(nlower, nlower), dtype="int64")
        mask_a = tf$linalg$band_part(ones, -1L, 0L)  # Upper triangular matrix of 0s and 1s
        mask_b = tf$linalg$band_part(ones, 0L, 0L)  # Diagonal matrix of 0s and 1s
        mask = tf$subtract(mask_a, mask_b) # Mask of upper triangle above diagonal
        
        zero = tf$constant(0L, dtype="int64")
        non_zero = tf$not_equal(mask, zero) #Conversion of mask to Boolean matrix
        non_zero = tf$reshape(non_zero, c(1L, dim(non_zero)))
        non_zero_tile <- tf$tile(non_zero, c(S_alpha, 1L, 1L))
        non_zero_tile <- tf$reshape(non_zero_tile, c(1L, dim(non_zero_tile)))
        non_zero_tile2 <- tf$tile(non_zero_tile, c(S, 1L, 1L, 1L))
        
        indices = tf$where(non_zero_tile2) # Extracting the indices of upper triangular elements
        shape <- tf$cast(c(S, S_alpha, nlower, nlower), dtype="int64")
        # shape_test <- tf$reshape(shape, c(dim(shape), 1L))
        # batch_shapes <- tf$tile(shape, c(S, 1L))
        out = tf$SparseTensor(indices, tf$reshape(grad2_zeta_offdiag, S*S_alpha*nlower), 
                              dense_shape = shape)
        grad2_zeta_lower_tri = tf$sparse$to_dense(out)
        # lower_tri_reshaped = tf$reshape(lower_tri, c(S, d, d))
        
        grad2_zeta_lower = grad2_zeta_diag + grad2_zeta_lower_tri
        
        grad2_zeta_tf = grad2_zeta_lower + tf$linalg$matrix_transpose(grad2_zeta_lower) - 
          tf$linalg$diag(tf$linalg$diag_part(grad2_zeta_lower))
        
        testbeta = tf$linalg$LinearOperatorFullMatrix(grad2_beta_tf)
        testzeta = tf$linalg$LinearOperatorFullMatrix(grad2_zeta_tf)
        hessian_tf = tf$linalg$LinearOperatorBlockDiag(list(testbeta, testzeta))
        hessian_tf2 = hessian_tf$to_dense()
      # })
      # grad_tf %<-% tape1$batch_jacobian(log_likelihood_tf, theta)
    # })
    # grad2_tf %<-% tape2$batch_jacobian(grad_tf, theta)
    
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
                llh_y = llh_y_i_tf,
                llh_alpha = llh_alpha_i_tf,
                grad = grad_tf,
                hessian = hessian_tf2,
                grad2_zeta_diag = grad2_zeta_diag,
                grad_Linv_zeta_11 = grad_Linv_zeta_11,
                grad_Linv_zeta_22 = grad_Linv_zeta_22,
                grad_Linv_zeta_21 = grad_Linv_zeta_21,
                grad_A_zeta_11 = grad_A_zeta_11,
                grad_A_zeta_22 = grad_A_zeta_22,
                grad_A_zeta_21 = grad_A_zeta_21,
                grad_B_zeta_11 = grad_B_zeta_11,
                grad_B_zeta_22 = grad_B_zeta_22,
                grad_B_zeta_21 = grad_B_zeta_21,
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
    numlower = tf$reshape(numlower, c(S*nlower, 1L))
    numlower = tf$squeeze(numlower)
    
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
    out = tf$SparseTensor(indices, numlower, 
                          dense_shape = shape)
    lower_tri = tf$sparse$to_dense(out)
    lower_tri_reshaped = tf$reshape(lower_tri, c(S, d, d))
    
    L = diag_mat + lower_tri_reshaped
    
    return(L)
  }#, reduce_retracing = T
# )

# fill_lower_tri_tf <- tf_function(
  fill_lower_tri <- function(dim, vals) {
    
    
    d <- as.integer(dim)
    S <- as.integer(nrow(vals))
    # vals_tf <- tf$constant(vals, dtype = "float64")
    # diag_mat <- tf$linalg$diag(tf$exp(vals[, 1:d]))
    diag_mat <- tf$linalg$diag(vals[, 1:d])
    
    # diag_mat_tiled <- tf$tile(diag_mat, c(S, 1L, 1L))
    
    nlower <- as.integer(d*(d-1)/2)
    numlower = vals[, (d+1):(d+nlower)]
    numlower = tf$reshape(numlower, c(S*nlower, 1L))
    numlower = tf$squeeze(numlower)
    
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
    out = tf$SparseTensor(indices, numlower, 
                          dense_shape = shape)
    lower_tri = tf$sparse$to_dense(out)
    lower_tri_reshaped = tf$reshape(lower_tri, c(S, d, d))
    
    L = diag_mat + lower_tri_reshaped
    
    return(L)
  }#, reduce_retracing = T
# )