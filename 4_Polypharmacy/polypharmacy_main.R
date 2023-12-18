## Logistic mixed model with POLYPHARMACY data ##

# setwd("~/R-VGAL/4_Polypharmacy/")

## Flags
rerun_rvgal <- F
save_rvgal_results <- F
rerun_stan <- F
save_hmc_results <- F
date <- "20230327_1" 
use_tempering <- T
reorder_data <- F
save_plots <- F

# reticulate::use_condaenv("tf2.11", required = TRUE)
library("tensorflow")
library("readxl") # part of tidyverse
library("dplyr")
library("mvtnorm")
library("rstan")
library("ggplot2")
library("gridExtra")
library("grid")
library("gtable")

source("./source/run_rvgal.R")
source("./source/run_stan_logmm.R")

# List physical devices
gpus <- tf$config$experimental$list_physical_devices('GPU')

if (length(gpus) > 0) {
  tryCatch({
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    tf$config$experimental$set_virtual_device_configuration(
      gpus[[1]],
      list(tf$config$experimental$VirtualDeviceConfiguration(memory_limit=4096))
    )
    
    logical_gpus <- tf$config$experimental$list_logical_devices('GPU')
    
    print(paste0(length(gpus), " Physical GPUs,", length(logical_gpus), " Logical GPUs"))
  }, error = function(e) {
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  })
}

if (reorder_data) {
  reorder_seed <- 2023
}

if (use_tempering) {
  n_obs_to_temper <- 10
  a_vals_temper <- rep(1/4, 4)
}

n_post_samples <- 20000 # desired number of posterior samples
S <- 200L ## number of Monte Carlo samples for approximating the expectations in R-VGAL
S_alpha <- 200L ## number of MC samples for approximating Fisher's/Louis' identities in R-VGA

####################################
##            Read data           ##
####################################

data <- read_excel("./data/polypharm.xls")
head(data)

## Split the variable MHV4 into MHV1, MHV2, MHV3
data$MHV1 <- ifelse(data$MHV4 == "1", 1, 0)
data$MHV2 <- ifelse(data$MHV4 == "2", 1, 0)
data$MHV3 <- ifelse(data$MHV4 == "3", 1, 0)

data$RACE_transf <- ifelse(data$RACE == "0", 0, 1)
data$INPTMHV <- ifelse(data$INPTMHV3 == "0", 0, 1)

## Logistic model: 
## y_ij = logit(beta_0 + beta_gender * Gender_i + beta_race * Race_i + beta_age * Age_ij
## + beta_M1 * MHV1_ij + beta_M2 * MHV2_ij + beta_M3 * MHV3_ij + beta_IM * INPTMHV_ij + u_i
## u_i ~ N(0, exp(2*tau)) is a subject level random intercept

param_names <- c("beta_0", "beta_gender", "beta_race", "beta_age",
                 "beta_M1", "beta_M2", "beta_M3", "beta_IM", "tau")

param_dim <- length(param_names)

## Split data by subjects (ID)
y_long = data[, c("ID", "POLYPHARMACY")]
y <- y_long %>% group_split(ID)
y <- lapply(y, function(x) { x["ID"] <- NULL; as.vector(data.matrix(x)) }) # get rid of the ID column then convert from df to matrix

intercept <- rep(1, length(y)) ## intercept term
fixed_effects <- c("GENDER", "RACE_transf", "AGE", "MHV1", "MHV2", "MHV3", "INPTMHV")
X_long = cbind(data[, "ID"], intercept, data[, fixed_effects])

## Also need a mapping of observations to subject... which in this case is the ID column
X <- X_long %>% group_split(ID) # split observations by ID
X <- lapply(X, function(x) { x["ID"] <- NULL; data.matrix(x) }) # get rid of the ID column then convert from df to matrix

## Define some variables for convenience
N <- length(unique(data$ID)) # number of subjects
n <- nrow(X[[1]]) # number of responses per subject

######################################
##        Maximum likelihood        ##
######################################
library(lme4)
glm_fit <- glmer(POLYPHARMACY ~ 1 + GENDER + RACE_transf + AGE + 
                                MHV1 + MHV2 + MHV3 + INPTMHV + (1 | ID),
                                family = "binomial", data = data)
ss <- getME(glm_fit, c("theta","fixef"))
update(glm_fit, start = ss, control = glmerControl(optimizer="bobyqa",
                            optCtrl = list(maxfun=2e5)))

fixef(glm_fit) ## returns fixed effects
# ranef(glm_fit) ## returns random effects
glm_params <- c(glm_fit@beta, glm_fit@theta)

#############################
##          R-VGA          ##
#############################

## Results directory
if (use_tempering) {
  temper_info <- paste0("_temper", n_obs_to_temper)
} else {
  temper_info <- ""
}

if (reorder_data) {
  reorder_info <- paste0("_seed", reorder_seed)
} else {
  reorder_info <- ""
}

result_directory <- "./results/"
results_file <- paste0("polypharmacy_rvga", temper_info, reorder_info,
                       "_S", S, "_Sa", S_alpha, "_", date, ".rds")
results_filepath <- paste0(result_directory, results_file)

## Initialise variational parameters
n_fixed_effects <- as.integer(ncol(X[[1]]))
param_dim <- n_fixed_effects + 1L

beta_0 <- rep(0, param_dim - 1L)
omega_0 <- 1 #log(0.5^2) 

mu_0 <- c(beta_0, omega_0)
P_0 <- diag(c(rep(10, n_fixed_effects), 1))

if (rerun_rvgal) {
  
  ## Reorder data if needed
  if (reorder_data) {
    set.seed(reorder_seed) 
    reordered_ind <- sample(1:length(y))
    print(head(reordered_ind))
    reordered_y <- lapply(reordered_ind, function(i) y[[i]])
    reordered_X <- lapply(reordered_ind, function(i) X[[i]])
    
    y <- reordered_y
    X <- reordered_X
  }
  
  rvgal_results <- run_rvgal(y, X, mu_0, P_0, 
                            n_post_samples = n_post_samples,
                            S = S, S_alpha = S_alpha,
                            use_tempering = use_tempering, 
                            n_temper = n_obs_to_temper, 
                            temper_schedule = a_vals_temper,
                            save_results = save_rvgal_results)
  
  if (save_rvgal_results) {
    saveRDS(rvgal_results, file = results_filepath)
  }
  
} else {
  rvgal_results <- readRDS(file = results_filepath)
}

rvga.post_samples <- rvgal_results$post_samples

####################
##      STAN      ##
####################
burn_in <- 5000
n_chains <- 2
hmc.iters <- n_post_samples/n_chains + burn_in

if (rerun_stan) {
  hmc.t1 <- proc.time()
  
  ## Data manipulation ##
  y_long <- data$POLYPHARMACY
  X_long <- cbind(intercept, data[, fixed_effects])
  
  logistic_code <- "./source/logistic_mm.stan"
  
  # logistic_data <- list(N = N * n, M = N, K = length(fixed_effects)+1, y = y, 
  #                       x = X, g = rep(1:N, each = n))
  
  hmc_results <- run_stan_logmm(iters = hmc.iters, burn_in = burn_in, 
                                n_chains = n_chains, data = y_long, 
                                grouping = rep(1:N, each = n), n_groups = N,
                                fixed_covariates = X_long)
  
  hmc.t2 <- proc.time()
  
  if (save_hmc_results) {
    saveRDS(hmc_results, file = paste0(result_directory, "polypharmacy_mm_hmc_", date, ".rds"))
  }
  
} else {
  hmc_results <- readRDS(file = paste0(result_directory, "polypharmacy_mm_hmc_", date, ".rds")) # for the experiements on starting points
}

hmc.fit <- hmc_results$post_samples[-(1:burn_in),,]
hmc.n_eff <- hmc_results$n_eff
hmc.Rhat <- hmc_results$Rhat

# traceplot(hfit, c("beta[1]","beta[2]","beta[3]","beta[4]",
#                   "beta[5]","beta[6]","beta[7]","beta[8]", "omega"),
#           ncol=1,nrow=5,inc_warmup=F)

#####################################################
##        Plot posterior density estimates         ##
#####################################################

rvgal.post_samples <- matrix(NA, nrow = n_post_samples, ncol = param_dim)
hmc.samples <- matrix(NA, n_post_samples, param_dim)

for (p in 1:param_dim) {
  
  if (p == param_dim) { # if the parameter is tau
    rvgal.post_samples[, p] <- sqrt(exp(rvgal_results$post_samples[, p])) # then back-transform
    hmc.samples[, p] <- sqrt(exp(hmc.fit[, , p]))
  } else {
    rvgal.post_samples[, p] <- rvgal_results$post_samples[, p]
    hmc.samples[, p] <- hmc.fit[, , p]
  }
}

rvgal.post_mean <- as.vector(apply(rvgal.post_samples, 2, mean))
rvgal.post_sd <- as.vector(apply(rvgal.post_samples, 2, sd))
hmc.post_mean <- as.vector(apply(hmc.samples, 2, mean))
hmc.post_sd <- as.vector(apply(hmc.samples, 2, sd))

## Put all results in a data frame
results <- data.frame(params = param_names, 
                      rvgal_mean = rvgal.post_mean, rvgal_sd = rvgal.post_sd,
                      hmc_mean = hmc.post_mean, hmc_sd = hmc.post_sd)
print(results)

## ggplot version
param_names <- c("beta1", "beta2", "beta3", "beta4", 
                 "beta5", "beta6", "beta7", "beta8", "tau")  
rvgal.df <- data.frame(beta = rvgal.post_samples) 
hmc.df <- data.frame(beta = hmc.samples)
colnames(rvgal.df) <- param_names
colnames(hmc.df) <- param_names

plots <- list()

for (p in 1:(param_dim-1)) {
  glm.df <- data.frame(param = param_names[p], val = glm_params[p])
  plot <- ggplot(rvgal.df, aes(x = .data[[param_names[p]]])) + 
    geom_density(col = "red", lwd = 1) +
    geom_density(data = hmc.df, col = "blue", lwd = 1) +
    geom_vline(data = glm.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth = 0.75) +
    labs(x = bquote(beta[.(p)])) +
    theme_bw() + 
    theme(axis.title = element_blank(), axis.text = element_text(size = 18)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 2)) 

  plots[[p]] <- plot  
}

glm.df <- data.frame(param = "tau", val = glm_fit@theta)
tau_plot <- ggplot(rvgal.df, aes(x = tau)) + 
  geom_density(col = "red", lwd = 1) +
  geom_density(data = hmc.df, col = "blue", lwd = 1) +
  geom_vline(data = glm.df, aes(xintercept=val),
               color="black", linetype="dashed", linewidth = 0.75) +
  labs(x = expression(tau)) +
  theme_bw() + 
  theme(axis.title = element_blank(), axis.text = element_text(size = 18)) +                               # Assign pretty axis ticks
  scale_x_continuous(breaks = scales::pretty_breaks(n = 2)) 

plots[[param_dim]] <- tau_plot


## Posterior covariance plot

## Function to map indices of plots inside a list to matrix (i,j) indices
## for arranging the plots in a lower triangular formation
n_lower_tri <- (param_dim^2 - param_dim)/2

index_to_i_j_colwise_nodiag <- function(k, n) {
  kp <- n * (n - 1) / 2 - k
  p  <- floor((sqrt(1 + 8 * kp) - 1) / 2)
  i  <- n - (kp - p * (p + 1) / 2)
  j  <- n - 1 - p
  c(i, j)
}

cov_plots <- list()
for (ind in 1:n_lower_tri) {
  mat_ind <- index_to_i_j_colwise_nodiag(ind, param_dim)
  p <- mat_ind[1]
  q <- mat_ind[2]
  glm.df <- data.frame(x = glm_params[q], y = glm_params[p])

  # cov_plot <- ggplot(rvgal.df, aes_string(x = param_names[p], y = param_names[q])) +
  cov_plot <- ggplot(rvgal.df, aes(x = .data[[param_names[q]]], y = .data[[param_names[p]]])) +
    stat_ellipse(col = "goldenrod", type = "norm", lwd = 1) +
    stat_ellipse(data = rvgal.df, col = "red", type = "norm", lwd = 1) +
    stat_ellipse(data = hmc.df, col = "blue", type = "norm", lwd = 1) +
    geom_point(data = glm.df, aes(x = x, y = y),
               shape = 4, color = "black", size = 5) +
    theme_bw() +
    theme(axis.title = element_blank(), axis.text = element_text(size = 18)) +                               # Assign pretty axis ticks
    scale_x_continuous(breaks = scales::pretty_breaks(n = 2)) 
  
  cov_plots[[ind]] <- cov_plot
}

m <- matrix(NA, param_dim, param_dim)
n_cov_plots <- param_dim * (param_dim-1)/2
m[lower.tri(m, diag = F)] <- 1:n_cov_plots
gr <- grid.arrange(grobs = cov_plots, layout_matrix = m)
gr2 <- gtable_add_cols(gr, unit(1, "null"), -1)
gr3 <- gtable_add_grob(gr2, grobs = lapply(plots, ggplotGrob), t = 1:9, l = 1:9)

# A list of text grobs - the labels
vars <- list(textGrob(bquote(beta[0])), textGrob(bquote(beta[gender])), 
             textGrob(bquote(beta[race])), textGrob(bquote(beta[age])), 
             textGrob(bquote(beta[M1])), textGrob(bquote(beta[M2])), 
             textGrob(bquote(beta[M3])), textGrob(bquote(beta[IM])), 
             textGrob(bquote(tau)))
vars <- lapply(vars, editGrob, gp = gpar(col = "black", fontsize = 20))

# So that there is space for the labels,
# add a row to the top of the gtable,
# and a column to the left of the gtable.
gp <- gtable_add_cols(gr3, unit(1.5, "lines"), 0)
gp <- gtable_add_rows(gp, unit(1.5, "lines"), -1) #0 adds on the top

# gtable_show_layout(gp)

# Add the label grobs.
# The labels on the left should be rotated; hence the edit.
# t and l refer to cells in the gtable layout.
# gtable_show_layout(gp) shows the layout.
gp <- gtable_add_grob(gp, lapply(vars[1:9], editGrob, rot = 90), t = 1:9, l = 1) # add column names to column 1, rows 2:9
gp <- gtable_add_grob(gp, vars[1:9], t = 10, l = 2:10) # add row names to row 6, columns 1:9

grid.newpage()
grid.draw(gp)

## Parameter trajectories
# par(mfrow = c(3,3))
# trajectories <- list()
# for (p in 1:param_dim) {
#   trajectories[[p]] <- sapply(rvgal_results$mu, function(e) e[p])
#   plot(trajectories[[p]], type = "l", xlab = "Iteration", ylab = param_names[p], main = "")
# }

if (save_plots) {
  plot_file <- paste0("polypharmacy_posterior", temper_info, reorder_info,
                      "_S", S, "_Sa", S_alpha, "_", date, ".png")
  png(paste0("./plots/", plot_file), width = 1300, height = 850)
  grid.newpage()
  grid.draw(gp)
  dev.off()
} 

## Time benchmark
hmc.time <- sum(colSums(hmc_results$time))
rvga.time <- rvgal_results$time_elapsed
print(hmc.time)
print(rvga.time)
