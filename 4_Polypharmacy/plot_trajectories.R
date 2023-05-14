## Compare trajectories of the parameters with and without tempering

library(ggplot2)
# library(ggmatplot)

date <- "20230327_1"
S <- 1000
S_alpha <- 1000
reorder_data <- T
save_images <- T

if (reorder_data) {
  reorder_seed <- 2023
  reorder_info <- paste0("_seed", reorder_seed)
} else {
  reorder_info <- ""
}

## 1. Original ordering of the data
plot_directory <- paste0("./var_test/results/")

results_temper <- readRDS(paste0(plot_directory, "var_test_temper10", reorder_info,
                       "_S", S, "_Sa", S_alpha, "_", date, ".rds"))
results_notemper <- readRDS(paste0(plot_directory, "var_test", reorder_info,
                            "_S", S, "_Sa", S_alpha, "_", date, ".rds"))

## Plot parameter trajectories
subscripts <- c("0", "gender", "race", "age", "M1", "M2", "M3", "IM")
param_dim <- length(results_temper[[1]]$mu)

trajectories_temper <- list()
trajectories_notemper <- list()

for (p in 1:param_dim) {
  trajectories_temper[[p]] <- sapply(results_temper, function(r) r$mean_trajectories[[p]])
  trajectories_notemper[[p]] <- sapply(results_notemper, function(r) r$mean_trajectories[[p]])
}

if (save_images) {
  filename2 = paste0("polypharmacy_trajectories", reorder_info,
                     "_S", S, "_Sa", S_alpha, "_", date, ".png")
  filepath2 = paste0("./var_test/plots/", filename2)

  png(filepath2, width = 1000, height = 600)
}

par(mfrow = c(3, 3), mai = c(0.7, 0.7, 0.2, 0.3))
for (p in 1:param_dim) {
  if (p == param_dim) { ## if the parameter is tau
    matplot(sqrt(exp(trajectories_notemper[[p]])), type = "l", 
            # xlab = "Iterations",
            ylab = "",
            main = "", col = "skyblue", cex.axis = 2) #,
            # cex.lab = 1.1, cex.axis = 1.25, line = 2)
    title(ylab=expression(tau), xlab = "Iterations", line = 3, cex.lab = 2)
    matlines(sqrt(exp(trajectories_temper[[p]])), col = "salmon")
  } else {
    matplot(trajectories_notemper[[p]], type = "l", 
            #xlab = "Iterations",
            ylab = "",
            main = "", col = "skyblue", cex.axis = 2) #,
            # cex.lab = 1.1, cex.axis = 1.25)
    title(ylab=bquote(beta[.(subscripts[p])]), xlab = "Iterations", line = 3, cex.lab = 2)
    
    matlines(trajectories_temper[[p]], col = "salmon")
  }
}

if (save_images) {
  dev.off()
}


## ggplot ver 
# plots <- list()
# 
# p <- 1
# param_names <- c("beta1", "beta2", "beta3", "beta4", 
#                  "beta5", "beta6", "beta7", "beta8", "tau") 
# sims <- rep(1:10, each = 501)
# notemper.df <- data.frame(sim = sims, 
#                           vals = c(trajectories_notemper[[p]]))
# 
# temper.df <- data.frame(sim = sims, 
#                           vals = c(trajectories_temper[[p]]))
# test <- rbind(trajectories_notemper[[p]], trajectories_temper[[p]])
# group <- rep(c("notemp", "temp"), each = nrow(trajectories_notemper[[p]]))
# notemper.df <- data.frame(g = group, val = test)
# 
# plot <- ggmatplot(notemper.df, col = g, plot_type = "line")
#         
# print(plot)
