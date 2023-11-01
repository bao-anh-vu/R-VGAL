#!/usr/bin/env Rscript
# args = commandArgs(trailingOnly=TRUE)
# sim = args[1]

setwd("~/R-VGAL/1_Linear/")

rm(list = ls())

# library("mvtnorm")
# library("Matrix")
library("rstan")
library("parallel")
# library("ggplot2")
# library("grid")
# library("gtable")
# library("gridExtra")
# library("reshape2")

# source("./source/generate_data.R")
# source("./source/run_rvgal.R")
source("./source/run_stan_lmm.R")
source("./multi_sims/run_multi_sims_hmc.R")

sims <- 1:100
parallel::mclapply(sims, run_multi_sims_hmc, mc.cores = 10L)