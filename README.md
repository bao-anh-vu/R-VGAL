# R-VGAL

This folder contains R code for the 4 examples in the R-VGAL manuscript: the linear mixed model with simulated data, the logistic mixed model with simulated data, and the logistic mixed model applied on the Six city and POLYPHARMACY datasets.

## Folder structure
Each example is stored in one folder, which contains separate sub-folders:
1. `source`, which contains the R code to run the R-VGAL and HMC algorithms on the model in the example
2. `results`, which contains the output from the source code (both R-VGAL and HMC outputs)
3. `plots`, which contains the plots of the posterior densities and bivariate posterior plots for each parameter in the model.
4. `var_test`, which contains the R code, output and plot for the test on the variance of the R-VGAL results for different Monte Carlo sample sizes (see Section S3 of the Supplementary material)

## Running the scripts
To reproduce the results in the manuscript, for example that of the Linear mixed model, download the `1_Linear` folder and run the `linear_mm_main.R` file. Note that the working directory needs to be set to the `1_Linear` folder for the filepaths to work properly. To reproduce the results based on pre-saved data and output, set `rerun_rvga` and `rerun_hmc` at the start of the `linear_mm_main.R` file to `FALSE`, and the script will produce output and plots from the `results` and `plots` folders automatically.

To re-run R-VGAL and HMC from scratch, set the flags `rerun_rvga` and `rerun_hmc` to `TRUE`. For this step, the `tensorflow` and `reticulate` libraries are required. The package versions required are listed below.

Results from other examples can be similarly reproduced by running the `*_main.R` file in each example's respective folder.

Results from the Supplementary material can be reproduced by running the `var_test_*.R` files in the `Logistic/var_test` and `Polypharmacy/var_test` folders. Inside each `var_test_*.R` files, there are flags to enable/disable variational tempering, enable/disable reordering the data, and to set the number of Monte Carlo samples $S$ and $S_\alpha$. Results for cases where the values of $S$ and $S_\alpha$ are taken from the set {50, 100, 500, 1000} are already saved so that they can be reproduced if the flag `rerun_test` is set to `FALSE`, but setting $S$ and $S_\alpha$ to any other values requires `rerun_test = TRUE`.

## Package requirements
Running the source code requires the following packages:
1. `reticulate` v1.27
2. `tensorflow` v2.11
3. `rstan` v2.21.7 (for instructions on how to install RStan, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started))
4. `ggplot2` v3.4.2
5. `gridExtra` v2.3
6. `gtable` v0.3.0         
         
         
         
