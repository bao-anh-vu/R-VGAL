# R-VGAL

This folder contains R code for the 4 examples in the R-VGAL manuscript: the linear mixed model with simulated data, the logistic mixed model with simulated data, and the logistic mixed model applied on the Six city and POLYPHARMACY datasets.

## Folder structure
Each example is stored in one folder, which contains separate sub-folders:
1. `scripts`, which contains the R code to run the R-VGAL and HMC algorithms on the model in the example
2. `results`, which contains the output from the scripts (both R-VGAL and HMC outputs)
3. `plots`, which contains the plots of the posterior densities and bivariate posterior plots for each parameter in the model.

In the logistic simulated data example and the POLYPHARMACY example, there are additional folders named `var_test`, which contains code and output for the tests on the variance of the R-VGAL posterior densities in the Supplementary material of the manuscript. The script for the variance tests are contained in the `scripts` folder, while the output and plot from the variance tests are stored in the `var_test/results` and `var_test/plots` folders, respectively.

## Running the scripts
To reproduce the results in the manuscript, for example that of the Linear mixed model, download the `Linear` folder and open the file `Linear.Rproj`. Run the `linear_mm_main.R` file inside `scripts`. By default, this file will read the data and output saved in the `Linear/data` and `Linear/results` folders. The `tensorflow` and `reticulate` libraries do not need to be installed for this step.

To re-run R-VGAL and HMC from scratch, set the flags `rerun_rvga` and `rerun_hmc` at the start of the `linear_mm_main.R` file to `TRUE`. For this step, the `tensorflow` and `reticulate` libraries are required. The package versions required are listed below.

Similarly, results from other examples involving the logistic mixed model can be reproduced by running the `*_main.R` file in each example's respective folder.

Results from the Supplementary material can be reproduced using the `var_test_*.R` files in the Logistic and Polypharmacy folders. Inside each `var_test_*.R` files, there are flags to enable/disable variational tempering, enable/disable reordering the data, and to set the number of Monte Carlo samples $S$ and $S_\alpha$. Results for cases where the values of $S$ and $S_\alpha$ are taken from the set {50, 100, 500, 1000} are already saved so that they can be reproduced if the flag `rerun_test` is set to `FALSE`, but setting $S$ and $S_\alpha$ to any other values requires `rerun_test = TRUE`.

## Package requirements
Running the scripts requires the following packages:
1. `reticulate` v1.27
2. `tensorflow` v2.11
3. `rstan` v2.21.7 (for instructions on how to install RStan, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started))
4. `ggplot2` v3.4.2
5. `gridExtra` v2.3
6. `gtable` v0.3.0         
         
         
         
