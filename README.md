# R-VGAL

This folder contains R code for the 4 examples in the R-VGAL manuscript: 
1. the linear mixed model with simulated data, 
2. the logistic mixed model with simulated data,
3. the logistic mixed model applied to the Six city dataset, and 
4. the logistic mixed model applied to the POLYPHARMACY dataset.

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

## RStudio version requirement
The R-VGAL code used to produce results in the manuscript requires package `tensorflow` version 2.11, which can be installed by first typing

```
install.packages("tensorflow")
```
Next type
```
install_tensorflow(version = "2.11")
```
which will install `tensorflow` v2.11. 

A quick installation guide along with system requirements for `tensorflow` in R can be found [here](https://tensorflow.rstudio.com/install). 

In order to run the HMC code, which was implemented in RStan, it is highly recommended that you install R version 4.0 or above. Note that prior to installing RStan, you need to configure your R installation to be able to compile C++ code. For instructions, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started)) under **Configuring C++ Toolchain**. Note that instructions vary depending on your operating system, and if you are using Windows, instructions will also vary depending on your R version (3.6/4.0/4.2). 

## Package version requirements
Running the source code requires the following packages:
1. `reticulate` v1.27
2. `tensorflow` v2.11
3. `rstan` v2.21.7 (for instructions on how to install RStan, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started))
4. `ggplot2` v3.4.2
5. `gridExtra` v2.3
6. `gtable` v0.3.0         
