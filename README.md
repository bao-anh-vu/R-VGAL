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
To reproduce the results in the manuscript, for example that of the Linear mixed model, download the `1_Linear` folder and run the `linear_mm_main.R` file. Note that the working directory needs to be set to the `1_Linear` folder for the filepaths to work properly (and similarly, for other examples, set the working directory to the folder containing that example). 
- To reproduce the results based on pre-saved data and output, set `rerun_rvga` and `rerun_hmc` at the start of the `linear_mm_main.R` file to `FALSE`, and the script will produce output and plots from the `results` and `plots` folders automatically. 
- To re-run R-VGAL and HMC from scratch, set the flags `rerun_rvga` and `rerun_hmc` to `TRUE`. 

Results from other examples can be similarly reproduced by running the `*_main.R` file in each example's respective folder.

Results from the Supplementary material can be reproduced by running the `var_test_*.R` files in the `Logistic/var_test` and `Polypharmacy/var_test` folders. Inside each `var_test_*.R` files, there are flags to enable/disable variational tempering, enable/disable reordering the data, and to set the number of Monte Carlo samples $S$ and $S_\alpha$. Results for cases where the values of $S$ and $S_\alpha$ are taken from the set {50, 100, 500, 1000} are already saved so that they can be reproduced if the flag `rerun_test` is set to `FALSE`, but setting $S$ and $S_\alpha$ to any other values requires `rerun_test = TRUE`.

The RStudio version and R packages required to run the code, along with installation instructions for these packages, can be found in the next section. 

## RStudio version requirements
In order to run the HMC code, which was implemented in RStan 2.21, it is highly recommended that you install R version 4.0 or 4.1. The latest released version of RStan at the time of writing is 2.21, which is not yet compatible with R 4.2 and above. There is an RStan development version, 2.26.x, which can be configured to work with R 4.2, but the code in this repository has not been tested on such a configuration.

Note that prior to installing RStan, you need to configure your R installation to be able to compile C++ code. For instructions, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started) under **Configuring C++ Toolchain**. Note that instructions vary depending on your operating system, and if you are using Windows, instructions will also vary depending on your R version (3.6/4.0/4.2). 

The R-VGAL code will work with R versions 3.6 onwards, provided that the package `tensorflow` version 2.11 is installed. First, install the `tensorflow` R package as follows:

```
install.packages("tensorflow")
```
Next, run the following lines of code:
```
library(tensorflow)
install_tensorflow(version = "2.11")
```
which will install `tensorflow` v2.11. If prompted to install Miniconda, select yes by typing 'Y'.

System requirements and a more detailed installation guide for `tensorflow` in R can be found [here](https://tensorflow.rstudio.com/install). 

## Package requirements 
Running the source code requires the following packages (along with their dependencies, which should be installed automatically):
1. `tensorflow` v2.11
2. `rstan` v2.21.7 (for instructions on how to install RStan, see [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started))
3. `ggplot2` v3.4.2
4. `gridExtra` v2.3
5. `gtable` v0.3.0         
6. `mvtnorm` v1.1
7. `dplyr` v1.1
8. `readxl` v1.4.2
