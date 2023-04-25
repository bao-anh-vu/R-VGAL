# R-VGAL

This folder contains R code for the 4 examples in the R-VGAL manuscript: the linear mixed model with simulated data, the logistic mixed model with simulated data, and the logistic mixed model applied on the Six city and POLYPHARMACY datasets.

## Folder structure
Each example is stored in one folder, which contains separate sub-folders:
1. scripts, which contains the R code to run the R-VGAL and HMC algorithms on the model in the example
2. results, which contains the output from the scripts (both R-VGAL and HMC outputs)
3. plots, which contains the plots of the posterior densities and bivariate posterior plots for each parameter in the model.

In the logistic simulated data example and the POLYPHARMACY example, there are additional folders named `var_test`, which contains code and output for the tests on the variance of the R-VGAL posterior densities in the Supplementary material of the manuscript. The script for the variance tests are contained in the {scripts} folder, while the output and plot from the variance tests are stored in the 'var_test/results' and 'var_test/plots' folders, respectively.

## Package requirements
Running R-VGAL requires the following packages:
1. reticulate v1.27
2. rstan v2.21.7
3. tensorflow v2.11
