<p align="center">
  <h3 align="center">SEM using Computation Graphs</h3>
  <h4 align="center">Example Code</h4>
</p>
<br/>

This repository contains the code to produce the figures and tables in the manuscript titled "SEM using Computation Graphs" by Erik-Jan van Kesteren and Daniel Oberski. 

### Prerequisites
This repository depends on the [computationgraph branch of `tensorsem`](https://github.com/vankesteren/tensorsem/tree/computationgraph). Please install `tensorsem` by following the instructions there.

### Contents
| File | Description |
| :--- | :---------- |
| [SEM estimation](./sem_estimation.R)              | Shows how structural equation models can be estimated using our method and compares it to [`lavaan`](http://lavaan.org). |
| [LAD estimation](./lad_estimation.R)              | Shows how estimation can be done using the least absolute deviation objective function. |
| [LASSO regression](./lasso_regression.R)          | Compares our method to [`regsem`](https://github.com/Rjacobucci/regsem) and [`glmnet`](https://cran.r-project.org/web/packages/glmnet/index.html). | 
| [Penalized MIMIC](./mimic_penalized.R)            | Shows how LASSO regression can be performed on latent outcomes. |
| [Regularized Factor Analysis](./fa_regularized.R) | Shows how sparsity can be induced in confirmatory factor analysis through LASSO and spike-and-slab penalties. |
