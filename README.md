<p align="center">
  <h3 align="center">Flexible Extensions to Structural Equation Models Using Computation Graphs</h3>
  <h4 align="center">Code repository</h4>
</p>
<br/>

This repository contains the code to produce the figures and tables in the article titled "Flexible Extensions to Structural Equation Models Using Computation Graphs" by Erik-Jan van Kesteren and Daniel Oberski. [DOI:10.1080/10705511.2021.1971527](https://doi.org/10.1080/10705511.2021.1971527)

### Prerequisites
- This repository depends on the [`tensorsem`](https://github.com/vankesteren/tensorsem/tree/computationgraph). Please install `tensorsem` by following the instructions there.
- This repository depends on several other packages, managed by `renv`. The first time you open `sem-computationgraphs.Rproj`, use `renv::restore()` to install the required dependencies.

### Contents
| Folder | Description |
| :--- | :---------- |
| [Lavaan comparison](./lavaan_comparison) | Shows how structural equation models can be estimated using our method and compares it to [`lavaan`](http://lavaan.org). |
| [LAD estimation](./lad_estimation) | Shows how estimation can be done using the least absolute deviation objective function. |
| [LASSO regression](./lasso_regression) | Compares our method to [`regsem`](https://github.com/Rjacobucci/regsem) and [`glmnet`](https://cran.r-project.org/web/packages/glmnet/index.html). | 
| [High-dimensional mediation (Generated)](./mediation_generated) | Performs ULS and LASSO-penalized ULS estimation for generated high-dimensional mediation data. |
| [High-dimensional mediation (Example)](./mediation_hidim) | Performs ULS and LASSO-penalized ULS estimation for real-world high-dimensional mediation data. |
| [Sparse Factor Analysis](./sparse_factor) | Shows how sparsity can be induced in confirmatory factor analysis through Bayesian LASSO estimation. |
