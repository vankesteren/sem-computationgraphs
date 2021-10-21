# LAD estimation example for the manuscript titled
# "Flexible extensions to SEM using Computation Graphs"
# Last edited: 25-05-2020 by Erik-Jan van Kesteren

library(MASS)
library(lavaan)
library(glmnet)
library(regsem)
library(tidyverse)
library(firatheme) # remotes::install_github("vankesteren/firatheme")
library(xtable)
library(tensorsem)

# generate data
set.seed(45)
S <- rWishart(1, 40, diag(20))[,,1]/20
X <- mvrnorm(1000, rep(0, 20), Sigma = S)
b <- c(rnorm(10, mean = 3), rep(0, 10))
y <- X %*% b + rnorm(1000, 0, sqrt(b %*% S %*% b))
mysd <- function(y) sqrt(sum((y - mean(y))^2) / length(y))
X    <- scale(X, scale = apply(X, 2, mysd))
y    <- scale(y, scale = mysd(y))

# prepare data format
colnames(X) <- paste0("X", 1:20)
colnames(y) <- "y"
dat  <- data.frame(X, y)
mod  <- paste("y ~", paste0("X", 1:20, collapse = " + "))

# Save for pytorch
torch_opts_to_file(syntax_to_torch_opts(mod), "regularized_regression/mod.pkl")
write_csv(dat, file = "regularized_regression/dat.csv")

# glmnet estimation
glmnet_mod <- glmnet(X, y, "gaussian", lambda = 0.028, intercept = FALSE)

# regsem first requires lavaan estimation
lavaan_mod <- sem(mod, dat, information = "observed")
regsem_mod <- regsem(lavaan_mod, lambda = 0.11, alpha = 0, gradFun = "ram")

# get coefficients for torch
pt_torch <- partable_from_torch(
  pars = read_csv("regularized_regression/params_pen.csv"), 
  syntax = mod
)


# comparison of coefficients
# create nice data frame for printing in the paper
lasso_df <- data.frame(
  round(rbind(glmnet_mod$beta[,1], 
              unlist(coef(regsem_mod)[-21]), 
              pt_torch$est[1:20]), 3),
  row.names = c("glmnet", "regsem", "pytorch")
)

# save the data frame to a file
save(lasso_df, file = "regularized_regression/lasso_df.rds")

# and as a latex table
xtable(lasso_df)
