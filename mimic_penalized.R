# LASSO MIMIC estimation example for the manuscript titled
# "SEM using Computation Graphs"
# Last edited: 08-10-2019 by Erik-Jan van Kesteren

# load packages 
# For installation of tensorsem see:
# https://github.com/vankesteren/tensorsem/tree/computationgraph
library(MASS)
library(lavaan)
library(regsem)
library(tensorsem)

# load packages for plotting (optional but recommended)
library(tidyverse)
library(firatheme) # remotes::install_github("vankesteren/firatheme")

# generate data following the MIMIC model
set.seed(45)
N <- 1000
S <- rWishart(1, 40, diag(20))[,,1]/20
X <- mvrnorm(N, rep(0, 20), Sigma = S)
b <- c(rnorm(10, mean = 3), rep(0, 10))
factor <- X %*% b + rnorm(N, 0, sqrt(b %*% S %*% b))
lambda <- rnorm(5, mean = 3)
Y <- factor %*% lambda + mvrnorm(N, rep(0, 5), diag(rep(var(factor),5)))
mysd <- function(y) sqrt(sum((y - mean(y))^2) / length(y))
X    <- scale(X, scale = apply(X, 2, mysd))
Y    <- scale(Y, scale = apply(Y, 2, mysd))

# prepare data format
dat  <- data.frame(X = X, Y = Y)
mod  <- paste(
  "Factor =~", paste0("Y.", 1:5, collapse = " + "), "\n",
  "Factor ~", paste0("X.", 1:20, collapse = " + ")
)

# regsem first requires lavaan estimation
lavaan_mod <- sem(mod, dat, information = "observed", std.lv = TRUE)
regsem_mod <- regsem(lavaan_mod, lambda = 0.11, alpha = 0)

# tensorsem estimation
tensem_mod <- tf_sem(mod, dat)
tensem_mod$penalties$lasso_beta <- 0.11
tensem_mod$train(4000)

# get coefficients
regsem_coef <- round(unlist(coef(regsem_mod)[6:25]), 3)
tensem_coef <- round(tensem_mod$Beta[1,-1], 3)

# comparison of coefficients
tibble(
  coef      = 1:20,
  regsem    = regsem_coef, 
  tensorsem = tensem_coef
) %>% 
  gather(key = "method", value = "val", -coef) %>% 
  mutate(val = abs(val)) %>% 
  ggplot(aes(x = coef, y = val, fill = method)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_fira() +
  scale_fill_fira() +
  scale_x_continuous(breaks = 1:20) +
  labs(x = "MIMIC cause coefficient", y = "Absolute value", fill = "Method") +
  theme(legend.position = "top")

# save plot
firaSave("img/mimic-results.pdf", width = 9, height = 5)
firaSave("img/tiff/mimic-results.tiff", "tiff", width = 9, height = 5, dpi = 300)
