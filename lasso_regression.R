# LASSO estimation example for the manuscript titled
# "SEM using Computation Graphs"
# Last edited: 08-10-2019 by Erik-Jan van Kesteren

# load packages 
# For installation of tensorsem see:
# https://github.com/vankesteren/tensorsem/tree/computationgraph
library(MASS)
library(lavaan)
library(glmnet)
library(regsem)
library(tensorsem)

# load packages for plotting (optional but recommended)
library(tidyverse)
library(firatheme) # remotes::install_github("vankesteren/firatheme")

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
dat  <- data.frame(X = X, y = y)
mod  <- paste("y ~", paste0("X.", 1:20, collapse = " + "))

# glmnet estimation
glmnet_mod <- glmnet(X, y, "gaussian", lambda = 0.028, intercept = FALSE)

# regsem first requires lavaan estimation
lavaan_mod <- sem(mod, dat, information = "observed")
regsem_mod <- regsem(lavaan_mod, lambda = 0.11, alpha = 0)

# tensorsem estimation
tensem_mod <- tf_sem(mod, dat)
tensem_mod$penalties$lasso_beta <- 0.11
tensem_mod$train(4000)

# get coefficients
glmnet_coef <- round(glmnet_mod$beta[,1], 3)
regsem_coef <- round(unlist(coef(regsem_mod)[-21]), 3)
tensem_coef <- round(tensem_mod$Beta[1,-1], 3)

# comparison of coefficients
tibble(
  coef      = 1:20,
  glmnet    = glmnet_coef,
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
  labs(x = "Regression coefficient", y = "Absolute value", fill = "Method") +
  theme(legend.position = "top")

# save plot
firaSave("img/reg-results.pdf", width = 9, height = 5)
firaSave("img/tiff/reg-results.tiff", "tiff", width = 9, height = 5, dpi = 300)
