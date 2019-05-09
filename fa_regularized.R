# LASSO FA example for the manuscript titled
# "SEM using Computation Graphs"
# Last edited: 09-05-2019 by Erik-Jan van Kesteren

# load packages 
# For installation of tensorsem see:
# https://github.com/vankesteren/tensorsem/tree/computationgraph
library(MASS)
library(lavaan)
library(tensorsem)

# load packages for plotting (optional but recommended)
library(tidyverse)
library(firatheme) # remotes::install_github("vankesteren/firatheme")

# generate data
set.seed(45)
p   <- 5
n   <- 300
Fct <- rnorm(n)
b   <- c(1, rnorm(p - 1), rep(0, p))
v   <- b^2
v[(p + 1):(2 * p)] <- 1
X <- tcrossprod(Fct, b) + mvrnorm(n, rep(0, 2*p), diag(v))
X <- data.frame(X)

# generate model
mod <- paste(
  paste0("Fact =~ ", paste0("X", 1:(p*2), collapse = " + "))
)

# estimate parameters in lavaan
lavaan_fa <- sem(mod, X, information = "observed")

# Estimate parameters in tensorsem
tensem_fa <- tf_sem(mod, X)
tensem_fa$train(10000)
loading_0 <- tensem_fa$Lambda

# add lasso penalty and retrain
tensem_fa$penalties$lasso_lambda <- 0.1
tensem_fa$train(1000)
loading_1 <- tensem_fa$Lambda

# add lasso penalty and retrain
tensem_fa$penalties$lasso_lambda <- 0.3
tensem_fa$train(1000)
loading_2 <- tensem_fa$Lambda

# Spike and slab prior and retrain
tensem_fa$penalties$lasso_lambda  <- 0    # remove lasso
tensem_fa$penalties$spike_lambda  <- 0.55 # spike
tensem_fa$penalties$slab_lambda   <- 0.05 # slab
tensem_fa$penalties$mixing_lambda <- 0.5  # proportion of loadings in spike
tensem_fa$train(5e3, verbose = TRUE)
loading_3 <- tensem_fa$Lambda

# create nice data frame for printing in the paper
lasso_fa_df <- data.frame(
  round(rbind(c(loading_0), c(loading_1), c(loading_2), c(loading_3)), 3),
  row.names = c("$\\lambda$ = 0", "$\\lambda$ = 0.1", "$\\lambda$ = 0.3", "spike-slab")
)

# save the data frame to a file
save(lasso_fa_df, file = "R/output/lasso_fa_df.Rdata")

# create a plot for the priors
dlap <- function(x, mu, l) {
  # laplace distributions
  exp(-abs(x - mu) / l) / 2
}
dspikeslab <- function(x, l1, l2, pi) {
  # spike-slab distribution (mixture of laplace & normal)
  pi * dlap(x, 0, 1/l1) + (1 - pi) * dnorm(x, 0, 1/l2)
}
xx <- seq(-10, 10, length.out = 1000)

tibble(
  "spike-slab"  = dspikeslab(xx, 1, .1, .5),
  "laplace (1)" = dlap(xx, 0, 1),
  "normal (10)" = dnorm(xx, 0, 10)
) %>% 
  mutate("theta" = xx) %>% 
  gather("prior", "p", -theta) %>% 
  mutate(prior = factor(prior, levels = c("laplace (1)", "spike-slab", "normal (10)"))) %>%
  ggplot(aes(x = theta, y = p, linetype = prior, colour = prior)) +
  geom_line(size = 1) +
  theme_fira() +
  scale_linetype_manual(values = c(1, 4, 3)) +
  scale_colour_manual(values = c(firaCols[3], firaCols[1], firaCols[5])) +
  labs(x = "Parameter", y = "Density", linetype = "Distribution", colour = "Distribution") +
  theme(legend.position = "top")

# save plot
firaSave("R/output/fa_priors.pdf", width = 9, height = 5)
