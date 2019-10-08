# LAD estimation example for the manuscript titled
# "SEM using Computation Graphs"
# Last edited: 08-10-2019 by Erik-Jan van Kesteren

# load packages 
# For installation of tensorsem see:
# https://github.com/vankesteren/tensorsem/tree/computationgraph
library(MASS)
library(expm)
library(lavaan)
library(tensorsem)

# load packages for plotting (optional but recommended)
library(tidyverse)
library(firatheme) # remotes::install_github("vankesteren/firatheme")

# generate data following exactly the distribution
LY <- cbind(c(1, 1.17, 1.18, 1.36, 1.40, 1.42, 1.34, 1.23, 0.89))
TE <- diag(9)
S <- (LY %*% t(LY)) + TE
X <- matrix(rnorm(1e3*9), 1e3)
X <- data.frame(X %*% sqrtm(solve(cov(X))) %*% sqrtm(S))

S_contam <- S
S_contam[3, 1] <- S_contam[1, 3] <- 2
S_contam[4, 2] <- S_contam[2, 4] <- 0.35
X_c <- matrix(rnorm(1e3*9), 1e3)
X_c <- data.frame(X_c %*% sqrtm(solve(cov(X_c))) %*% sqrtm(S_contam))

mod <- "
F1 =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9
"

# lavaan model for comparison
lav_mod <- sem(mod, X, information = "observed")

# maximum likelihood training
fml_mod <- tf_sem(mod, X, fit_fun = "ml")
fml_mod$train(1e4)

# least absolute deviation training
lad_mod <- tf_sem(mod, X, fit_fun = "lad")
lad_mod$train(1e4)

# contaminated lavaan
lav_mod_c <- sem(mod, X_c, information = "observed")

# contaminated maximum likelihood training
fml_mod_c <- tf_sem(mod, X_c, fit_fun = "ml")
fml_mod_c$train(1e4)

# contaminated least absolute deviation training
lad_mod_c <- tf_sem(mod, X_c, fit_fun = "lad")
lad_mod_c$train(1e4)

# get parameter estimates
lav_lambdas <- lav_mod@Model@GLIST$lambda[1:9, 1]
fml_lambdas <- fml_mod$Lambda[1:9, 1]
lad_lambdas <- lad_mod$Lambda[1:9, 1]

lav_lambdas_c <- lav_mod_c@Model@GLIST$lambda[1:9, 1]
fml_lambdas_c <- fml_mod_c$Lambda[1:9, 1]
lad_lambdas_c <- lad_mod_c$Lambda[1:9, 1]

# create nice data frame for printing in the paper
lad_df <- data.frame(
  round(rbind(
    lav_lambdas,   fml_lambdas,   lad_lambdas, 
    fml_lambdas_c, lad_lambdas_c
  ), 3),
  row.names = c(
    "lavaan (ML)", "tensorsem (ML)", "tensorsem (LAD)",
    "Contaminated (ML)", "Contaminated (LAD)")
)

# save the data frame to a file
save(lad_df, file = "R/output/lad_df.Rdata")

# Plot the loss
lad_mod_c$loss_vec %>% 
  enframe %>% 
  ggplot(aes(x = name, y = value)) +
  geom_line(col = firaCols[1], size = 1) + 
  theme_fira() + 
  labs(x = "Iteration", y = "Loss value (LAD)")

# save plot
firaSave("img/loss-plot.pdf", width = 9, height = 4)
firaSave("img/tiff/loss-plot.tiff", "tiff", width = 9, height = 4, dpi = 300)