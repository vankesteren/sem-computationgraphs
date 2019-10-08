# SEM estimation example for the manuscript titled
# "SEM using Computation Graphs"
# Last edited: 08-10-2019 by Erik-Jan van Kesteren

# load packages 
# For installation of tensorsem see:
# https://github.com/vankesteren/tensorsem/tree/computationgraph
library(lavaan)
library(tensorsem)

# load packages for plotting (optional but recommended)
library(tidyverse)
library(firatheme) # remotes::install_github("vankesteren/firatheme")

# Construct the model and perform estimation in lavaan
mod <- "
F1 =~ x1 + x2 + x3
F2 =~ x4 + x5 + x6
F1 ~ F2
"
sem_lav <- sem(mod, HolzingerSwineford1939, information = "observed", 
               std.lv = TRUE)

# Create a tensorsem object and estimate the parameters
sem_mod <- tf_sem(mod, HolzingerSwineford1939)
sem_mod$train(7000)

# Compare the parameter estimates
est_tf  <- sem_mod$delta_free[c(2:7, 1, 8:13)]
est_lav <- coef(sem_lav)

# Compute standard errors
se_tf  <- sqrt(diag(sem_mod$ACOV))[c(2:7, 1, 8:13)]
se_lav <- sqrt(diag(vcov(sem_lav)))

lo_tf  <- est_tf  - 1.96 * se_tf
hi_tf  <- est_tf  + 1.96 * se_tf
lo_lav <- est_lav - 1.96 * se_lav
hi_lav <- est_lav + 1.96 * se_lav

# Parameter names
parnames <- 
c("Lambda[21]", "Lambda[31]", 
  "Lambda[52]", "Lambda[62]", 
  "Beta[0~12]", 
  "Theta[11]",  "Theta[22]",  "Theta[33]",
  "Theta[44]",  "Theta[55]",  "Theta[66]",
  "Psi[11]",    "Psi[22]")

# Plot the results
pd <- position_dodge(0.55)
tibble(
  param = rep(names(est_lav), 2),
  est = c(est_tf, est_lav),
  lo  = c(lo_tf,  lo_lav), 
  hi  = c(hi_tf,  hi_lav),
  method = rep(c("tensorsem", "lavaan"), each = length(est_lav))
) %>% 
  ggplot(aes(x = param, y = est, ymin = lo, ymax = hi, colour = method)) +
  geom_errorbar(size = 1, position = pd, width = 0.5) +
  geom_point(position = pd, size = 2.5) +
  theme_fira() +
  scale_colour_fira() +
  theme(panel.grid.major.x = element_blank(), legend.position = "top") +
  labs(x = "Parameter", y = "Estimate", colour = "Method") +
  scale_x_discrete(labels = parse(text = parnames))

# save plot
firaSave("img/sem-results.pdf", width = 9, height = 5)
firaSave("img/tiff/sem-results.tiff", "tiff", width = 9, height = 5, dpi = 300)
