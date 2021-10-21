# LAD estimation example for the manuscript titled
# "Flexible extensions to SEM using Computation Graphs"
# Last edited: 15-05-2020 by Erik-Jan van Kesteren

library(tidyverse)
library(firatheme)
library(lavaan)
library(tensorsem)

# data from UCI machine learning repository
dat_raw <- read_csv("sparse_factor/parkinsons.csv")
dat <- dat_raw %>% 
  select(status, everything(), -name) %>% 
  set_names("status", paste0("X", 1:22))

# look at skewness
dat %>% 
  pivot_longer(cols = everything())  %>% 
  ggplot(aes(x = value)) +
  geom_density(fill = "grey") +
  facet_wrap(~as_factor(name), scales = "free")

# fix the skewness and check!
dat %>% 
  mutate(across(paste0("X", 2:15), log)) %>% 
  pivot_longer(cols = everything()) %>% 
  ggplot(aes(x = value)) +
  geom_density(fill = "grey") +
  facet_wrap(~as_factor(name), scales = "free")

# apply the fix and scale the data
dat <- 
  dat %>% 
  mutate(across(paste0("X", 2:15), log)) %>% 
  mutate(across(paste0("X", 1:22), function(x) c(scale(x)))) %>% 
  mutate(status = c(scale(status, center = TRUE, scale = FALSE)))%>% 
  as_tibble()


# create a model
mod <- paste("Factor =~", paste(paste0("X", 1:22), sep = "", collapse = " + "), 
             "\n Factor ~ status")

torch_opts_to_file(syntax_to_torch_opts(mod), 
                   filename = "sparse_factor/mod.pkl")
write_csv(dat, file = "sparse_factor/dat.csv")

# lavaan model for comparison
fit <- lavaan::sem(mod, dat, std.lv = TRUE, information = "observed", 
                   fixed.x = FALSE, control = list(iter.max = 500, x.tol = 1.5e-10))

### RUN PYTORCH ###
# now run sparse_factor.py


# Get the results for all the fitted models
pt_lav <- lavMatrixRepresentation(partable(fit)) %>% 
  filter(mat %in% c("lambda", "beta")) %>% 
  mutate(method = "lavaan")

pt_ml <- lavMatrixRepresentation(partable_from_torch(
    pars = read_csv("sparse_factor/params_ml.csv"),
    syntax = mod
  )) %>% 
  filter(mat %in% c("lambda", "beta")) %>% 
  mutate(method = "ML (pytorch)")

pt_pen <- lavMatrixRepresentation(partable_from_torch(
    pars = read_csv("sparse_factor/params_pen.csv"),
    syntax = mod
  )) %>% 
  filter(mat %in% c("lambda", "beta")) %>% 
  mutate(method = "Bayesian LASSO")

pt_pen$se <- NA

pt_lav$est <- -pt_lav$est


# why are they so different?
# hypothesis: lavaan does not optimise properly
# openmx model for comparison
library(umx)
results <- umxRAM(mod, data = as.data.frame(dat), std.lv = TRUE)
respar  <- summary(results)$parameters

# openmx leads to the same results as pytorch.
xtable::xtable(data.frame(
  pytorch = pt_ml$est[1:22],
  OpenMX  = respar$Estimate[2:23],
  lavaan  = -pt_lav$est[1:22], 
  row.names = paste0("X", 1:22)
))

df_omx <- data.frame(
  rhs = paste0("X", 1:22),
  est = respar$Estimate[2:23],
  se = respar$Std.Error[2:23],
  method = "ML (OpenMx)"
)

df_gg <- bind_rows(
  df_omx, 
  pt_ml  %>% filter(mat == "lambda") %>% select(rhs, est, se, method),
  pt_pen %>% filter(mat == "lambda") %>% select(rhs, est, se, method)
)

# create plot for printing in the paper
loadings_plot <- 
  df_gg %>% 
  ggplot(aes(
    x      = as_factor(rhs), 
    y      = est , 
    ymin   = est - 1.96*se, 
    ymax   = est + 1.96*se, 
    colour = as_factor(method)
  )) +
  geom_pointrange(position = position_dodge(0.5)) +
  theme_fira() +
  scale_colour_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Indicator", y = "Loading (95% CI)", colour = "Method",
       title = "Factor loadings")

loadings_plot
firaSave("sparse_factor/sparse_loadings.pdf", width = 12, height = 6)


score_plot <-
  tibble(fac = c(scale(as.matrix(dat[,1:22]) %*% pt_ml$est[1:22]), 
               scale(as.matrix(dat[,1:22]) %*% pt_pen$est[1:22])),
       met = as_factor(rep(c("ML", "LASSO"), each = 195)),
       class = factor(rep(dat$status, 2), labels = c("0", "1"))) %>% 
  ggplot(aes(x = class, y = fac)) +
  geom_point(position = position_jitter(0.07), colour = "#343434") +
  geom_boxplot(colour = "black", fill = "transparent", width = 0.5, 
               outlier.colour = "transparent") +
  facet_wrap(~met) +
  theme_fira() +
  labs(x = "Parkinson status", y = "Standardized factor score",
       title = "Factor scores") +
  theme(panel.background = element_rect(colour = "#343434"))

firaSave("sparse_factor/factor_scores.pdf", width = 6, height = 6)

loadings_plot + score_plot + 
  plot_layout(2, 1, widths = c(0.65, 0.35))
firaSave("sparse_factor/sparse_loadings.pdf", width = 12, height = 5)
