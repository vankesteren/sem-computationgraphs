# stress-test data generation
# devtools::install_github("vankesteren/cmfilter")
library(tidyverse)
library(firatheme)
library(cmfilter)
library(lavaan)
library(tensorsem)

### GENERATE DATA ###
set.seed(45)
dat <- generateMed(a = c(rep(.5, 10), rep(0, 100)),
                   b = c(rep(.5, 10), rep(0, 100)),
                   n = 40)
dat <- scale(dat, scale = FALSE)
write.csv(dat, file = "mediation_generated/mediation.csv", row.names = FALSE)

model <- "
M.1 + M.2 + M.3 + M.4 + M.5 + M.6 + M.7 + M.8 + M.9 + M.10 + M.11 + M.12 + M.13 + M.14 + M.15 + M.16 + M.17 + M.18 + M.19 + M.20 + M.21 + M.22 + M.23 + M.24 + M.25 + M.26 + M.27 + M.28 + M.29 + M.30 + M.31 + M.32 + M.33 + M.34 + M.35 + M.36 + M.37 + M.38 + M.39 + M.40 + M.41 + M.42 + M.43 + M.44 + M.45 + M.46 + M.47 + M.48 + M.49 + M.50 + M.51 + M.52 + M.53 + M.54 + M.55 + M.56 + M.57 + M.58 + M.59 + M.60 + M.61 + M.62 + M.63 + M.64 + M.65 + M.66 + M.67 + M.68 + M.69 + M.70 + M.71 + M.72 + M.73 + M.74 + M.75 + M.76 + M.77 + M.78 + M.79 + M.80 + M.81 + M.82 + M.83 + M.84 + M.85 + M.86 + M.87 + M.88 + M.89 + M.90 + M.91 + M.92 + M.93 + M.94 + M.95 + M.96 + M.97 + M.98 + M.99 + M.100 + M.101 + M.102 + M.103 + M.104 + M.105 + M.106 + M.107 + M.108 + M.109 + M.110 ~ x
y ~ M.1 + M.2 + M.3 + M.4 + M.5 + M.6 + M.7 + M.8 + M.9 + M.10 + M.11 + M.12 + M.13 + M.14 + M.15 + M.16 + M.17 + M.18 + M.19 + M.20 + M.21 + M.22 + M.23 + M.24 + M.25 + M.26 + M.27 + M.28 + M.29 + M.30 + M.31 + M.32 + M.33 + M.34 + M.35 + M.36 + M.37 + M.38 + M.39 + M.40 + M.41 + M.42 + M.43 + M.44 + M.45 + M.46 + M.47 + M.48 + M.49 + M.50 + M.51 + M.52 + M.53 + M.54 + M.55 + M.56 + M.57 + M.58 + M.59 + M.60 + M.61 + M.62 + M.63 + M.64 + M.65 + M.66 + M.67 + M.68 + M.69 + M.70 + M.71 + M.72 + M.73 + M.74 + M.75 + M.76 + M.77 + M.78 + M.79 + M.80 + M.81 + M.82 + M.83 + M.84 + M.85 + M.86 + M.87 + M.88 + M.89 + M.90 + M.91 + M.92 + M.93 + M.94 + M.95 + M.96 + M.97 + M.98 + M.99 + M.100 + M.101 + M.102 + M.103 + M.104 + M.105 + M.106 + M.107 + M.108 + M.109 + M.110
y ~ x
"
opts <- syntax_to_torch_opts(model)
torch_opts_to_file(opts, "mediation_generated/med_mod.pkl")


### PYTHON STEP ###
# Run the file mediation.py to generate output


### READ OUTPUT ###
pt_uls <- lavMatrixRepresentation(partable_from_torch(
  pars = read_csv("mediation_generated/params_uls.csv"),
  model = model
))

pt_lasso <- lavMatrixRepresentation(partable_from_torch(
  pars = read_csv("mediation_generated/params_lasso.csv"),
  model = model
))

tibble(
  "LASSO Estimate" = pt_lasso[1:110, "est"]*pt_lasso[111:220, "est"],
  "ULS Estimate" = pt_uls[1:110, "est"]*pt_uls[111:220, "est"],
  "True effect" = c(rep(.5, 10), rep(0, 100))^2,
  mediator = as_factor(pt_lasso[1:110, "lhs"])
) %>% 
  pivot_longer(-mediator) %>% 
  ggplot(aes(x = mediator, y = value, colour = as_factor(name), shape = as_factor(name))) +
  geom_hline(yintercept = 0) +
  geom_point() + 
  scale_colour_fira() +
  theme_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "", y = "Indirect effect", 
       title = "High-dimensional sparse mediation analysis: simulated data",
       colour = "", shape = "") +
  theme(legend.position = "top")
firaSave("mediation_generated/mediation_results.pdf", width = 13, height = 5)
