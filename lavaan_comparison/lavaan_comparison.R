# Validation for SEM in pytorch
library(tidyverse)
library(firatheme)
library(patchwork)
library(lavaan)
library(tensorsem)


### SAVE MODELS AND DATASETS TO FILE ###
# Default political democracy model, but without constrained loadings
mod_poldem <- "
  # latent variable definitions
     ind60 =~ x1 + x2 + x3
     dem60 =~ y1 + y2 + y3 + y4
     dem65 =~ y5 + y6 + y7 + y8

  # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60

  # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
"

opt_poldem <- syntax_to_torch_opts(mod_poldem)
dat_poldem <- scale(PoliticalDemocracy)

torch_opts_to_file(opt_poldem, "lavaan_comparison/poldem_mod.pkl")
write.csv(dat_poldem, "lavaan_comparison/poldem.csv", row.names = FALSE)


# Default Holzinger-Swineford model
mod_hs <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9 
"

opt_hs <- syntax_to_torch_opts(mod_hs)
dat_hs <- scale(HolzingerSwineford1939[,7:15])

torch_opts_to_file(opt_hs, "lavaan_comparison/hs_mod.pkl")
write.csv(dat_hs, "lavaan_comparison/hs.csv", row.names = FALSE)


### PYTHON STEP ###
# Run the file lavaan_comparison.py to generate output


### COMPARE LAVAAN TO TENSORSEM OUTPUT ###
fit_poldem <- sem(mod_poldem, dat_poldem, std.lv = TRUE, 
                  information = "observed", fixed.x = FALSE)
pt_poldem_lav   <- partable(fit_poldem) %>% mutate(method = "lavaan")
pt_poldem_torch <- partable_from_torch(
  pars  = read_csv("lavaan_comparison/poldem_pars.csv"), 
  model = mod_poldem
) %>% mutate(method = "pytorch")

plt_poldem <- 
  bind_rows(lavMatrixRepresentation(pt_poldem_lav), 
            lavMatrixRepresentation(pt_poldem_torch)) %>% 
  filter(mat != "psi") %>% 
  ggplot(aes(
    x      = paste0(lhs, op, rhs), 
    y      = est,
    ymin   = est - 1.96*se,
    ymax   = est + 1.96*se,
    colour = method
  )) +
  geom_pointrange(position = position_dodge(0.5)) +
  theme_fira() +
  scale_colour_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Parameter", y = "Estimate", colour = "Estimator", 
       title = "Political democracy") +
  facet_wrap(vars(mat), scales = "free")

fit_hs <- sem(mod_hs, dat_hs, std.lv = TRUE, information = "observed")
pt_hs_lav   <- partable(fit_hs) %>% mutate(method = "lavaan")
pt_hs_torch <- partable_from_torch(
  pars  = read_csv("lavaan_comparison/hs_pars.csv"),
  model = mod_hs
) %>% mutate(method = "pytorch")

plt_hs <- 
  bind_rows(
    lavMatrixRepresentation(pt_hs_lav), 
    lavMatrixRepresentation(pt_hs_torch)
  ) %>% 
  ggplot(aes(
    x      = paste0(lhs, op, rhs), 
    y      = est, 
    ymin   = est - 1.96*se, 
    ymax   = est + 1.96*se, 
    colour = method
  )) +
  geom_pointrange(position = position_dodge(0.5)) +
  theme_fira() +
  scale_colour_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Parameter", y = "Estimate", colour = "Estimator",
       title = "Holzinger-Swineford") +
  facet_wrap(vars(mat), scales = "free")

firaSave(plot = plt_hs / plt_poldem, filename = "lavaan_comparison/par_comp.pdf",
         width = 16, height = 9, dpi = 300)
