# high-dimensional mediation analysis
library(tidyverse)
library(firatheme)
library(ggrepel)
library(lavaan)
library(tensorsem)

# read the data, for preprocessing see cmfilter paper
med_dat <- read_rds("mediation_hidim/med_dat.rds") %>% 
  mutate_all(function(x) c(scale(x)))

write.csv(med_dat, file = "mediation_hidim/med_dat.csv", row.names = FALSE)

# create model
med_mod <- paste0(
  paste(colnames(med_dat[,-1:-2]), collapse = " + "), " ~ x\n",
  "y ~ ", paste(colnames(med_dat[,-1:-2]), collapse = " + "), " + x"
)

opts <- syntax_to_torch_opts(med_mod)
torch_opts_to_file(opts, "mediation_hidim/med_mod.pkl")

# note: 3003 free parameters:
sum(opts$delta_free)

### PYTHON STEP ###
# Run the file mediation_hidim.py to generate output


### READ OUTPUT ###
pt_uls <- lavMatrixRepresentation(partable_from_torch(
  pars = read_csv("mediation_hidim/params_uls.csv"),
  syntax = med_mod
))

pt_lasso <- lavMatrixRepresentation(partable_from_torch(
  pars = read_csv("mediation_hidim/params_lasso.csv"),
  syntax = med_mod
))


tibble(
  "LASSO Estimate" = pt_lasso[1:1000, "est"]*pt_lasso[1001:2000, "est"],
  "ULS Estimate" = pt_uls[1:1000, "est"]*pt_uls[1001:2000, "est"],
  mediator = pt_lasso[1:1000, "lhs"],
  rowid = 1:1000
) %>% 
  pivot_longer(-c(mediator, rowid)) %>% 
  mutate(label = ifelse(name == "LASSO Estimate" & abs(value) > 0.006, mediator, "")) %>% 
  ggplot(aes(x = rowid, y = abs(value), colour = as_factor(name), shape = as_factor(name), alpha = as_factor(name))) +
  geom_hline(yintercept = 0) +
  geom_point() + 
  geom_text_repel(aes(label = label), color = "black") +
  scale_colour_fira() +
  scale_alpha_manual(values = c("LASSO Estimate" = 1, "ULS Estimate" = 0.5), guide = "none") +
  theme_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Mediator", y = "Absolute indirect effect", 
       title = "High-dimensional sparse mediation analysis: Gene methylation",
       colour = "", shape = "") +
  theme(legend.position = "top")
firaSave("mediation_hidim/hidim_results.pdf", width = 13, height = 5)
