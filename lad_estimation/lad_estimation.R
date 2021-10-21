# LAD estimation example for the manuscript titled
# "Flexible extensions to SEM using Computation Graphs"
# Last edited: 14-05-2020 by Erik-Jan van Kesteren

# load packages 
library(MASS)
library(expm)
library(lavaan)
library(tidyverse)
library(xtable)
library(tensorsem)


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

opts <- syntax_to_torch_opts(mod)
torch_opts_to_file(opts, "lad_estimation/mod.pkl")

write.csv(X, file = "lad_estimation/dat.csv", row.names = FALSE)
write.csv(X_c, file = "lad_estimation/dat_c.csv", row.names = FALSE)

# lavaan model for comparison
lav_mod <- sem(mod, X, information = "observed")
lav_mod_c <- sem(mod, X_c, information = "observed")

### RUN PYTORCH ###
# now run lad_estimation.py

# Get the results for all the fitted models
pt_lav <- lavMatrixRepresentation(partable(lav_mod)) %>% 
  filter(mat == "lambda") %>% 
  mutate(method = "ML", cov = "uncontaminated")

pt_lav_c <- lavMatrixRepresentation(partable(lav_mod_c)) %>% 
  filter(mat == "lambda") %>% 
  mutate(method = "ML", cov = "contaminated")

pt_torch <- lavMatrixRepresentation(
    partable_from_torch(
      pars = read_csv("lad_estimation/params.csv"),
      model = mod
    )
  ) %>% 
  filter(mat == "lambda") %>% 
  mutate(method = "LAD", cov = "uncontaminated")

pt_torch_c <- lavMatrixRepresentation(
    partable_from_torch(
      pars = read_csv("lad_estimation/params_c.csv"),
      model = mod
    )
  ) %>% 
  filter(mat == "lambda") %>% 
  mutate(method = "LAD", cov = "contaminated")


# create nice data frame for printing in the paper
lad_df <- data.frame(
  round(rbind(pt_lav$est, pt_lav_c$est, pt_torch$est, pt_torch_c$est), 3),
  row.names = c("Uncontaminated (ML)", "Contaminated (ML)", 
                "Uncontaminated (LAD)", "Contaminated (LAD)")
)

# save the data frame to a file
saveRDS(lad_df, file = "lad_estimation/results.rds")

# and as a latex table
xtable(lad_df)
