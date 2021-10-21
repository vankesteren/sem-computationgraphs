# -*- coding: utf-8 -*-
# Comparing torch to lavaan for political democracy and holzinger-swineford models
import torch
import pandas as pd
import tensorsem as ts
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path("./lavaan_comparison")

### OPTIMIZATION PARAMETERS ###
LRATE = 0.01  # Adam learning rate
TOL = 1e-20  # loss change tolerance
MAXIT = 10000  # maximum epochs

### POLITICAL DEMOCRACY MODEL ###
# The poldem model but without the parameter constraints.
opt_pol = ts.SemOptions.from_file(WORK_DIR / "poldem_mod.pkl")
df_pol = pd.read_csv(WORK_DIR / "poldem.csv")[opt_pol.ov_names]  # order the columns, important step!
df_pol -= df_pol.mean(0)  # center the data
N, P = df_pol.shape
dat_pol = torch.tensor(df_pol.values, dtype = torch.float64, requires_grad = False)

# create the model and optimize it
mod_pol = ts.StructuralEquationModel(opt_pol, dtype = torch.float64)
optim_pol = torch.optim.Adam(mod_pol.parameters(), lr = LRATE)
ll_pol = []
for epoch in range(MAXIT):
    if epoch % 1000 == 1:
        print("Epoch:", epoch, " ll:", ll_pol[-1])
    optim_pol.zero_grad()
    loss = ts.mvn_negloglik(dat_pol, mod_pol())
    loss.backward()
    ll_pol.append(-loss.item())
    optim_pol.step()
    if epoch > 1:
        if abs(ll_pol[-1] - ll_pol[-2]) < TOL:
            break

plt.plot(ll_pol)
plt.ylabel("Log-likelihood")
plt.xlabel("Epoch")
plt.title("Adam for PolDem Model\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "poldem_optim.png")
plt.close()

# write parameter estimates and se
est = mod_pol.free_params
se = mod_pol.Inverse_Hessian(ts.mvn_negloglik(dat_pol, mod_pol())).diag().sqrt()
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "poldem_pars.csv")

### HOLZINGER-SWINEFORD MODEL ###
# The holzinger-swineford model
opt_hs = ts.SemOptions.from_file(WORK_DIR / "hs_mod.pkl")
df_hs = pd.read_csv(WORK_DIR / "hs.csv")[opt_hs.ov_names]  # order the columns, important step!
df_hs -= df_hs.mean(0)  # center the data
dat_hs = torch.tensor(df_hs.values, dtype = torch.float32, requires_grad = False)

# create the model and optimize it
mod_hs = ts.StructuralEquationModel(opt_hs, dtype = torch.float64)
optim_hs = torch.optim.Adam(mod_hs.parameters(), lr = LRATE)
ll_hs = []
for epoch in range(MAXIT):
    if epoch % 1000 == 1:
        print("Epoch:", epoch, " ll:", ll_hs[-1])
    optim_hs.zero_grad()
    loss = ts.mvn_negloglik(dat_hs, mod_hs())
    ll_hs.append(-loss.item())
    loss.backward()
    optim_hs.step()
    if epoch > 1:
        if abs(ll_hs[-1] - ll_hs[-2]) < TOL:
            break

plt.plot(ll_hs)
plt.ylabel("Log-likelihood")
plt.xlabel("Epoch")
plt.title("Adam for Holzinger-Swineford Model\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "hs_optim.png")
plt.close()

# write parameter estimates and se
est = mod_hs.free_params
se = mod_hs.Inverse_Hessian(ts.mvn_negloglik(dat_hs, mod_hs())).diag().sqrt()
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "hs_pars.csv")
