# -*- coding: utf-8 -*-
# Comparing torch to lavaan for political democracy and holzinger-swineford models
# Case-wise optimization
import torch
import pandas as pd
import tensorsem as ts
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path("./experiments")

### OPTIMIZATION PARAMETERS ###
LRATE = 0.0001  # Adam learning rate
TOL = 1e-20  # loss change tolerance
MAXIT = 300  # maximum epochs

### POLITICAL DEMOCRACY MODEL ###
# The poldem model but without the parameter constraints.
opt_pol = ts.SemOptions.from_file(WORK_DIR / "lavaan_comparison/poldem_mod.pkl")
df_pol = pd.read_csv(WORK_DIR / "lavaan_comparison/poldem.csv")[opt_pol.ov_names]  # order the columns, important step!
df_pol -= df_pol.mean(0)  # center the data
N, P = df_pol.shape
dat_pol = torch.tensor(df_pol.values, dtype = torch.float64, requires_grad = False)

# Full case-wise stochastic gradient descent
mod_pol = ts.StructuralEquationModel(opt_pol, dtype = torch.float64)
optim_pol = torch.optim.Adam(mod_pol.parameters(), lr = LRATE)
ll_pol = []
for epoch in range(MAXIT):
    idx = torch.randperm(N)
    loss = 0
    for i in range(N):
        optim_pol.zero_grad()
        loss_i = ts.mvn_negloglik(dat_pol[idx[i], :], mod_pol())
        loss_i.backward()
        optim_pol.step()
        loss += loss_i
    ll_pol.append(-loss.item())
    print("Epoch:", epoch, "| LL:", -loss.item())
    
plt.plot(ll_pol)
plt.ylabel("Case-wise log-likelihood")
plt.xlabel("Epoch")
plt.title("Adam for PolDem Model\nLearning rate = " + str(LRATE))
plt.close()

est = mod_pol.free_params
se = mod_pol.Inverse_Hessian(ts.mvn_negloglik(dat_pol, mod_pol())).diag().sqrt()
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "lavaan_comparison/poldem_pars_casewise.csv")
