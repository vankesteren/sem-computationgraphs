# -*- coding: utf-8 -*-
# Running high-dimensional mediation
import torch
import pandas as pd
import tensorsem as ts
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path("./experiments/lad_estimation")

### OPTIMIZATION PARAMETERS ###
LRATE = 0.0001  # Adam learning rate
TOL = 1e-20  # loss change tolerance
MAXIT = 15000  # maximum epochs

### DATA LOADING ###
opt = ts.SemOptions.from_file(WORK_DIR / "mod.pkl")
df = pd.read_csv(WORK_DIR / "dat.csv")[opt.ov_names]  # order the columns, important step!
df -= df.mean(0)  # center the columns, important step too!
df_c = pd.read_csv(WORK_DIR / "dat_c.csv")[opt.ov_names]  # order the columns, important step!
df_c -= df_c.mean(0)  # center the columns, important step too!

N, P = df.shape

dat = torch.tensor(df.values, dtype = torch.float32, requires_grad = False)
s = ts.vech(dat.t().mm(dat).div(N))
dat_c = torch.tensor(df_c.values, dtype = torch.float32, requires_grad = False)
s_c = ts.vech(dat_c.t().mm(dat_c).div(N))

### OPTIMIZATION ###
mod = ts.StructuralEquationModel(opt)
optim = torch.optim.Adam(mod.parameters(), lr = LRATE)
loss_values = []
for epoch in range(MAXIT):
    if epoch % 1000 == 1:
        print("Epoch:", epoch, " loss:", loss_values[-1])
    optim.zero_grad()
    r = s - ts.vech(mod())
    loss = r.abs().sum()
    loss_values.append(loss.item())
    loss.backward()
    optim.step()
    if epoch > 1:
        if abs(loss_values[-1] - loss_values[-2]) < TOL:
            break

plt.plot(loss_values)
plt.ylabel("Log-likelihood")
plt.xlabel("Epoch")
plt.title("Factor model optimization\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "lad_optim.png")
plt.close()

# save params to csv
est = mod.free_params
se = mod.Inverse_Hessian(ts.mvn_negloglik(dat, mod())).diag().sqrt()  # takes a while because we are inverting a big big matrix
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params.csv")


### OPTIMIZATION with CONTAMINATION ###
mod = ts.StructuralEquationModel(opt)
optim = torch.optim.Adam(mod.parameters(), lr = LRATE)
loss_values = []
for epoch in range(MAXIT):
    if epoch % 1000 == 1:
        print("Epoch:", epoch, " loss:", loss_values[-1])
    optim.zero_grad()
    r = s_c - ts.vech(mod())
    loss = r.abs().sum()
    loss_values.append(loss.item())
    loss.backward()
    optim.step()

plt.plot(loss_values)
plt.ylabel("Log-likelihood")
plt.xlabel("Epoch")
plt.title("Contaminated factor model optimization\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "lad_optim_c.png")
plt.close()

# save params to csv
est = mod.free_params
se = mod.Inverse_Hessian(ts.mvn_negloglik(dat, mod())).diag().sqrt()  # takes a while because we are inverting a big big matrix
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params_c.csv")