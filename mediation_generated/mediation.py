# -*- coding: utf-8 -*-
# Running high-dimensional mediation
import torch
import pandas as pd
import tensorsem as ts
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path("./mediation_generated")

### OPTIMIZATION PARAMETERS ###
LRATE = 0.01  # Adam learning rate
TOL = 1e-20  # loss change tolerance
MAXIT = 5000  # maximum epochs

### DATA LOADING ###
opt = ts.SemOptions.from_file(WORK_DIR / "med_mod.pkl")
df = pd.read_csv(WORK_DIR / "mediation.csv")[opt.ov_names]  # order the columns, important step!
df -= df.mean(0)  # center the columns, important step too!
dat = torch.tensor(df.values, dtype = torch.float32, requires_grad = False)

# ULS estimation requirements
N, P = dat.shape
s = ts.vech(dat.t().mm(dat) / N)  # vech observed covariance

### ULS ESTIMATION ###
mod_uls = ts.StructuralEquationModel(opt)
optim = torch.optim.Adam(mod_uls.parameters(), lr = LRATE)
loss_uls = []
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", loss_uls[-1])
    optim.zero_grad()
    r = s - ts.vech(mod_uls())
    loss = r.t().mm(r)
    loss_uls.append(loss.item())
    loss.backward()
    optim.step()
    if epoch > 1:
        if abs(loss_uls[-1] - loss_uls[-2]) < TOL:
            break

plt.plot(loss_uls)
plt.ylabel("Loss value (ULS)")
plt.xlabel("Epoch")
plt.title("Mediation model optimization\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "optim_uls.png")
plt.close()

# save params to csv
est = mod_uls.free_params
se = torch.zeros(est.shape)  # no standard errors because they are take a long time
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params_uls.csv")


### LASSO ESTIMATION ###
mod_lasso = ts.StructuralEquationModel(opt)
optim = torch.optim.Adam(mod_lasso.parameters(), lr = LRATE)
loss_lasso = []
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", loss_lasso[-1])
    optim.zero_grad()
    r = s - ts.vech(mod_lasso())
    paths = torch.cat((mod_lasso.B_0[110, :-2], mod_lasso.B_0[:-2, 111]))
    loss = r.t().mm(r) + paths.abs().sum()
    loss_lasso.append(loss.item())
    loss.backward()
    optim.step()
    if epoch > 1:
        if abs(loss_lasso[-1] - loss_lasso[-2]) < TOL:
            break

plt.plot(loss_lasso)
plt.ylabel("Loss value (ULS + LASSO)")
plt.xlabel("Epoch")
plt.title("Mediation model optimization\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "optim_lasso.png")
plt.close()

# save params to csv
est = mod_lasso.free_params
se = torch.zeros(est.shape)  # no standard errors because they are inconsistent for LASSO
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params_lasso.csv")