# -*- coding: utf-8 -*-
# Running sparse factor analysis
import torch
import pandas as pd
import tensorsem as ts
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path("./regularized_regression")

### OPTIMIZATION PARAMETERS ###
LRATE = 0.001  # Adam learning rate
TOL = 1e-20  # loss change tolerance
MAXIT = 5000  # maximum epochs

### DATA LOADING ###
opt = ts.SemOptions.from_file(WORK_DIR / "mod.pkl")
df = pd.read_csv(WORK_DIR / "dat.csv")[opt.ov_names]  # order the columns, important step!
df -= df.mean(0)
N, P = df.shape

dat = torch.tensor(df.values, requires_grad = False, dtype = torch.float32)
S = dat.t().mm(dat).div(N)

### LASSO OPTIMIZATION ###
mod_pen = ts.StructuralEquationModel(opt)
loss_values = []
penalty = torch.tensor(0.11)
optim = torch.optim.Adam((mod_pen.dlt_vec, penalty), lr = LRATE)  # also optimize the penalty!
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", loss_values[-1])
    optim.zero_grad()
    Sigma = mod_pen()
    loss = ts.sem_fitfun(S, Sigma) + mod_pen.B_0.flatten()[range(1, 21)].abs().sum().mul(penalty)
    loss_values.append(loss.item())
    loss.backward()
    optim.step()

# Save optimization plot
plt.plot(loss_values)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("LASSO optimization\nLearning rate = " + str(LRATE) + ", lambda = .11")
plt.savefig(WORK_DIR / "lasso_optim.png")
plt.close()

# Save params for comparison
est = mod_pen.free_params
se = mod_pen.Inverse_Hessian(ts.mvn_negloglik(dat, mod_pen())).diag().sqrt()
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params_pen.csv")
