# -*- coding: utf-8 -*-
# Running sparse factor analysis
import torch
import pandas as pd
import tensorsem as ts
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path("./experiments/sparse_factor")

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

### MAXIMUM LIKELIHOOD ESTIMATION ###
mod_ml = ts.StructuralEquationModel(opt)
ll_values = []
optim = torch.optim.Adam(mod_ml.parameters(), lr = LRATE)
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", ll_values[-1])
    optim.zero_grad()
    ll = ts.mvn_negloglik(dat, mod_ml())
    ll_values.append(-ll.item())
    ll.backward()
    optim.step()

# Save optimization plot
plt.plot(ll_values)
plt.ylabel("Log-likelihood")
plt.xlabel("Epoch")
plt.title("ML optimization\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "ml_optim.png")
plt.close()

# Save params for comparison
est = mod_ml.free_params
se = mod_ml.Inverse_Hessian(ts.mvn_negloglik(dat, mod_ml())).diag().sqrt()
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params_ml.csv")

### FULLY BAYESIAN LASSO OPTIMIZATION ###
mod_pen = ts.StructuralEquationModel(opt)
loss_values = []

# prior for the penalty
hyperprior = torch.distributions.Gamma(1.78, 1)  # from park & casella
penalty = hyperprior.mean.clone().detach().requires_grad_(True)
prior = torch.distributions.Laplace(0, penalty)

# optimize (see penalty on first 22 elements of lambda!)
optim = torch.optim.Adam((mod_pen.dlt_vec, penalty), lr = LRATE)  # also optimize the penalty!
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", loss_values[-1])
    optim.zero_grad()
    Sigma = mod_pen()
    penalty.data = penalty.clamp_min(1e-20).data  # force penalty positive
    loss = ts.sem_fitfun(S, Sigma) - \
           prior.log_prob(mod_pen.Lam.t().flatten()[range(22)]).sum() - \
           hyperprior.log_prob(penalty)
    loss_values.append(loss.item())
    loss.backward()
    optim.step()

# Save optimization plot
plt.plot(loss_values)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Bayesian LASSO optimization\nLearning rate = " + str(LRATE))
plt.savefig(WORK_DIR / "lasso_optim.png")
plt.close()


# Save params for comparison
est = mod_pen.free_params
se = mod_pen.Inverse_Hessian(ts.mvn_negloglik(dat, mod_pen())).diag().sqrt()
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "params_pen.csv")


# visualize sparsity
plt.plot(mod_ml.Lam.t().flatten()[range(22)].detach())
plt.plot(mod_pen.Lam.t().flatten()[range(22)].detach())
plt.ylabel("Value")
plt.xlabel("Factor loading")
plt.title("Sparsity due to LASSO optimization")
plt.savefig(WORK_DIR / "sparsity.png")
plt.close()
