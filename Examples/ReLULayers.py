# %% [markdown]
# # ReLU Layers
# 
# We can write a ReLU layer $z = \max(Wx+b, 0)$ as the
# convex optimization problem
# \begin{equation}
# \begin{array}{ll}
# \mbox{minimize} & \|z-\tilde Wx - b\|_2^2 \\[.2cm]
# \mbox{subject to} & z \geq 0, \\
# & \tilde W = W,
# \end{array}
# \label{eq:prob}
# \end{equation}
# with variables $z$ and $\tilde W$,
# and parameters $W$, $b$, and $x$.
# (Note that we have added an extra variable $\tilde W$ so
# that the problem is DPP.)
# 
# We can embed this problem into a PyTorch `Module` and use it
# as a layer in a sequential neural network.
# We note that this example is purely illustrative;
# one can implement a ReLU layer much more efficiently
# by directly performing the matrix multiplication, vector addition,
# and then taking the positive part.

# %%
import os

import pyomo.environ as pyo
import numpy as np
from pyomolayers import PyomoOptLayer
import torch
import time
import torch.nn as nn
# import matplotlib.pyplot as plt

this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(this_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# %%
def create_model(nominal_W, nominal_b, nominal_x):
    # Create a concrete model
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    m.Wtilde = pyo.Var(range(D_out), range(D_in), within=pyo.Reals)
    
    # Define parameters
    m.W = pyo.Var(range(D_out), range(D_in), within=pyo.Reals)
    m.b = pyo.Var(range(D_out), within=pyo.Reals)
    m.x = pyo.Var(range(D_in), within=pyo.Reals)

    # Define variables
    m.z = pyo.Var(range(D_out), within=pyo.NonNegativeReals)
    
    for i in range(D_out):
        for j in range(D_in):
            m.W[i, j].fix(nominal_W[i, j])
            
    for i in range(D_out):
        m.b[i].fix(nominal_b[i])

    for i in range(D_in):
        m.x[i].fix(nominal_x[i])

    m.equ_constraints = pyo.ConstraintList()
    for i in range(D_out):
        for j in range(D_in):
            m.equ_constraints.add(m.Wtilde[i, j] == m.W[i, j])

    # Define the objective function
    def objective_rule(m):
        tol_term = 0
        for i in range(D_out):
            tol_term += (m.z[i] - sum(m.Wtilde[i, j] * m.x[j] for j in range(D_in)) - m.b[i])**2
        return tol_term

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return m

def model_instance():
    np.random.seed(1)
    nominal_W, nominal_x, nominal_b = np.random.rand(D_out, D_in), np.random.rand(D_in), np.random.rand(D_out)
    model = create_model(nominal_W, nominal_b, nominal_x)
    return model
# %%
class ReluLayer(torch.nn.Module):
    def __init__(self, D_in, D_out, Pyomolayer):
        super(ReluLayer, self).__init__()
        np.random.seed(0)
        self.W = torch.nn.Parameter(1e-3*torch.from_numpy(np.random.randn(D_out, D_in)))
        self.b = torch.nn.Parameter(1e-3*torch.from_numpy(np.random.randn(D_out)))
        self.layer = Pyomolayer

    def forward(self, x):
        # when x is batched, repeat W and b 
        if x.ndim == 2:
            batch_size = x.shape[0]
            out, _, _ = self.layer(self.W.repeat(batch_size, 1, 1), self.b.repeat(batch_size, 1), x)
            z = out
            return z
        else:
            out, _, _ = self.layer(self.W, self.b, x)
            z = out
            return z
# %% Test the Pyomo model
# D_out, D_in = 20, 20
# np.random.seed(1)
# nominal_W, nominal_x, nominal_b = np.random.rand(D_out, D_in), np.random.rand(D_in), np.random.rand(D_out)

# model = create_model(nominal_W, nominal_b, nominal_x)
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())
# %%
def main():
    global D_out, D_in
    D_out, D_in = 20, 20
    sample_num = 300
    epoch = 50
    learning_rate = 1e-2
    
    model = model_instance()
    variables = [model.z]
    parameters = [model.W, model.b, model.x]
    Pyomolayer = PyomoOptLayer(model, variables, parameters)

    torch.manual_seed(0)
    net = torch.nn.Sequential(
    ReluLayer(D_in = D_in, D_out = D_out, Pyomolayer = Pyomolayer),
    ReluLayer(D_in = D_in, D_out = D_out, Pyomolayer = Pyomolayer),
    torch.nn.Linear(D_out, 1)
    )
    torch.manual_seed(0)
    X = torch.randn(sample_num, D_in)
    Y = torch.randn(sample_num, 1)

    training_loss = []
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for _ in range(epoch):
        opt.zero_grad()
        loss = torch.nn.MSELoss()(net(X), Y)
        print(loss.item())
        
        loss.backward()
        training_loss.append(loss.item())
        opt.step()
        
    np.save(os.path.join(results_dir, "ReLU_training_loss"), training_loss)
if __name__ == "__main__":
    main()
