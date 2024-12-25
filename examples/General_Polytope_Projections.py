# %% [markdown]
# # General Polytope Projections
# \begin{align}
# \text{min}_x & \frac{1}{2}\sum_{i = 1}^{n} (x_i - p_i)^2, 
# \\
# \\
# \text{s.t.} \quad & \mathbf{G} \mathbf{x} \leq \mathbf{h} \\
# \end{align}
# with the variable $\mathbf{x}$ and parameter $\mathbf{h}$.

# %%
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir) # Add the parent directory to Python's search path
results_dir = os.path.join(current_dir, "examples/results")

import pyomo.environ as pyo
import numpy as np
from pyomolayer import PyomoOptLayer
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.spatial import HalfspaceIntersection
from matplotlib.patches import Polygon
import torch.optim as optim
# %%
def create_model(nominal_G, nominal_h, nominal_p):
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # Define parameters
    m.G = pyo.Var(range(p), range(n), within=pyo.Reals)  # Inequality constraint matrix
    m.p = pyo.Var(range(n), within=pyo.Reals) 
    m.h = pyo.Var(range(p), within=pyo.Reals) 

    # Define variables
    m.x = pyo.Var(range(n), within=pyo.Reals)  # y is between 0 and 1

    # Define fake constraints for the parameter
    for i in range(p):
        for j in range(n):
            m.G[i, j].fix(nominal_G[i, j])
            
    for i in range(p):
        m.h[i].fix(nominal_h[i])
    
    for i in range(n):
        m.p[i].fix(nominal_p[i])
        
    # Define objective
    def objective_rule(m):
        return 0.5 * sum((m.x[i] - m.p[i])**2 for i in range(n))
        
    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Define inequality constraints
    m.inequ_constraints = pyo.ConstraintList()
    for i in range(p):
        m.inequ_constraints.add(sum(m.G[i, j] * m.x[j] for j in range(n)) <= m.h[i])
    return m

def model_instance():
    nominal_G = np.random.rand(p, n)
    nominal_h = np.random.rand(p)
    nominal_p = np.random.rand(n)
    model = create_model(nominal_G, nominal_h, nominal_p)
    return model
# %% Pyomo model   
# n, p = 2, 20

# nominal_G = np.random.rand(p, n)
# nominal_h = np.random.rand(p)
# nominal_p = np.random.rand(n)

# model = create_model(nominal_G, nominal_h, nominal_p)
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
#model.pprint()
# print(model.obj())
# %%
class PolytopeProjection(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.G = nn.Parameter(torch.randn(m, n))
        self.h = nn.Parameter(torch.ones(m))

        self.layer = Pyomolayer
    
    def forward(self, x):
        return self.layer(self.G.expand(x.shape[0], *self.G.shape), self.h.expand(x.shape[0], *self.h.shape), x)

def plot_polytope(G, h, color, label):
    hs = HalfspaceIntersection(np.hstack((G.detach(),  -h[:,None].detach())), np.array([0.,0.]))
    pts = hs.intersections - hs.interior_point
    pts = pts[np.argsort(np.arctan2(pts[:,1],pts[:,0])),:] + hs.interior_point
    plt.fill(pts[:,0], pts[:,1], color=color, alpha=0.5, label = label)

def main():
    global n, p, Pyomolayer
    n, p = 2, 20
    batch = 100
    eporch_num = 40
    lr = 1e-1
    lr_schedule, thre_epoch = 1e-2, int(eporch_num/2)
    
    model = model_instance()
    variables_name = [model.x]
    parameters_name = [model.G, model.h, model.p]
    
    Pyomolayer = PyomoOptLayer(model, variables_name, parameters_name, solver = 'ipopt')
    X = torch.randn(batch, n)
    Y = X / X.norm(dim=1).clamp(min=1)[:,None]
    plt.figure(dpi=150)
    plt.plot(Y[:,0], Y[:,1], 'bx', label = "Initial ramdom data")

    torch.manual_seed(0)
    layer = PolytopeProjection(n, p)
    plot_polytope(layer.G, layer.h, "orange", "NN initial poly")

    opt = optim.Adam(layer.parameters(), lr=lr)
    for i in range(eporch_num):
        if i == thre_epoch:
            opt.param_groups[0]["lr"] = lr_schedule
        primal, _, _, _ = layer(X)
        loss = nn.MSELoss()(primal, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

    plot_polytope(layer.G, layer.h, "red", "NN optimal poly")
    plt.axis('equal')
    plt.savefig(os.path.join(results_dir, "General_Polytope_Projections.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()