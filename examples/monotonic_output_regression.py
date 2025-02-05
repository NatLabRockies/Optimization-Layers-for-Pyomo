# %% [markdown]
# # Monotonic Output Regression
# 
# This notebook accompanies the paper [Learning Convex Optimization Models](https://web.stanford.edu/~boyd/papers/learning_copt_models.html).

# %%
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir) # Add the parent directory to Python's search path
results_dir = os.path.join(current_dir, "examples/results_12172024")

import pyomo.environ as pyo
import numpy as np
from Opt_Layer.pyomolayer import PyomoOptLayer
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from Opt_Layer.algorithms import fit
# %%
def create_model(nominal_y):
    # Create a concrete model
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    m.x = pyo.Var(range(n), within=pyo.Reals)
    # Define parameters
    m.y = pyo.Var(range(n), within=pyo.Reals)

    for i in range(n):
        m.y[i].fix(nominal_y[i])
    
    # Add constraints directly to enforce non-decreasing condition
    def non_decreasing_constraint(m, i):
        if i < n - 1:  # Ensure we only add the constraint for consecutive elements
            return m.x[i+1] - m.x[i] >= 0
        return pyo.Constraint.Skip  # Skip for the last element to avoid index error
    m.non_decreasing_constraint = pyo.Constraint(range(n-1), rule=non_decreasing_constraint)

    # Define objective function: 
    def objective_rule(m):
        tol_term = 0
        for i in range(n):
            q_term = (m.y[i] - m.x[i])**2
            tol_term += q_term
        #Different from cvxpy here, remove the sqrt  
        return tol_term
        
    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return m

def model_instance():
    nominal_y = np.random.rand(n) 
    model = create_model(nominal_y)
    return model
# %% Test the Pyomo model
# n = 10
# nominal_y = np.random.rand(n) 
# model = create_model(nominal_y)
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())
# %% Get data
def get_data(N, n1, n, theta):
    torch.manual_seed(0)
    X = torch.randn(N, n1)
    input = tuple([X @ theta + torch.randn(N, n)])
    primal, _, _, _ = Pyomolayer(*input)
    return X, primal

def loss(X, Y, theta, mse):
    input = tuple([X @ theta])
    primal, _, _, _ = Pyomolayer(*input)
    return mse(primal, Y)
# %% Main function  
def main():
    global n, Pyomolayer
    n = 10
    n1 = 20
    sample_num = 100
    valid_sample_num = 50
    mse_loss = torch.nn.MSELoss()

    model = model_instance()
    variables = [model.x]
    parameters = [model.y]
    Pyomolayer = PyomoOptLayer(model, variables, parameters, solver = 'ipopt')

    torch.manual_seed(0)
    theta_true = torch.randn(n1, n)
    X, Y = get_data(sample_num, n1, n, theta_true)
    Xval, Yval = get_data(valid_sample_num, n1, n, theta_true)

    theta_lstsq = torch.linalg.solve(X.t() @ X, X.t() @ Y)
    lstsq_val_loss = mse_loss(Xval @ theta_lstsq, Yval).item()
    theta = torch.zeros_like(theta_lstsq) 
    theta.requires_grad_(True)

    input = tuple([Xval @ theta_true])
    primal, _, _, _ = Pyomolayer(*input)
    bayes_val_loss = mse_loss(primal, Yval).item()

    val_losses, train_losses = fit(lambda X, Y: loss(X, Y, theta, mse_loss), [theta], X, Y, Xval, Yval,
                               opt=torch.optim.Adam, opt_kwargs={"lr": 1e-1},
                               batch_size=16, epochs=50, verbose=True)
    
    np.save(os.path.join(results_dir, "val_losses"), val_losses)
    _, ax = plt.subplots(1, 2)

    ax[0].axhline(lstsq_val_loss, linestyle='-.', c='red', label='LR')
    ax[0].plot(val_losses, c='k', label='Pyomo')
    ax[0].axhline(bayes_val_loss, linestyle='--', c='blue', label='true')
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("validation loss")
    ax[0].set_ylim(0)
    ax[0].legend()

    ax[1].plot(Xval[13] @ theta_lstsq, '-.', c='red', label='LR')
    input = tuple([(Xval[13] @ theta).unsqueeze(0)])
    output, _, _, _ = Pyomolayer(*input)
    ax[1].plot(output[0].detach().numpy(), c='k', label='Pyomo')
    ax[1].plot(Yval[13].numpy(), '--', c='blue', label='true')
    ax[1].legend()
    ax[1].set_xlabel('$i$')
    ax[1].set_ylabel("$\phi(x;\\theta)$")

    plt.savefig(os.path.join(results_dir, "validation_loss_MOR.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()