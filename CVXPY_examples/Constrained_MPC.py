# %% [markdown]
# # Constrained MPC
# 
# This notebook accompanies the paper [Learning Convex Optimization Models](https://web.stanford.edu/~boyd/papers/learning_copt_models.html).

# %%
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir) # Add the parent directory to Python's search path
results_dir = os.path.join(current_dir, "examples/results")

import pyomo.environ as pyo
import numpy as np
from Opt_Layer.pyomolayer import PyomoOptLayer
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt

# %%
def create_model(nominal_x):
    # Create a concrete model
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    model.states = pyo.Var(range(T), range(n), domain=pyo.Reals)

    # Define parameters
    model.x = pyo.Var(range(n), within=pyo.Reals)

    # Define variables
    model.controls = pyo.Var(range(T), range(m), within=pyo.Reals)

    for i in range(n):
        model.x[i].fix(nominal_x[i])

    model.equ_constraints = pyo.ConstraintList()
    for i in range(n):
        model.equ_constraints.add(model.x[i] == model.states[0, i])

    # Separate constraints for upper and lower bounds of controls
    def upper_bound_rule(model, t):
        return sum(model.controls[t, j]**2 for j in range(m)) <= beta

    model.norm_upper = pyo.Constraint(range(T), rule=upper_bound_rule)

    # Constraint: states[t] == A_np @ states[t-1] + B_np @ controls[t-1]
    def state_transition_rule(model, t, i):
        return model.states[t, i] == sum(A_np[i, j] * model.states[t-1, j] for j in range(n)) + sum(B_np[i, k] * model.controls[t-1, k] for k in range(m))

    model.state_transition = pyo.Constraint(range(1, T), range(n), rule=state_transition_rule)

    # Define the objective function
    def objective_rule(model):
        obj = 0
        for t in range(T):
            obj += sum(weights_np[i] * model.states[t, i]**2 for i in range(n)) + sum(model.controls[t, i]**2 for i in range(m))
        return obj
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return model

#Pyomo model
def model_instance():
    np.random.seed(1)
    nominal_x = np.random.rand(n)
    model = create_model(nominal_x)
    return model
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())
def create_model_mpc(nominal_x, nominal_weights):
    # Create a concrete model
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    model.controls = pyo.Var(range(T), range(m), within=pyo.Reals)
    model.states = pyo.Var(range(T), range(n), domain=pyo.Reals)
    # Define parameters
    model.weights = pyo.Var(range(n), within=pyo.NonNegativeReals)
    model.x = pyo.Var(range(n), within=pyo.Reals)

    for i in range(n):
        model.x[i].fix(nominal_x[i])

    for i in range(n):
        model.weights[i].fix(nominal_weights[i])

    model.equ_constraints = pyo.ConstraintList()
    for i in range(n):
        model.equ_constraints.add(model.x[i] == model.states[0, i])

    # Separate constraints for upper and lower bounds of controls
    def upper_bound_rule(model, t):
        return sum(model.controls[t, j]**2 for j in range(m)) <= beta

    # def lower_bound_rule(model, t, i):
    #     return model.controls[t, i] >= -beta

    model.norm_upper = pyo.Constraint(range(T), rule=upper_bound_rule)
    # model.norm_lower = pyo.Constraint(range(T), range(m), rule=lower_bound_rule)
    # Constraint: states[t] == A_np @ states[t-1] + B_np @ controls[t-1]
    def state_transition_rule(model, t, i):
        return model.states[t, i] == sum(A_np[i, j] * model.states[t-1, j] for j in range(n)) + sum(B_np[i, k] * model.controls[t-1, k] for k in range(m))

    model.state_transition = pyo.Constraint(range(1, T), range(n), rule=state_transition_rule)

    # Define the objective function
    def objective_rule(model):
        obj = 0
        for t in range(T):
            obj += sum(model.weights[i] * model.states[t, i]**2 for i in range(n)) + sum(model.controls[t, i]**2 for i in range(m))
        return obj
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return model

def model_instance_mpc():
    np.random.seed(1)
    nominal_x, nominal_weights = np.random.rand(n), np.abs(np.random.rand(n))
    model = create_model_mpc(nominal_x, nominal_weights)
    return model
#Pyomo model
# np.random.seed(1)
# n = 10
# nominal_x, nominal_weights = np.random.rand(n), np.abs(np.random.rand(n))
# model = create_model_mpc(nominal_x, nominal_weights)
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())
class FF(torch.nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=n, out_features=n)
        self.fc2 = torch.nn.Linear(in_features=n, out_features=m)
        
    def forward(self, x):
        h1 = self.fc1(x).relu()
        return self.fc2(h1).clamp(-beta, beta)
    
def dynamics(xt, ut):
    return A @ xt + B @ ut + torch.randn(xt.shape)

def stage_cost(xt, ut):
    cost = 0.    
    return (weights*(xt.pow(2))).sum() + ut.pow(2).sum()

def simulate(policy, n_iters=1000, seed=0, weights_tch = None, NN = None):
    torch.random.manual_seed(seed)
    x0 = torch.randn(n)
    states = [x0]
    controls = []
    costs = []
    for t in range(n_iters):
        xt = states[-1]
        if weights_tch:
            primal, _, _ = policy(xt.unsqueeze(0), weights_tch)
            ut = primal[:, :m].squeeze(0) 
        elif NN:
            ut = policy(xt)[0] 
        else:
            primal, _, _  = policy(xt.unsqueeze(0))
            ut = primal[:, :m].squeeze(0)                                                    
        controls.append(ut)
        costs.append(stage_cost(xt.squeeze(0), ut).item())
        states.append(dynamics(xt.squeeze(0), ut))
    return states[:-1], controls, costs

def simulate_adp(policy, weights_tch, n_iters=1000, seed=0):
    torch.random.manual_seed(seed)
    x0 = torch.randn(n)
    states = [x0]
    controls = []
    costs = []
    for t in range(n_iters):
        xt = states[-1]
        primal, _, _ = policy(xt.unsqueeze(0), weights_tch)
        ut = primal[:, :m].squeeze(0)                                                    
        controls.append(ut)
        costs.append(stage_cost(xt.squeeze(0), ut).item())
        states.append(dynamics(xt.squeeze(0), ut))
    return states[:-1], controls, costs

def simulate_NN(policy, n_iters=1000, seed=0):
    torch.random.manual_seed(seed)
    x0 = torch.randn(n)
    states = [x0]
    controls = []
    costs = []
    for t in range(n_iters):
        xt = states[-1]
        ut = policy(xt)[0] 
        controls.append(ut)
        costs.append(stage_cost(xt, ut).item())
        states.append(dynamics(xt, ut))
    return states[:-1], controls, costs

def mse(preds, actual):
    preds = torch.stack(preds, dim=0)
    actual = torch.stack(actual, dim=0)
    return (preds - actual).pow(2).mean(axis=1).mean(axis=0).item()

def plot_weights(weights_tch, weights_np):
    fig = plt.figure()
    fig.set_size_inches((10, 3))
    plt.plot(weights_tch.squeeze(0).detach().numpy(), linestyle='-', color='k', label='learned')
    plt.plot(weights_np, linestyle='--', color='k', label='true')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\theta_i$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "weight_CMPC.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_losses(val_nn_losses, val_losses, true_mse):
    fig = plt.figure()
    fig.set_size_inches((10, 3))
    plt.axhline(val_nn_losses[-1], color='k', linestyle='-.', label='NN')    
    plt.plot(np.arange(len(val_losses)) + 1, val_losses, color='k', label='Pyomo')
    plt.axhline(true_mse, color='k', linestyle='--', label='true')
    plt.xticks([1, 5, 10, 15, 20])
    plt.xlabel('iteration')
    plt.ylabel('validation loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_CMPC.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    global A, B, A_np, B_np, beta, m, n, T, weights, weights_np
    n = 10
    m = 4
    beta = 0.5
    T = 5
    scale = 0.31622776601
    epoch_num = 20
    learning_rate = 3e-4
    val_seed = 243

    np.random.seed(0)
    torch.manual_seed(0)
    A_np = np.random.randn(n, n)
    A_np /= np.max(np.abs(np.linalg.eig(A_np)[0]))
    B_np = np.random.randn(n, m)

    A = torch.tensor(A_np)
    B = torch.tensor(B_np)
    weights = torch.randn(n).abs()
    weights_np = weights.numpy()
    weights_tch = torch.ones(1, n, requires_grad=True)

    opt = torch.optim.Adam([weights_tch], lr=learning_rate)

    model = model_instance()
    variables = [model.controls]
    parameters = [model.x]
    
    Pyomolayer = PyomoOptLayer(model, variables, parameters)

    states, controls, costs = simulate(Pyomolayer)
    
    torch.manual_seed(1)
    for control in controls:
        control.add_(scale*torch.randn(control.shape))
        control.clamp_(-beta, beta)

    _, val_mpc_controls, val_mpc_costs = simulate(Pyomolayer, seed=val_seed)
    for control in val_mpc_controls:
        control.add_(scale*torch.randn(control.shape))
        control.clamp_(-beta, beta)

    _, true_copt_controls, _ = simulate(Pyomolayer, seed=val_seed)
    true_mse = mse(true_copt_controls, val_mpc_controls)
    print("True MSE", true_mse)

    model_mpc = model_instance_mpc()
    variables = [model_mpc.controls]
    parameters = [model_mpc.x, model_mpc.weights]
    
    Pyomolayer_adp = PyomoOptLayer(model_mpc, variables, parameters, solver = 'ipopt')
    
    val_losses = []
    losses = []
    with torch.no_grad():
        _, initial_preds, _ = simulate_adp(Pyomolayer_adp, weights_tch, seed=val_seed)
        val_losses.append(mse(initial_preds, val_mpc_controls))
        print(val_losses[-1])
    
    for epoch in range(epoch_num):
        print('Epoch: ', epoch)
        for xt, ut in zip(states, controls):
            opt.zero_grad()
            primal, _, _ = Pyomolayer_adp(xt.unsqueeze(0), weights_tch)
            ut_hat = primal[:, :m].squeeze(0) 
            loss = (ut - ut_hat).pow(2).mean()
            loss.backward()
            losses.append(loss.item())
            opt.step()
        with torch.no_grad():
            weights_tch.data = weights_tch.relu()
            _, pred_ctrls, _ = simulate_adp(Pyomolayer_adp, weights_tch, seed=val_seed)
            val_losses.append(mse(pred_ctrls, val_mpc_controls))
        print("Validation loss", val_losses[-1])

    plot_weights(weights_tch, weights_np)
    plt.show()

    torch.random.manual_seed(0)
    ff = FF()
    epochs = 100
    nn_losses = []
    val_nn_losses = []
    opt = torch.optim.Adam(ff.parameters(), lr=3e-4)
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for xt, ut in zip(states, controls):
            opt.zero_grad()
            ut_hat = ff(xt)
            loss = (ut - ut_hat).pow(2).mean()
            loss.backward()
            nn_losses.append(loss.item())
            opt.step()

        _, val_preds, _ = simulate_NN(lambda x: [ff(x)], seed=val_seed)
        with torch.no_grad():
            val_nn_losses.append(mse(val_preds, val_mpc_controls))
        print("Validation loss for NN", val_nn_losses[-1])

    plot_losses(val_nn_losses, val_losses, true_mse)

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 3)

    ax[1].plot(weights_tch.squeeze(0).detach().numpy(), linestyle='-', label='learned')
    ax[1].plot(weights_np, linestyle='--', label='true')
    ax[1].set_xlabel(r'$i$')
    ax[1].set_ylabel(r'$\theta_i$')
    ax[1].set_xticks([0, 2,4, 6, 8])
    ax[1].legend()
        
    ax[0].axhline(val_nn_losses[-1], linestyle='-.', label='NN')
    ax[0].plot(np.arange(len(val_losses)) + 1, val_losses, label='COM')
    ax[0].axhline(true_mse, linestyle='--', label='true')
    ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('validation loss')
    ax[0].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(results_dir, "CMPC.png"), dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    main()
