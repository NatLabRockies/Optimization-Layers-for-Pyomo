# %% [markdown]
# # Convex approximate dynamic programming
# 
# We consider a stochastic control problem of the form
# \begin{equation}
# \begin{array}{ll}
# \mbox{minimize} & \underset{T \to \infty}\lim {\mathbb E} \left[\frac{1}{T} \sum_{t=0}^{T-1} \|{x_t}\|_2^2 + \|{\phi(x_t)}\|_2^2\right]\\[.2cm]
# \mbox{subject to} & x_{t+1} = Ax_t + B\phi(x_t) + \omega_t,
# \end{array}
# \label{eq:adp}
# \end{equation}
# where $x_t\in\mathbf{R}^n$ is the state, $\phi:\mathbf{R}^n \to \mathcal U \subseteq \mathbf{R}^m$ is
# the policy, $\mathcal U$ is a convex set representing the allowed set of controls,
# and $\omega_t\in\Omega$ is a (random, i.i.d.) disturbance.
# Here the variable is the policy $\phi$, and the expectation is taken over
# disturbances and the initial state $x_0$. If $\mathcal U$ is not an affine
# set, then this problem is in general very difficult to solve.
# 
# A common heuristic for solving stochastic control problems is
# approximate dynamic programming (ADP), which parametrizes $\phi$
# and replaces the minimization over functions $\phi$ with a minimization over parameters.
# In this example, we take $\mathcal U$ to be the unit ball and we represent $\phi$
# as a particular quadratic *control-Lyapunov* policy.
# Evaluating $\phi$ corresponds to solving the SOCP
# \begin{equation}
# \begin{array}{ll}
# \mbox{minimize} & u^T P u + x_t^T Q u + q^T u \\
# \mbox{subject to} & \|{u}\|_2 \leq 1,
# \end{array}
# \label{eq:policy}
# \end{equation}
# with variable $u$ and parameters $P$, $Q$, $q$, and $x_t$. We can run
# gradient descent (SGD) on $P$, $Q$, and $q$ to
# approximately solve the original problem, which requires requires differentiating
# through the quadratic policy. Note that if $u$ were unconstrained, the original problem
# could be solved exactly, via linear quadratic regulator (LQR) theory.

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
from algorithms import fit
from scipy.linalg import solve_discrete_are
from scipy.linalg import sqrtm

# %%
def create_model(nominal_q, nominal_P21, nominal_Psqrt, nominal_x):
    # Create a concrete model
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    model.u = pyo.Var(range(m), range(1), within=pyo.Reals)
    # Define parameters
    model.q = pyo.Var(range(m), range(1), within=pyo.Reals)
    model.P21 = pyo.Var(range(n), range(m), within=pyo.Reals)
    model.Psqrt = pyo.Var(range(m), range(m), within=pyo.Reals)
    model.x = pyo.Var(range(n), range(1), within=pyo.Reals)
    # Define variables
    model.y = pyo.Var(range(n), range(1), within=pyo.Reals)

    for i in range(m):
        model.q[i, 0].fix(nominal_q[i, 0])
    for i in range(n):
        for j in range(m):
            model.P21[i, j].fix(nominal_P21[i, j])
    for i in range(m):
        for j in range(m):
            model.Psqrt[i, j].fix(nominal_Psqrt[i, j])
    for i in range(n):
        model.x[i, 0].fix(nominal_x[i, 0])

    model.equ_constraints = pyo.ConstraintList()
    model.equ_constraints.add(sum(model.u[i, 0]**2 for i in range(m)) <= 1)
    for i in range(n):
        model.equ_constraints.add(model.y[i, 0] == sum(model.P21[i, j] * model.u[j, 0] for j in range(m)))
    # Define objective function: 
    # Objective function
    def objective_rule(model):
        # Quadratic term: 0.5 * ||P_sqrt @ u||^2
        quad_term = 0.5 * sum(sum(model.Psqrt[i, j] * model.u[j, 0] for j in range(m)) ** 2 for i in range(m))
        # Linear term: x.T @ y
        linear_term1 = sum(model.x[i, 0] * model.y[i, 0] for i in range(n))
        # Linear term: q.T @ u
        linear_term2 = sum(model.q[i, 0] * model.u[i, 0] for i in range(m))
    
        return quad_term + linear_term1 + linear_term2

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize) 
    return model

def model_instance():
    # pyomo model
    nominal_q, nominal_P21, nominal_Psqrt, nominal_x = np.random.rand(m, 1),  np.random.rand(n, m), np.random.rand(m, m), np.random.rand(n, 1)
    model = create_model(nominal_q, nominal_P21, nominal_Psqrt, nominal_x)
    return model
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())
# %%
def main():
    def g(x, u):
        return (x.t() @ Q_tch @ x + u.t() @ R_tch @ u).squeeze()

    def evaluate(x0, P_sqrt, P_21, q, T):
        x = x0
        cost = 0.
        for _ in range(T):
            input = tuple([q.unsqueeze(0), P_21.unsqueeze(0), P_sqrt.unsqueeze(0), x.unsqueeze(0)])
            primal, _, _, _= Pyomolayer(*input)
            u = primal.T
            cost += g(x, u) / T
            x = A_tch @ x + B_tch @ u + .2 * torch.randn(n, 1).double()
        return cost

    def eval_loss(N=8, T=25):
        return sum([evaluate(torch.zeros(n, 1).double(), P_sqrt, P_21, q, T=T) for _ in range(N)]) / N

    # Generate data
    torch.manual_seed(1)
    np.random.seed(1)
    global m, n
    n = 2
    m = 3
    iters=100

    A = np.eye(n) + 1e-2 * np.random.randn(n, n)
    B = 1e-2 / 3 * np.random.randn(n, m)
    Q = np.eye(n)
    R = np.eye(m)

    # Compute LQR control policy
    P_lqr = solve_discrete_are(A, B, Q, R)
    P = R + B.T@P_lqr@B
    P_sqrt_lqr = sqrtm(P)

    # Initialize with LQR control lyapunov function
    P_sqrt = torch.from_numpy(P_sqrt_lqr).requires_grad_(True)
    P_21 = torch.from_numpy(A.T @ P_lqr @ B).requires_grad_(True)
    q = torch.zeros((m, 1), dtype=torch.double, requires_grad=True)
    variables = [P_sqrt, P_21, q]
    A_tch, B_tch, Q_tch, R_tch = map(torch.from_numpy, [A, B, Q, R])

    variables_name = ["u"]
    # variables_size = {'u':[m, 1], "y":[n, 1]}
    parameters_name = ["q", "P21", "Psqrt", "x"]
    # parameters_size = {'q':[m, 1], "P21" : [n, m], "Psqrt" : [m, m], "x": [n, 1]}
    model = model_instance()
    Pyomolayer = PyomoOptLayer(model, variables_name, parameters_name, solver = 'ipopt')

    results = []
    optimizer = torch.optim.SGD(variables, lr=.02, momentum=.9)
    for i in range(iters):
        # use same seeds each iteration to get pretty training plot
        torch.manual_seed(1)
        np.random.seed(1)
        optimizer.zero_grad()
        loss = eval_loss()
        loss.backward()
        optimizer.step()
        results.append(loss.item())
        print("(iter %d) loss: %g " % (i, results[-1]))

    np.save(os.path.join(results_dir, "cadp_results"), results)
    cadp_cvx_results = np.load(os.path.join(results_dir, "cadp_cvx_results.npy"))
    plt.plot(results, label = "Pyomo")
    plt.plot(cadp_cvx_results, label = "CVX")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "loss_CADP.png"), dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    main()




