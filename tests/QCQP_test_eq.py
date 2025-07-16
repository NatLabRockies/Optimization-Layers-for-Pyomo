# %% [markdown]
# # QP
# \begin{align}
# \text{min}_\mathbf{x} & \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{q}^\top\mathbf{x}, 
# \\
# \\
# \text{s.t.} \quad & \mathbf{A} \mathbf{x} = \mathbf{b} \\
# & \mathbf{G}\mathbf{x} \leq \mathbf{h}
# \end{align}
# with the variable $\mathbf{x}$ and parameter $\mathbf{p}$.
# %%
import sys
sys.path.append('/projects/drl4dsr/kchen2/PyomoLayer')  # Add the parent directory to Python's search path
import pyomo.environ as pyo
import numpy as np
from pyomolayer import PyomoOptLayer
import torch
import time
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
# %%
def create_model(nominal_Psqrt, nominal_q, nominal_A, nominal_b, nominal_d, nominal_F, nominal_g):
    # Create the model
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    n = nominal_Psqrt.shape[0]
    m = nominal_A.shape[0]
    p = nominal_F.shape[0]

    # Define the decision variables
    model.x = pyo.Var(range(n), domain=pyo.Reals, bounds = (-1, 1))
    # Define parameters
    model.Psqrt = pyo.Var(range(n), range(n), within=pyo.Reals)
    model.q = pyo.Var(range(n), within=pyo.Reals) # Linear term vector
    model.A = pyo.Var(range(m), range(n), range(n), within=pyo.Reals)
    model.b = pyo.Var(range(m), range(n), within=pyo.Reals)
    model.d = pyo.Var(range(m), within=pyo.Reals)
    model.F = pyo.Var(range(p), range(n), within=pyo.Reals)
    model.g = pyo.Var(range(p), within=pyo.Reals)

    for i in range(n):
        for j in range(n):
            model.Psqrt[i, j].fix(nominal_Psqrt[i, j])

    for i in range(n):
        model.q[i].fix(nominal_q[i])

    for i in range(m):
        for j in range(n):
            for k in range(n):
                model.A[i, j, k].fix(nominal_A[i, j, k])

    for i in range(m):
        for j in range(n):
            model.b[i, j].fix(nominal_b[i, j])

    for i in range(m):
        model.d[i].fix(nominal_d[i])

    for i in range(p):
        for j in range(n):
            model.F[i, j].fix(nominal_F[i, j])

    for i in range(p):
        model.g[i].fix(nominal_g[i])
        
    # Define objective function: 
    def objective_rule(model):
        tol_term = 0
        for i in range(n):
            q_term = 0
            for j in range(n):
                q_term += model.Psqrt[i, j] * model.x[j]
            tol_term += q_term**2
            
        return 0.5 * tol_term + sum(model.q[i] * model.x[i] for i in range(n))

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Linear equality constraints: F @ x == g
    model.equ_constraints = pyo.ConstraintList()
    for i in range(p):
        model.equ_constraints.add(sum(model.F[i, j] * model.x[j] for j in range(n)) == model.g[i])

    # QC constraints
    def quadratic_constraint_rule(model, i):
        tol_term = 0
        for j in range(n):
            q_term = 0
            for k in range(n):
                q_term += model.A[i, j, k] * model.x[k]
            tol_term += q_term**2

        return (None, 0.5 * tol_term + sum(model.b[i, j] * model.x[j] for j in range(n)) + model.d[i], 1)
        
    model.inequ_constraints = pyo.ConstraintList()
    for i in range(m):
        model.inequ_constraints.add(quadratic_constraint_rule(model, i))
        
    return model

def model_instance(m, n, p):
    nominal_Psqrt = np.random.rand(n, n)
    nominal_q = np.random.rand(n)

    nominal_A = np.random.randn(m, n, n)
    nominal_b = np.random.randn(m, n)
    nominal_d = np.random.randn(m)

    nominal_F = np.random.randn(p, n)
    nominal_g = np.random.randn(p)

    model = create_model(nominal_Psqrt, nominal_q, nominal_A, nominal_b, nominal_d, nominal_F, nominal_g)
    return model
# %% Gradient obtained by Pyomo
def QP_grad_pyomo(n, p, sample_num, batch_size, alg = "pyomo", val_seed=0):
    m = 1
    if alg == "pyomo":
        variables_name = ["x"]
        variables_size = {'x':[n]}
        parameters_name = ["Psqrt", "q", "A", "b", "d", "F", "g"]
        parameters_size = {'Psqrt':[n, n], 'q':[n], "A" : [m, n, n], "b" : [m, n], 'd':[m], "F": [p, n], "g": [p]}
        model = model_instance(m, n, p)
        Layer = PyomoOptLayer(model, variables_name, parameters_name, solver = 'ipopt')
    elif alg == "cvxpy":
        x = cp.Variable(n)
        Q_sqrt = cp.Parameter((n, n))
        q = cp.Parameter(n)
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        d = cp.Parameter(1)
        F = cp.Parameter((p, n))
        g = cp.Parameter(p)

        obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt @ x) + q @ x)
        cons = [F @ x == g, 0.5*cp.sum_squares(A @ x) + b @ x + d - 1 <= 0, x - 1 <= 0, -1 - x <= 0]
        prob = cp.Problem(obj, cons)
        Layer = CvxpyLayer(prob, parameters=[Q_sqrt, q, A, b, d, F, g], variables=[x])

    torch.manual_seed(val_seed)
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64)
    qval = torch.randn(sample_num, n, dtype=torch.float64)
    Aval = torch.randn(sample_num, m, n, n, dtype=torch.float64)
    bval = torch.randn(sample_num, m, n, dtype=torch.float64)
    dval = torch.randn(sample_num, m, dtype=torch.float64)
    if alg == "cvxpy":
        Aval = Aval[:, 0]
        bval = bval[:, 0]

    Fval = torch.randn(sample_num, p, n, dtype=torch.float64)
    gval= torch.randn(sample_num, p, dtype=torch.float64)

    primal, duals, Psqrtgrad, qgrad, Agrad, bgrad, dgrad, Fgrad, ggrad = [], [], [], [], [], [], [], [], []
    dataset = torch.utils.data.TensorDataset(Psqrt, qval, Aval, bval, dval, Fval, gval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()
    for batch_idx, (P_batch, q_batch, A_batch, b_batch, d_batch, F_batch, g_batch) in enumerate(dataloader):
        P_batch = P_batch.detach().clone().requires_grad_(True)
        q_batch = q_batch.detach().clone().requires_grad_(True)
        A_batch = A_batch.detach().clone().requires_grad_(True)
        b_batch = b_batch.detach().clone().requires_grad_(True)
        d_batch = d_batch.detach().clone().requires_grad_(True)
        F_batch = F_batch.detach().clone().requires_grad_(True)
        g_batch = g_batch.detach().clone().requires_grad_(True)
        input = tuple([P_batch, q_batch, A_batch, b_batch, d_batch, F_batch, g_batch])
        if alg == "pyomo":
            primal_batch, dual_batch, _, _ = Layer(*input)
            duals.append(dual_batch.detach().numpy())
        elif alg == "cvxpy":
            primal_batch, = Layer(*input)
        primal_batch.sum().backward()
        primal.append(primal_batch.detach().numpy())
        Psqrtgrad.append(P_batch.grad.detach().numpy())
        qgrad.append(q_batch.grad.detach().numpy())
        Agrad.append(A_batch.grad.detach().numpy())
        bgrad.append(b_batch.grad.detach().numpy())
        dgrad.append(d_batch.grad.detach().numpy())
        Fgrad.append(F_batch.grad.detach().numpy())
        ggrad.append(g_batch.grad.detach().numpy())
    primal = np.concatenate(primal, axis=0)
    if alg == "pyomo":
        duals = np.concatenate(duals, axis=0)
    Psqrtgrad = np.concatenate(Psqrtgrad, axis=0)
    qgrad = np.concatenate(qgrad, axis=0)
    Agrad = np.concatenate(Agrad, axis=0)
    bgrad = np.concatenate(bgrad, axis=0)
    dgrad = np.concatenate(dgrad, axis=0)
    Fgrad = np.concatenate(Fgrad, axis=0)
    ggrad = np.concatenate(ggrad, axis=0)
    # input = tuple([Psqrt, qval, Aval, bval, Gval, hval])
    # primal, _, _, _ = Pyomolayer(*input)
    # primal.sum().backward()
    end = time.time()
    pyomo_time = end - start
    return Psqrtgrad, qgrad, Agrad, bgrad, dgrad, Fgrad, ggrad, pyomo_time, primal, duals

def get_duals(n, p, sample_num, val_seed=0):
    x = cp.Variable(n)
    Q_sqrt = cp.Parameter((n, n))
    q = cp.Parameter(n)
    A = cp.Parameter((n, n))
    b = cp.Parameter(n)
    d = cp.Parameter(1)
    F = cp.Parameter((p, n))
    g = cp.Parameter(p)

    obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt @ x) + q @ x)
    cons = [F @ x == g, 0.5*cp.sum_squares(A @ x) + b @ x + d - 1 <= 0, x - 1 <= 0, -1 - x <= 0]
    prob = cp.Problem(obj, cons)

    torch.manual_seed(val_seed)
    m = 1
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64).numpy()
    qval = torch.randn(sample_num, n, dtype=torch.float64).numpy()
    Aval = torch.randn(sample_num, m, n, n, dtype=torch.float64).numpy()
    bval = torch.randn(sample_num, m, n, dtype=torch.float64).numpy()
    dval = torch.randn(sample_num, m, dtype=torch.float64).numpy()
    Aval = Aval[:, 0]
    bval = bval[:, 0]
    Fval = torch.randn(sample_num, p, n, dtype=torch.float64).numpy()
    gval= torch.randn(sample_num, p, dtype=torch.float64).numpy()
    duals = []
    for i in range(sample_num):
        Q_sqrt.value, q.value, A.value, b.value, d.value, F.value, g.value = Psqrt[i], qval[i], Aval[i], bval[i], dval[i], Fval[i], gval[i]
        prob.solve()
        dual = np.concatenate((cons[0].dual_value, cons[1].dual_value))
        duals.append(dual)
    return np.array(duals)


def calculate_error(actual, predicted):
    """
    Calculate the absolute relative error.

    Args:
        actual (float): The true or actual value.
        predicted (float): The predicted or estimated value.

    Returns:
        float: The absolute error
    """

    #calculate the total absolute error
    if actual.ndim > 3:
        tol_diff = np.sum(np.abs(actual - predicted), axis=(1, 2, 3))
    elif actual.ndim > 2:
        tol_diff = np.sum(np.abs(actual - predicted), axis=(1, 2))
    else:
        tol_diff = np.sum(np.abs(actual - predicted), axis=1)
    return tol_diff

def grad_diff(n: int = 1, p: int = 1, sample_num: int = 32, batch_size: int = 32, alg: str = "pyomo", val_seed: int = 0):
    PsqrtG, qvalG, AvalG, bvalG, dvalG, FvalG, gvalG, pyomo_time, primal, duals = QP_grad_pyomo(n, p, sample_num, batch_size, alg = "pyomo", val_seed=val_seed)
    PsqrtG_ref, qvalG_ref, AvalG_ref, bvalG_ref, dvalG_ref, FvalG_ref, gvalG_ref, cvxpy_time, primal_ref, _ = QP_grad_pyomo(n, p, sample_num, batch_size, alg = "cvxpy", val_seed=val_seed)
    duals_ref = get_duals(n, p, sample_num, val_seed)
    inputs = [(primal, primal_ref), (duals, duals_ref), (PsqrtG, PsqrtG_ref), (qvalG, qvalG_ref), (AvalG, AvalG_ref), (bvalG, bvalG_ref), (dvalG, dvalG_ref), (FvalG, FvalG_ref), (gvalG, gvalG_ref)]
    diffs = [calculate_error(P, C) for P, C in inputs]
    Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, dvalG_diff, FvalG_diff, gvalG_diff =  diffs

    return Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, dvalG_diff, FvalG_diff, gvalG_diff

def test_grad():
    primal_diff_threshold = 1e-2
    grad_diff_threshold = 1e-2
    Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, dvalG_diff, FvalG_diff, gvalG_diff = grad_diff(n = 10, p = 1, sample_num = 100, batch_size= 20, alg = "pyomo", val_seed = 0)
    success_rate_primal = np.mean(Primal_diff < primal_diff_threshold)
    success_rate_dual = np.mean(dual_diff < primal_diff_threshold)
    G_diff = np.stack((PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, dvalG_diff, FvalG_diff, gvalG_diff), axis=1)
    success_rate_grad = np.mean(np.all(G_diff < grad_diff_threshold, axis=1))
    dual_grad_correct = np.sum((dual_diff < primal_diff_threshold) & np.all(G_diff < grad_diff_threshold, axis=1)) / np.sum(dual_diff < primal_diff_threshold) 
    print("!!!success_rate_primal!!!", success_rate_primal)
    print("!!!success_rate_dual!!!", success_rate_dual)
    print("!!!success_rate_grad!!!", success_rate_grad)
    print("!!!success_rate_dual_grad!!!", dual_grad_correct)
    assert success_rate_primal > 0.95, f"Primal success rate exceeds {95} %"
    assert success_rate_grad  > 0.95, f"Grad success rate exceeds {95} %"

if __name__ == "__main__":
    test_grad()