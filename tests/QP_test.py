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
import os
import datetime
current_dir = os.getcwd()
sys.path.append(current_dir) # Add the parent directory to Python's search path
results_dir = os.path.join(current_dir, "tests", "results_" + datetime.date.today().strftime("%Y%m%d"))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import numpy as np
from Opt_Layer.pyomolayer import PyomoOptLayer
import torch
import time
cp, cp_available = attempt_import(name="cvxpy")
if cp_available:
    from cvxpylayers.torch import CvxpyLayer
# %%
def create_model(nominal_Psqrt, nominal_q, nominal_A, nominal_b, nominal_G, nominal_h):
    # Create a concrete model
    m = pyo.ConcreteModel()

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    n = nominal_Psqrt.shape[0]
    t = nominal_A.shape[0]
    p = nominal_G.shape[0]
    
    # Define parameters
    m.q = pyo.Var(range(n), within=pyo.Reals)            # Linear term vector
    m.G = pyo.Var(range(p), range(n), within=pyo.Reals)  # Inequality constraint matrix
    m.h = pyo.Var(range(p), within=pyo.Reals) 
    m.A = pyo.Var(range(t), range(n), within=pyo.Reals)  # Equality constraint matrix
    m.b = pyo.Var(range(t), within=pyo.Reals)            # Equality constraint vector
    m.Psqrt = pyo.Var(range(n), range(n), within=pyo.Reals)

    # Define variables
    m.x = pyo.Var(range(n), within=pyo.Reals)

    for i in range(n):
        for j in range(n):
            m.Psqrt[i, j].fix(nominal_Psqrt[i, j])

    for i in range(n):
        m.q[i].fix(nominal_q[i])
    
    for i in range(t):
        for j in range(n):
            m.A[i, j].fix(nominal_A[i, j])
            
    for i in range(t):  
        m.b[i].fix(nominal_b[i])

    for i in range(p):
        for j in range(n):
            m.G[i, j].fix(nominal_G[i, j])

    for i in range(p):
        m.h[i].fix(nominal_h[i])

    # Define equality constraints
    m.equ_constraints = pyo.ConstraintList()
    for i in range(t):
        m.equ_constraints.add(sum(m.A[i, j] * m.x[j] for j in range(n)) == m.b[i])

    # Define inequality constraints
    m.inequ_constraints = pyo.ConstraintList()
    for i in range(p):
        m.inequ_constraints.add(sum(m.G[i, j] * m.x[j] for j in range(n)) - m.h[i] <= 0)
    
    # Define objective function: 
    def objective_rule(m):
        tol_term = 0
        for i in range(n):
            q_term = 0
            for j in range(n):
                q_term += m.Psqrt[i, j] * m.x[j]
            tol_term += q_term**2
            
        return 0.5 * tol_term + sum(m.q[i] * m.x[i] for i in range(n))
        
    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return m

def model_instance(n, t, p):
    nominal_Psqrt = np.random.rand(n, n)
    nominal_q = np.random.rand(n)
    nominal_A = np.random.rand(t, n)
    nominal_b = np.random.rand(t)
    nominal_G = np.random.rand(p, n)
    nominal_h = np.random.rand(p)
    model = create_model(nominal_Psqrt, nominal_q, nominal_A, nominal_b, nominal_G, nominal_h)
    return model
# %% Gradient obtained by Pyomo
def QP_grad_pyomo(n, t, p, sample_num, batch_size, alg = "pyomo", val_seed=0):
    
    if alg == "pyomo":
        model = model_instance(n, t, p)
        variables_name = [model.x]
        parameters_name = [model.Psqrt, model.q, model.A, model.b, model.G, model.h]
        if partial:
            free_parameters_name = [model.b]
        else:
            free_parameters_name = None
        Layer = PyomoOptLayer(model, variables_name, parameters_name, free_parameters_name)
    elif alg == "cvxpy":
        x = cp.Variable(n)
        Q_sqrt = cp.Parameter((n, n))
        q = cp.Parameter(n)
        A = cp.Parameter((t, n))
        b = cp.Parameter(t)
        G = cp.Parameter((p, n))
        h = cp.Parameter(p)

        obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt@x) + q @ x)
        cons = [A @ x == b, G @ x <= h]
        prob = cp.Problem(obj, cons)
        Layer = CvxpyLayer(prob, parameters=[Q_sqrt, q, A, b, G, h], variables=[x])

    torch.manual_seed(val_seed)
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64)
    qval = torch.randn(sample_num, n, dtype=torch.float64)
    Aval = torch.randn(sample_num, t, n, dtype=torch.float64)
    bval = torch.randn(sample_num, t, dtype=torch.float64)
    Gval = torch.randn(sample_num, p, n, dtype=torch.float64)
    hval = torch.randn(sample_num, p, dtype=torch.float64)

    primal, duals, Psqrtgrad, qgrad, Agrad, bgrad, Ggrad, hgrad = [], [], [], [], [], [], [], []
    dataset = torch.utils.data.TensorDataset(Psqrt, qval, Aval, bval, Gval, hval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()
    for batch_idx, (P_batch, q_batch, A_batch, b_batch, G_batch, h_batch) in enumerate(dataloader):
        P_batch = P_batch.detach().clone().requires_grad_(True)
        q_batch = q_batch.detach().clone().requires_grad_(True)
        A_batch = A_batch.detach().clone().requires_grad_(True)
        b_batch = b_batch.detach().clone().requires_grad_(True)
        G_batch = G_batch.detach().clone().requires_grad_(True)
        h_batch = h_batch.detach().clone().requires_grad_(True)
        input = tuple([P_batch, q_batch, A_batch, b_batch, G_batch, h_batch])
        if alg == "pyomo":
            primal_batch, _, dual_batch, _, _ = Layer(*input)
            duals.append(dual_batch.detach().numpy())
        elif alg == "cvxpy":
            primal_batch, = Layer(*input)
        primal_batch.sum().backward()
        primal.append(primal_batch.detach().numpy())
        Psqrtgrad.append(P_batch.grad.detach().numpy())
        qgrad.append(q_batch.grad.detach().numpy())
        Agrad.append(A_batch.grad.detach().numpy())
        bgrad.append(b_batch.grad.detach().numpy())
        if not partial:
            Ggrad.append(G_batch.grad.detach().numpy())
        hgrad.append(h_batch.grad.detach().numpy())
    primal = np.concatenate(primal, axis=0)
    if alg == "pyomo":
        duals = np.concatenate(duals, axis=0)
    Psqrtgrad = np.concatenate(Psqrtgrad, axis=0)
    qgrad = np.concatenate(qgrad, axis=0)
    Agrad = np.concatenate(Agrad, axis=0)
    bgrad = np.concatenate(bgrad, axis=0)
    if not partial:
        Ggrad = np.concatenate(Ggrad, axis=0)
    hgrad = np.concatenate(hgrad, axis=0)
    end = time.time()
    pyomo_time = end - start
    return Psqrtgrad, qgrad, Agrad, bgrad, Ggrad, hgrad, pyomo_time, primal, duals

def get_duals(n, t, p, sample_num, val_seed=0):
    x = cp.Variable(n)
    Q_sqrt = cp.Parameter((n, n))
    q = cp.Parameter(n)
    A = cp.Parameter((t, n))
    b = cp.Parameter(t)
    G = cp.Parameter((p, n))
    h = cp.Parameter(p)

    obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt@x) + q @ x)
    cons = [A @ x == b, G @ x <= h]
    prob = cp.Problem(obj, cons)

    torch.manual_seed(val_seed)
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64).numpy()
    qval = torch.randn(sample_num, n, dtype=torch.float64).numpy()
    Aval = torch.randn(sample_num, t, n, dtype=torch.float64).numpy()
    bval = torch.randn(sample_num, t, dtype=torch.float64).numpy()
    Gval = torch.randn(sample_num, p, n, dtype=torch.float64).numpy()
    hval = torch.randn(sample_num, p, dtype=torch.float64).numpy()
    duals = []
    for i in range(sample_num):
        Q_sqrt.value, q.value, A.value, b.value, G.value, h.value = Psqrt[i], qval[i], Aval[i], bval[i], Gval[i], hval[i]
        prob.solve()
        if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            dual = [float('inf')] * (t + p)
        else:
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
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: array1 has shape {actual.shape}, array2 has shape {predicted.shape}")
    # if actual.ndim > 3:
    #     tol_diff = np.sum(np.abs(actual - predicted), axis=(1, 2, 3))
    # elif actual.ndim > 2:
    #     tol_diff = np.sum(np.abs(actual - predicted), axis=(1, 2))
    # else:
    #     tol_diff = np.sum(np.abs(actual - predicted), axis=1)
    return np.max(np.abs(actual - predicted), axis=tuple(range(1, actual.ndim)))

def write_results(n: int = 10, t: int = 5, p: int = 2, sample_num: int = 32, batch_size: int = 32, alg: str = "pyomo", val_seed: int = 0):
    PsqrtG, qvalG, AvalG, bvalG, GvalG, hvalG, pyomo_time, primal, duals = QP_grad_pyomo(n, t, p, sample_num, batch_size, alg = "pyomo", val_seed=val_seed)
    np.save(os.path.join(results_dir, "QP_PsqrtG.npy"), PsqrtG)
    np.save(os.path.join(results_dir, "QP_qvalG.npy"), qvalG)
    np.save(os.path.join(results_dir, "QP_AvalG.npy"), AvalG)
    np.save(os.path.join(results_dir, "QP_bvalG.npy"), bvalG)
    np.save(os.path.join(results_dir, "QP_GvalG.npy"), GvalG)
    np.save(os.path.join(results_dir, "QP_hvalG.npy"), hvalG)
    np.save(os.path.join(results_dir, "QP_primal.npy"), primal)
    np.save(os.path.join(results_dir, "QP_dual.npy"), duals)

def grad_diff(n: int = 10, t: int = 5, p: int = 2, sample_num: int = 32, batch_size: int = 32, alg: str = "pyomo", val_seed: int = 0):
    PsqrtG, qvalG, AvalG, bvalG, GvalG, hvalG, pyomo_time, primal, duals = QP_grad_pyomo(n, t, p, sample_num, batch_size, alg = "pyomo", val_seed=val_seed)
    np.save(os.path.join(results_dir, "QP_PsqrtG.npy"), PsqrtG)
    np.save(os.path.join(results_dir, "QP_qvalG.npy"), qvalG)
    np.save(os.path.join(results_dir, "QP_AvalG.npy"), AvalG)
    np.save(os.path.join(results_dir, "QP_bvalG.npy"), bvalG)
    np.save(os.path.join(results_dir, "QP_GvalG.npy"), GvalG)
    np.save(os.path.join(results_dir, "QP_hvalG.npy"), hvalG)
    np.save(os.path.join(results_dir, "QP_primal.npy"), primal)
    np.save(os.path.join(results_dir, "QP_dual.npy"), duals)
    print("pyomo_time", pyomo_time)
    PsqrtG_ref, qvalG_ref, AvalG_ref, bvalG_ref, GvalG_ref, hvalG_ref, cvxpy_time, primal_ref, _ = QP_grad_pyomo(n, t, p, sample_num, batch_size, alg = "cvxpy", val_seed=val_seed)
    print("cvxpy_time", cvxpy_time)
    duals_ref = get_duals(n, t, p, sample_num, val_seed)
    if partial:
        inputs = [(primal, primal_ref), (duals, duals_ref), (PsqrtG, PsqrtG_ref), (qvalG, qvalG_ref), (AvalG, AvalG_ref), (bvalG, bvalG_ref), (hvalG, hvalG_ref)]
    else:
        inputs = [(primal, primal_ref), (duals, duals_ref), (PsqrtG, PsqrtG_ref), (qvalG, qvalG_ref), (AvalG, AvalG_ref), (bvalG, bvalG_ref), (GvalG, GvalG_ref), (hvalG, hvalG_ref)]
    diffs = [calculate_error(P, C) for P, C in inputs]
    if partial:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, hvalG_diff =  diffs
        return Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, hvalG_diff 
    else:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, GvalG_diff, hvalG_diff =  diffs
        return Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, GvalG_diff, hvalG_diff 

def test_grad():
    primal_diff_threshold = 1e-2
    grad_diff_threshold = 1e-2
    if partial:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, hvalG_diff = grad_diff(n = 10, t = 4, p = 2, sample_num = 1000, batch_size= 32, alg = "pyomo", val_seed = 0)
    else:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, GvalG_diff, hvalG_diff = grad_diff(n = 10, t = 4, p = 2, sample_num = 1000, batch_size= 32, alg = "pyomo", val_seed = 0)
    success_rate_primal = np.mean(Primal_diff < primal_diff_threshold)
    success_rate_dual = np.mean(dual_diff < primal_diff_threshold)
    if partial:
        G_diff = np.stack((PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, hvalG_diff), axis=1)
    else:
        G_diff = np.stack((PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff, GvalG_diff, hvalG_diff), axis=1)
    success_rate_grad = np.mean(np.all(G_diff < grad_diff_threshold, axis=1))
    dual_grad_correct = np.sum((dual_diff < primal_diff_threshold) & np.all(G_diff < grad_diff_threshold, axis=1)) / np.sum(dual_diff < primal_diff_threshold) 
    print("!!!success_rate_primal!!!", success_rate_primal)
    print("!!!success_rate_dual!!!", success_rate_dual)
    print("!!!success_rate_grad!!!", success_rate_grad)
    print("!!!success_rate_dual_grad!!!", dual_grad_correct)
    assert success_rate_primal > 0.99, f"Primal success rate exceeds {99} %"
    assert dual_grad_correct > 0.99, f"Grad success rate exceeds {99} %"

if __name__ == "__main__":
    global partial
    partial = False
    if len(sys.argv) > 1 and sys.argv[1]:
        write_results(n=10, t=4, p=2, sample_num=1000, batch_size=32, alg="pyomo", val_seed=0)
    else:
        test_grad()