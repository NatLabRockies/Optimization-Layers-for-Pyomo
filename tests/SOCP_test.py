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
def create_model(nominal_Psqrt, nominal_q, nominal_A, nominal_b):
    # Create the model
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    n = nominal_Psqrt.shape[0]
    m = nominal_A.shape[0]

    # Define the decision variables
    model.x = pyo.Var(range(n), domain=pyo.Reals, bounds = (-1, 1))
    # Define parameters
    model.Psqrt = pyo.Var(range(n), range(n), within=pyo.Reals)
    model.q = pyo.Var(range(n), within=pyo.Reals) # Linear term vector
    model.A = pyo.Var(range(m), range(n), within=pyo.Reals)
    model.b = pyo.Var(range(m), within=pyo.Reals)

    for i in range(n):
        for j in range(n):
            model.Psqrt[i, j].fix(nominal_Psqrt[i, j])

    for i in range(n):
        model.q[i].fix(nominal_q[i])

    for i in range(m):
        for j in range(n):
            model.A[i, j].fix(nominal_A[i, j])

    for i in range(m):
        model.b[i].fix(nominal_b[i])
    
    model.equ_constraints = pyo.ConstraintList()
    for i in range(m):
        model.equ_constraints.add(sum(model.A[i, j] * model.x[j] for j in range(n)) == model.b[i])
    # Norm 1 constraints
    model.inequ_constraints = pyo.ConstraintList()
    model.inequ_constraints.add((sum(model.x[i]**2 for i in range(n))) <= 1)
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
        
    return model

def model_instance(n, m):
    # Generate random parameters
    nominal_Psqrt = np.random.rand(n, n)
    nominal_q = np.random.randn(n)
    nominal_A = np.random.randn(m, n)
    nominal_b = np.random.randn(m)
    model = create_model(nominal_Psqrt, nominal_q, nominal_A, nominal_b)
    return model
# %% Gradient obtained by Pyomo
def QP_grad_pyomo(n, m, sample_num, batch_size, alg = "pyomo", val_seed=0):
    if alg == "pyomo":
        model = model_instance(n, m)
        variables_name = [model.x]
        parameters_name = [model.Psqrt, model.q, model.A, model.b]
        if partial:
            grad_parameters_name = [model.Psqrt, model.q, model.A]
        else:
            grad_parameters_name = None
        Layer = PyomoOptLayer(model, variables_name, parameters_name, grad_parameters_name)
    elif alg == "cvxpy":
        x = cp.Variable(n)
        Q_sqrt = cp.Parameter((n, n))
        q = cp.Parameter(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)

        obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt @ x) + q @ x)
        cons = [A@x == b, cp.sum_squares(x) <= 1, x - 1 <= 0, -1 - x <= 0]
        prob = cp.Problem(obj, cons)
        Layer = CvxpyLayer(prob, parameters=[Q_sqrt, q, A, b], variables=[x])

    torch.manual_seed(val_seed)
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64)
    qval = torch.randn(sample_num, n, dtype=torch.float64)
    Aval = torch.randn(sample_num, m, n, dtype=torch.float64)
    bval = torch.randn(sample_num, m, dtype=torch.float64)

    primal, duals, Psqrtgrad, qgrad, Agrad, bgrad = [], [], [], [], [], []
    dataset = torch.utils.data.TensorDataset(Psqrt, qval, Aval, bval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()
    infeasible_list = []
    for batch_idx, (P_batch, q_batch, A_batch, b_batch) in enumerate(dataloader):
        P_batch = P_batch.detach().clone().requires_grad_(True)
        q_batch = q_batch.detach().clone().requires_grad_(True)
        A_batch = A_batch.detach().clone().requires_grad_(True)
        b_batch = b_batch.detach().clone().requires_grad_(True)
        input = tuple([P_batch, q_batch, A_batch, b_batch])
        infeasible = []
        if alg == "pyomo":
            primal_batch, dual_batch, _, _, infeasible = Layer(*input)
            infeasible_list.append(infeasible)
            duals.append(dual_batch.detach().numpy())
        elif alg == "cvxpy":
            primal_batch, = Layer(*input)
        
        primal_batch.sum().backward()
        primal.append(primal_batch.detach().numpy())
        Psqrtgrad.append(P_batch.grad.detach().numpy())
        qgrad.append(q_batch.grad.detach().numpy())
        Agrad.append(A_batch.grad.detach().numpy())
        if not partial:
            bgrad.append(b_batch.grad.detach().numpy())

    primal = np.concatenate(primal, axis=0)
    if alg == "pyomo":
        duals = np.concatenate(duals, axis=0)
    Psqrtgrad = np.concatenate(Psqrtgrad, axis=0)
    qgrad = np.concatenate(qgrad, axis=0)
    Agrad = np.concatenate(Agrad, axis=0)
    if not partial:
        bgrad = np.concatenate(bgrad, axis=0)
    end = time.time()
    pyomo_time = end - start
    return Psqrtgrad, qgrad, Agrad, bgrad, pyomo_time, primal, duals, infeasible_list


def QP_grad_cvxpy(n, m, sample_num, batch_size, infeasible_list, alg = "cvxpy", val_seed=0):
    x = cp.Variable(n)
    Q_sqrt = cp.Parameter((n, n))
    q = cp.Parameter(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt @ x) + q @ x)
    cons = [A@x == b, cp.sum_squares(x) <= 1, x - 1 <= 0, -1 - x <= 0]
    prob = cp.Problem(obj, cons)
    Layer = CvxpyLayer(prob, parameters=[Q_sqrt, q, A, b], variables=[x])

    torch.manual_seed(val_seed)
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64)
    qval = torch.randn(sample_num, n, dtype=torch.float64)
    Aval = torch.randn(sample_num, m, n, dtype=torch.float64)
    bval = torch.randn(sample_num, m, dtype=torch.float64)

    infeasible = []
    for batch in range(len(infeasible_list)):
        infeasible.append((np.array(infeasible_list[batch]) + batch_size * batch).tolist())
    infeasible = [item for sublist in infeasible for item in sublist]

    mask = torch.ones(sample_num, dtype=torch.bool)
    mask[infeasible] = False

    Psqrt = Psqrt[mask]
    qval = qval[mask]
    Aval = Aval[mask]
    bval = bval[mask]

    primal, duals, Psqrtgrad, qgrad, Agrad, bgrad = [], [], [], [], [], []
    dataset = torch.utils.data.TensorDataset(Psqrt, qval, Aval, bval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()

    for batch_idx, (P_batch, q_batch, A_batch, b_batch) in enumerate(dataloader):
        P_batch = P_batch.detach().clone().requires_grad_(True)
        q_batch = q_batch.detach().clone().requires_grad_(True)
        A_batch = A_batch.detach().clone().requires_grad_(True)
        b_batch = b_batch.detach().clone().requires_grad_(True)
        input = tuple([P_batch, q_batch, A_batch, b_batch])
        infeasible = []

        primal_batch, = Layer(*input)
        primal_batch.sum().backward()
        primal.append(primal_batch.detach().numpy())
        Psqrtgrad.append(P_batch.grad.detach().numpy())
        qgrad.append(q_batch.grad.detach().numpy())
        Agrad.append(A_batch.grad.detach().numpy())
        bgrad.append(b_batch.grad.detach().numpy())

    primal = np.concatenate(primal, axis=0)
    Psqrtgrad = np.concatenate(Psqrtgrad, axis=0)
    qgrad = np.concatenate(qgrad, axis=0)
    Agrad = np.concatenate(Agrad, axis=0)
    bgrad = np.concatenate(bgrad, axis=0)
    end = time.time()
    pyomo_time = end - start
    return Psqrtgrad, qgrad, Agrad, bgrad, pyomo_time, primal, duals, infeasible_list


def get_duals(n, m, sample_num, val_seed=0):
    x = cp.Variable(n)
    Q_sqrt = cp.Parameter((n, n))
    q = cp.Parameter(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    obj = cp.Minimize(0.5*cp.sum_squares(Q_sqrt @ x) + q @ x)
    cons = [A@x == b, cp.sum_squares(x) <= 1, x - 1 <= 0, -1 - x <= 0]
    prob = cp.Problem(obj, cons)

    torch.manual_seed(val_seed)
    Psqrt = torch.randn(sample_num, n, n, dtype=torch.float64).numpy()
    qval = torch.randn(sample_num, n, dtype=torch.float64).numpy()
    Aval = torch.randn(sample_num, m, n, dtype=torch.float64).numpy()
    bval = torch.randn(sample_num, m, dtype=torch.float64).numpy()
    duals = []
    for i in range(sample_num):
        Q_sqrt.value, q.value, A.value, b.value = Psqrt[i], qval[i], Aval[i], bval[i]
        prob.solve()
        if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            dual = [float('inf')] * (m + 1)
        else:
            dual = []
            for j in range(m + 1):
                dual.append(cons[j].dual_value.item())
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
    actual = actual.squeeze()
    predicted = predicted.squeeze()
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: array1 has shape {actual.shape}, array2 has shape {predicted.shape}")
    # if actual.ndim > 3:
    #     tol_diff = np.sum(np.abs(actual - predicted), axis=(1, 2, 3))
    # elif actual.ndim > 2:
    #     tol_diff = np.sum(np.abs(actual - predicted), axis=(1, 2))
    # else:
    #     tol_diff = np.sum(np.abs(actual - predicted), axis=1)
    return np.max(np.abs(actual - predicted), axis=tuple(range(1, actual.ndim)))


def remove_infeasible(PsqrtG, qvalG, AvalG, bvalG, primal, duals, batch_size, infeasible_list, alg = "cvxpy"):
    infeasible = []
    for batch in range(len(infeasible_list)):
        infeasible.append((np.array(infeasible_list[batch]) + batch_size * batch).tolist())
    infeasible = [item for sublist in infeasible for item in sublist]
    if alg == "pyomo":
        PsqrtG = np.delete(PsqrtG, infeasible, axis=0)
        qvalG = np.delete(qvalG, infeasible, axis=0)
        AvalG = np.delete(AvalG, infeasible, axis=0)
        if not partial:
            bvalG = np.delete(bvalG, infeasible, axis=0)
    elif alg == "cvxpy":
        duals = np.delete(duals, infeasible, axis=0)
        
    return PsqrtG, qvalG, AvalG, bvalG, primal, duals

def write_results(n: int = 1, m: int = 1, sample_num: int = 32, batch_size: int = 32, alg: str = "pyomo", val_seed: int = 0):
    PsqrtG, qvalG, AvalG, bvalG, pyomo_time, primal, duals, infeasible_list = QP_grad_pyomo(n, m, sample_num, batch_size, alg = "pyomo", val_seed=val_seed)
    np.save(os.path.join(results_dir, "SOCP_PsqrtG1.npy"), PsqrtG)
    np.save(os.path.join(results_dir, "SOCP_qvalG1.npy"), qvalG)
    np.save(os.path.join(results_dir, "SOCP_AvalG1.npy"), AvalG)
    np.save(os.path.join(results_dir, "SOCP_bvalG1.npy"), bvalG)
    np.save(os.path.join(results_dir, "SOCP_primal1.npy"), primal)
    np.save(os.path.join(results_dir,"SOCP_dual1.npy"), duals)

def grad_diff(n: int = 1, m: int = 1, sample_num: int = 32, batch_size: int = 32, alg: str = "pyomo", val_seed: int = 0):
    PsqrtG, qvalG, AvalG, bvalG, pyomo_time, primal, duals, infeasible_list = QP_grad_pyomo(n, m, sample_num, batch_size, alg = "pyomo", val_seed=val_seed)
    np.save(os.path.join(results_dir, "SOCP_PsqrtG1.npy"), PsqrtG)
    np.save(os.path.join(results_dir, "SOCP_qvalG1.npy"), qvalG)
    np.save(os.path.join(results_dir, "SOCP_AvalG1.npy"), AvalG)
    np.save(os.path.join(results_dir, "SOCP_bvalG1.npy"), bvalG)
    np.save(os.path.join(results_dir, "SOCP_primal1.npy"), primal)
    np.save(os.path.join(results_dir,"SOCP_dual1.npy"), duals)
    PsqrtG_ref, qvalG_ref, AvalG_ref, bvalG_ref, cvxpy_time, primal_ref, _, _ = QP_grad_cvxpy(n, m, sample_num, batch_size, infeasible_list, alg = "cvxpy", val_seed=val_seed)
    duals_ref = get_duals(n, m, sample_num, val_seed)
    # remove infeasible list
    # PsqrtG, qvalG, AvalG, bvalG, primal, duals = remove_infeasible(PsqrtG, qvalG, AvalG, bvalG, primal, duals, batch_size, infeasible_list, alg = "pyomo")
    # PsqrtG_ref, qvalG_ref, AvalG_ref, bvalG_ref, primal_ref, duals_ref = remove_infeasible(PsqrtG_ref, qvalG_ref, AvalG_ref, bvalG_ref, primal_ref, duals_ref, batch_size, infeasible_list, alg = "cvxpy")
    if partial:
        inputs = [(primal, primal_ref), (duals, duals_ref), (PsqrtG, PsqrtG_ref), (qvalG, qvalG_ref), (AvalG, AvalG_ref)]
    else:
        inputs = [(primal, primal_ref), (duals, duals_ref), (PsqrtG, PsqrtG_ref), (qvalG, qvalG_ref), (AvalG, AvalG_ref), (bvalG, bvalG_ref)]
    diffs = [calculate_error(P, C) for P, C in inputs]
    if partial:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff =  diffs
        return Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff
    else:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff =  diffs
        return Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff

def test_grad():
    primal_diff_threshold = 1e-2
    grad_diff_threshold = 1e-2
    if partial:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff = grad_diff(n = 5, m = 1, sample_num = 1000, batch_size=32, alg = "pyomo", val_seed = 0)
    else:
        Primal_diff, dual_diff, PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff = grad_diff(n = 5, m = 1, sample_num = 1000, batch_size=32, alg = "pyomo", val_seed = 0)
    success_rate_primal = np.mean(Primal_diff < primal_diff_threshold)
    success_rate_dual = np.mean(dual_diff < primal_diff_threshold)
    if partial:
        G_diff = np.stack((PsqrtG_diff, qvalG_diff, AvalG_diff), axis=1)
    else:
        G_diff = np.stack((PsqrtG_diff, qvalG_diff, AvalG_diff, bvalG_diff), axis=1)
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
        write_results(n=1, m=1, sample_num=32, batch_size=32, alg="pyomo", val_seed=0)
    else:
        test_grad()