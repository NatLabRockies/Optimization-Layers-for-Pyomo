# %% [markdown]
# # Signal denoising
# 
# This notebook accompanies the paper [Learning Convex Optimization Models](https://web.stanford.edu/~boyd/papers/learning_copt_models.html).

# %%
import os
import pyomo.environ as pyo
import numpy as np
from pyomolayers import PyomoOptLayer
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader
this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(this_dir, "results")
os.makedirs(results_dir, exist_ok=True)
torch.set_default_dtype(torch.double)
# %%
def fit(loss, params, X, Y, Xval, Yval, opt = torch.optim.Adam, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None):
    """
    Description:
    The function aims to minimize the loss of :math:`X` and :math:`Y` by updating the parameters using the gradient descent algorithm.

    Args:
        - loss (required):
            A criterion that measures the difference between :math:`X` and target :math:`Y`. The output is a scalar by default for backpropagation. 
        - params (``List[Tensor]``, required): 
            A list of parameters to optimize. 
        - X and Y (``Tensor``, required):
            Input and output tensors of training dataset.
        - Xval and Yval (``Tensor``, required):
            Input and output tensors of validation dataset.
        - opt (``Pytorch optimizer``, optional): 
            Pytorch optimizer is an optimization technique for gradient descent. Default: torch.optim.Adam.
        - opt_kwargs (``Dict[str, any]``, optional):
            A dict containing default values of optimization options. The default value for learning rate (float) is :math:`10^{-3}`.
        - batch_size (``int``, optional): 
            Training mini-batch size. Default: 128.
        - epochs (``int``, optional): 
            Training epochs. Default: 100.
        - verbose (``bool``, optional): 
            If set to True, display the training loss for each epoch. Default: False.
        - callback (``PyTorch callback``, optional): 
            PyTorch callback is a function triggered at specific points during model training. Default: None.

    Shape:
        - X: :math:`(*, )` where :math:`*` means any number of dimensions.
        - Y: :math:`(*, )` where the first dimension is the same shape as X.
        - Xval: :math:`(**, )` where :math:`**` means any number of dimensions.
        - Yval: :math:`(**, )` where the first dimension is the same shape as Xval.

    Returns:
        A list of validation loss for each epoch, a list of training loss for each epoch.

    Return type:
        ``List[loss scalar]``, ``List[loss scalar]``.

    Raises:
        ExceptionName: Description of the error raised, if any.

    Examples:
        >>> val_losses, train_losses = example_function(loss, params, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None)
    """

    train_dset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    opt = opt(params, **opt_kwargs)

    train_losses = []
    batch_loss = []
    val_losses = []
    try:
        for epoch in range(epochs):
            with torch.no_grad():
                val_losses.append(loss(Xval, Yval).item())
            if verbose:
                print("%03d | %3.5f" % (epoch + 1, val_losses[-1]))
            batch_loss = []
            for Xbatch, Ybatch in train_loader:
                opt.zero_grad()
                l = loss(Xbatch, Ybatch)
                l.backward()
                opt.step()
                batch_loss.append(l.item())
                if callback is not None:
                    callback()
            train_losses.append(np.mean(batch_loss))
            if verbose:
                print("epoch %03d | %3.5f" % (epoch, np.mean(batch_loss)))
    except Exception as e:
        print(f"Training stopped due to an error: {e}")
    finally:
        # Ensure valid_loss is returned even if training stops
        return val_losses, train_losses

# %%
def create_model(nominal_theta_param, nominal_x_param, nominal_lambda_param):
    # Create a concrete model
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    m.y_cp = pyo.Var(range(n), within=pyo.Reals)
    m.x_minus_y = pyo.Var(range(n), within=pyo.Reals)
    # Define parameters
    m.theta_param = pyo.Var(range(n), range(n), within=pyo.Reals)
    m.x_param = pyo.Var(range(n), within=pyo.Reals)
    m.lambda_param = pyo.Var(range(1), within=pyo.PositiveReals)

    for i in range(n):
        for j in range(n):
            m.theta_param[i, j].fix(nominal_theta_param[i, j])

    for i in range(n):
        m.x_param[i].fix(nominal_x_param[i])
            
    for i in range(1):
        m.lambda_param[i].fix(nominal_lambda_param[i])

    m.equ_constraints = pyo.ConstraintList()
    for i in range(n):
        m.equ_constraints.add(m.x_minus_y[i] == m.x_param[i] - m.y_cp[i])
    
    # Objective Function
    def objective_rule(m):
        # First term: sum of squares of theta_param * x_minus_y
        first_term = sum(sum(m.theta_param[i, j] * m.x_minus_y[j] for j in range(n))**2 for i in range(n))
        # Second term: lambda_param * sum of squares of differences in y_cp
        second_term = m.lambda_param[0] * sum((m.y_cp[i+1] - m.y_cp[i])**2 for i in range(n-1))
        return first_term + second_term

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return m

def model_instance():
    nominal_theta_param, nominal_x_param, nominal_lambda_param = np.random.rand(n, n), np.random.rand(n), np.abs(np.random.rand(1))
    model = create_model(nominal_theta_param, nominal_x_param, nominal_lambda_param)
    return model
# Pyomo model
# np.random.seed(1)
# n = 10
# nominal_theta_param, nominal_x_param, nominal_lambda_param = np.random.rand(n, n), np.random.rand(n), np.abs(np.random.rand(1))

# model = create_model(nominal_theta_param, nominal_x_param, nominal_lambda_param)
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())

def create_model_withouttheta(nominal_x_param, nominal_lambda_param):
    # Create a concrete model
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Define variables
    m.y_cp = pyo.Var(range(n), within=pyo.Reals)
    m.x_minus_y = pyo.Var(range(n), within=pyo.Reals)
    # Define parameters
    m.x_param = pyo.Var(range(n), within=pyo.Reals)
    m.lambda_param = pyo.Var(range(1), within=pyo.PositiveReals)

    for i in range(n):
        m.x_param[i].fix(nominal_x_param[i])
            
    for i in range(1):
        m.lambda_param[i].fix(nominal_lambda_param[i])

    m.equ_constraints = pyo.ConstraintList()
    for i in range(n):
        m.equ_constraints.add(m.x_minus_y[i] == m.x_param[i] - m.y_cp[i])
    
    # Objective Function
    def objective_rule(m):
        # First term: sum of squares of x - y_cp
        first_term = sum((m.x_param[i] - m.y_cp[i])**2 for i in range(n))
        # Second term: lambda_param * sum of squares of differences in y_cp
        second_term = m.lambda_param[0] * sum((m.y_cp[i+1] - m.y_cp[i])**2 for i in range(n-1))
        return first_term + second_term

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
     
    return m

def model_instance_withouttheta():
    nominal_x_param, nominal_lambda_param = np.random.rand(n), np.abs(np.random.rand(1))
    model = create_model_withouttheta(nominal_x_param, nominal_lambda_param)
    return model
#Pyomo model
# np.random.seed(1)
# n = 10
# nominal_x_param, nominal_lambda_param = np.random.rand(n), np.abs(np.random.rand(1))

# model = create_model_withouttheta(nominal_x_param, nominal_lambda_param)
# opt = pyo.SolverFactory('ipopt', tee=True)
# results = opt.solve(model) 
# print(results)
# model.pprint()
# print(model.obj())

def loss_fn(X, actual):
    batch_size = X.shape[0]
    preds, _ , _ = Pyomolayer(theta_tch.repeat(batch_size, 1, 1), X, lambda_tch.repeat(batch_size, 1))
    y_cp = preds
    mse_per_example = (y_cp - actual).pow(2).mean(axis=1)
    return mse_per_example.mean()

def main():
    global n, Pyomolayer, theta_tch, lambda_tch
    torch.random.manual_seed(0)
    np.random.seed(0)
    N_train = 500
    N_val = 20
    n = 100
    epochs_num = 50
    batch_size = 64
    leanring_rate = 1e-2
    Sigma_sqrt = 0.1 * np.random.randn(n, n)
    Sigma = Sigma_sqrt.T @ Sigma_sqrt
    normal = torch.distributions.MultivariateNormal(loc=torch.zeros(n), covariance_matrix=torch.tensor(Sigma))
    sample_idx = np.random.randint(low=0, high=N_val)
    theta_tch = torch.eye(n, requires_grad=True)
    lambda_tch = torch.tensor(0.5, requires_grad=True)

    inputs = []
    outputs = []
    eval_pts = torch.linspace(0, 2*math.pi, n)
    for i in range(N_train + N_val):
        b = np.random.uniform(low=1, high=3)
        y = torch.cos(b*eval_pts)
        x = y.clone()
        x += normal.sample()
        inputs.append(x)
        outputs.append(y)

    inputs = torch.stack(inputs)
    outputs = torch.stack(outputs)
    # plt.plot(inputs[sample_idx])
    # plt.plot(outputs[sample_idx])
    X_train = inputs[:N_train]
    Y_train = outputs[:N_train]
    X_val = inputs[N_train:]
    Y_val = outputs[N_train:]

    model = model_instance()
    variables = [model.y_cp]
    parameters = [model.theta_param, model.x_param, model.lambda_param]
    
    Pyomolayer = PyomoOptLayer(model, variables, parameters)

    params = [theta_tch, lambda_tch]
    val_losses, train_losses = fit(loss_fn, params, X_train, Y_train, X_val, Y_val, \
                                   opt=torch.optim.Adam, opt_kwargs={"lr": leanring_rate}, batch_size=batch_size, epochs=epochs_num, verbose=True)
    # plt.figure(figsize=(4, 4))
    # plt.imshow(theta_tch.detach().numpy(), cmap='rainbow')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.tight_layout()

    with torch.no_grad():
        val_preds = Pyomolayer(theta_tch.repeat(X_val.shape[0], 1, 1), X_val, lambda_tch.repeat(X_val.shape[0], 1))

    model_withouttheta = model_instance_withouttheta()
    variables = [model_withouttheta.y_cp]
    parameters = [model_withouttheta.x_param, model_withouttheta.lambda_param]    
    Pyomolayer_withouttheta = PyomoOptLayer(model_withouttheta, variables, parameters)

    lambda_values = torch.linspace(1e-5, 20, 10)
    best_lambda = None
    lowest_loss = np.inf
    for value in lambda_values:
        batch_size = X_train.shape[0]
        primal, _, _ = Pyomolayer_withouttheta(X_train, value.repeat(batch_size, 1))
        preds = primal
        mse_per_example = (preds - Y_train).pow(2).mean(axis=1)
        mse = mse_per_example.mean()
        print('mse', mse)
        if mse < lowest_loss:
            print('lowest mse yet')
            lowest_loss = mse
            best_lambda = value
    print("best_lambda", best_lambda)
    print("lowest_loss", lowest_loss)
    primal, _, _ = Pyomolayer_withouttheta(X_val, best_lambda.repeat(X_val.shape[0], 1))
    one_param_preds = primal
    mse_per_example = (one_param_preds - Y_val).pow(2).mean(axis=1)
    one_param_mse = mse_per_example.mean()
    print("one_param_mse", one_param_mse)

    plt.figure(figsize=(10., 3.5))
    ax = plt.gca()
    plt.plot(val_losses, color='k', label='convex optimization model')
    ax.axhline(one_param_mse, linestyle='-.', label='least squares')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "signal_denoising_val_loss.png"), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10., 3.5))
    plt.plot(X_val[sample_idx], label='input', color='silver', linestyle='-.')
    plt.plot(one_param_preds[sample_idx], label='least squares', color='gray', linestyle='-',)
    plt.plot(val_preds[0][sample_idx, :n], label='convex optimization model', linestyle='-')
    plt.plot(Y_val[sample_idx], label='true output', linestyle='--')
    plt.xlabel('$i$')
    plt.ylabel('$y_i$')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "signal_denoising.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
