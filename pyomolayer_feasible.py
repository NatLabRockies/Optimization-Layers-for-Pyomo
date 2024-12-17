import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('..')  # Add the parent directory to Python's search path
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import copy
from utilities import get_sen
torch.set_default_dtype(torch.float64)

class PyomoOptLayer(nn.Module):
    def __init__(self, create_model, variables_name, parameters_name, grad_parameters_name = None, solver = 'ipopt'):
        super().__init__()
        # takes a pyomo model
        self.concrete_model = create_model
        self.solver = pyo.SolverFactory(solver)
        self.variables_name = variables_name
        self.parameters_name = parameters_name
        self.grad_parameters_name = grad_parameters_name if grad_parameters_name else parameters_name
        self.init_pyomo_varorder()
    
    def init_pyomo_varorder(self):
        self.parameters_size = {}
        for var in self.concrete_model.component_objects(pyo.Var, active=True):
            # Scaler
            if not var.is_indexed():
                self.parameters_size[var.name] = [0]
            else: 
                index_set = var.index_set()
                self.parameters_size[var.name] = [len(s) for s in index_set.subsets()]

        self.variables_name_index = []
        for var in self.variables_name:
            # Get the variable object using component()
            var_index = self.concrete_model.component(var)
            # Retrieve indexed names
            self.variables_name_index += [var_index[i].name for i in var_index.keys()]

    def forward(self, *batch_params):
        f = PyomoLayerFn(concrete_model = self.concrete_model, variables_name = self.variables_name,  \
                        variables_name_index = self.variables_name_index,\
                        parameters_name = self.parameters_name, parameters_size = self.parameters_size, \
                        grad_parameters_name = self.grad_parameters_name, solver = self.solver)

        sol = f(*batch_params)

        return sol
        
def PyomoLayerFn(concrete_model, variables_name, variables_name_index, parameters_name, parameters_size, grad_parameters_name, solver):
    class PyomoLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *batch_params):
            J, lhs_J, rhs_J, primal_out, dual_out, infeasible = [], [], [], [], [], []
            # solve over minibatch by just iterating
            for batch in range(batch_params[0].shape[0]):
                params = [p[batch] for p in batch_params]
                with torch.no_grad():
                    params_ = [p.detach().clone().double().numpy() for p in params]
                    for index, p_name in enumerate(parameters_name):
                        p_size = copy.deepcopy(parameters_size[p_name]) 
                        p_attr = getattr(concrete_model, p_name)
                        if len(p_size) == 3:
                            for i in range(p_size[0]):
                                for j in range(p_size[1]):
                                    for k in range(p_size[2]):
                                        p_attr[i, j, k].fix(params_[index][i, j, k])
                                        p_attr[i, j, k].value = params_[index][i, j, k]
                        elif len(p_size) == 2:
                            for i in range(p_size[0]):
                                for j in range(p_size[1]):
                                    p_attr[i, j].fix(params_[index][i, j])
                                    p_attr[i, j].value = params_[index][i, j]
                        else:
                            for i in range(p_size[0]):
                                p_attr[i].fix(params_[index][i])
                                p_attr[i].value = params_[index][i]

                    # concrete_model = model(*params_)
                    results = solver.solve(concrete_model, tee=False)
                    if results.solver.termination_condition in [TerminationCondition.infeasible, TerminationCondition.unbounded]:
                        print(f"Problem {batch} is infeasible or unbounded. Skipping.")
                        infeasible.append(batch)
                        continue 
                    # should add a post-processing function to help project to the closest decision variables. 
                    # decision variables
                    #concrete_model.pprint
                    #concrete_model.display()
                    # concrete_model.obj()
                    duals = []    
                    for constraint in concrete_model.component_objects(pyo.Constraint, active=True):
                        for index in constraint:
                            if constraint[index].active:
                                duals.append(-concrete_model.dual[constraint[index]])
                    duals = torch.tensor(duals).type_as(params[0]).view(-1)

                    vars = []
                    for v_name in variables_name:
                        var = concrete_model.component(v_name) 
                        vars  += [var[i].value for i in var]
                    vars = torch.tensor(vars).type_as(params[0]).view(-1).requires_grad_(True)
                    
                # if batch == 0:
                #     primal_out = torch.zeros((batch_params[0].shape[0], len(vars)))
                #     dual_out = torch.zeros((batch_params[0].shape[0], len(duals)))
                 
                # A Jacobian matrix of the (decision variables) with respect to parameters
                dfullvar_dp, lhs_Jac, rhs_Jac, variables_nameindex = get_sen(concrete_model, grad_parameters_name, parameters_size, variables_name_index)
                primal_out.append(vars)
                dual_out.append(duals)
                J.append(dfullvar_dp[variables_nameindex, :])
                lhs_J.append(lhs_Jac)
                rhs_J.append(rhs_Jac)
            ctx.save_for_backward(torch.tensor(np.array(J), dtype=torch.float64))
            ctx.infeasible = infeasible
            return torch.stack(primal_out), torch.stack(dual_out), lhs_J, rhs_J, infeasible

        @staticmethod
        def backward(ctx, grad_out, dual_dummy = 0, Jac_dummy = 0, rhs_dummy = 0, infeasible_dummy = 0):
            Jac, = ctx.saved_tensors
            infeasible = ctx.infeasible
            batch_size = grad_out.shape[0]
            full_batch_size = batch_size + len(infeasible)
            feasible_list = [item for item in list(range(full_batch_size)) if item not in infeasible]
            grad = (Jac.transpose(1, 2).bmm(grad_out.unsqueeze(-1))).squeeze(-1)
            grad_full = torch.zeros(int(grad.shape[0] + len(infeasible)), grad.shape[1])
            grad_full[feasible_list] = grad
            grad_reshape = []
            start = 0
            for _, p in enumerate(parameters_name):
                if p not in grad_parameters_name:
                    grad_reshape.append(None)
                    continue
                size = copy.deepcopy(parameters_size[p])
                param_length = len(list(concrete_model.component(p).keys()))
                size.insert(0, full_batch_size)
                grad_reshape.append(grad_full[:, start:start+param_length].reshape(size))
                start += param_length
            grad_reshape = tuple(grad_reshape)
            return *grad_reshape, None

    return PyomoLayerFnFn.apply