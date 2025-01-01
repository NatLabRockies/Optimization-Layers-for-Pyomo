import numpy as np
import torch
import copy
import torch.nn as nn
import sys
sys.path.append('..')  # Add the parent directory to Python's search path
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.opt import TerminationCondition
import copy
from utilities import get_sen
torch.set_default_dtype(torch.float64)

class PyomoOptLayer(nn.Module):
    """
    Descriptions:
        A subclass of torch.nn.Module. 

    Args:
        - create_model (``Pyomo model creation function``, required)
            The function of creating the Pyomo model. The input is the parameter value and the output is the concrete Pyomo model.
        - variables_name (``List[objective]``, required)
            A list of variable defined in Pyomo.
        - parameters_name(``List[objective]``, required)
            A list of parameter defined in Pyomo.
        - free_parameters_name(``List[objective]``, optional)
            A list of parameters that do not require gradient, which a subset of parameters_name. By setting to ``None``, grad_parameters_name = parameters_name.
        - solver (``str`` , optional)
            The optimization solver. The default solver is ``ipopt``.

    Examples:
        >>> def create_model(A_value, b_value): 
            concrete_model = pyo.ConcreteModel()
            concrete_model.x = pyo.Var(range(n), within=pyo.Reals)
            concrete_model.A = pyo.Var(range(n), within=pyo.Reals)
            concrete_model.b = pyo.Var(range(m), within=pyo.Reals)
            concrete_model.A[i].fix(A_value[i]) for i in range(n)
            concrete_model.b[i].fix(b_value[i]) for i in range(m)
            return concrete_model
            
        >>> variables_name = [concrete_model.x]
        >>> parameters_name = [concrete_model.A, concrete_model.b]
        >>> grad_parameters_name = [concrete_model.b]

        >>> Layer = PyomoOptLayer(create_model, variables_name, parameters_name, free_parameters_name, solver = 'ipopt')
    """

    def __init__(self, concrete_model, variables_name, parameters_name, free_parameters_name = None, solver = 'ipopt'):
        super().__init__()
        # takes a pyomo model
        self.concrete_model = concrete_model
        self.solver = pyo.SolverFactory(solver)
        self.variables_name = variables_name
        self.parameters_name = parameters_name
        self.free_parameters_name = free_parameters_name
        self.nlp_init = PyomoNLP(concrete_model)
        self.init_pyomo_varorder()
        self.get_nlp_full()

    def get_nlp_full(self):
        model = copy.deepcopy(self.concrete_model)
        # Unfix the param variables for Jac/Hessian evaluation
        for param in model.component_objects(pyo.Var):
            if param in self.grad_parameters_name:
                for index in param:
                    param[index].unfix() 
        self.nlp_full = PyomoNLP(model)
    
    def init_pyomo_varorder(self):
        if self.free_parameters_name:
            self.grad_parameters_name = [item for item in self.parameters_name if item not in self.free_parameters_name] 
        else:
            self.grad_parameters_name = self.parameters_name
        assert len(self.grad_parameters_name) > 0 

        self.parameters_size = {}
        for var in self.concrete_model.component_objects(pyo.Var, active=True):
            # Scaler
            if not var.is_indexed():
                self.parameters_size[var.name] = [0]
            else: 
                index_set = var.index_set()
                self.parameters_size[var.name] = [len(s) for s in index_set.subsets()]

        self.variables_name_index = []
        for var_index in self.variables_name:
            self.variables_name_index += [var_index[i].name for i in var_index.keys()]

        self.vars_to_indices = {}
        for p in self.parameters_name:
            if not p.is_indexed():
                self.vars_to_indices[p.name] = 0
            else:
                self.vars_to_indices[p] = pyo.ComponentMap()
                for idx, var in enumerate(p.values()):
                    self.vars_to_indices[p][var] = idx

        self.param_order = []
        for p_name in self.grad_parameters_name:
            for i in p_name.index_set():
                self.param_order.append(str(p_name[i]))
        
        variables_name_full = []
        self.variables_nameindex_full = []
        self.vars_obj = []
        pyomo_variables = self.nlp_init.get_pyomo_variables()
        for var in pyomo_variables:
            self.variables_nameindex_full.append(var.name)
            var_name = var.parent_component().name
            if var_name not in variables_name_full:
                variables_name_full.append(var_name)
                v = getattr(self.concrete_model, var_name)
                self.vars_obj.append(v)

        self.variables_index_order = []
        for var in self.variables_name_index:
            self.variables_index_order.append(self.variables_nameindex_full.index(var))

    def forward(self, *batch_params):
        """
        Descriptions:
            The forward function which takes in the parameter values and outputs the primal variables. 

        Args:
            batch_params (Tuple(List[Tensor]), required):
                A batch of parameter values.
        Returns:
            - primal_out: The primal variables that require gradients.
            - dual_out: The dual variables that do not require gradients.
            - lhs_J: Left hand side KKT matrix.
            - rhs_J: Right hand side vector.

        Return type:
            - primal_out (Tensor): :math:`(*, N_p)` where :math:`*` means the batch dimension and :math:`N_p` denotes the flattened primal variable values.
            - dual_out (Tensor): :math:`(*, N_d)` where :math:`N_d` denotes the flattened dual variable values.
            - lhs_J (List[Matrix]): :math:`(*, )`.
            - rhs_J (List[Vector]): :math:`(*, )`.
        
        Examples:
            >>> b_batch = torch.randn(sample_num, m, dtype=torch.float64, requires_grad=True)
            >>> input_param = tuple([b_batch])
            >>> primal_out, dual_out, _, _ = Layer(*input_param)
            >>> primal_out.sum().backward()
            >>> print(b_batch.grad)
        """
        f = PyomoLayerFn(concrete_model = self.concrete_model, variables_name = self.variables_name,  \
                        parameters_name = self.parameters_name, parameters_size = self.parameters_size, vars_to_indices = self.vars_to_indices, \
                        grad_parameters_name = self.grad_parameters_name, nlp_init = self.nlp_init, nlp_full = self.nlp_full, \
                        param_order = self.param_order, vars_obj = self.vars_obj,  \
                        variables_index_order = self.variables_index_order, solver = self.solver)

        sol = f(*batch_params)

        return sol
        
def PyomoLayerFn(concrete_model, variables_name, parameters_name, parameters_size, vars_to_indices, grad_parameters_name, nlp_init, nlp_full, param_order, vars_obj, variables_index_order, solver):
    """
    Descriptions: 
        The forward and backward function for the PyomoOptLayer module.
    """
    class PyomoLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *batch_params):
            J = []
            lhs_J = []
            rhs_J = []
            infeasible = []
            # solve over minibatch by just iterating
            for batch in range(batch_params[0].shape[0]):
                params = [p[batch] for p in batch_params]
                with torch.no_grad():
                    params_ = [p.detach().clone().double().numpy() for p in params]
                    for index, p_name in enumerate(parameters_name):
                        params_flat = params_[index].reshape(-1)
                        if not p_name.is_indexed():
                            p_name.fix(params_flat)
                        else:
                            for var, idx in vars_to_indices[p_name].items():
                                var.fix(params_flat[idx])
            
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
                    vars = []
                    for var in variables_name:
                        vars += [var[i].value for i in var]
                    vars = torch.tensor(vars).type_as(params[0]).view(-1).requires_grad_(True)
                 
                # A Jacobian matrix of the (decision variables) with respect to parameters
                dfullvar_dp, lhs_Jac, rhs_Jac, duals = get_sen(concrete_model, grad_parameters_name, param_order, vars_obj)
                if batch == 0:
                    primal_out = torch.zeros((batch_params[0].shape[0], len(vars)))
                    dual_out = torch.zeros((batch_params[0].shape[0], len(duals)))
                primal_out[batch] = vars
                dual_out[batch] = torch.tensor(duals)
                J.append(dfullvar_dp[variables_index_order, :])
                lhs_J.append(lhs_Jac)
                rhs_J.append(rhs_Jac)
            ctx.save_for_backward(torch.tensor(np.array(J), dtype=torch.float64))
            ctx.infeasible = infeasible
            return primal_out, dual_out, lhs_J, rhs_J, infeasible

        @staticmethod
        def backward(ctx, grad_out, dual_dummy = 0, Jac_dummy = 0, rhs_dummy = 0):
            
            batch_size = grad_out.shape[0]
            Jac, = ctx.saved_tensors
            infeasible = ctx.infeasible
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
                size = copy.deepcopy(parameters_size[p.name])
                param_length = len(list(concrete_model.component(p).keys()))
                size.insert(0, full_batch_size)
                grad_reshape.append(grad[:, start:start+param_length].reshape(size))
                start += param_length
            grad_reshape = tuple(grad_reshape)
            return *grad_reshape, None

    return PyomoLayerFnFn.apply