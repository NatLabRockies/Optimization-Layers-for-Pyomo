import numpy as np
import torch
import copy
import torch.nn as nn
import sys
sys.path.append('..')  # Add the parent directory to Python's search path
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import copy
from Opt_Layer.utilities import get_sen
torch.set_default_dtype(torch.float64)
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

class PyomoOptLayer(nn.Module):
    """
    Descriptions:
        A subclass of torch.nn.Module. 

    Args:
        - create_model (``Pyomo model creation function``, required)
            The function of creating the Pyomo model. The input is the parameter value and the output is the concrete Pyomo model.
        - variables (``List[objective]``, required)
            A list of variable defined in Pyomo.
        - parameters(``List[objective]``, required)
            A list of parameter defined in Pyomo.
        - free_parameters(``List[objective]``, optional)
            A list of parameters that do not require gradient, which a subset of parameters. By setting to ``None``, grad_parameters = parameters.
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
            
        >>> variables = [concrete_model.x]
        >>> parameters = [concrete_model.A, concrete_model.b]
        >>> grad_parameters = [concrete_model.b]

        >>> Layer = PyomoOptLayer(create_model, variables, parameters_name, free_parameters_name, solver = 'ipopt')
    """

    def __init__(self, concrete_model, variables, parameters, free_parameters = None, known_parameters = None, solver = 'ipopt'):
        super().__init__()
        # takes a pyomo model
        self.concrete_model = concrete_model
        self.solver = pyo.SolverFactory(solver)
        self.variables = variables
        self.parameter = parameters
        self.known_parameters = known_parameters
        self.free_parameters = free_parameters
        self.nlp_var = PyomoNLP(concrete_model)
        self.init_pyomo_varorder()
        self.training = True
        
    def get_nlp_full(self):
        model = self.concrete_model
        # Unfix the param variables for Jac/Hessian evaluation
        for param in model.component_objects(pyo.Var):
            if param in self.grad_parameters:
                for index in param:
                    param[index].unfix() 
        self.nlp_full = PyomoNLP(model)
        # Fix the param variables
        for param in model.component_objects(pyo.Var):
            if param in self.grad_parameters:
                for index in param:
                    param[index].fix()
    
    def init_pyomo_varorder(self):
        if self.free_parameters:
            self.grad_parameters = [item for item in self.parameter if item not in self.free_parameters] 
        else:
            self.grad_parameters = self.parameter
        assert len(self.grad_parameters) > 0 

        self.parameters_size = {}
        for var in self.concrete_model.component_objects(pyo.Var, active=True):
            # Scaler
            if not var.is_indexed():
                self.parameters_size[var.name] = [0]
            else: 
                index_set = var.index_set()
                self.parameters_size[var.name] = [len(s) for s in index_set.subsets()]

        self.vars_to_indices = {}
        for p in self.parameter:
            if not p.is_indexed():
                self.vars_to_indices[p.name] = 0
            else:
                self.vars_to_indices[p] = pyo.ComponentMap()
                for idx, var in enumerate(p.values()):
                    self.vars_to_indices[p][var] = idx

        self.param_order = []
        for p_name in self.grad_parameters:
            for i in p_name.index_set():
                self.param_order.append(str(p_name[i]))
        
        self.get_nlp_full()
        self.pyomo_cons = self.nlp_full.get_pyomo_constraints()
        self.pyomo_vars_full = self.nlp_full.get_pyomo_variables()
        pyomo_variables_vars = self.nlp_var.get_pyomo_variables()
        pyomo_variables_vars_name = {var.name for var in pyomo_variables_vars} 
        # Obtain the variables index based on nlp_full.get_pyomo_variable, 
        variables_name_full = set()
        # Use nlp_var = ProjectedExtendedNLP(nlp_full, vars_order) in get_sen() function, make sure the var_order matches nlp_full.pyomo_variables()
        self.vars_order = []
        self.vars_obj = [] # nlp.extract_submatrix_hessian_lag(vars_obj) follows nlp_full pyomo_variables order
        for var in self.pyomo_vars_full:
            # actual vars
            if var.name in pyomo_variables_vars_name:
                self.vars_order.append(var.name)
                var_name = var.parent_component().name
                if var_name not in variables_name_full:
                    variables_name_full.add(var_name)
                    v = getattr(self.concrete_model, var_name)
                    self.vars_obj.append(v)
        
        # obtain the order difference of nlp_full.get_pyomo_variable and variables given by the user
        vars_name_index = []
        for var_index in self.variables:
            vars_name_index += [var_index[i].name for i in var_index.keys()]
        self.variables_index_order = []
        for var in vars_name_index:
            self.variables_index_order.append(self.vars_order.index(var))

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
        if self.training:
            f = PyomoLayerFn(concrete_model = self.concrete_model, variables = self.variables,  \
                            parameters = self.parameter, parameters_size = self.parameters_size, vars_to_indices = self.vars_to_indices, \
                            known_parameters = self.known_parameters, grad_parameters = self.grad_parameters, \
                            param_order = self.param_order, vars_obj = self.vars_obj, vars_order = self.vars_order,  \
                            variables_index_order = self.variables_index_order, nlp_full = self.nlp_full, nlp_var = self.nlp_var, \
                            pyomo_cons = self.pyomo_cons, pyomo_vars_full = self.pyomo_vars_full, solver = self.solver)

            sol = f(*batch_params)
        else:
            with torch.no_grad(): 
                f = PyomoLayerFn_eval(concrete_model = self.concrete_model, variables = self.variables,  \
                            parameters = self.parameter, vars_to_indices = self.vars_to_indices, \
                            known_parameters = self.known_parameters, solver = self.solver)
                sol = f(*batch_params)
        return sol
        
def PyomoLayerFn(concrete_model, variables, parameters, parameters_size, vars_to_indices, known_parameters, grad_parameters, \
                 param_order, vars_obj, vars_order, variables_index_order, nlp_full, nlp_var, pyomo_cons, pyomo_vars_full, solver):
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
            # solve over minibatch by just iterating
            for batch in range(batch_params[0].shape[0]):
                params = [p[batch] for p in batch_params]
                with torch.no_grad():
                    params_ = [p.detach().clone().double().numpy() for p in params]
                    for index, p_name in enumerate(parameters):
                        params_flat = params_[index].reshape(-1)
                        if not p_name.is_indexed():
                            p_name.fix(params_flat)
                        else:
                            for var, idx in vars_to_indices[p_name].items():
                                var.fix(params_flat[idx])
                    end_ = index + 1
                    if known_parameters:
                        for index, p_name in enumerate(known_parameters):
                            params_flat = params_[end_ + index].reshape(-1)
                            for idx, p_idx in enumerate(p_name.keys()):
                                p_name[p_idx] = params_flat[idx] 
                            
                    # concrete_model = model(*params_)
                    solver.solve(concrete_model, tee=False)
                    # TODO mini slack -> training obj is 0 -> 
                    # if not pyo.check_optimal_termination(results):
                    #     raise RuntimeWarning("IPOPT failed to converge! Watch out!")
                    # should add a post-processing function to help project to the closest decision variables. 
                    # decision variables
                    #concrete_model.pprint
                    # concrete_model.display()
                    # concrete_model.obj()
                    vars = []
                    for var in variables:
                        vars += [var[i].value for i in var]
                    vars = torch.tensor(vars).type_as(params[0]).view(-1).requires_grad_(True)
                 
                # A Jacobian matrix of the (decision variables) with respect to parameters
                dfullvar_dp, lhs_Jac, rhs_Jac, duals = get_sen(concrete_model, grad_parameters, param_order, vars_obj, vars_order, nlp_full, nlp_var, pyomo_cons, pyomo_vars_full)
                if batch == 0:
                    primal_out = torch.zeros((batch_params[0].shape[0], len(vars)))
                    dual_out = torch.zeros((batch_params[0].shape[0], len(duals)))
                primal_out[batch] = vars
                dual_out[batch] = torch.tensor(duals)
                J.append(dfullvar_dp[variables_index_order, :])
                lhs_J.append(lhs_Jac)
                rhs_J.append(rhs_Jac)
            ctx.save_for_backward(torch.tensor(np.array(J), dtype=torch.float64))
            return primal_out, dual_out, lhs_J, rhs_J

        @staticmethod
        def backward(ctx, grad_out, dual_dummy = 0, Jac_dummy = 0, rhs_dummy = 0):
            
            batch_size = grad_out.shape[0]
            Jac, = ctx.saved_tensors
            grad = (Jac.transpose(1, 2).bmm(grad_out.unsqueeze(-1))).squeeze(-1)
            grad_reshape = []
            start = 0
            for _, p in enumerate(parameters):
                if p not in grad_parameters:
                    grad_reshape.append(None)
                    continue
                size = copy.deepcopy(parameters_size[p.name])
                param_length = len(list(concrete_model.component(p).keys()))
                size.insert(0, batch_size)
                grad_reshape.append(grad[:, start:start+param_length].reshape(size))
                start += param_length
            if known_parameters:
                grad_reshape.extend([None] * len(known_parameters))
            grad_reshape = tuple(grad_reshape)
            return *grad_reshape, None

    return PyomoLayerFnFn.apply

def PyomoLayerFn_eval(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver):
    """
    Descriptions: 
        The forward and backward function for the PyomoOptLayer module.
    """
    class PyomoLayerFnFn_eval(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *batch_params):
            # solve over minibatch by just iterating
            for batch in range(batch_params[0].shape[0]):
                params = [p[batch] for p in batch_params]
                with torch.no_grad():
                    params_ = [p.detach().clone().double().numpy() for p in params]
                    for index, p_name in enumerate(parameters):
                        params_flat = params_[index].reshape(-1)
                        if not p_name.is_indexed():
                            p_name.fix(params_flat)
                        else:
                            for var, idx in vars_to_indices[p_name].items():
                                var.fix(params_flat[idx])
                    end_ = index + 1
                    if known_parameters:
                        for index, p_name in enumerate(known_parameters):
                            params_flat = params_[end_ + index].reshape(-1)
                            for idx, p_idx in enumerate(p_name.keys()):
                                p_name[p_idx] = params_flat[idx] 
                            
                    # concrete_model = model(*params_)
                    solver.solve(concrete_model, tee=False)
                    # TODO mini slack -> training obj is 0 -> 
                    # if not pyo.check_optimal_termination(results):
                    #     raise RuntimeWarning("IPOPT failed to converge! Watch out!")
                    # should add a post-processing function to help project to the closest decision variables. 
                    # decision variables
                    #concrete_model.pprint
                    # concrete_model.display()
                    # concrete_model.obj()
                    vars = []
                    for var in variables:
                        vars += [var[i].value for i in var]
                    vars = torch.tensor(vars).type_as(params[0]).view(-1).requires_grad_(True)

                if batch == 0:
                    primal_out = torch.zeros((batch_params[0].shape[0], len(vars)))
                primal_out[batch] = vars
            return primal_out, None, None, None

        @staticmethod
        def backward(ctx, grad_out, dual_dummy = 0, Jac_dummy = 0, rhs_dummy = 0):
            return None

    return PyomoLayerFnFn_eval.apply