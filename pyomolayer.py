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
    """
    Descriptions:
        A subclass of torch.nn.Module. 

    Args:
        - create_model (``Pyomo model creation function``, required)
            The function of creating the Pyomo model. The input is the parameter value and the output is the concrete Pyomo model.
        - variables_name (``List[str]``, required)
            A list of variable name defined in Pyomo.
        - parameters_name(``List[str]``, required)
            A list of parameter name defined in Pyomo.
        - grad_parameters_name(``List[str]``, optional)
            A list of parameter name defined in Pyomo, a subset of parameters_name. By setting to ``None``, grad_parameters_name = parameters_name.
        - solver (``str`` , optional)
            The optimization solver. The default solver is ``ipopt``.

    Examples:
        >>> def create_model(b_value): 
            concrete_model = pyo.ConcreteModel()
            concrete_model.b = pyo.Var(range(m), within=pyo.Reals)
            concrete_model.b[i].fix(b_value[i]) for i in range(m)
            return concrete_model
            
        >>> variables_name = ["x"]
        >>> parameters_name = ["b"]
        >>> grad_parameters_name = ["b"]

        >>> Layer = PyomoOptLayer(create_model, variables_name, parameters_name, grad_parameters_name, solver = 'ipopt')
    """

    def __init__(self, create_model, variables_name, parameters_name, grad_parameters_name = None, solver = 'ipopt'):
        super().__init__()
        # takes a pyomo model
        self.concrete_model = create_model
        self.solver = pyo.SolverFactory(solver)
        self.variables_name = variables_name
        self.parameters_name = parameters_name
        # self.parameters_size = parameters_size
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
                        variables_name_index = self.variables_name_index,\
                        parameters_name = self.parameters_name, parameters_size = self.parameters_size, \
                        grad_parameters_name = self.grad_parameters_name, solver = self.solver)

        sol = f(*batch_params)

        return sol
        
def PyomoLayerFn(concrete_model, variables_name, variables_name_index, parameters_name, parameters_size, grad_parameters_name, solver):
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
            index = 0
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
                            #scalar
                            if p_size[0] < 1:
                                p_attr.fix(params_[index])
                                p_attr.value = params_[index]
                            else:
                                for i in range(p_size[0]):
                                    p_attr[i].fix(params_[index][i])
                                    p_attr[i].value = params_[index][i]

                    # concrete_model = model(*params_)
                    solver.solve(concrete_model, tee=False)
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
                    
                if batch == 0:
                    primal_out = torch.zeros((batch_params[0].shape[0], len(vars)))
                    dual_out = torch.zeros((batch_params[0].shape[0], len(duals)))
                 
                # A Jacobian matrix of the (decision variables) with respect to parameters
                dfullvar_dp, lhs_Jac, rhs_Jac, variables_nameindex = get_sen(concrete_model, grad_parameters_name, parameters_size, variables_name_index)
                primal_out[batch] = vars
                dual_out[batch] = duals
                J.append(dfullvar_dp[variables_nameindex, :])
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
            for _, p in enumerate(parameters_name):
                if p not in grad_parameters_name:
                    grad_reshape.append(None)
                    continue
                size = copy.deepcopy(parameters_size[p])
                param_length = len(list(concrete_model.component(p).keys()))
                size.insert(0, batch_size)
                grad_reshape.append(grad[:, start:start+param_length].reshape(size))
                start += param_length
            grad_reshape = tuple(grad_reshape)
            return *grad_reshape, None

    return PyomoLayerFnFn.apply