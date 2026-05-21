import numpy as np
import torch
# from mpi4py import MPI
import copy
import warnings
import torch.nn as nn
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomolayers.utilities import Sensitivity
torch.set_default_dtype(torch.float64)
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

# comm = MPI.COMM_WORLD  # Initialize MPI
# rank = comm.Get_rank()  # Process ID
# size = comm.Get_size()  # Total MPI processes

# print(f"Process {rank}/{size}: MPI initialized", flush=True)
size = 1
rank = 0

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
            A list of parameter defined in Pyomo, whose gradients are needed.
        - free_parameters(``List[objective]``, optional)
            A list of parameters that do not require gradient, which a subset of parameters. By setting to ``None``, grad_parameters = parameters.
        - known_parameters(``List[objective]``, optional)
            A list of actual parameters that do not require gradient.
        - ipopt_options (``dict`` , optional)
            Options for Ipopt. Default is None.

    Examples:
        >>> def create_model(A_value, b_value): 
            concrete_model = pyo.ConcreteModel()
            concrete_model.x = pyo.Var(range(n), within=pyo.Reals)
            concrete_model.A = pyo.Var(range(n), within=pyo.Reals)
            concrete_model.b = pyo.Var(range(m), within=pyo.Reals)
            concrete_model.d = pyo.Param(range(m), within=pyo.Reals)
            concrete_model.A[i].fix(A_value[i]) for i in range(n)
            concrete_model.b[i].fix(b_value[i]) for i in range(m)
            return concrete_model
            
        >>> variables = [concrete_model.x]
        >>> parameters = [concrete_model.A, concrete_model.b]
        >>> free_parameters = [concrete_model.b]
        >>> known_parameters = [concrete_model.d]
        >>> Layer = PyomoOptLayer(create_model, variables, parameters, free_parameters, known_parameters, solver = 'ipopt')
    """

    def __init__(self, concrete_model, variables, parameters, free_parameters = None, known_parameters = None, ipopt_options = None):
        super().__init__()
        # takes a pyomo model
        self.concrete_model = concrete_model
        self.solver = pyo.SolverFactory("ipopt")
        if ipopt_options is not None:
            for k, v in ipopt_options.items():
                self.solver.options[k] = v
        if "honor_original_bounds" not in self.solver.options:
            self.solver.options["honor_original_bounds"] = "no"
        elif self.solver.options["honor_original_bounds"].lower() == "yes":
            raise RuntimeWarning("Ipopt configured with `honor_original_bounds == 'yes'` "
                                 "This can often lead to errors in KKT evaluations.")
        if "bound_relax_factor" not in self.solver.options:
            self.solver.options["bound_relax_factor"] = 1e-08

        self.variables = self.expand_complist_to_compdata_list(variables)
        self.parameter = self.expand_complist_to_compdata_list(parameters)
        self.known_parameters = self.expand_complist_to_compdata_list(known_parameters)
        self.free_parameters = self.expand_complist_to_compdata_list(free_parameters)

        self.obj_weight = None
        self.nlp_var = PyomoNLP(concrete_model)
        self.init_pyomo_varorder()
        self.parameters_parent = self.collapse_to_components(parameters)
        self.grad_parameters_parent = self.collapse_to_components(self.grad_parameters)
    # Expand variable list to vardata list
    def expand_complist_to_compdata_list(self, varlist):
        if varlist is None:
            return None
        if isinstance(varlist, (pyo.Component, IndexedComponent_slice)):
            varlist = [varlist]
        vardatalist = []
        for var in varlist:
            if isinstance(var, IndexedComponent_slice):
                vardatalist.extend(var.__iter__())
            elif var.is_indexed():
                vardatalist.extend(var.values())
            else:
                vardatalist.append(var)
        return vardatalist
    def collapse_to_components(self, params):
        comps = []
        seen = set()
        for p in params:
            comp = p.parent_component()
            if id(comp) not in seen:
                comps.append(comp)
                seen.add(id(comp))
        return comps
    def get_nlp_full(self, para):
        model = self.concrete_model
        # Unfix the param variables for Jac/Hessian evaluation
        for param in model.component_data_objects(pyo.Var):
            if any(param is v for v in para):
                param.unfix()

        nlp_full = PyomoNLP(model)
        # Fix the param variables
        for param in model.component_data_objects(pyo.Var):
            if any(param is v for v in para):
                param.fix()
        return nlp_full
    
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
                self.parameters_size[var.name] = [1]
            else: 
                index_set = var.index_set()
                self.parameters_size[var.name] = [len(s) for s in index_set.subsets()]

        self.vars_to_indices = {}
        for p in self.parameter:
            if not p.is_indexed():
                pass
                # self.vars_to_indices[p.name] = 0
            else:
                self.vars_to_indices[p] = pyo.ComponentMap()
                for idx, var in enumerate(p.values()):
                    self.vars_to_indices[p][var] = idx

        self.nlp_full = self.get_nlp_full(para = self.parameter)
            
        # self.pyomo_cons = self.nlp_full.get_pyomo_constraints()
        pyomo_vars_full = self.nlp_full.get_pyomo_variables()
        pyomo_variables_vars = ComponentSet(self.nlp_var.get_pyomo_variables())
        # Use nlp_var = ProjectedExtendedNLP(nlp_full, var_names) in get_sen() function, make sure the var_order matches nlp_full.pyomo_variables()
        self.var_names = []
        self.vars_objs = [] # nlp.extract_submatrix_hessian_lag(vars_objs) follows nlp_full pyomo_variables order

        for var in pyomo_vars_full:
            # actual vars
            if var in pyomo_variables_vars:
                self.var_names.append(var.name)
                self.vars_objs.append(var)

        var_to_idx = pyo.ComponentMap()
        for idx, var in enumerate(self.vars_objs):
            var_to_idx[var] = idx

        # obtain the order difference of nlp_full.get_pyomo_variable and variables given by the user
        self.variables_index_order = []
        for var in self.variables:
            if var.is_indexed():
                for v in var.values():
                    self.variables_index_order.append(var_to_idx[v])
            else:
                self.variables_index_order.append(var_to_idx[var])
        self.sensitivity = Sensitivity(self) 

    def forward(self, *batch_params):
        """
        Descriptions:
            The forward function, which takes in the parameter values and outputs the primal variables. 
            Layer.train(): train mode with gradient computation for backward
            Layer.eval(): test mode without gradient computation
        Args:
            batch_params (Tuple(List[Tensor]), required):
                A batch of parameter values.
        Returns:
            - primal_out: The primal variables that require gradients.
            - lhs_J: Left-hand side KKT matrix.
            - rhs_J: Right-hand side vector.
            - dual_out: The dual variables that do not require gradients.

        Return type:
            - primal_out (Tensor): :math:`(*, N_p)` where :math:`*` means the batch dimension and :math:`N_p` denotes the flattened primal variable values.
            - lhs_J (List[Matrix]): :math:`(*, )`.
            - rhs_J (List[Vector]): :math:`(*, )`.
            - dual_out (Tensor): :math:`(*, N_d)` where :math:`N_d` denotes the flattened dual variable values.
        
        Examples:
            >>> b_batch = torch.randn(sample_num, m, dtype=torch.float64, requires_grad=True)
            >>> input_param = tuple([b_batch])
            >>> primal_out, _, _, dual_out = Layer(*input_param)
            >>> primal_out.sum().backward()
            >>> print(b_batch.grad)
        """
        if self.training:
            f = PyomoLayerFn(concrete_model = self.concrete_model, variables = self.variables,  \
                            parameters = self.parameter, parameters_size = self.parameters_size, vars_to_indices = self.vars_to_indices, \
                            known_parameters = self.known_parameters, grad_parameters = self.grad_parameters, \
                            variables_index_order = self.variables_index_order, solver = self.solver, sensitivity = self.sensitivity, \
                            parameters_parent = self.parameters_parent, grad_parameters_parent = self.grad_parameters_parent)
            sol = f(*batch_params)
        else:
            with torch.no_grad(): 
                f = PyomoLayerFn_eval(concrete_model = self.concrete_model, variables = self.variables,  \
                            parameters = self.parameter, vars_to_indices = self.vars_to_indices, \
                            known_parameters = self.known_parameters, solver = self.solver)
                sol = f(*batch_params)
        return sol
        
def PyomoLayerFn(concrete_model, variables, parameters, parameters_size, vars_to_indices, known_parameters, grad_parameters, \
                 variables_index_order, solver, sensitivity, parameters_parent, grad_parameters_parent):
    """
    Descriptions: 
        The forward and backward functions for the PyomoOptLayer module.
    """
    class PyomoLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *batch_params):
            local_primal_out = []
            local_dual_out = []
            local_J = []
            batch_size = batch_params[0].shape[0]
            # Split batch indices across MPI ranks
            # local_batches = [i for i in range(batch_size) if i % size == rank]  # More balanced distribution 
            local_batches = rank_partition(rank, size, batch_size)
            # Each MPI process solves only its assigned batches
            for batch in local_batches:
                params = [p[batch] for p in batch_params]  # Get batch parameters
                var_values = single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params)
                dfullvar_dp, lhs_Jac, rhs_Jac, duals = sensitivity.get_sen(concrete_model)
                grad = dfullvar_dp[variables_index_order, :]
                local_primal_out.append(var_values)
                local_dual_out.append(duals)
                local_J.append(grad)
            
            if size == 1:
                all_primal_out = [np.array(local_primal_out) for _ in range(size)]
                all_dual_out = [np.array(local_dual_out) for _ in range(size)]
                all_J = [np.array(local_J) for _ in range(size)]
            else:
                all_primal_out = comm.allgather(np.array(local_primal_out))
                all_dual_out = comm.allgather(np.array(local_dual_out))
                all_J = comm.allgather(np.array(local_J))
            
            all_primal_out = [arr for arr in all_primal_out if arr.size > 0]
            all_dual_out = [arr for arr in all_dual_out if arr.size > 0]
            all_J = [arr for arr in all_J if arr.size > 0]

            primal_out = torch.from_numpy(np.concatenate(all_primal_out, axis=0)).requires_grad_(True)
            dual_out = torch.from_numpy(np.concatenate(all_dual_out, axis=0)).requires_grad_(False)
            J = torch.from_numpy(np.concatenate(all_J, axis=0)).requires_grad_(False)
            ctx.save_for_backward(J)
            return primal_out, dual_out, J

        @staticmethod
        def backward(ctx, grad_out, dual_dummy = 0, J_dummy = 0):
            batch_size = grad_out.shape[0]
            Jac, = ctx.saved_tensors
            Jac = Jac.to(dtype=grad_out.dtype)
            grad = (Jac.transpose(1, 2).bmm(grad_out.unsqueeze(-1))).squeeze(-1)
  
            grad_reshape = []
            start = 0
            for _, p in enumerate(parameters_parent):
                if not any(p is gp for gp in grad_parameters_parent):
                    grad_reshape.append(None)
                    continue
                size = copy.deepcopy(parameters_size[p.name])
                param_length = len(p)
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
        The forward and backward functions for the PyomoOptLayer module.
    """
    class PyomoLayerFnFn_eval(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *batch_params):
            batch_size = batch_params[0].shape[0]
            # Split batch indices across MPI ranks
            # local_batches = [i for i in range(batch_size) if i % size == rank]  # More balanced distribution 
            local_batches = rank_partition(rank, size, batch_size)
            # Each MPI process solves only its assigned batches
            local_results = []
            for batch in local_batches:
                params = [p[batch] for p in batch_params]  # Get batch parameters
                var_values = single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params)
                local_results.append(var_values)

            all_results = comm.allgather(np.array(local_results))
            all_results = [arr for arr in all_results if arr.size > 0]
            primal_out = torch.from_numpy(np.concatenate(all_results, axis=0)).requires_grad_(False)

            return primal_out, None, None, None

        @staticmethod
        def backward(ctx, grad_out, dual_dummy = 0, Jac_dummy = 0, rhs_dummy = 0):
            return None

    return PyomoLayerFnFn_eval.apply

def rank_partition(rank, size, batch_size):
    base = batch_size // size
    remainder = batch_size % size

    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return list(range(start, end))

def single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params):
    # solve over minibatch by just iterating
    with torch.no_grad():
        params_ = [p.detach().clone().double().numpy() for p in params]
        params_flat = np.concatenate([p.flatten() for p in params_])
        for index, p_name in enumerate(parameters):
            p_name.fix(params_flat[index])

        end_ = index + 1
        if known_parameters:
            for index, p_name in enumerate(known_parameters):
                p_name[index] = params_flat[index + end_] 
        
        result = solver.solve(concrete_model, tee=False)
        
        if not pyo.check_optimal_termination(result):
            warnings.warn("IPOPT failed to converge!")
        var_values = []
        for var in variables:
            var_values.append(var.value)

    return var_values

