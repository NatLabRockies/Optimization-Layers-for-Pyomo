import numpy as np
import torch
from mpi4py import MPI
import copy
import torch.nn as nn
import sys
sys.path.append('..')  # Add the parent directory to Python's search path
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import copy
from Opt_Layer.utilities import Sensitivity
torch.set_default_dtype(torch.float64)
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
import time

comm = MPI.COMM_WORLD  # Initialize MPI
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total MPI processes

print(f"Process {rank}/{size}: MPI initialized", flush=True)

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

    def __init__(self, concrete_model, variables, parameters, free_parameters = None, known_parameters = None, slacks = None, ipopt_options = None):
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
        self.variables = variables
        self.parameter = parameters
        self.known_parameters = known_parameters
        self.free_parameters = free_parameters
        self.slacks = slacks
        self.obj_weight = None
        self.nlp_var = PyomoNLP(concrete_model)
        self.init_pyomo_varorder()

    def get_nlp_full(self, para):
        model = self.concrete_model
        # Unfix the param variables for Jac/Hessian evaluation
        for param in model.component_objects(pyo.Var):
            if any(param is v for v in para):
                for index in param:
                    param[index].unfix() 
        nlp_full = PyomoNLP(model)
        # Fix the param variables
        for param in model.component_objects(pyo.Var):
            if any(param is v for v in para):
                for index in param:
                    param[index].fix()
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

        if self.slacks:
            self.nlp_full = self.get_nlp_full(para = [*self.parameter, *self.slacks])
            self.nlp_var_slack = self.get_nlp_full(para = [*self.slacks])
            pyomo_variables_vars_slack = ComponentSet(self.nlp_var_slack.get_pyomo_variables())
            self.obj_weight = self.concrete_model.obj_weight.value
        else:
            self.nlp_full = self.get_nlp_full(para = self.parameter)
            self.slacks_num = None
            
        # self.pyomo_cons = self.nlp_full.get_pyomo_constraints()
        pyomo_vars_full = self.nlp_full.get_pyomo_variables()
        pyomo_variables_vars = ComponentSet(self.nlp_var.get_pyomo_variables())
        # Use nlp_var = ProjectedExtendedNLP(nlp_full, var_names) in get_sen() function, make sure the var_order matches nlp_full.pyomo_variables()
        self.var_names = []
        self.vars_objs = [] # nlp.extract_submatrix_hessian_lag(vars_objs) follows nlp_full pyomo_variables order
        self.var_slack_names = []
        self.vars_objs_slacks = []
        for var in pyomo_vars_full:
            # actual vars
            if var in pyomo_variables_vars:
                self.var_names.append(var.name)
                self.vars_objs.append(var)

            if self.slacks and (var in pyomo_variables_vars_slack):
                self.var_slack_names.append(var.name)
                self.vars_objs_slacks.append(var)

        var_to_idx = pyo.ComponentMap()
        for idx, var in enumerate(self.vars_objs):
            var_to_idx[var] = idx

        var_slack_to_idx = pyo.ComponentMap()
        for idx, var in enumerate(self.vars_objs_slacks):
            var_slack_to_idx[var] = idx
        # obtain the order difference of nlp_full.get_pyomo_variable and variables given by the user
        self.variables_index_order = []
        for var in self.variables:
            if var.is_indexed():
                for v in var.values():
                    self.variables_index_order.append(var_to_idx[v])
            else:
                self.variables_index_order.append(var_to_idx[var])
        self.variables_slacks_index_order = []
        self.slacks_index_order = []
        if self.slacks:
            for var in self.variables:
                if var.is_indexed():
                    for v in var.values():
                        self.variables_slacks_index_order.append(var_slack_to_idx[v])
                else:
                    self.variables_slacks_index_order.append(var_slack_to_idx[var])
            
            for var in self.slacks:
                if var.is_indexed():
                    for v in var.values():
                        self.slacks_index_order.append(var_slack_to_idx[v])
                else:
                    self.slacks_index_order.append(var_slack_to_idx[var])
            self.slacks_num = len(self.slacks_index_order)
        self.sensitivity = Sensitivity(self) 

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
                            variables_index_order = self.variables_index_order, variables_slacks_index_order = self.variables_slacks_index_order,  \
                            slacks_index_order = self.slacks_index_order, slacks_num = self.slacks_num, slacks = self.slacks, obj_weight = self.obj_weight, solver = self.solver, sensitivity = self.sensitivity)
            sol = f(*batch_params)
        else:
            with torch.no_grad(): 
                f = PyomoLayerFn_eval(concrete_model = self.concrete_model, variables = self.variables,  \
                            parameters = self.parameter, vars_to_indices = self.vars_to_indices, \
                            known_parameters = self.known_parameters, slacks = self.slacks, obj_weight = self.obj_weight, solver = self.solver)
                sol = f(*batch_params)
        return sol
        
def PyomoLayerFn(concrete_model, variables, parameters, parameters_size, vars_to_indices, known_parameters, grad_parameters, \
                 variables_index_order, variables_slacks_index_order, slacks_index_order, slacks_num, slacks, obj_weight, solver, sensitivity):
    """
    Descriptions: 
        The forward and backward function for the PyomoOptLayer module.
    """
    class PyomoLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *batch_params):
            local_primal_out = []
            local_dual_out = []
            local_J = []
            local_J_slack = []
            local_slack = []
            local_lhs_Jac = []
            local_rhs_Jac = []
            flag = False
            batch_size = batch_params[0].shape[0]
            # Split batch indices across MPI ranks
            # local_batches = [i for i in range(batch_size) if i % size == rank]  # More balanced distribution 
            local_batches = rank_partition(rank, size, batch_size)
            # Each MPI process solves only its assigned batches
            for batch in local_batches:
                params = [p[batch] for p in batch_params]  # Get batch parameters
                # TODO single_solve(concrete_modified_model)
                vars, correction, vars_slack = single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params, slacks, obj_weight)
                # vars = torch.tensor(vars, dtype=batch_params[0].dtype, device=batch_params[0].device).view(-1).requires_grad_(True)
                dfullvar_dp, lhs_Jac, rhs_Jac, duals = sensitivity.get_sen(concrete_model, correction)
                # use the updated model for this sample
                if correction:
                    grad = dfullvar_dp[variables_slacks_index_order, :]
                    grad_slack = dfullvar_dp[slacks_index_order, :]
                    local_J_slack.append(grad_slack)
                    flag = True
                # use the original model for this sample, but the grad_slack should be returned just in case other batch samples need slack_grad
                elif slacks_num is not None:
                    grad = dfullvar_dp[variables_index_order, :]
                    grad_slack = np.zeros((slacks_num, grad.shape[1]))
                    local_J_slack.append(grad_slack)
                # the user does not want slack mode    
                else:
                    grad = dfullvar_dp[variables_index_order, :]
                # print("grad", grad)
                local_primal_out.append(vars)
                local_dual_out.append(duals)
                local_J.append(grad)
                local_slack.append(vars_slack)
                local_lhs_Jac.append(lhs_Jac)
                local_rhs_Jac.append(rhs_Jac)
            
            if size == 1:
                all_primal_out = [np.array(local_primal_out, dtype=np.float32) for _ in range(size)]
                all_dual_out = [np.array(local_dual_out, dtype=np.float32) for _ in range(size)]
                all_J = [np.array(local_J, dtype=np.float32) for _ in range(size)]
            else:
                all_primal_out = comm.allgather(np.array(local_primal_out, dtype=np.float32))
                all_dual_out = comm.allgather(np.array(local_dual_out, dtype=np.float32))
                all_J = comm.allgather(np.array(local_J, dtype=np.float32))

            # all_lhs_Jac = comm.allgather(local_lhs_Jac)
            # all_rhs_Jac = comm.allgather(local_rhs_Jac)
            all_lhs_Jac, all_rhs_Jac = None, None
            slack_out, J_slack = None, None
            # at least one data sample is corrected
            if flag:
                all_J_slack = comm.allgather(np.array(local_J_slack))
                all_J_slack = [arr for arr in all_J_slack if arr.size > 0]
                J_slack = torch.from_numpy(np.concatenate(all_J_slack, axis=0)).requires_grad_(False)
                all_slack = comm.allgather(np.array(local_slack))
                all_slack = [arr for arr in all_slack if arr.size > 0]
                slack_out = torch.from_numpy(np.concatenate(all_slack, axis=0)).requires_grad_(True)
            all_primal_out = [arr for arr in all_primal_out if arr.size > 0]
            all_dual_out = [arr for arr in all_dual_out if arr.size > 0]
            all_J = [arr for arr in all_J if arr.size > 0]

            primal_out = torch.from_numpy(np.concatenate(all_primal_out, axis=0)).requires_grad_(True)
            dual_out = torch.from_numpy(np.concatenate(all_dual_out, axis=0)).requires_grad_(False)
            J = torch.from_numpy(np.concatenate(all_J, axis=0)).requires_grad_(False)
            ctx.save_for_backward(J, J_slack)
            return primal_out, slack_out, dual_out, all_lhs_Jac, all_rhs_Jac, J

        @staticmethod
        def backward(ctx, grad_out, grad_out_slack, dual_dummy = 0, Jac_dummy = 0, rhs_dummy = 0, J_dummy = 0):
            batch_size = grad_out.shape[0]
            Jac, Jac_slack = ctx.saved_tensors
            Jac = Jac.to(dtype=grad_out.dtype)
            grad = (Jac.transpose(1, 2).bmm(grad_out.unsqueeze(-1))).squeeze(-1)
            # print(grad)
            # if Jac_slack is not None:
            #     grad_slack = Jac_slack.mean(dim=1)
            #     grad += grad_slack
            if Jac_slack is not None:
                grad_slack = (Jac_slack.transpose(1, 2).bmm(grad_out_slack.unsqueeze(-1))).squeeze(-1)
                grad += grad_slack
  
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

def PyomoLayerFn_eval(concrete_model, variables, parameters, vars_to_indices, known_parameters, slacks, obj_weight, solver):
    """
    Descriptions: 
        The forward and backward function for the PyomoOptLayer module.
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
            local_results_slack = []
            for batch in local_batches:
                params = [p[batch] for p in batch_params]  # Get batch parameters
                vars, _, vars_slack = single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params, slacks, obj_weight)
                local_results.append(vars)
                if len(vars_slack) > 0:
                    local_results_slack.append(vars_slack)

            all_results = comm.allgather(np.array(local_results))
            all_results = [arr for arr in all_results if arr.size > 0]
            primal_out = torch.from_numpy(np.concatenate(all_results, axis=0)).requires_grad_(False)

            slack_out = None
            if len(vars_slack) > 0:
                all_results_slack = comm.allgather(np.array(local_results_slack))
                all_results_slack = [arr for arr in all_results_slack if arr.size > 0]
                slack_out = torch.from_numpy(np.concatenate(all_results_slack, axis=0)).requires_grad_(False)

            # for batch in range(batch_params[0].shape[0]):
            #     params = [p[batch] for p in batch_params]
            #     vars = single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params)
            #     vars = torch.tensor(vars).type_as(params[0]).view(-1).requires_grad_(False)

            #     if batch == 0:
            #         primal_out = torch.zeros((batch_params[0].shape[0], len(vars)))
            #     primal_out[batch] = vars
            return primal_out, slack_out, None, None, None, None

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

def single_solve(concrete_model, variables, parameters, vars_to_indices, known_parameters, solver, params, slacks, obj_weight):
    # solve over minibatch by just iterating
    with torch.no_grad():
        params_ = [p.detach().clone().double().numpy() for p in params]
        for index, p_name in enumerate(parameters):
            params_flat = params_[index].reshape(-1)
            if not p_name.is_indexed():
                p_name.fix(params_flat.item())
            else:
                for var, idx in vars_to_indices[p_name].items():
                    var.fix(params_flat[idx])
        end_ = index + 1
        if known_parameters:
            for index, p_name in enumerate(known_parameters):
                params_flat = params_[end_ + index].reshape(-1)
                for idx, p_idx in enumerate(p_name.keys()):
                    p_name[p_idx] = params_flat[idx] 
    
        correction = False
        if slacks is not None:
            for slack in slacks:
                slack.fix(0.0)
            concrete_model.obj_weight = obj_weight   
        result = solver.solve(concrete_model, tee=False)
        
        if not pyo.check_optimal_termination(result):
            # print("slack model")
            if slacks is not None:
                for slack in slacks:
                    slack.unfix() 
                correction = True
                # TODO feasibility
                concrete_model.obj_weight = 0  
                result = solver.solve(concrete_model, tee=False)
            else:
                print("IPOPT failed to converge! Please consider the slack model")
                # raise RuntimeWarning("IPOPT failed to converge! Please consider the slack model")
        # else:
        #     print("normal")
        vars = []
        for var in variables:
            vars += [var[i].value for i in var]
        vars_slack = []
        if slacks is not None:
            for var in slacks:
                vars_slack += [var[i].value for i in var]
    return vars, correction, vars_slack

