import numpy as np
import torch
import copy
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedExtendedNLP
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.common.timing import HierarchicalTimer

torch.set_default_dtype(torch.float64)

class InteriorPointInterface(object):
    """
    Descriptions:
        A modified class based on ``pyomo.contrib.interior_point.interface`` to obtain the left hand side matrix for KKT optimality condition.

    Args: 
        nlp: A pyomo nonlinear program interface (``pyomo.contrib.pynumero.interfaces.pyomo_nlp.PyomoNLP``) instance of the Pyomo model (``pyomo.environ.ConcreteModel``), required.
            Initialize the primal and dual bounds based on nlp for the ``InteriorPointInterface`` module.

    Raises:
        ExceptionName: Description of the error raised, if any.

    Examples:
        >>> concrete_model = pyo.ConcreteModel()
        >>> nlp = PyomoNLP(concrete_model)
        >>> IPOPT_module = InteriorPointInterface(nlp)
    """

    def __init__(self, nlp):

        self._nlp = nlp
        self._slacks = self._nlp.evaluate_ineq_constraints()

        # set the init_duals_primals_lb/ub from ipopt_zL_out, ipopt_zU_out if available
        # need to compress them as well and initialize the duals_primals_lb/ub
        (self._init_duals_primals_lb, self._init_duals_primals_ub) = (self._get_full_duals_primals_bounds())
        self._init_duals_primals_lb[np.isneginf(self._nlp.primals_lb())] = 0
        self._init_duals_primals_ub[np.isinf(self._nlp.primals_ub())] = 0
        self._duals_primals_lb = self._init_duals_primals_lb.copy()
        self._duals_primals_ub = self._init_duals_primals_ub.copy()

        # set the init_duals_slacks_lb/ub from the init_duals_ineq
        # need to be compressed and set according to their sign
        # (-) value indicates it the upper is active, while (+) indicates
        # that lower is active
        self._duals_slacks_lb = -self._nlp.get_duals_ineq().copy()
        self._duals_slacks_lb[self._duals_slacks_lb < 0] = 0
        self._duals_slacks_ub = -self._nlp.get_duals_ineq().copy()
        self._duals_slacks_ub[self._duals_slacks_ub > 0] = 0
        self._duals_slacks_ub *= -1.0


    def n_primals(self):
        return self._nlp.n_primals()

    def n_eq_constraints(self):
        return self._nlp.n_eq_constraints()

    def n_ineq_constraints(self):
        return self._nlp.n_ineq_constraints()

    def get_primals(self):
        return self._nlp.get_primals()

    def primals_lb(self):
        return self._nlp.primals_lb()

    def primals_ub(self):
        return self._nlp.primals_ub()

    def ineq_lb(self):
        return self._nlp.ineq_lb()

    def ineq_ub(self):
        return self._nlp.ineq_ub()

    def evaluate_primal_dual_kkt_matrix(self, timer=None):
        """
        Descriptions: 
            Evaluate the KKT matrix based on the first-order optimality condition.

        Returns:
            The left hand side KKT matrix.

        Return type:
            BlockMatrix

        """
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('eval hess')
        hessian = self._nlp.evaluate_hessian_lag()
        timer.stop('eval hess')
        timer.start('eval jac')
        jac_eq = self._nlp.evaluate_jacobian_eq()
        jac_ineq = self._nlp.evaluate_jacobian_ineq()
        timer.stop('eval jac')

        duals_primals_lb = self._duals_primals_lb
        duals_primals_ub = self._duals_primals_ub
        duals_slacks_lb = self._duals_slacks_lb
        duals_slacks_ub = self._duals_slacks_ub
        primals = self._nlp.get_primals()
        timer.start('hess block')
        data = duals_primals_lb / (primals - self._nlp.primals_lb()) + duals_primals_ub / (self._nlp.primals_ub() - primals)
        n = self._nlp.n_primals()
        indices = np.arange(n)
        hess_block = coo_matrix((data, (indices, indices)), shape=(n, n))
        hess_block += hessian
        timer.stop('hess block')

        timer.start('slack block')
        data = duals_slacks_lb / (
            self._slacks - self._nlp.ineq_lb()
        ) + duals_slacks_ub / (self._nlp.ineq_ub() - self._slacks)
        n = self._nlp.n_ineq_constraints()
        indices = np.arange(n)
        slack_block = coo_matrix((data, (indices, indices)), shape=(n, n))
        timer.stop('slack block')
        
        timer.start('set block')
        kkt = BlockMatrix(4, 4)
        kkt.set_block(0, 0, hess_block)
        kkt.set_block(1, 1, slack_block)
        kkt.set_block(2, 0, jac_eq)
        kkt.set_block(0, 2, jac_eq.transpose())
        kkt.set_block(3, 0, jac_ineq)
        kkt.set_block(0, 3, jac_ineq.transpose())
        kkt.set_block(3, 1, -identity(self._nlp.n_ineq_constraints(), format='coo'))
        kkt.set_block(1, 3, -identity(self._nlp.n_ineq_constraints(), format='coo'))
        timer.stop('set block')
        return kkt

    def evaluate_jacobian_eq(self):
        return self._nlp.evaluate_jacobian_eq()

    def evaluate_jacobian_ineq(self):
        return self._nlp.evaluate_jacobian_ineq()

    def _get_full_duals_primals_bounds(self):
        """
        Descriptions: 
            Obtain the primal and dual bounds from the Pyomo model.

        Args:
            ipopt_zL_out and ipopt_zU_out are attributes used to retrieve the dual variables associated with the lower and upper bounds of variables. 

        Returns:
            full_duals_primals_lb, full_duals_primals_ub

        Examples:
            >>> model = pyo.ConcreteModel()
            >>> model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            >>> model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            >>> model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            
        """
        full_duals_primals_lb = None
        full_duals_primals_ub = None
        # Check in case _nlp was constructed as an AmplNLP (from an nl file)
        if hasattr(self._nlp, 'pyomo_model') and hasattr(self._nlp, 'get_pyomo_variables'):
            pyomo_model = self._nlp.pyomo_model()
            pyomo_variables = self._nlp.get_pyomo_variables()
            if not hasattr(pyomo_model, 'dual'):
                raise Exception("Unknown duals. Please define dual in Pyomo model, i.e., model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)")
            if hasattr(pyomo_model, 'ipopt_zL_out'):
                zL_suffix = pyomo_model.ipopt_zL_out
                full_duals_primals_lb = np.empty(self._nlp.n_primals())
                for i, v in enumerate(pyomo_variables):
                    if v in zL_suffix:
                        full_duals_primals_lb[i] = zL_suffix[v]
            if hasattr(pyomo_model, 'ipopt_zU_out'):
                zU_suffix = pyomo_model.ipopt_zU_out
                full_duals_primals_ub = np.empty(self._nlp.n_primals())
                for i, v in enumerate(pyomo_variables):
                    if v in zU_suffix:
                        full_duals_primals_ub[i] = zU_suffix[v]
        if full_duals_primals_lb is None or full_duals_primals_ub is None:
            raise Exception("Unknown duals_primals_lb/ub. Please define ipopt_zL_out and ipopt_zU_out in Pyomo model, \
                            i.e., model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT) \
                            and model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)")

        return full_duals_primals_lb, full_duals_primals_ub

    def pyomo_model(self):
        return self._nlp.pyomo_model()

    def get_pyomo_variables(self):
        return self._nlp.get_pyomo_variables()

    def get_pyomo_constraints(self):
        return self._nlp.get_pyomo_constraints()

def get_sen(concrete_model, parameters_name, parameters_size, variables_name_index): 
    """
    Descriptions: 
        Obtain the sensitivity matrix of decision variables with respect to the parameters.
    
    Args:
        - concrete_model (``Pyomo model``, required)
            The concrete instance has been solved by IPOPT, and the optimal values of primal and dual variables have been obtained.
        - vars_obj (``List[objective]``, required)
            A list of variable object obatined by v_obj = getattr(concrete_model, v_name).
        - parameters_name(``List[str]``, required)
            A list of parameter name defined in Pyomo.
        - parameters_size (``Dict[str: List[int]]``, required)
            A dict of parameter name (key) and the size (value). 
    
    Returns:
        The sensitivity matrix, left hand side of KKT matrix, right hand side of vector.
    
    Return type:
        Matrix.
    
    Examples:
        >>> concrete_model = pyo.ConcreteModel()
        >>> solver.solve(concrete_model, tee=False)
        >>> variables_name = ["x"]
        >>> variables_size = {'x':[n]}
        >>> parameters_name = ["Psqrt", "q", "A", "b"]
        >>> parameters_size = {'Psqrt':[n, n], 'q':[n], "A" : [m, n], "b" : [m]}
        >>> dvar_dp, lhs_Jac, rhs_Jac = get_sen(concrete_model, variables_name, vars_obj, variables_size, parameters_name, parameters_size)
    """
    nlp = PyomoNLP(concrete_model)
    variables_name_full = []
    variables_nameindex_full = []
    vars_obj = []
    pyomo_variables = nlp.get_pyomo_variables()
    for var in pyomo_variables:
        variables_nameindex_full.append(var.name)
        var_name = var.parent_component().name
        if var_name not in variables_name_full:
            variables_name_full.append(var_name)
            v = getattr(concrete_model, var_name)
            vars_obj.append(v)
    
    variables_nameindex = []
    for var in variables_name_index:
        variables_nameindex.append(variables_nameindex_full.index(var))

    ineq_duals = []
    duals = []
    for constraint in nlp.get_pyomo_constraints():
        if constraint.active:
            duals.append(concrete_model.dual[constraint])

    for constraint in nlp.get_pyomo_inequality_constraints():
        if constraint.active:
            ineq_duals.append(concrete_model.dual[constraint])
    ineq_duals = np.array(ineq_duals, dtype = np.float64)
    duals = - np.array(duals, dtype = np.float64)
    nlp.set_duals(duals)
    IPOPT = InteriorPointInterface(nlp) 
    kkt = IPOPT.evaluate_primal_dual_kkt_matrix()

    # Unfix the param variables for Jac/Hessian evaluation
    for param in concrete_model.component_objects(pyo.Var):
        if param.name in parameters_name:
            param.unfix()

    param_order = []
    params_obj = []
    for p_name in parameters_name:
        p_attr = getattr(concrete_model, p_name)
        params_obj.append(p_attr)
        size = copy.deepcopy(parameters_size[p_name])
        if len(size) == 3:
            for i in range(size[0]):
                for j in range(size[1]):
                    for k in range(size[2]):
                        param_order.append(str(p_attr[i, j, k]))
        elif len(size) == 2:
            for i in range(size[0]):
                for j in range(size[1]):
                    param_order.append(str(p_attr[i, j]))
        elif len(size) == 1:
            for i in range(size[0]):
                param_order.append(str(p_attr[i]))

    nlp = PyomoNLP(concrete_model)

    # Obtain the duals
    ineq_duals = []
    duals = []
    for constraint in nlp.get_pyomo_constraints():
        if constraint.active:
            duals.append(concrete_model.dual[constraint])
            
    for constraint in nlp.get_pyomo_inequality_constraints():
        if constraint.active:
            ineq_duals.append(concrete_model.dual[constraint])
            
    ineq_duals = - np.array(ineq_duals)
    duals = - np.array(duals)
    nlp.set_duals(duals)

    nlp_params = ProjectedExtendedNLP(nlp, param_order)

    # Build the right hand side vector
    Hessian = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=vars_obj, pyomo_variables_cols=params_obj)
    H_ineq = nlp_params.evaluate_jacobian_ineq()        
    H_eq = nlp_params.evaluate_jacobian_eq()
    
    kkt_rhs = BlockMatrix(4, 1)
    kkt_rhs.set_block(0, 0, Hessian)
    kkt_rhs.set_block(1, 0, coo_matrix((nlp.n_ineq_constraints(), Hessian.shape[1])))
    kkt_rhs.set_block(2, 0, H_eq)
    kkt_rhs.set_block(3, 0, H_ineq)
    ds = spsolve(kkt.tocsc(), -kkt_rhs.tocsc())
    dfullvar_dp = np.array(ds.todense())[:len(variables_nameindex_full), :]
    return dfullvar_dp, kkt.tocsc().todense(), kkt_rhs.tocsc().todense(), variables_nameindex