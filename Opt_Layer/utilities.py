import numpy as np
import torch
import copy
from scipy.sparse import coo_matrix, identity, diags
from scipy.sparse.linalg import spsolve, lsqr 
from Opt_Layer.parapint.interior_point import IPOptions, ip_solve_optimal
from Opt_Layer.parapint.mumps_interface import MumpsInterface
from Opt_Layer.parapint.scipy_interface import ScipyInterface

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedExtendedNLP
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.common.timing import HierarchicalTimer
Debug = False
is_first_call = True
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

    def __init__(self, concreate_model, nlp, nlp_full, pyomo_vars_full):

        self.concreate_model = concreate_model
        self._nlp = nlp
        self._nlp_full = nlp_full
        self.pyomo_vars_full = pyomo_vars_full
        self._slacks = self._nlp.evaluate_ineq_constraints()

        # set the init_duals_primals_lb/ub from ipopt_zL_out, ipopt_zU_out if available
        # need to compress them as well and initialize the duals_primals_lb/ub
        # only have the duals_primals_lb and ub for the vars.
        (self._init_duals_primals_lb, self._init_duals_primals_ub) = (self._get_full_duals_primals_bounds())
        self._init_duals_primals_lb[np.isneginf(self._nlp.primals_lb())] = 0
        self._init_duals_primals_ub[np.isinf(self._nlp.primals_ub())] = 0
        self._duals_primals_lb = self._init_duals_primals_lb.copy()
        self._duals_primals_ub = self._init_duals_primals_ub.copy()

        # set the init_duals_slacks_lb/ub from the init_duals_ineq
        # need to be compressed and set according to their sign
        # (-) value indicates it the upper is active, while (+) indicates
        # that lower is active
        # the nlp does not have get_duals_ineq() property.
        self._duals_slacks_lb = -self._nlp_full.get_duals_ineq().copy()
        self._duals_slacks_lb[self._duals_slacks_lb < 0] = 0
        self._duals_slacks_ub = -self._nlp_full.get_duals_ineq().copy()
        self._duals_slacks_ub[self._duals_slacks_ub > 0] = 0
        self._duals_slacks_ub *= -1.0

        self.bounds_relaxation_factor = 1e-8

    def get_bounds_relaxation_factor(self) -> float:
        return self.bounds_relaxation_factor

    def set_bounds_relaxation_factor(self, val: float):
        self.bounds_relaxation_factor = val

    def n_primals(self):
        return self._nlp.n_primals()

    def n_eq_constraints(self):
        return self._nlp.n_eq_constraints()

    def n_ineq_constraints(self):
        return self._nlp.n_ineq_constraints()

    def get_primals(self):
        return self._nlp.get_primals()

    def primals_lb(self):
        lbs = self._nlp.primals_lb()
        if self.bounds_relaxation_factor == 0:
            return lbs
        eye = np.ones(lbs.size)
        lbs_mod = lbs - self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(lbs)]), axis=0)
        return lbs_mod

    def primals_ub(self):
        ubs = self._nlp.primals_ub()
        if self.bounds_relaxation_factor == 0:
            return ubs
        eye = np.ones(ubs.size)
        ubs_mod = ubs + self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(ubs)]), axis=0)
        return ubs_mod

    def ineq_lb(self):
        lbs = self._nlp.ineq_lb()
        if self.bounds_relaxation_factor == 0:
            return lbs
        eye = np.ones(lbs.size)
        lbs_mod = lbs - self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(lbs)]), axis=0)
        return lbs_mod

    def ineq_ub(self):
        ubs = self._nlp.ineq_ub()
        if self.bounds_relaxation_factor == 0:
            return ubs
        eye = np.ones(ubs.size)
        ubs_mod = ubs + self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(ubs)]), axis=0)
        return ubs_mod

    def set_barrier_parameter(self, barrier):
        self._barrier = barrier

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
        hess_block = self._nlp.evaluate_hessian_lag()
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
        # if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        #     print(f"{duals_primals_lb=}")
        #     print(f"{duals_primals_ub=}")
        #     print(f"{primals=}")
        #     print(f"{self._nlp.primals_ub()=}")
        #     print(f"{self._nlp.primals_lb()=}")
        #     print("data1", data)
        n = self._nlp.n_primals()
        indices = np.arange(n)
        hess_block.row = np.concatenate([hess_block.row, indices])
        hess_block.col = np.concatenate([hess_block.col, indices])
        hess_block.data = np.concatenate([hess_block.data, data])
        timer.stop('hess block')

        timer.start('slack block')
        data = duals_slacks_lb / (
            self._slacks - self._nlp_full.ineq_lb()
        ) + duals_slacks_ub / (self._nlp_full.ineq_ub() - self._slacks)
        n = self._nlp.n_ineq_constraints()
        indices = np.arange(n)
        slack_block = coo_matrix((data, (indices, indices)), shape=(n, n))
        timer.stop('slack block')
        
        timer.start('regularization block')
        eq_reg_blk = identity(self._nlp.n_eq_constraints(), format='coo')
        eq_reg_blk.data.fill(0)
        ineq_reg_blk = identity(self._nlp.n_ineq_constraints(), format='coo')
        ineq_reg_blk.data.fill(0)
        timer.stop('regularization block')

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
        kkt.set_block(2, 2, eq_reg_blk)
        kkt.set_block(3, 3, ineq_reg_blk)
        timer.stop('set block')
        return kkt

    def evaluate_jacobian_eq(self):
        return self._nlp.evaluate_jacobian_eq()

    def evaluate_jacobian_ineq(self):
        return self._nlp.evaluate_jacobian_ineq()

    def regularize_equality_gradient(self, kkt, coef, copy_kkt=True):
        # Not technically regularizing the equality gradient ...
        # Replace this with a regularize_diagonal_block function?
        # Then call with kkt matrix and the value of the perturbation?

        # Use a constant perturbation to regularize the equality constraint
        # gradient
        if copy_kkt:
            kkt = kkt.copy()
        reg_coef = coef
        eq_ptb = (reg_coef *
                  identity(self._nlp.n_eq_constraints(),
                                        format='coo'))
        ineq_ptb = (reg_coef *
                    identity(self._nlp.n_ineq_constraints(),
                                          format='coo'))

        kkt.set_block(2, 2, eq_ptb)
        kkt.set_block(3, 3, ineq_ptb)
        return kkt
    
    def regularize_hessian(self, kkt, coef, copy_kkt=True):
        if copy_kkt:
            kkt = kkt.copy()

        hess = kkt.get_block(0, 0)
        ptb = coef * identity(self._nlp.n_primals(), format='coo')
        hess += ptb
        kkt.set_block(0, 0, hess)
        return kkt
    
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
        if hasattr(self._nlp_full, 'pyomo_model') and hasattr(self._nlp_full, 'get_pyomo_variables'):
            # pyomo_model = self._nlp_full.pyomo_model()
            pyomo_model = self.concreate_model
            pyomo_variables = self.pyomo_vars_full
            # pyomo_variables = self._nlp_full.get_pyomo_variables()
            # pyomo_variables contains vars and params, but onlt set the duals for vars (self._nlp.n_primals())
            if not hasattr(pyomo_model, 'dual'):
                raise Exception("Unknown duals. Please define dual in Pyomo model, i.e., model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)")
            if hasattr(pyomo_model, 'ipopt_zL_out'):
                zL_suffix = pyomo_model.ipopt_zL_out
                full_duals_primals_lb = np.zeros(self._nlp.n_primals())
                k = 0
                for _, v in enumerate(pyomo_variables):
                    if v in zL_suffix:
                        full_duals_primals_lb[k] = zL_suffix[v]
                        k += 1
            if hasattr(pyomo_model, 'ipopt_zU_out'):
                zU_suffix = pyomo_model.ipopt_zU_out
                full_duals_primals_ub = np.zeros(self._nlp.n_primals())
                k = 0
                for _, v in enumerate(pyomo_variables):
                    if v in zU_suffix:
                        full_duals_primals_ub[k] = zU_suffix[v]
                        k += 1
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

def get_sen(concrete_model, parameters_name, param_order, vars_obj, vars_order, nlp_full, nlp_var, pyomo_cons, pyomo_vars_full): 
    """
    Descriptions: 
        Obtain the sensitivity matrix of decision variables with respect to the parameters.
    
    Args:
        - concrete_model (``Pyomo model``, required)
            The concrete instance has been solved by IPOPT, and the optimal values of primal and dual variables have been obtained.
        - parameters_name(``List[str]``, required)
            A list of parameter name defined in Pyomo.
        - variables_name_index(``List[str]``, required)
            A list of indexed variable names
    Returns:
        The sensitivity matrix, left hand side of KKT matrix, right hand side of vector, and the order of variables
    
    Return type:
        Matrix.
    
    Examples:
        >>> concrete_model = pyo.ConcreteModel()
        >>> solver.solve(concrete_model, tee=False)
        >>> parameters_name = ["Psqrt", "q", "A", "b"]
        >>> variables_name_index = ["x[0]", "x[1]", "x[3]", "y"]
        >>> param_order = ["Psqrt[0, 0]",... "q[0]"..., "A[0, 0]"..., "b[0]"...]
        >>> vars_obj = [x, y] based on nlp.get_pyomo_variables()
        >>> dvar_dp, lhs_Jac, rhs_Jac, variables_index_order = get_sen(concrete_model, parameters_name, variables_name_index)
    """
    # Unfix the param variables for Jac/Hessian evaluation
    for param in concrete_model.component_objects(pyo.Var):
        if param in parameters_name:
            for index in param:
                param[index].unfix() 

    primals = []
    for var in pyomo_vars_full:
        primals.append(var.value)
    nlp_full.set_primals(np.array(primals))

    duals = []
    for constraint in pyomo_cons:
        duals.append(concrete_model.dual[constraint])
    duals = - np.array(duals, dtype = np.float64)
    nlp_full.set_duals(duals)

    nlp_vars = ProjectedExtendedNLP(nlp_full, vars_order)

    IPOPT = InteriorPointInterface(concrete_model, nlp_vars, nlp_full, pyomo_vars_full) 
    kkt = IPOPT.evaluate_primal_dual_kkt_matrix()

    nlp_params = ProjectedExtendedNLP(nlp_full, param_order)

    # Build the right hand side vector
    Hessian = nlp_full.extract_submatrix_hessian_lag(pyomo_variables_rows=vars_obj, pyomo_variables_cols=parameters_name)
    H_ineq = nlp_params.evaluate_jacobian_ineq()        
    H_eq = nlp_params.evaluate_jacobian_eq()
    
    kkt_rhs = BlockMatrix(4, 1)
    kkt_rhs.set_block(0, 0, Hessian)
    kkt_rhs.set_block(1, 0, coo_matrix((nlp_full.n_ineq_constraints(), Hessian.shape[1])))
    kkt_rhs.set_block(2, 0, H_eq)
    kkt_rhs.set_block(3, 0, H_ineq)

    if np.any(np.isnan(kkt.tocsc().todense())) or np.any(np.isinf(kkt.tocsc().todense())):
        raise RuntimeError("This is a runtime error")
    global is_first_call
    global IPsolver
    try:
        if is_first_call:
            IPsolver = IPOptions()
            IPsolver.use_inertia_correction = True
            # IPsolver.linalg.solver = MumpsInterface()
            IPsolver.linalg.solver = ScipyInterface(compute_inertia=IPsolver.use_inertia_correction)
            ds, IPsolver = ip_solve_optimal(interface=IPOPT, kkt = kkt, rhs = -kkt_rhs.toarray(), 
                                    is_first_call = is_first_call, IPoptions = IPsolver)
            is_first_call = False
        else:     
            ds, _ = ip_solve_optimal(interface=IPOPT, kkt = kkt, rhs = -kkt_rhs.toarray(), 
                                        is_first_call = is_first_call, IPoptions = IPsolver)
    except:
        rhs = -kkt_rhs.toarray()
        ds = np.zeros_like(rhs)
        
    dfullvar_dp = ds[:len(vars_order), :]
    # ds = spsolve(kkt.tocsc(), -kkt_rhs.tocsc())
    # dfullvar_dp = np.array(ds.todense())[:len(vars_order), :]

    if Debug:
        return dfullvar_dp, kkt.tocsc().todense(), kkt_rhs.tocsc().todense(), duals
    else:
        return dfullvar_dp, 0, 0, duals