import numpy as np
import torch
import copy
from scipy.sparse import coo_matrix, identity, diags
from scipy.sparse.linalg import spsolve, lsqr 
from Opt_Layer.parapint.interior_point import IPOptions, ip_solve_optimal
from Opt_Layer.parapint.interface import BaseInteriorPointInterface
from Opt_Layer.parapint.mumps_interface import MumpsInterface
from Opt_Layer.parapint.scipy_interface import ScipyInterface
from Opt_Layer.parapint.ma27_interface import InteriorPointMA27Interface

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedExtendedNLP
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.common.timing import HierarchicalTimer
torch.set_default_dtype(torch.float64)
from pyomo.common.dependencies import attempt_import
import time
mpi4py, mpi4py_available = attempt_import("mpi4py", error_message="mpi4py is not available")

class InteriorPointInterface:
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

    def __init__(self, concreate_model, nlp, nlp_full, pyomo_vars):

        self.concreate_model = concreate_model
        self._nlp = nlp
        self._nlp_full = nlp_full
        self.pyomo_vars = pyomo_vars
        self._slacks = self._nlp.evaluate_ineq_constraints()

        # set the init_duals_primals_lb/ub from ipopt_zL_out, ipopt_zU_out if available
        # need to compress them as well and initialize the duals_primals_lb/ub
        # only have the duals_primals_lb and ub for the vars.
        (self._init_duals_primals_lb, self._init_duals_primals_ub) = (self._get_full_duals_primals_bounds())
        # print("self._init_duals_primals_lb", self._init_duals_primals_lb)
        # print("self._init_duals_primals_ub", self._init_duals_primals_ub)
        # print("primal value", self._nlp.get_primals())
        # print("self._nlp.primals_lb()", self._nlp.primals_lb())
        self._init_duals_primals_lb[np.isneginf(self._nlp.primals_lb())] = 0
        self._init_duals_primals_ub[np.isinf(self._nlp.primals_ub())] = 0
        self._duals_primals_lb = self._init_duals_primals_lb.copy()
        self._duals_primals_ub = self._init_duals_primals_ub.copy()
        # print("self._duals_primals_lb", self._duals_primals_lb)
        
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
        lbs = self._nlp_full.ineq_lb()
        if self.bounds_relaxation_factor == 0:
            return lbs
        eye = np.ones(lbs.size)
        lbs_mod = lbs - self.bounds_relaxation_factor * np.max(np.array([eye, np.abs(lbs)]), axis=0)
        return lbs_mod

    def ineq_ub(self):
        ubs = self._nlp_full.ineq_ub()
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
        jac_eq = self.evaluate_jacobian_eq()
        jac_ineq = self.evaluate_jacobian_ineq()
        timer.stop('eval jac')

        duals_primals_lb = self._duals_primals_lb
        duals_primals_ub = self._duals_primals_ub
        duals_slacks_lb = self._duals_slacks_lb
        duals_slacks_ub = self._duals_slacks_ub
        primals = self._nlp.get_primals()

        timer.start('hess block')
        data = duals_primals_lb / (primals - self.primals_lb()) + duals_primals_ub / (self.primals_ub() - primals)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"{duals_primals_lb=}")
            print(f"{duals_primals_ub=}")
            print(f"{primals=}")
            print(f"{self.primals_ub()=}")
            print(f"{self.primals_lb()=}")
            print("primals block", data)
            raise RuntimeError
        n = self._nlp.n_primals()
        indices = np.arange(n)
        hess_block.row = np.concatenate([hess_block.row, indices])
        hess_block.col = np.concatenate([hess_block.col, indices])
        hess_block.data = np.concatenate([hess_block.data, data])
        timer.stop('hess block')

        timer.start('slack block')
        data = duals_slacks_lb / (
            self._slacks - self.ineq_lb()
        ) + duals_slacks_ub / (self.ineq_ub() - self._slacks)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"{duals_slacks_lb=}")
            print(f"{duals_slacks_ub=}")
            print(f"{self._slacks=}")
            print(f"{self.ineq_ub()=}")
            print(f"{self.ineq_lb()=}")
            print("slack block", data)
            raise RuntimeError
        n = self.n_ineq_constraints()
        indices = np.arange(n)
        slack_block = coo_matrix((data, (indices, indices)), shape=(n, n))
        timer.stop('slack block')
        
        timer.start('regularization block')
        eq_reg_blk = identity(self.n_eq_constraints(), format='coo')
        eq_reg_blk.data.fill(0)
        ineq_reg_blk = identity(self.n_ineq_constraints(), format='coo')
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
        kkt.set_block(3, 1, -identity(self.n_ineq_constraints(), format='coo'))
        kkt.set_block(1, 3, -identity(self.n_ineq_constraints(), format='coo'))
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
                  identity(self.n_eq_constraints(),
                                        format='coo'))
        ineq_ptb = (reg_coef *
                    identity(self.n_ineq_constraints(),
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
            pyomo_variables = self.pyomo_vars
            # pyomo_variables = self._nlp_full.get_pyomo_variables()
            # pyomo_variables contains vars and params, but only set the duals for vars (self._nlp.n_primals())
            if not hasattr(pyomo_model, 'dual'):
                raise Exception("Unknown duals. Please define dual in Pyomo model, i.e., model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)")
            if hasattr(pyomo_model, 'ipopt_zL_out'):
                zL_suffix = pyomo_model.ipopt_zL_out
                full_duals_primals_lb = np.zeros(self._nlp.n_primals())
                for i, v in enumerate(pyomo_variables):
                    if v in zL_suffix:
                        full_duals_primals_lb[i] = zL_suffix[v]
            if hasattr(pyomo_model, 'ipopt_zU_out'):
                zU_suffix = pyomo_model.ipopt_zU_out
                full_duals_primals_ub = np.zeros(self._nlp.n_primals())
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


class Sensitivity:
    def __init__(self, pyomo_layer, comm=None):
        self.grad_parameters = pyomo_layer.grad_parameters
        self.nlp_full = pyomo_layer.nlp_full
        self.nlp_var = pyomo_layer.nlp_var

        if mpi4py_available:
            from mpi4py import MPI
            if comm is None:
                comm = MPI.COMM_WORLD
            rank_comm = comm.Split(comm.rank, 0)
        else:
            rank_comm = None

        self.param_order = []
        for p_name in self.grad_parameters:
            for i in p_name.index_set():
                self.param_order.append(str(p_name[i]))

        self.pyomo_cons = self.nlp_full.get_pyomo_constraints()
        self.pyomo_vars_full = self.nlp_full.get_pyomo_variables()
        self.pyomo_vars = self.nlp_var.get_pyomo_variables()
        
        self.var_names = pyomo_layer.var_names
        self.vars_objs = pyomo_layer.vars_objs 
        self.bounds_relaxation_factor = pyomo_layer.solver.options["bound_relax_factor"]

        self.is_first_call = True
        self.debug = True
        self.IPsolver = IPOptions()
        self.IPsolver.use_inertia_correction = True
        self.IPsolver.linalg.solver = MumpsInterface(comm=rank_comm)
        # self.IPsolver.linalg.solver = InteriorPointMA27Interface()
        # self.IPsolver.linalg.solver = ScipyInterface(compute_inertia=self.IPsolver.use_inertia_correction)

    def get_sen(self, concrete_model):
        """
        Descriptions: 
            Obtain the sensitivity matrix of decision variables with respect to the parameters.
        
        Args:
            - concrete_model (``Pyomo model``, required)
                The concrete instance has been solved by IPOPT, and the optimal values of primal and dual variables have been obtained.
            - grad_parameters(``List[str]``, required)
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
            >>> grad_parameters = ["Psqrt", "q", "A", "b"]
            >>> variables_name_index = ["x[0]", "x[1]", "x[3]", "y"]
            >>> param_order = ["Psqrt[0, 0]",... "q[0]"..., "A[0, 0]"..., "b[0]"...]
            >>> vars_objs = [x, y] based on nlp.get_pyomo_variables()
            >>> dvar_dp, lhs_Jac, rhs_Jac, variables_index_order = get_sen(concrete_model, grad_parameters, variables_name_index)
        """
        # Unfix the param variables for Jac/Hessian evaluation
        for param in concrete_model.component_objects(pyo.Var):
            if any(param is v for v in self.grad_parameters):
                for index in param:
                    param[index].unfix() 

        primals = []
        for var in self.pyomo_vars_full:
            primals.append(var.value)
        self.nlp_full.set_primals(np.array(primals))
        duals = []
        for constraint in self.pyomo_cons:
            duals.append(concrete_model.dual[constraint])
        duals = - np.array(duals, dtype = np.float64)
        self.nlp_full.set_duals(duals)

        nlp_vars = ProjectedExtendedNLP(self.nlp_full, self.var_names)
        pyomo_vars = self.pyomo_vars

        IPOPT = InteriorPointInterface(concrete_model, nlp_vars, self.nlp_full, pyomo_vars) 
        IPOPT.set_bounds_relaxation_factor(self.bounds_relaxation_factor)
        kkt = IPOPT.evaluate_primal_dual_kkt_matrix()

        nlp_params = ProjectedExtendedNLP(self.nlp_full, self.param_order)

        # Build the right hand side vector
        Hessian = self.nlp_full.extract_submatrix_hessian_lag(pyomo_variables_rows=self.vars_objs, pyomo_variables_cols = self.grad_parameters)
        H_ineq = nlp_params.evaluate_jacobian_ineq()        
        H_eq = nlp_params.evaluate_jacobian_eq()
        
        kkt_rhs = BlockMatrix(4, 1)
        kkt_rhs.set_block(0, 0, Hessian)
        kkt_rhs.set_block(1, 0, coo_matrix((self.nlp_full.n_ineq_constraints(), Hessian.shape[1])))
        kkt_rhs.set_block(2, 0, H_eq)
        kkt_rhs.set_block(3, 0, H_ineq)
        if np.any(np.isnan(kkt.tocsc().data)) or np.any(np.isinf(kkt.tocsc().data)):
            raise RuntimeError("This is a runtime error")
        try:
            if self.is_first_call:
                ds, self.IPsolver = ip_solve_optimal(interface=IPOPT, kkt = kkt, rhs = -kkt_rhs.toarray(), 
                                        is_first_call = self.is_first_call, IPoptions = self.IPsolver)
                self.is_first_call = False
            else:   
                ds, _ = ip_solve_optimal(interface=IPOPT, kkt = kkt, rhs = -kkt_rhs.toarray(), 
                                            is_first_call = self.is_first_call, IPoptions = self.IPsolver)
        except:
            rhs = -kkt_rhs.toarray()
            ds = np.zeros_like(rhs)
            raise RuntimeWarning("zero_grad returned due to failed KKT")
        dfullvar_dp = ds[:len(self.var_names), :]

        if self.debug:
            return dfullvar_dp, kkt.toarray(), kkt_rhs.toarray(), duals
        else:
            return dfullvar_dp, 0, 0, duals