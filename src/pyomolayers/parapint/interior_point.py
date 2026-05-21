import sys
from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix
import numpy as np
import logging
import time
from pyomo.common.timing import HierarchicalTimer
import enum
from typing import Optional
from pyomo.common.config import ConfigDict, ConfigValue, PositiveFloat, NonNegativeInt, NonNegativeFloat
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from .mumps_interface import MumpsInterface
from .scipy_interface import ScipyInterface
from .interface import BaseInteriorPointInterface
from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus
"""
Interface Requirements
----------------------
1) duals_primals_lb[i] must always be 0 if primals_lb[i] is -inf
2) duals_primals_ub[i] must always be 0 if primals_ub[i] is inf
3) duals_slacks_lb[i] must always be 0 if ineq_lb[i] is -inf
4) duals_slacks_ub[i] must always be 0 if ineq_ub[i] is inf
"""

###############################################################################
# Parapint
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
# This software is distributed under the Revised BSD License
# Parapint also leverages a variety of third-party software packages, which have separate licensing policies.

# This file was originally part of Parapint, available: https://github.com/sandialabs/parapint
# Copied with modification from https://github.com/sandialabs/parapint/blob/main_branch/parapint/algorithms/interior_point.py
# ip_solve_optimal() is the function to obtain the gradient matrix at the optimal solution.
###############################################################################
###############################################################################
# Pyomo
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC. 
# Under the terms of Contract DE-NA0003525 with National Technology and 
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.
# This software is distributed under the 3-clause BSD License.
###############################################################################
logger = logging.getLogger(__name__)

class InteriorPointStatus(enum.Enum):
    optimal = 0
    error = 1


class InertiaCorrectionOptions(ConfigDict):
    """
    Attributes
    ----------
    init_coef: float
    factor_increase: float
    factor_decrease: float
    max_coef: float
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('init_coef', ConfigValue(domain=PositiveFloat))
        self.declare('factor_increase', ConfigValue(domain=PositiveFloat))
        self.declare('factor_decrease', ConfigValue(domain=PositiveFloat))
        self.declare('max_coef', ConfigValue(domain=PositiveFloat))

        self.init_coef = 1e-8
        self.factor_increase = 10
        self.factor_decrease = 1/3
        self.max_coef = 1e9


class LinalgOptions(ConfigDict):
    """
    Attributes
    ----------
    solver: LinearSolverInterface
    reallocation_factor: float
    max_num_reallocations: int
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('solver', ConfigValue())
        self.declare('reallocation_factor', ConfigValue(domain=PositiveFloat))
        self.declare('max_num_reallocations', ConfigValue(domain=NonNegativeInt))

        self.solver = None
        self.reallocation_factor = 2
        self.max_num_reallocations = 5


class LineSearchOptions(ConfigDict):
    """
    Attributes
    ----------
    max_iter: int
    disable: bool
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('max_iter', ConfigValue(domain=NonNegativeInt))
        self.declare('disable', ConfigValue(domain=bool))
        self.declare('step_anyway', ConfigValue(domain=bool))

        self.max_iter: int = 4
        self.disable: bool = True
        self.step_anyway: bool = True


class IPOptions(ConfigDict):
    """
    Attributes
    ----------
    max_iter: int
    tol: float
    init_barrier_parameter: float
    minimum_barrier_parameter: float
    report_timing: bool
    use_inertia_correction: bool
    inertia_correction: InertiaCorrectionOptions
    linalg: LinalgOptions
    line_search: LineSearchOptions
    unified_step: bool
    error_scaling: float
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('max_iter', ConfigValue(domain=NonNegativeInt))
        self.declare('tol', ConfigValue(domain=PositiveFloat))
        self.declare('init_barrier_parameter', ConfigValue(domain=PositiveFloat))
        self.declare('minimum_barrier_parameter', ConfigValue(domain=PositiveFloat))
        self.declare('barrier_decrease', ConfigValue(domain=PositiveFloat))
        self.declare('report_timing', ConfigValue(domain=bool))
        self.declare('use_inertia_correction', ConfigValue(domain=bool))
        self.declare('inertia_correction', InertiaCorrectionOptions())
        self.declare('linalg', LinalgOptions())
        self.declare('line_search', LineSearchOptions())
        self.declare('unified_step', ConfigValue(domain=bool))
        self.declare('error_scaling', ConfigValue(domain=PositiveFloat))
        self.declare('bounds_relaxation_factor', ConfigValue(domain=NonNegativeFloat))

        self.max_iter = 1000
        self.tol = 1e-8
        self.init_barrier_parameter = 0.1
        self.minimum_barrier_parameter = 1e-9
        self.barrier_decrease = 10
        self.report_timing = False
        self.use_inertia_correction = True
        self.inertia_correction = InertiaCorrectionOptions()
        self.linalg = LinalgOptions()
        self.line_search: LineSearchOptions = LineSearchOptions()
        self.unified_step: bool = False
        self.error_scaling: float = 100
        self.bounds_relaxation_factor: float = 1e-8

def line_search(
    interface: BaseInteriorPointInterface,
    barrier: float,
    options: LineSearchOptions,
    delta_primals,
    delta_slacks,
    delta_duals_eq,
    delta_duals_ineq,
    delta_duals_primals_lb,
    delta_duals_primals_ub,
    delta_duals_slacks_lb,
    delta_duals_slacks_ub,
    timer: Optional[HierarchicalTimer] = None,
) -> Optional[float]:
    raise NotImplementedError('This is just a placeholder for a line search')


def numeric_factorization(interface: BaseInteriorPointInterface,
                          kkt,
                          options: IPOptions,
                          inertia_coef,
                          timer: Optional[HierarchicalTimer] = None):
    logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10}{pos_eig:<10}'
                 '{neg_eig:<10}{zero_eig:<10}{status:<10}'.format(reg_iter='reg_iter', num_realloc='# realloc',
                                                                  reg_coef='reg_coef', pos_eig='pos_eig',
                                                                  neg_eig='neg_eig', zero_eig='zero_eig',
                                                                  status='status'))
    status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                             linear_solver=options.linalg.solver,
                                                             reallocation_factor=options.linalg.reallocation_factor,
                                                             max_iter=options.linalg.max_num_reallocations,
                                                             symbolic_or_numeric='numeric',
                                                             timer=timer)
    final_inertia_coef = 0
    if not options.use_inertia_correction:
        # logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
        #              '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
        #              '{status:<10}'.format(reg_iter=0, num_realloc=num_realloc, reg_coef=final_inertia_coef,
        #                                    pos_eig=None, neg_eig=None, zero_eig=None, status=str(status)))
        if status != LinearSolverStatus.successful:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))
    else:
        if status not in {LinearSolverStatus.successful, LinearSolverStatus.singular}:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))

        pos_eig, neg_eig, zero_eig = None, None, None
        _iter = 0
        while final_inertia_coef <= options.inertia_correction.max_coef:
            if status == LinearSolverStatus.successful:
                pos_eig, neg_eig, zero_eig = options.linalg.solver.get_inertia()
            else:
                pos_eig, neg_eig, zero_eig = None, None, None
            # logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
            #              '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
            #              '{status:<10}'.format(reg_iter=_iter, num_realloc=num_realloc, reg_coef=final_inertia_coef,
            #                                    pos_eig=str(pos_eig), neg_eig=str(neg_eig), zero_eig=str(zero_eig),
            #                                    status=str(status)))
            if ((zero_eig == 0) and (status == LinearSolverStatus.successful)):
                break
            if _iter == 0:
                kkt = kkt.copy()
            kkt = interface.regularize_equality_gradient(kkt=kkt, coef=-inertia_coef, copy_kkt=False)
            # TODO: may need to reularize hessian to overcome singularity, seems not.
            #kkt = interface.regularize_hessian(kkt=kkt, coef=inertia_coef, copy_kkt=False)
            status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                                     linear_solver=options.linalg.solver,
                                                                     reallocation_factor=options.linalg.reallocation_factor,
                                                                     max_iter=options.linalg.max_num_reallocations,
                                                                     symbolic_or_numeric='numeric',
                                                                     timer=timer)
            final_inertia_coef = inertia_coef
            inertia_coef *= options.inertia_correction.factor_increase
            _iter += 1
        #TODO final_inertia_coef 1000000000.0, neg_eig converted from 2 to 1,  interface.regularize_hessian plays the role.
        if ((zero_eig != 0) or (status != LinearSolverStatus.successful)):
            raise RuntimeError('Exceeded maximum inertia correciton')
    
    return final_inertia_coef

def ip_solve_optimal(interface, kkt, rhs, is_first_call, IPoptions):
    options = IPoptions
    timer = HierarchicalTimer()
    interface.set_bounds_relaxation_factor(options.bounds_relaxation_factor)
    
    barrier_parameter = options.init_barrier_parameter
    inertia_coef = options.inertia_correction.init_coef
    used_inertia_coef = 0
    interface.set_barrier_parameter(barrier_parameter)
    timer.start('factorize')
    if is_first_call:
        timer.start('symbolic')
        sym_fact_status, sym_fact_iter = try_factorization_and_reallocation(kkt=kkt,
                                                                            linear_solver=options.linalg.solver,
                                                                            reallocation_factor=options.linalg.reallocation_factor,
                                                                            max_iter=options.linalg.max_num_reallocations,
                                                                            symbolic_or_numeric='symbolic',
                                                                            timer=timer)                                                          
        timer.stop('symbolic')
        if sym_fact_status != LinearSolverStatus.successful:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(sym_fact_status))
    timer.start('numeric')

    used_inertia_coef = numeric_factorization(interface=interface,
                                                kkt=kkt,
                                                options=options,
                                                inertia_coef=inertia_coef,
                                                timer=timer)
    inertia_coef = used_inertia_coef * options.inertia_correction.factor_decrease
    if inertia_coef < options.inertia_correction.init_coef:
        inertia_coef = options.inertia_correction.init_coef
    timer.stop('numeric')
    timer.stop('factorize')

    timer.start('back solve')
    if type(options.linalg.solver).__name__ == "ScipyInterface":
        ds = options.linalg.solver.do_back_solve(rhs)
    else:
        ds = np.zeros_like(rhs)
        for i in range(rhs.shape[1]):
            ds[:, i] = options.linalg.solver.do_back_solve(rhs[:, i])
    timer.stop('back solve')
    if is_first_call:
        return ds, options
    return ds, None

def try_factorization_and_reallocation(kkt, linear_solver: LinearSolverInterface, reallocation_factor, max_iter,
                                       symbolic_or_numeric, timer=None):
    if timer is None:
        timer = HierarchicalTimer()

    assert max_iter >= 1
    if symbolic_or_numeric == 'numeric':
        method = linear_solver.do_numeric_factorization
    else:
        assert symbolic_or_numeric == 'symbolic'
        method = linear_solver.do_symbolic_factorization
    for count in range(max_iter):
        res = method(matrix=kkt, raise_on_error=False, timer=timer)
        status = res.status
        if status == LinearSolverStatus.not_enough_memory:
            linear_solver.increase_memory_allocation(reallocation_factor)
        else:
            break
    return status, count



