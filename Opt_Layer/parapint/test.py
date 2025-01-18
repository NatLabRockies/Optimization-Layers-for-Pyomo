import pyomo.environ as pe
from interface import InteriorPointInterface
from interior_point import InteriorPointStatus, IPOptions, ip_solve
from mumps_interface import MumpsInterface

def main(linear_solver):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.x**2 + m.y**2)
    m.c1 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
    m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))

    interface = InteriorPointInterface(m)
    options = IPOptions()
    options.linalg.solver = linear_solver
    status = ip_solve(interface=interface, options=options)
    assert status == InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()
    m.x.pprint()
    m.y.pprint()
    return m


if __name__ == '__main__':
    # cntl[1] is the MA27 pivot tolerance
    linear_solver = MumpsInterface(cntl_options={1: 1e-6})
    main(linear_solver=linear_solver)