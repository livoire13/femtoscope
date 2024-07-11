import numpy as np
from matplotlib import pyplot as plt

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.solvers import NonLinearSolver
from femtoscope.core.weak_form import WeakForm
from femtoscope.inout.meshfactory import generate_uniform_1d_mesh
from femtoscope.core.nonlinear_monitoring import NonLinearMonitor

rho_min = 1
rho_max = 1e2
alpha = 0.1
Rcut = 6.0
fem_order = 2


def create_wf_int():
    """Create the linearized weak form (instance of `WeakForm`)."""

    # Mesh creation
    pre_mesh = generate_uniform_1d_mesh(0, Rcut, 500 + 1, 'mesh_1d')

    # Terms
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2
        return {'val': val.reshape(-1, 1, 1)}

    t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=alpha)

    def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2 * vec_qp ** (-4)
        return {'val': val.reshape(-1, 1, 1)}

    t2 = PreTerm('dw_volume_dot', mat=mat2, tag='mod', prefactor=3)

    def mat3(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2 * vec_qp ** (-3)
        return {'val': val.reshape(-1, 1, 1)}

    t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='mod', prefactor=-4)

    def mat4(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        r = coors.squeeze()
        rho = np.where(r <= 1.0, rho_max, rho_min)
        val = r ** 2 * rho
        return {'val': val.reshape(-1, 1, 1)}

    t4 = PreTerm('dw_volume_integrate', mat=mat4, tag='cst', prefactor=1)

    # Vertex selection
    def right_boundary(coors, domain=None):
        return np.where(coors.squeeze() == Rcut)[0]

    dim_func_entities = [(0, right_boundary, 0)]

    # WeakForm creation
    args_dict = {
        'name': 'wf_chameleon_1d',
        'dim': 1,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4],
        'dim_func_entities': dim_func_entities,
        'fem_order': fem_order,
        'pre_ebc_dict': {('vertex', 0): rho_min ** (-1 / 3)}
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf


def create_wf_res():
    """Create the residual weak form (instance of `WeakForm`)."""

    # Mesh creation
    pre_mesh = generate_uniform_1d_mesh(0, Rcut, 500 + 1, 'mesh_1d')

    # Terms
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2
        return {'val': val.reshape(-1, 1, 1)}
    t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=alpha)

    def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2 * vec_qp ** (-3)
        return {'val': val.reshape(-1, 1, 1)}
    t2 = PreTerm('dw_volume_integrate', mat=mat2, tag='mod', prefactor=-1)

    def mat3(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        r = coors.squeeze()
        rho = np.where(r <= 1, rho_max, rho_min)
        val = r ** 2 * rho
        return {'val': val.reshape(-1, 1, 1)}
    t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='cst', prefactor=1)

    def mat4(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        r = coors.squeeze()
        val = np.zeros((coors.shape[0], 1, 1))
        val[:, 0, 0] = r ** 2
        return {'val': val}
    t4 = PreTerm('dw_surface_flux', mat=mat4, tag='cst', prefactor=-alpha,
                 region_key=('vertex', 0))

    # Vertex selection
    def right_boundary(coors, domain=None):
        return np.where(coors.squeeze() == Rcut)[0]

    dim_func_entities = [(0, right_boundary, 0)]

    # WeakForm creation
    args_dict = {
        'name': 'wf_residual_1d',
        'dim': 1,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4],
        'dim_func_entities': dim_func_entities,
        'fem_order': fem_order,
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf


def create_nonlinear_solver(wf_int, wf_res):
    wf_dict = {'wf_int': wf_int, 'wf_residual': wf_res}
    phi_min = rho_max**(-1/3)
    phi_max = rho_min**(-1/3)
    rr = wf_int.field.coors.squeeze()
    initial_guess = np.where(rr <= 1, phi_min, phi_max)
    initial_guess_dict = {'int': initial_guess}
    sol_min = rho_max**(-1/3)
    sol_max = rho_min**(-1/3)
    solver = NonLinearSolver(wf_dict, initial_guess_dict,
                             sol_bounds=[sol_min, sol_max])
    return solver


def create_nonlinear_monitor(nonlinear_solver):
    criteria = (
        {
            'name': 'RelativeDeltaSolutionNorm2',
            'threshold': 1e-6,
            'look': True,
            'active': False
        },
        {
            'name': 'ResidualVector',
            'threshold': -1,
            'look': True,
            'active': False
        },
        {
            'name': 'ResidualVectorNorm2',
            'threshold': -1,
            'look': True,
            'active': False
        },
    )
    args_dict = {
        'minimum_iter_num': 5,
        'maximum_iter_num': 10,
        'criteria': criteria
    }
    monitor = NonLinearMonitor.from_scratch(args_dict)
    return monitor


wf_int = create_wf_int()
wf_res = create_wf_res()
solver = create_nonlinear_solver(wf_int, wf_res)
monitor = create_nonlinear_monitor(solver)
monitor.link_monitor_to_solver(solver)
# solver.solve(pause_iter_num=5)
# solver.resume(force=True, new_maximum_iter_num=15)
solver.solve()
rr = solver.wf_int.field.coors.squeeze()
sol = solver.sol
