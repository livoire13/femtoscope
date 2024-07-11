import numpy as np
from matplotlib import pyplot as plt

from femtoscope.core.solvers import NonLinearSolver
from femtoscope.core.nonlinear_monitoring import NonLinearMonitor
from femtoscope.core.weak_form import WeakForm
from femtoscope.core.pre_term import PreTerm
from femtoscope.inout.meshfactory import generate_uniform_1d_mesh

# parameters
rho_min = 1
rho_max = 1e2
alpha = 0.1
npot = 2
Rc = 5.0
fem_order = 1
phi_max = rho_min ** (-1 / (npot + 1))
phi_min = rho_max ** (-1 / (npot + 1))


def create_wf_int():
    """Create the linearized weak form (instance of `WeakForm`)."""

    # Mesh creation
    pre_mesh = generate_uniform_1d_mesh(0, Rc, 1000 + 1, 'mesh_1d')

    # Terms
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2
        return {'val': val.reshape(-1, 1, 1)}
    t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=alpha)

    def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2 * vec_qp ** (-(npot+2))
        return {'val': val.reshape(-1, 1, 1)}
    t2 = PreTerm('dw_volume_dot', mat=mat2, tag='mod', prefactor=npot+1)

    def mat3(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2 * vec_qp ** (-(npot+1))
        return {'val': val.reshape(-1, 1, 1)}
    t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='mod', prefactor=-(npot+2))

    def mat4(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        r = coors.squeeze()
        rho = np.where(r <= 1.0, rho_max, rho_min)
        val = r ** 2 * rho
        return {'val': val.reshape(-1, 1, 1)}
    t4 = PreTerm('dw_volume_integrate', mat=mat4, tag='cst', prefactor=1)

    # Vertex selection
    def right_boundary(coors, domain=None):
        return np.where(coors.squeeze() == Rc)[0]

    dim_func_entities = [(0, right_boundary, 0)]

    # WeakForm creation
    args_dict = {
        'name': 'wf_chameleon_int',
        'dim': 1,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4],
        'dim_func_entities': dim_func_entities,
        'fem_order': fem_order,
        'is_exterior': False
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf

def create_wf_ext():
    """Create the exterior weak form (instance of `WeakForm`)."""

    # Mesh creation
    pre_mesh = generate_uniform_1d_mesh(0, Rc, 1000 + 1, 'mesh_1d')

    # Terms
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        eta = coors.squeeze()
        val = eta ** 4 / Rc ** 2 * (5 - 4 * eta / Rc)
        return {'val': val.reshape(-1, 1, 1)}
    t1 = PreTerm('dw_laplace', tag='cst', prefactor=alpha, mat=mat1)

    def mat2(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        eta = coors.squeeze()
        val = 20 * eta ** 3 / Rc ** 2 * (1 - eta / Rc)
        return {'val': val.reshape(eta.shape[0], 1, 1)}
    t2 = PreTerm('dw_s_dot_mgrad_s', tag='cst', prefactor=alpha, mat=mat2)

    def mat3(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        eta = coors.squeeze()
        val = Rc ** 2 * (5 - 4 * eta / Rc) * vec_qp ** (-(npot + 2))
        return {'val': val.reshape(-1, 1, 1)}
    t3 = PreTerm('dw_volume_dot', tag='mod', prefactor=npot+1, mat=mat3)

    def mat4(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        eta = coors.squeeze()
        val = Rc ** 2 * (5 - 4 * eta / Rc) * vec_qp ** (-(npot + 1))
        return {'val': val.reshape(-1, 1, 1)}
    t4 = PreTerm('dw_volume_integrate', tag='mod', prefactor=-(npot+2), mat=mat4)

    def mat5(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        eta = coors.squeeze()
        val = Rc ** 2 * (5 - 4 * eta / Rc) * rho_min
        return {'val': val.reshape(-1, 1, 1)}
    t5 = PreTerm('dw_volume_integrate', tag='cst', prefactor=1, mat=mat5)

    pre_terms = [t1, t2, t3, t4, t5]

    # Vertex selection
    def right_boundary(coors, domain=None):
        return np.where(coors.squeeze() == Rc)[0]

    def left_boundary(coors, domain=None):
        return np.where(coors.squeeze() == 0)[0]

    dim_func_entities = [(0, right_boundary, 0), (0, left_boundary, 1)]

    # WeakForm creation
    args_dict = {
        'name': 'wf_chameleon_ext',
        'dim': 1,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4, t5],
        'dim_func_entities': dim_func_entities,
        'fem_order': fem_order,
        'pre_ebc_dict': {('vertex', 1): phi_max},
        'is_exterior': True
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf


def create_wf_res():
    """Create the residual weak form (instance of `WeakForm`)."""

    # Mesh creation
    pre_mesh = generate_uniform_1d_mesh(0, Rc, 1000 + 1, 'mesh_1d')

    # Terms
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2
        return {'val': val.reshape(-1, 1, 1)}

    t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=alpha)

    def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        val = coors.squeeze() ** 2 * vec_qp ** (-(npot+1))
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
        return np.where(coors.squeeze() == Rc)[0]

    dim_func_entities = [(0, right_boundary, 0)]

    # WeakForm creation
    args_dict = {
        'name': 'wf_residual_1d',
        'dim': 1,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4],
        'dim_func_entities': dim_func_entities,
        'fem_order': fem_order,
        'is_exterior': False
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf


def create_nonlinear_solver(wf_int, wf_ext, wf_res):
    wf_dict = {'wf_int': wf_int, 'wf_ext': wf_ext, 'wf_residual': wf_res}
    rr = wf_int.field.coors.squeeze()
    eta = wf_ext.field.coors.squeeze()
    initial_guess_int = np.where(rr <= 1, phi_min, phi_max)
    initial_guess_ext = phi_max * np.ones_like(eta)
    initial_guess_dict = {'int': initial_guess_int, 'ext': initial_guess_ext}
    solver = NonLinearSolver(wf_dict, initial_guess_dict,
                             sol_bounds=[phi_min, phi_max],
                             region_key_int=('vertex', 0),
                             region_key_ext=('vertex', 0))
    return solver


def create_nonlinear_monitor(nonlinear_solver):
    criteria = (
        {'name': 'RelativeDeltaSolutionNorm2', 'threshold': 1e-6, 'look': True,'active': False},
        {'name': 'ResidualVector', 'threshold': -1, 'look': True, 'active': False},
        {'name': 'ResidualVectorNorm2', 'threshold': -1, 'look': True, 'active': False},
        {'name': 'ResidualReductionFactor', 'threshold': -1, 'look': True, 'active': False},
    )
    args_dict = {
        'minimum_iter_num': 0,
        'maximum_iter_num': 20,
        'criteria': criteria
    }
    monitor = NonLinearMonitor.from_scratch(args_dict)
    return monitor


wf_int = create_wf_int()
wf_ext = create_wf_ext()
wf_res = create_wf_res()
solver = create_nonlinear_solver(wf_int, wf_ext, wf_res)
monitor = create_nonlinear_monitor(solver)
monitor.link_monitor_to_solver(solver)

# Solve Klein-Gordon equation
solver.solve(verbose=True)

rr = wf_int.field.coors.squeeze()
eta = wf_ext.field.coors.squeeze()
sol_int = solver.sol_int
sol_ext = solver.sol_ext

plt.figure()
plt.plot(rr, sol_int)
plt.show()

# Save data to pickle file
import pickle
data_dict = {
    'r': rr,
    'eta': eta,
    'sol_int': sol_int,
    'sol_ext': sol_ext
}

with open("chameleon_radial_test.pkl", 'wb') as f:
    pickle.dump(data_dict, f)
