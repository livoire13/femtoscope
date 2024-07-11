# Importation of relevant module

import numpy as np
from matplotlib import pyplot as plt

from femtoscope.inout.meshfactory import generate_uniform_1d_mesh
from femtoscope.core.weak_form import WeakForm
from femtoscope.core.solvers import NonLinearSolver
from femtoscope.core.nonlinear_monitoring import NonLinearMonitor
from femtoscope.core.pre_term import PreTerm

# Mesh parameters

x_inf = 7.0
x_cut = 5.0
n_points = 1000 + 1
fem_order = 2

a_right = x_inf
b_right = x_inf - x_cut
a_left = -a_right
b_left = -b_right
ab = a_right * b_right

def get_a(coors):
    eta = coors.squeeze()
    if (eta < 0).all():
        a = a_left
    elif (eta > 0).all():
        a = a_right
    else:
        raise ValueError("Impossible to retrieve 'a' coefficient!")
    return a


# Physical parameters

alpha = 5e-2
rho_left = 1e3
rho_right = 1e-1
npot = 2
phi_max = rho_right ** (-1 / (npot + 1))
phi_min = rho_left ** (-1 / (npot + 1))

# Mesh creation, weak form creation

# Mesh
pre_mesh = generate_uniform_1d_mesh(-x_inf, +x_inf, n_points)

dim_func_entities = []
pre_terms = []

# Region selections
def right_boundary(coors, domain=None):
    return np.where(coors.squeeze() == x_inf)[0]
dim_func_entities.append((0, right_boundary, 2))

def left_boundary(coors, domain=None):
    return np.where(coors.squeeze() == -x_inf)[0]
dim_func_entities.append((0, left_boundary, 1))

def right_bulk(coors, domain=None):
    x = coors.squeeze()
    return np.where((x >= 0) & (x <= x_cut))[0]
dim_func_entities.append((1, right_bulk, 302))

def left_bulk(coors, domain=None):
    x = coors.squeeze()
    return np.where((x <= 0) & (x >= -x_cut))[0]
dim_func_entities.append((1, left_bulk, 301))

def bulk(coors, domain=None):
    x = coors.squeeze()
    return np.where((x >= -x_cut) & (x <= x_cut))[0]
dim_func_entities.append((1, bulk, 300))

def right_side(coors, domain=None):
    return np.where(coors.squeeze() >= x_cut)[0]
dim_func_entities.append((1, right_side, 402))

def left_side(coors, domain=None):
    return np.where(coors.squeeze() <= -x_cut)[0]
dim_func_entities.append((1, left_side, 401))

def gamma_bulk(coors, domain=None):
    x = coors.squeeze()
    return np.array([np.argmin(abs(x - (-x_cut))), np.argmin(abs(x - x_cut))])
dim_func_entities.append((0, gamma_bulk, 10))

# Terms bulk
t1_bulk = PreTerm('dw_laplace', tag='cst', region_key=('subomega', 300),
                  prefactor=alpha)
pre_terms.append(t1_bulk)

def mat2bulk(ts, coors, mode=None, vec_qp=None, **kwargs):
    if mode != 'qp': return
    val = vec_qp ** (-(npot+2))
    return {'val': val.reshape(-1, 1, 1)}
t2_bulk = PreTerm('dw_volume_dot', mat=mat2bulk, tag='mod', region_key=('subomega', 300),
                  prefactor=npot+1)
pre_terms.append(t2_bulk)

def mat3bulk(ts, coors, mode=None, vec_qp=None, **kwargs):
    if mode != 'qp': return
    val = vec_qp ** (-(npot+1))
    return {'val': val.reshape(-1, 1, 1)}
t3_bulk = PreTerm('dw_volume_integrate', mat=mat3bulk, tag='mod', prefactor=-(npot+2),
                 region_key=('subomega', 300))
pre_terms.append(t3_bulk)

trho_right_bulk = PreTerm('dw_volume_integrate', tag='cst', prefactor=rho_right,
                          region_key=('subomega', 302))
pre_terms.append(trho_right_bulk)
trho_left_bulk = PreTerm('dw_volume_integrate', tag='cst', prefactor=rho_left,
                         region_key=('subomega', 301))
pre_terms.append(trho_left_bulk)

# Terms side
def matside1(ts, coors, mode=None, **kwargs):
    if mode != 'qp': return
    a = get_a(coors)
    val = (a - coors.squeeze()) ** 2 / ab
    return {'val': val.reshape(-1, 1, 1)}
t1_left_side = PreTerm('dw_laplace', tag='cst', prefactor=alpha, mat=matside1,
                      region_key=('subomega', 401))
pre_terms.append(t1_left_side)
t1_right_side = PreTerm('dw_laplace', tag='cst', prefactor=alpha, mat=matside1,
                        region_key=('subomega', 402))
pre_terms.append(t1_right_side)

def matside2(ts, coors, mode=None, vec_qp=None, **kwargs):
    if mode != 'qp': return
    a = get_a(coors)
    val = ab / (a - coors.squeeze()) ** 2 * vec_qp ** (-(npot+2))
    return {'val': val.reshape(-1, 1, 1)}
t2_left_side = PreTerm('dw_volume_dot', tag='mod', prefactor=npot+1, mat=matside2,
                       region_key=('subomega', 401))
pre_terms.append(t2_left_side)
t2_right_side = PreTerm('dw_volume_dot', tag='mod', prefactor=npot+1, mat=matside2,
                       region_key=('subomega', 402))
pre_terms.append(t2_right_side)

def matside3(ts, coors, mode=None, vec_qp=None, **kwargs):
    if mode != 'qp': return
    a = get_a(coors)
    val = ab / (a - coors.squeeze()) ** 2 * vec_qp ** (-(npot+1))
    return {'val': val.reshape(-1, 1, 1)}
t3_left_side = PreTerm('dw_volume_integrate', tag='mod', prefactor=-(npot+2),
                      mat=matside3, region_key=('subomega', 401))
pre_terms.append(t3_left_side)
t3_right_side = PreTerm('dw_volume_integrate', tag='mod', prefactor=-(npot+2),
                        mat=matside3, region_key=('subomega', 402))
pre_terms.append(t3_right_side)

def matside4(ts, coors, mode=None, **kwargs):
    if mode != 'qp': return
    a = get_a(coors)
    val = ab / (a - coors.squeeze()) ** 2
    return {'val': val.reshape(-1, 1, 1)}
t4_left_side = PreTerm('dw_volume_integrate', tag='cst', mat=matside4,
                      prefactor=rho_left, region_key=('subomega', 401))
pre_terms.append(t4_left_side)
t4_right_side = PreTerm('dw_volume_integrate', tag='cst', mat=matside4,
                       prefactor=rho_right, region_key=('subomega', 402))
pre_terms.append(t4_right_side)

# Boundary conditions
pre_ebc_dict = {('vertex', 1): phi_min, ('vertex', 2): phi_max}

args_dict = {
    'name': 'wfx',
    'dim': 1,
    'pre_mesh': pre_mesh,
    'pre_terms': pre_terms,
    'dim_func_entities': dim_func_entities,
    'fem_order': fem_order,
    'pre_ebc_dict': pre_ebc_dict
}
wf = WeakForm.from_scratch(args_dict)


# Residual weakform in the bulk

tres1 = PreTerm('dw_laplace', tag='cst', region_key=('subomega', 300),
                prefactor=alpha)

tres2 = PreTerm('dw_surface_flux', tag='cst', region_key=('vertex', 10),
               prefactor=-alpha)

def matres3(ts, coors, mode=None, vec_qp=None, **kwargs):
    if mode != 'qp': return
    val = vec_qp ** (-(npot + 1))
    return {'val': val.reshape(-1, 1, 1)}
tres3 = PreTerm('dw_volume_integrate', tag='mod', mat=matres3, prefactor=-1,
               region_key=('subomega', 300))

trhores_right = PreTerm('dw_volume_integrate', tag='cst', prefactor=rho_right,
                        region_key=('subomega', 302))

trhores_left = PreTerm('dw_volume_integrate', tag='cst', prefactor=rho_left,
                       region_key=('subomega', 301))

pre_terms_res = [tres1, tres2, tres3, trhores_left, trhores_right]

args_dict = {
    'name': 'wf_res',
    'dim': 1,
    'pre_mesh': pre_mesh,
    'pre_terms': pre_terms_res,
    'dim_func_entities': dim_func_entities,
    'fem_order': fem_order
}
wf_res = WeakForm.from_scratch(args_dict)

# Solver
wf_dict = {'wf_int': wf, 'wf_residual': wf_res}
phi_min = phi_min
phi_max = phi_max
x = wf.field.coors.squeeze()
initial_guess = np.where(x <= 0, phi_min, phi_max)
initial_guess_dict = {'int': initial_guess}
solver = NonLinearSolver(wf_dict, initial_guess_dict,
                         sol_bounds=[phi_min, phi_max])

# Monitor
criteria = (
    {'name': 'RelativeDeltaSolutionNorm2', 'threshold': 1e-6, 'look': True, 'active': False},
    {'name': 'ResidualVector', 'threshold': -1, 'look': True, 'active': False},
    {'name': 'ResidualVectorNorm2', 'threshold': -1, 'look': True, 'active': False},
    {'name': 'ResidualReductionFactor', 'threshold': -1, 'look': True, 'active': False},
)
args_dict = {
    'minimum_iter_num': 0,
    'maximum_iter_num': 15,
    'criteria': criteria
}
monitor = NonLinearMonitor.from_scratch(args_dict)
monitor.link_monitor_to_solver(solver)

# Solving
solver.solve()
sol = solver.sol
dofs_bulk = wf.field.get_dofs_in_region(wf.region_dict[('subomega', 300)])
dofs_right_side = wf.field.get_dofs_in_region(wf.region_dict[('subomega', 402)])
sol_bulk = sol[dofs_bulk]
x_bulk = wf.field.coors.squeeze()[dofs_bulk]
sol_right = sol[dofs_right_side]
eta_right = wf.field.coors.squeeze()[dofs_right_side]


idx = np.argsort(x_bulk)
plt.figure()
plt.plot(x_bulk[idx], sol_bulk[idx])
plt.show()

plt.figure()
plt.plot(eta_right, sol_right)
plt.show()


