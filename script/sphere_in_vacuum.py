import numpy as np
import matplotlib.pyplot as plt

from femtoscope.physics.physical_problems import Chameleon
from femtoscope.misc.unit_conversion import compute_beta, compute_alpha
from femtoscope.inout.meshfactory import (generate_uniform_1d_mesh,
                                          generate_1d_mesh_from_array)
from femtoscope.misc.analytical import chameleon_radial
from femtoscope.misc.constants import *

# FEM parameters
fem_order = 2
npot = 1
L0 = 1  # m
rho0 = 1  # kg/m^3
Rc = 5
Rsat = 0.95  # m
Msat = 675  # kg
rho_vac = 6e-19  # dimensionless
rho_sat = Msat / (4/3*pi*Rsat**3) / rho0  # dimensionless
phi_min = rho_sat ** (-1 / (npot + 1))
phi_max = rho_vac ** (-1 / (npot + 1))
alpha = 2e-7

param_dict = {
    'alpha': alpha, 'npot': npot, 'rho_min': rho_vac, 'rho_max': rho_sat
}

# pre_mesh_int = generate_uniform_1d_mesh(0, Rc, int(5e4))
# array_ext = Rc * np.linspace(0, 1, int(5e4)) ** 2
# pre_mesh_ext = generate_1d_mesh_from_array(array_ext)

pre_mesh_int = generate_uniform_1d_mesh(0, Rc, int(1e2*Rc)+1)
# pre_mesh_ext = generate_uniform_1d_mesh(0, Rc, int(5e3*Rc)+1)
array_ext = Rc * np.linspace(0, 1, len(pre_mesh_int.coors)) ** (1.5)
pre_mesh_ext = generate_1d_mesh_from_array(array_ext)

partial_args_dict_int = {
    'dim': 1,
    'name': 'wf_int',
    'pre_mesh': pre_mesh_int,
    'fem_order': fem_order,
}

partial_args_dict_ext = {
    'dim': 1,
    'name': 'wf_ext',
    'pre_mesh': pre_mesh_ext,
    'fem_order': fem_order,
}


def right_boundary(coors, domain=None):
    return np.where(coors.squeeze() == Rc)[0]


def left_boundary(coors, domain=None):
    return np.where(coors.squeeze() == 0)[0]


dim_func_entities = [(0, right_boundary, 0), (0, left_boundary, 1)]


def density(coors):
    return np.where(coors.squeeze() <= Rsat, rho_sat, rho_vac)


density_dict = {('omega', -1): density}

partial_args_dict_int['dim_func_entities'] = dim_func_entities
partial_args_dict_ext['dim_func_entities'] = dim_func_entities
partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 1): phi_max}
chameleon = Chameleon(param_dict, 1, Rc=Rc)
chameleon.set_wf_int(partial_args_dict_int, density_dict)
chameleon.set_wf_residual(partial_args_dict_int, density_dict)
chameleon.set_wf_ext(partial_args_dict_ext, density=rho_vac)

initial_guess_dict = {
    'int': phi_max * np.ones_like(chameleon.wf_int.field.coors.squeeze()),
    'ext': phi_max * np.ones_like(chameleon.wf_ext.field.coors.squeeze())
}

chameleon.set_default_solver(
    relax_param=0.5, guess=initial_guess_dict,
    region_key_int=('vertex', 0), region_key_ext=('vertex', 0))
chameleon.set_default_monitor(max_iter=50, min_iter=5)
solver = chameleon.default_solver
# solver.relax_method = 'line-search'
monitor = solver.nonlinear_monitor
solver.solve(verbose=True,)

sol_int = solver.sol_int
rr = solver.wf_int.field.coors.squeeze()
sol_ext = solver.sol_ext
eta = solver.wf_ext.field.coors.squeeze()

fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)
axs[0].scatter(rr, sol_int, s=1)
axs[1].scatter(eta, sol_ext, s=1)
plt.show()

ana_int = chameleon_radial(rr, Rsat, rho_sat, rho_vac, alpha, npot, plot=False)
ana_ext = chameleon_radial(Rc**2/eta[1:], Rsat, rho_sat, rho_vac, alpha, npot)

fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)
axs[0].scatter(rr, ana_int, s=1)
axs[1].scatter(eta[1:], ana_ext, s=1)
plt.show()