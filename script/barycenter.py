# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:28:55 2024

Solving chameleon Klein-Gordon equation for two spheres of arbitrary radius,
relative positions and density.

@author: hlevy
"""

from pathlib import Path

import numpy as np

from femtoscope.inout.meshfactory import generate_mesh_from_geo
from femtoscope.inout.meshfactory import adjust_boundary_nodes
from femtoscope.physics.physical_problems import Chameleon
from femtoscope.misc.unit_conversion import compute_alpha
from femtoscope.misc.constants import G_GRAV, LAMBDA_DE
from femtoscope import MESH_DIR

# Options
verbose = 1
rebuild_mesh = True
show_mesh = True

# Chameleon parameters
rho0 = 1  # kg/m^3
L0 = 0.1  # m
npot = 1
Lambda = LAMBDA_DE
beta = 1e2
alpha = compute_alpha(Lambda, beta, npot, L0, rho0)

# Mesh parameters
R1 = 0.5  # radius of sphere no 1
R2 = 1.5  # radius of sphere no 2
d = 3.2  # distance between the two spheres
Rc = 6  # truncation radius
minSize = 0.01
maxSize = 0.2
Ngamma = 170

# Densities & chameleon parameters
rho_sp1 = 2700
rho_sp2 = 1e2
rho_vac = 1e-1
rho_max = max([rho_sp1, rho_sp2])
phi_max = rho_vac ** (-1 / (npot + 1))
mass_sph = 4/3 * np.pi * (R1 * L0)**3 * rho_sp1

assert d > R1 + R2  # non-overlapping spheres
assert abs((R1/R2)**3 - rho_sp2/rho_sp1) < 1e-5

param_dict_int = {'R1': R1,
                  'R2': R2,
                  'd': d,
                  'Rcut': Rc,
                  'Ngamma': Ngamma,
                  'minSize': minSize,
                  'maxSize': maxSize}

param_dict_ext = {'Rcut': Rc,
                  'Ngamma': Ngamma,
                  'minSize': minSize,
                  'maxSize': maxSize}

if rebuild_mesh:
    if verbose:
        print("...rebuilding the meshes...")
    mesh_int = generate_mesh_from_geo('spheres-2d_int.geo', show_mesh=show_mesh,
                                      param_dict=param_dict_int)
    pre_mesh_int = Path(mesh_int).name
    mesh_ext = generate_mesh_from_geo('spheres-2d_ext.geo', show_mesh=show_mesh,
                                      param_dict=param_dict_ext)
    pre_mesh_ext = Path(mesh_ext).name
else:
    if verbose:
        print("...reading existing meshes...")
    pre_mesh_int = 'spheres-2d_int.vtk'
    pre_mesh_ext = 'spheres-2d_ext.vtk'

# Adjust boundary nodes of the exterior mesh to match those of the interior mesh
file_ref = str(Path(MESH_DIR / pre_mesh_int))
file_mod = str(Path(MESH_DIR / pre_mesh_ext))
adjust_boundary_nodes(file_ref, file_mod, 200, 200, verbose=verbose)

# Chameleon problem
param_dict = {'alpha': alpha, 'npot': npot,
              'rho_min': rho_vac, 'rho_max': rho_max}
chameleon = Chameleon(param_dict, 2, Rc=Rc, coorsys='cylindrical')
partial_args_dict_int = {
    'dim': 2,
    'name': 'wf_int',
    'pre_mesh': pre_mesh_int,
    'fem_order': 2,
}
partial_args_dict_ext = {
    'dim': 2,
    'name': 'wf_ext',
    'pre_mesh': pre_mesh_ext,
    'fem_order': 2,
    'pre_ebc_dict': {('vertex', 0): phi_max}
}
region_key_int = ('facet', 200)
region_key_ext = ('facet', 200)
density_dict = {('subomega', 300): rho_sp1, ('subomega', 301): rho_sp2,
                ('subomega', 302): rho_vac}
chameleon.set_wf_int(partial_args_dict_int, density_dict)
chameleon.set_wf_residual(partial_args_dict_int, density_dict)
chameleon.set_wf_ext(partial_args_dict_ext, density=rho_vac)
chameleon.set_default_solver(relax_param=0.95,
                             region_key_int=region_key_int,
                             region_key_ext=region_key_ext)
chameleon.set_default_monitor(50, 5)

# Solving stage
solver = chameleon.default_solver
monitor = solver.nonlinear_monitor
solver.solve(verbose=verbose)
solver.save_results("barycenter-fem-2")
