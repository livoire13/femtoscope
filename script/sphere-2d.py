import numpy as np
import matplotlib.pyplot as plt

from femtoscope.physics.physical_problems import Chameleon
from femtoscope.misc.unit_conversion import compute_beta
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.misc.analytical import chameleon_radial

# FEM parameters
fem_order = 1
npot = 1
L0 = 1  # m
rho0 = 1  # kg/m^3
Rc = 5
Rsat = 1
rho_vac = 1e-2  # dimensionless
rho_sat = 1e2  # dimensionless
phi_min = rho_sat ** (-1 / (npot + 1))
phi_max = rho_vac ** (-1 / (npot + 1))
alpha = 1e-1
coorsys = 'cylindrical'

meshint_name = 'mesh_test_sphere_int.vtk'
meshext_name = 'mesh_test_sphere_ext.vtk'

param_dict = {
    'alpha': alpha, 'npot': npot, 'rho_min': rho_vac, 'rho_max': rho_sat
}


def mesh2dcyl_int():
    mt = MeshingTools(2)
    s1 = mt.create_ellipse(rx=Rsat, ry=Rsat)
    mt.create_subdomain(cell_size_min=0.05, cell_size_max=0.2,
                        dist_min=0.0, dist_max=4.0)
    s2 = mt.create_disk_from_pts(Rc, N=200)
    mt.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    return mt.generate_mesh(meshint_name, cylindrical_symmetry=True,
                            show_mesh=True, ignored_tags=[200])


def mesh2dcyl_ext():
    mt = MeshingTools(2)
    mt.create_disk_from_pts(Rc, N=200)
    mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    origin_rf = [0.07, 0.2, 0.1, 3.0]
    return mt.generate_mesh(
        meshext_name, cylindrical_symmetry=True, show_mesh=True,
        embed_origin=True, origin_rf=origin_rf)


pre_mesh_int, pre_mesh_ext = mesh2dcyl_int(), mesh2dcyl_ext()
partial_args_dict_int = {
    'dim': 2,
    'name': 'wf_int',
    'pre_mesh': pre_mesh_int,
    'fem_order': fem_order,
}
partial_args_dict_ext = {'dim': 2, 'name': 'wf_ext', 'pre_mesh': pre_mesh_ext,
                         'fem_order': fem_order,
                         'pre_ebc_dict': {('vertex', 0): phi_max}}

region_key_int = ('facet', 201)
region_key_ext = ('facet', 200)

density_dict = {('subomega', 300): rho_sat, ('subomega', 301): rho_vac}

chameleon = Chameleon(param_dict, 2, Rc=Rc, coorsys=coorsys)
chameleon.set_wf_int(partial_args_dict_int, density_dict)
chameleon.set_wf_residual(partial_args_dict_int, density_dict)
chameleon.set_wf_ext(partial_args_dict_ext, density=rho_vac)
chameleon.set_default_solver(region_key_int=region_key_int,
                             region_key_ext=region_key_ext)
chameleon.set_default_monitor(max_iter=50, min_iter=5)
solver = chameleon.default_solver
monitor = solver.nonlinear_monitor

# Solve Chameleon problem
solver.solve(verbose=True)
sol_int_test = solver.sol_int
sol_ext_test = solver.sol_ext

xx = np.linspace(0, Rc, 100)
yy = np.zeros_like(xx)
cc = np.concatenate((xx[:, np.newaxis], yy[:, np.newaxis]), axis=1)
field_ext = solver.wf_ext.field
line_ext = field_ext.evaluate_at(cc, sol_ext_test[:, np.newaxis]).squeeze()

plt.figure()
plt.plot(xx, line_ext)
plt.show()