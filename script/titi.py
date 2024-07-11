# Module imports
import numpy as np
from numpy import pi
from femtoscope.physics.physical_problems import Poisson, Chameleon
from femtoscope.inout.meshfactory import MeshingTools, generate_mesh_from_geo

# Meshes
def get_pre_meshes():
    return mesh2dpol_int(), mesh2dpol_ext()

def mesh2dpol_int():
    param_dict = {'Rts': 0.0, 'Rc': Rc, 'sa': sa, 'size': 0.1, 'Ngamma': 40}
    return generate_mesh_from_geo('test_theta_int.geo', param_dict=param_dict,
                                  ignored_tags=[200, 202, 203], show_mesh=True,
                                  mesh_name=meshint_name)


def mesh2dpol_ext():
    param_dict = {'Rc': Rc, 'size': 0.1, 'Ngamma': 40}
    return generate_mesh_from_geo('test_theta_ext.geo', param_dict=param_dict,
                                  ignored_tags=[200, 202], show_mesh=True,
                                  mesh_name=meshext_name)

# def mesh2dcyl_int():
#     mt = MeshingTools(2)
#     s1 = mt.create_ellipse(rx=sa, ry=sc)
#     mt.create_subdomain(cell_size_min=0.05, cell_size_max=0.2,
#                         dist_min=0.0, dist_max=4.0)
#     s2 = mt.create_disk_from_pts(Rc, N=200)
#     mt.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
#     mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
#     return mt.generate_mesh(meshint_name, cylindrical_symmetry=True,
#                             show_mesh=False, ignored_tags=[200])
#
# def mesh2dcyl_ext():
#     mt = MeshingTools(2)
#     mt.create_disk_from_pts(Rc, N=200)
#     mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
#     origin_rf = [0.07, 0.2, 0.1, 3.0]
#     return mt.generate_mesh(meshext_name, cylindrical_symmetry=True,
#                             show_mesh=False, embed_origin=True,
#                             origin_rf=origin_rf)

# parameters
sa = 1.0
sc = 1.0
Rc = 5.0
meshint_name = 'mesh_sphere_int.vtk'
meshext_name = 'mesh_sphere_ext.vtk'
fem_order = 2
dim = 2
coorsys = 'polar'
alpha = 0.1
npot = 2
rho_min = 1.0
rho_max = 1e2
phi_min = rho_max ** (-1 / (npot + 1))
phi_max = rho_min ** (-1 / (npot + 1))
param_dict = {'alpha': alpha, 'npot': npot,
              'rho_min': rho_min, 'rho_max': rho_max}

pre_mesh_int, pre_mesh_ext = get_pre_meshes()

partial_args_dict_int = {
        'dim': dim,
        'name': 'wf_int',
        'pre_mesh': pre_mesh_int,
        'fem_order': 2,
    }
density_int = {('subomega', 300): rho_max, ('subomega', 301): rho_min}

partial_args_dict_ext = {
    'dim': dim,
    'name': 'wf_ext',
    'pre_mesh': pre_mesh_ext,
    'fem_order': 2,
    'pre_ebc_dict': {('facet', 203): phi_max}
}
density_ext = rho_min

region_key_int = ('facet', 201)
region_key_ext =('facet', 201)

# Creation and setting of the 'Chameleon' object
chameleon = Chameleon(param_dict, dim, Rc=Rc, coorsys=coorsys)
chameleon.set_wf_int(partial_args_dict_int, density_int)
chameleon.set_wf_residual(partial_args_dict_int, density_int)
chameleon.set_wf_ext(partial_args_dict_ext, density=density_ext)
chameleon.set_default_solver(region_key_int=region_key_int,
                             region_key_ext=region_key_ext)
chameleon.set_default_monitor(10)

solver = chameleon.default_solver
solver.solve(verbose=True)
solver.display_results()