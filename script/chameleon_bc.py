# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:42:53 2022

@author: hlevy
"""

import numpy as np
from femtoscope.inout.meshfactory import MeshingTools, mesh_from_geo
from femtoscope.physics.chameleon import ChameleonBounded, ChameleonPicard
from matplotlib import pyplot as plt
from numpy import sin, cos, pi, sqrt
from femtoscope import MESH_DIR

verbose = 1 # {False/0, True/1, 2}

####################################
#       Simulation Parameters      #
####################################
Rcut = 5.0 # radius for the separation of the two domains
sa = 1.0
sc = 1.0
ecc = sqrt(1-(sc/sa)**2)
rho_bounds = [1e-15, 1e3] # = [rho_min, rho_max]
alpha = 1e-10
npot = 1
phi_max = min(rho_bounds)**(-1/(npot+1))
fem_order = 1
coorsys = 'cartesian' # 'cartesian' or 'polar' or 'polar_mu'
dimension = 2
line_search_bool = False
print_kappa = False
update_mesh = True

analytic_params = {'R_A' : (sa+sc)/2,
                  'rho_in' : rho_bounds[1],
                  'rho_vac' : rho_bounds[0]}

if coorsys == 'polar' and dimension == 2:
    meshfile = "mesh_theta.vtk"
    if update_mesh:
        try:
            assert sa == sc, "flat ellipsoid in cartesian coor. only"
            mesh_from_geo('mesh_theta.geo', show_mesh=True,
                          param_dic={'size' : 0.08, 'Rc' : Rcut, 'sa' : sa})
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_theta.geo cannot be found!")
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(dx=Rcut, dy=pi, centered=False)
            MT.create_subdomain(CellSizeMin=0.05, CellSizeMax=0.05, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202, 203])

    def rho_func(coors):
        value = rho_bounds[0] * np.ones(coors.shape[0])
        norm2 = coors[:,0]**2
        theta = coors[:, 1]
        boolin = np.where(norm2*((sin(theta)/sa)**2 + (cos(theta)/sc)**2)<1)[0]
        value[boolin] = rho_bounds[1]
        return value

    densities = [rho_func]

    pb = ChameleonBounded(alpha, npot, densities, [phi_max], rho_bounds,
                          meshfile, coorsys=coorsys, fem_order=fem_order,
                          verbose=verbose, relax=0.30, max_iter=200,
                          analytic_params=analytic_params,
                          print_kappa=print_kappa)
    pb.solve(save_all_newton=False)
    
elif coorsys == 'polar_mu' and dimension == 2:
    meshfile = "mesh_mu.vtk"
    if update_mesh:
        try:
            assert sa == sc, "flat ellipsoid in cartesian coor. only"
            mesh_from_geo('mesh_mu.geo', show_mesh=True,
                          param_dic={'size' : 0.5, 'Rc' : Rcut, 'sa' : sa})
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_mu.geo cannot be found!")
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(xll=0, yll=-1, dx=Rcut, dy=2, centered=False)
            MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202, 203])
            
    def rho_func(coors):
        value = rho_bounds[0] * np.ones(coors.shape[0])
        norm2 = coors[:,0]**2
        mu = coors[:, 1]
        sin_theta = sqrt(1-mu**2)
        boolin = np.where(norm2*((sin_theta/sa)**2 + (mu/sc)**2)<1)[0]
        value[boolin] = rho_bounds[1]
        return value

    densities = [rho_func]

    pb = ChameleonBounded(alpha, npot, densities, [phi_max], rho_bounds,
                          meshfile, coorsys=coorsys, fem_order=fem_order,
                          verbose=verbose, relax=0.20, max_iter=200,
                          analytic_params=analytic_params,
                          print_kappa=print_kappa)
    pb.solve(save_all_newton=False)

elif coorsys == 'polar' and dimension == 1:
    assert sa == sc, "sa must be equal to sc in 1D!"
    # from sfepy.discrete.fem import Mesh
    from sfepy.examples.dg.example_dg_common import get_gen_1D_mesh_hook
    X1 = 0.0
    XN = Rcut
    n_nod = int(Rcut*500) + 1
    n_el = n_nod - 1
    mesh = get_gen_1D_mesh_hook(X1, XN, n_nod).read(None)

    def rho_func(r):
        return np.where(r<=sa, rho_bounds[1], rho_bounds[0])

    def bord_Rcut(r, domain=None):
        return np.where(r==Rcut)[0]

    ent_func = [(0, bord_Rcut)]
    densities = [rho_func]
    pb = ChameleonBounded(alpha, npot, densities, [phi_max], rho_bounds,
                          mesh, coorsys=coorsys, fem_order=fem_order,
                          entity_functions=ent_func, print_kappa=print_kappa,
                          line_search_bool=line_search_bool, max_iter=200,
                          min_iter=50, analytic_params=analytic_params)
    pb.solve(save_all_newton=False)

    sol1D = pb.solver.sols[0]
    res_vec = pb.solver.criteria.res_vec
    res_mtx, res_rho, res_pow = list(pb.solver.criteria.res_vec_parts.values())
    rr = pb.solver.weakforms[0].field.coors.squeeze()
    ind = rr.argsort()
    rr = rr[ind]
    sol1D = sol1D[ind]
    res_vec = res_vec[ind]
    res_mtx = res_mtx[ind]
    res_rho = res_rho[ind]
    res_pow = res_pow[ind]

    # plt.figure()
    # plt.scatter(rr, abs(res_mtx), label='mtx', alpha=0.75, s=5.8)
    # plt.scatter(rr, abs(res_rho), label='rho', alpha=0.75, s=5.8)
    # plt.scatter(rr, abs(res_pow), label='pow', alpha=0.75, s=5.8)
    # plt.scatter(rr, abs(res_vec), label='res', alpha=0.35, s=5.8)
    # plt.yscale('log')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=True)
    # plt.show()

    plt.figure()
    plt.scatter(rr[:-3], (res_mtx[:-3]), label='mtx', alpha=0.75, s=5.8)
    plt.scatter(rr[:-3], (res_rho[:-3]), label='rho', alpha=0.75, s=5.8)
    plt.scatter(rr[:-3], (res_pow[:-3]), label='pow', alpha=0.75, s=5.8)
    plt.scatter(rr[:-3], (res_vec[:-3]), label='res', alpha=0.35, s=5.8)
    # plt.yscale('log')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=True)
    plt.show()

elif coorsys == 'cartesian' and dimension == 2:

    # Meshes creation
    Ngamma = 300 # Number of vertices on the shared boundary

    '''
    The interior mesh and the exterior mesh are different:
        - the interior mesh contains a subdomain representing the main body
        - the exterior mesh represents vacum extending from Rcut up to
        infinity. Thus, it must contain a node at (0,0,0) so as to impose a
        Dirichlet boundary condition at infinity.
    In order to make it possible for the two meshes to exchange information
    at their common boundary, one must ensure that the two meshes match
    perfectly at the interface.
    Another issue that has to be dealt with are facet physical groups lying
    inside omega. Because a node can belong only to one group, such facets
    are simply ignored when generating the mesh for Sfepy.
    '''

    meshfile = 'cir.vtk'
    if not (MESH_DIR / 'cir.vtk').is_file() or update_mesh:
        MT = MeshingTools(dimension=2)
        s1 = MT.create_ellipse(sa, sc)
        # m = MT.create_ellipse(sa/10, sa/5, xc=0, yc=sa)
        # s = MT.add_shapes(s1, m)
        s = s1
        MT.create_subdomain(CellSizeMin=0.015, CellSizeMax=0.2, DistMin=sa/5,
                            DistMax=3*sa)
        s2 = MT.create_disk_from_pts(Rcut, N=Ngamma) # impose vertices on gamma
        s12 = MT.subtract_shapes(s2, s, removeObject=True, removeTool=False)
        MT.create_subdomain(CellSizeMin=0.15, CellSizeMax=0.2, DistMax=3)
        MT.generate_mesh(meshfile, ignoreTags=[200], show_mesh=True)

    densities = [rho_bounds[1], rho_bounds[0]]

    pb = ChameleonBounded(alpha, npot, densities, [phi_max], rho_bounds,
                          meshfile, coorsys=coorsys, fem_order=fem_order,
                          verbose=verbose, analytic_params=analytic_params,
                          print_kappa=print_kappa)

    pb.solve(save_all_newton=False)