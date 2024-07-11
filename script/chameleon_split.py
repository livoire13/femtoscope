# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:17:04 2022

@author: hlevy
"""

import numpy as np
from femtoscope.inout.meshfactory import MeshingTools, mesh_from_geo
from femtoscope.physics.chameleon import ChameleonSplit
from numpy import sin, cos, pi, sqrt
from matplotlib import pyplot as plt

from femtoscope import MESH_DIR

verbose = 1 # {False/0, True/1, 2}

####################################
#       Simulation Parameters      #
####################################
Rcut = 5.0 # radius for the separation of the two domains
sa = 1.0
sc = 1.0
ecc = sqrt(1-(sc/sa)**2)
rho_bounds = [1e-1, 1e2] # = [rho_min, rho_max]
alpha = 1e-1
npot = 1
fem_order = 2
coorsys = 'polar' # 'cartesian' or 'polar' or 'polar_mu'
conn = 'connected' # 'connected' or 'ping-pong'
dimension = 1 # {1, 2, 3}
symmetry = True
update_mesh = True
print_kappa = False
analytic_params = {'R_A' : sa,
                   'rho_in' : max(rho_bounds),
                   'rho_vac' : min(rho_bounds)}

if coorsys == 'polar' and dimension == 2:
    meshfile_int = "mesh_theta_int.vtk"
    meshfile_ext = "mesh_theta_ext.vtk"
    if update_mesh:
        try:
            assert sa == sc, "flat ellipsoid in cartesian coor. only"
            size = 0.1
            mesh_from_geo('mesh_theta_int.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(1.5*pi/size),
                                     'Rc' : Rcut, 'sa' : sa, 'better_gamma' : 0})
            mesh_from_geo('mesh_theta_ext.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(1.5*pi/size),
                                     'Rc' : Rcut, 'better_gamma' : 0})
            meshfiles = [meshfile_int, meshfile_ext]
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_theta_int.geo cannot be found!")
            # The same mesh is used for both the interior and the exterior regions
            meshfiles = "rec.vtk"
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(dx=Rcut, dy=pi, centered=False)
            MT.create_subdomain(CellSizeMin=0.05, CellSizeMax=0.05, DistMax=0.0)
            MT.generate_mesh(meshfiles, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202])

    def rho_func(coors):
        value = rho_bounds[0] * np.ones(coors.shape[0])
        norm2 = coors[:,0]**2
        theta = coors[:, 1]
        boolin = np.where(norm2*((sin(theta)/sa)**2 + (cos(theta)/sc)**2)<1)[0]
        value[boolin] = rho_bounds[1]
        return value

    densities_int = rho_func
    densities_ext = rho_bounds[0]

    pb = ChameleonSplit(alpha, npot, densities_int, densities_ext, rho_bounds,
                        meshfiles, coorsys=coorsys, conn=conn, fem_order=fem_order,
                        analytic_params=analytic_params, relax=0.5, max_iter=200,
                        min_iter=3, verbose=verbose, print_kappa=print_kappa)
    pb.solve(save_all_newton=False)
    # pb.save()
    pb.plot(save=False)

if coorsys == 'polar_mu' and dimension == 2:

    meshfile_int = "mesh_mu_int.vtk"
    meshfile_ext = "mesh_mu_ext.vtk"
    if update_mesh:
        try:
            assert sa == sc, "flat ellipsoid in cartesian coor. only"
            size = 0.1
            mesh_from_geo('mesh_mu_int.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(1.5*2/size),
                                     'Rc' : Rcut, 'sa' : sa, 'better_gamma' : 0})
            mesh_from_geo('mesh_mu_ext.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(1.5*2/size),
                                     'Rc' : Rcut, 'better_gamma' : 0})
            meshfiles = [meshfile_int, meshfile_ext]
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_mu_int.geo cannot be found!")
            meshfile = "rec.vtk"
            # The same mesh is used for both the interior and the exterior regions
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(xll=0, yll=-1, dx=Rcut, dy=2, centered=False)
            MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202])
            meshfiles = [meshfile, meshfile]

    def rho_func(coors):
        value = rho_bounds[0] * np.ones(coors.shape[0])
        norm2 = coors[:,0]**2
        mu = coors[:, 1]
        sin_theta = sqrt(1-mu**2)
        boolin = np.where(norm2*((sin_theta/sa)**2 + (mu/sc)**2)<1)[0]
        value[boolin] = rho_bounds[1]
        return value

    densities_int = rho_func
    densities_ext = rho_bounds[0]

    pb = ChameleonSplit(alpha, npot, densities_int, densities_ext, rho_bounds,
                        meshfiles, coorsys=coorsys, conn=conn, relax=1,
                        fem_order=fem_order, max_iter=200, initial_guess='min_pot',
                        analytic_params=analytic_params, verbose=verbose,
                        print_kappa=print_kappa)
    pb.solve(save_all_newton=True)
    # pb.plot(save=False)

elif coorsys == 'polar' and dimension == 1:
    from sfepy.examples.dg.example_dg_common import get_gen_1D_mesh_hook
    X1 = 0.0
    XN = Rcut
    n_nod = int(Rcut*1500) + 1
    n_el = n_nod - 1
    mesh = get_gen_1D_mesh_hook(X1, XN, n_nod).read(None)

    def bord_zero(r, domain=None):
        return np.where(r==0)[0]

    def bord_Rcut(r, domain=None):
        return np.where(r==Rcut)[0]

    def rho_func(r):
        return np.where(r<=sa, rho_bounds[1], rho_bounds[0])

    densities = [rho_func, rho_bounds[0]]

    ent_func_in = [(0, bord_Rcut)]
    ent_func_out = [(0, bord_zero), (0, bord_Rcut)]

    analytic_params = {'R_A' : sa,
                       'rho_in' : rho_bounds[1],
                       'rho_vac' : rho_bounds[0]}
    pb = ChameleonSplit(alpha, npot, rho_func, rho_bounds[0], rho_bounds,
                        [mesh, mesh], coorsys=coorsys, conn=conn, Rcut=Rcut,
                        entity_functions_in=ent_func_in,
                        entity_functions_out=ent_func_out,
                        fem_order=fem_order, verbose=verbose,
                        analytic_params=analytic_params,
                        max_iter=200, print_kappa=print_kappa)

    pb.solve(save_all_newton=False)

    sol_cham = pb.solver.sols[0]
    rr = pb.solver.weakforms[0].field.coors.squeeze()
    ind = rr.argsort()
    rr = rr[ind]
    sol_cham = sol_cham[ind]
    plt.figure()
    xmin, xmax = 0, Rcut
    plt.plot(rr, sol_cham)
    plt.xlim(0, 4)
    plt.show()

    sol_cham = pb.solver.sols[1]
    eta = pb.solver.weakforms[1].field.coors.squeeze()
    ind = eta.argsort()
    eta = eta[ind]
    sol_cham = sol_cham[ind]
    plt.figure()
    xmin, xmax = 0, Rcut
    plt.plot(eta, sol_cham)
    plt.show()

    # # Investigation of the residual
    # rr = pb.solver.weakforms[0].field.coors[:-1]
    # res_vec = pb.solver.criteria.res_vec
    # plt.figure()
    # plt.scatter(rr, res_vec, s=1)
    # plt.title(r"$\alpha = $ {}".format(alpha), fontsize=17)
    # plt.xlabel(r"$\hat{r}$", fontsize=15)
    # plt.ylabel(r"Residual vector", fontsize=15)
    # plt.ylim([-1e-6, 1e-6])
    # plt.show()

    # # terms of the residual
    # rr = pb.solver.weakforms[0].field.coors.squeeze()
    # idx = np.where(rr==5.0)[0]
    # t1 = pb.solver.mtxvec_dic['mtx_cst_res']
    # t1[idx, :] = 0.0
    # t1[:, idx] = 0.0
    # t1 = t1.dot(pb.solver.sols[0])
    # t1[abs(t1)>1] = np.nan
    # t2 = pb.solver.mtxvec_dic['rhs_cst_res']
    # t2[abs(t2)>1] = np.nan
    # t3 = pb.solver.mtxvec_dic['rhs_mod_res']
    # plt.figure()
    # plt.scatter(rr, t1, s=1)
    # plt.scatter(rr, t2+t3, s=1)
    # # plt.scatter(rr, t3, s=1)
    # plt.title(r"$\alpha = $ {}".format(alpha), fontsize=17)
    # plt.xlabel(r"$\hat{r}$", fontsize=15)
    # plt.show()

    # plt.figure()
    # plt.scatter(rr, abs(t1-(t2+t3))/abs(t1), s=1)
    # plt.title(r"$\alpha = $ {}".format(alpha), fontsize=17)
    # plt.xlabel(r"$\hat{r}$", fontsize=15)
    # plt.show()

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

    meshfile1 = 'cir_in.vtk'
    meshfile2 = 'cir_out.vtk'
    if not (MESH_DIR / 'cir_in.vtk').is_file() or update_mesh:
        MT = MeshingTools(dimension=2)
        s1 = MT.create_ellipse(sa, sc)
        s = s1
        # m = MT.create_ellipse(sa/10, sa/5, xc=0, yc=sa)
        # s = MT.add_shapes(s1, m)
        MT.create_subdomain(CellSizeMin=0.05, CellSizeMax=0.2, DistMin=sa/5,
                            DistMax=3*sa)
        s2 = MT.create_disk_from_pts(Rcut, N=Ngamma) # impose vertices on gamma
        s12 = MT.subtract_shapes(s2, s, removeObject=True, removeTool=False)
        MT.create_subdomain(CellSizeMin=0.15, CellSizeMax=0.2, DistMax=3)
        MT.generate_mesh(meshfile1, show_mesh=True, symmetry=symmetry,
                         ignoreTags=[200])

        MT = MeshingTools(dimension=2)
        MT.create_disk_from_pts(Rcut, N=Ngamma) # impose vertices on gamma
        MT.create_subdomain(CellSizeMin=0.15, CellSizeMax=0.4, DistMin=0.0,
                          DistMax=2)
        center_rf = [0.05, 0.15, 0.1, 3]
        MT.generate_mesh(meshfile2, show_mesh=True, embed_center=True,
                         symmetry=symmetry, center_rf=center_rf)

    densities_int = [rho_bounds[1], rho_bounds[0]]
    densities_ext = rho_bounds[0]
    meshfiles = [meshfile1, meshfile2]

    pb = ChameleonSplit(alpha, npot, densities_int, densities_ext, rho_bounds,
                        meshfiles, coorsys=coorsys, conn=conn,
                        fem_order=fem_order, verbose=verbose,
                        print_kappa=print_kappa)
    pb.solve(save_all_newton=False)
    pb.plot(save=False)


elif coorsys == 'cartesian' and dimension == 3:
    densities_int = [rho_bounds[1], rho_bounds[0]]
    densities_ext = rho_bounds[0]
    meshfile = 'sphere.vtk'
    center_rf = [1, 1, 1, 1]
    MT = MeshingTools(dimension=3)
    s1 = MT.create_ellipsoid(sa, sa, sa)
    MT.create_subdomain(CellSizeMin=0.7, CellSizeMax=0.7)
    s2 = MT.create_ellipsoid(5, 5, 5)
    s12 = MT.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    MT.create_subdomain(CellSizeMin=0.7, CellSizeMax=0.7)
    out = MT.generate_mesh(meshfile, show_mesh=True, convert=True,
                           exterior=True, center_rf=center_rf,
                           ignoreTags=[200])
    pb = ChameleonSplit(alpha, npot, densities_int, densities_ext, rho_bounds,
                        out, coorsys=coorsys, conn=conn, print_kappa=print_kappa,
                        fem_order=fem_order, verbose=verbose)
    pb.solve(save_all_newton=False)






