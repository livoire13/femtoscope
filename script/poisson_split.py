# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:16:23 2022

Solving Poisson's equation for ellipsoidal shapes in 2D using two different
coordinate-systems. A weight function is introduced so as to kill the
singularity at virtual infinity.

@author: hlevy
"""
import numpy as np
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.inout.meshfactory import mesh_from_geo
from femtoscope.physics.poisson import PoissonSplit
from femtoscope.misc.analytical import potential_sphere, potential_ellipsoid
from numpy import sin, cos, pi, sqrt, arccos
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from pathlib import Path

verbose = True # {False/0, True/1, 2}
compute_mesh = True

####################################
#       Simulation Parameters      #
####################################
Rcut = 5.0 # radius for the separation of the two domains
sa = 1.0 # semi-major axis
sc = 1.0 # semi-minor axis
ecc = sqrt(1-(sc/sa)**2) # eccentricity
# mass = 10.0
rho = 1.0 #3*mass/(4*pi*sqrt(1-ecc**2)*sa**3)
mass = 4*pi*sa**3/3 * rho
alpha = 4*pi
fem_order = 2
coorsys = 'polar' # 'cartesian' or 'polar' or 'polar_mu'
conn = 'connected' # 'connected' or 'ping-pong'
dimension = 2
symmetry = True
print_kappa = False
ana_comparison = True

if dimension >= 2:
# Random sampling
    np.random.seed(503)
    X_rand = np.random.uniform(-Rcut, Rcut, 10000)
    np.random.seed(1165)
    Y_rand = np.random.uniform(-Rcut, Rcut, 10000)
    coors_cart_rand = np.concatenate(
        (X_rand[:, np.newaxis], Y_rand[:, np.newaxis]), axis=1)
    coors_cart_rand = coors_cart_rand[np.where(
        np.linalg.norm(coors_cart_rand, axis=1)<0.95*Rcut)]
    X_rand, Y_rand = coors_cart_rand[:, 0], coors_cart_rand[:, 1]
    coors_cart_2d = np.ascontiguousarray(coors_cart_rand)
    coors_cart_3d = np.ascontiguousarray(np.insert(coors_cart_rand, 1, 0, axis=1))
    rr = np.linalg.norm(coors_cart_2d, axis=1)
    if ecc == 0 or sc == sa:
        sol_ana = potential_sphere(rr, sa, 1.0, rho=rho).squeeze()
    else:
        sol_ana = potential_ellipsoid(coors_cart_3d, sa, 1.0, rho=rho, sc=sc)

if coorsys == 'polar' and dimension == 2:
    
    meshfile_int = "mesh_theta_int.vtk"
    meshfile_ext = "mesh_theta_ext.vtk"
    if compute_mesh:
        try:
            assert sa == sc
            size = 0.3
            mesh_from_geo('mesh_theta_int.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(1.5*pi/size),
                                     'Rc' : Rcut, 'sa' : sa})
            mesh_from_geo('mesh_theta_ext.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(1.5*pi/size),
                                     'Rc' : Rcut})
            meshfiles = [meshfile_int, meshfile_ext]
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_theta_int.geo cannot be found!")
            meshfile = "rec.vtk"
            # The same mesh is used for both the interior and the exterior regions
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(dx=Rcut, dy=pi, centered=False)
            MT.create_subdomain(CellSizeMin=0.05, CellSizeMax=0.03, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202])
            meshfiles = [meshfile, meshfile]

    def rho_func(coors):
        value = np.zeros(coors.shape[0])
        norm2 = coors[:,0]**2
        theta = coors[:, 1]
        boolin = np.where(norm2*((sin(theta)/sa)**2+(cos(theta)/sc)**2)<1)[0]
        value[boolin] = rho
        return value

    densities_int = [rho_func]
    densities_ext = None
    pb = PoissonSplit(alpha, densities_int, densities_ext, meshfiles,
                      coorsys=coorsys, conn=conn, fem_order=fem_order,
                      relax_pp=0.86, max_iter_pp=30, verbose=verbose,
                      print_kappa=print_kappa)
    pb.solve()
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    theta = arccos(Y_rand/rr)
    coors_eval = np.ascontiguousarray(
        np.concatenate((rr[:, np.newaxis], theta[:, np.newaxis]), axis=1))
    fem_2 = field_2.evaluate_at(coors_eval, sol_2[:, np.newaxis]).squeeze()
    print("#DOFs = {}".format(sol_2.shape[0]+pb.solver.weakforms[1].field.coors.shape[0]))
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))
    
if coorsys == 'polar_mu' and dimension == 2:
    
    meshfile_int = "mesh_mu_int.vtk"
    meshfile_ext = "mesh_mu_ext.vtk"
    if compute_mesh:
        try:
            assert sa == sc
            size = 0.1
            mesh_from_geo('mesh_mu_int.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(3*2/size),
                                     'Rc' : Rcut, 'sa' : sa, 'better_gamma' : 1})
            mesh_from_geo('mesh_mu_ext.geo', show_mesh=True,
                          param_dic={'size' : size, 'Ngamma' : int(3*2/size),
                                     'Rc' : Rcut, 'better_gamma' : 1})
            meshfiles = [meshfile_int, meshfile_ext]
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_mu.geo cannot be found!")
            meshfile = "rec.vtk"
            # The same mesh is used for both the interior and the exterior regions
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(xll=0, yll=-1, dx=Rcut, dy=2, centered=False)
            MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202])
            meshfiles = [meshfile, meshfile]

    def rho_func(coors):
        value = np.zeros(coors.shape[0])
        norm2 = coors[:,0]**2
        theta = coors[:, 1]
        boolin = np.where(norm2*((sin(theta)/sa)**2+(cos(theta)/sc)**2)<1)[0]
        value[boolin] = rho
        return value

    densities_int = [rho_func]
    densities_ext = None
    pb = PoissonSplit(alpha, densities_int, densities_ext, meshfiles,
                      coorsys=coorsys, conn=conn, fem_order=fem_order,
                      relax_pp=0.86, max_iter_pp=30, verbose=verbose,
                      print_kappa=print_kappa)
    pb.solve()
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    theta = arccos(Y_rand/rr)
    mu = cos(theta)
    coors_eval = np.ascontiguousarray(
        np.concatenate((rr[:, np.newaxis], mu[:, np.newaxis]), axis=1))
    fem_2 = field_2.evaluate_at(coors_eval, sol_2[:, np.newaxis]).squeeze()
    print("#DOFs = {}".format(sol_2.shape[0]+pb.solver.weakforms[1].field.coors.shape[0]))
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))

if coorsys == 'polar' and dimension == 1:
    assert sa == sc, "sa must be equal to sc in 1D!"
    # from sfepy.discrete.fem import Mesh
    from sfepy.examples.dg.example_dg_common import get_gen_1D_mesh_hook
    X1 = 0.0
    XN = Rcut
    n_nod = 101
    n_el = n_nod - 1
    mesh = get_gen_1D_mesh_hook(X1, XN, n_nod).read(None)

    def rho_func(r):
        return np.where(r<=sa, rho, 0.0)

    def bord_zero(r, domain=None):
        return np.where(r==0)[0]

    def bord_Rcut(r, domain=None):
        return np.where(r==Rcut)[0]

    ent_func_int = [(0, bord_Rcut)]
    ent_func_ext = [(0, bord_zero), (0, bord_Rcut)]

    pb = PoissonSplit(alpha, rho_func, None, [mesh, mesh], Rcut=Rcut,
                      coorsys=coorsys, conn=conn, fem_order=fem_order,
                      entity_functions_int=ent_func_int, verbose=verbose,
                      entity_functions_ext=ent_func_ext,
                      print_kappa=print_kappa)
    pb.solve()

# Connected meshes in cartesian coordinates
if coorsys == 'cartesian' and dimension == 2:

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

    meshfile1 = 'cir_int.vtk'
    MT = MeshingTools(dimension=2)
    s1 = MT.create_ellipse(sa, sc, xc=0, yc=0)
    MT.create_subdomain(CellSizeMin=0.03, CellSizeMax=0.2, DistMin=sa/5,
                        DistMax=3*sa)
    s2 = MT.create_disk_from_pts(Rcut, N=Ngamma) # impose vertices on gamma
    s12 = MT.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.7, DistMax=0.0)
    MT.generate_mesh(meshfile1, show_mesh=True, ignoreTags=[200],
                     symmetry=symmetry)

    meshfile2 = 'cir_ext.vtk'
    MT = MeshingTools(dimension=2)
    MT.create_disk_from_pts(Rcut, N=Ngamma) # impose vertices on gamma
    MT.create_subdomain(CellSizeMin=0.2, CellSizeMax=0.2, DistMin=0.0,
                        DistMax=3, NumPointsPerCurve=Ngamma)
    center_rf = [0.07, 0.2, 0.1, 3]
    MT.generate_mesh(meshfile2, show_mesh=True, embed_center=True,
                     center_rf=center_rf, symmetry=symmetry)

    densities_int = rho
    pb = PoissonSplit(alpha, densities_int, 0, [meshfile1, meshfile2],
                      coorsys=coorsys, conn=conn, fem_order=fem_order,
                      verbose=verbose, print_kappa=print_kappa)
    pb.solve()


####################################
#          Comparison              #
####################################
figsize = (12.0, 7.0)

if dimension == 1:
    sol = pb.solver.sols[0]
    rr = pb.solver.weakforms[0].field.coors
    rr = rr.squeeze()
    ind = rr.argsort()
    rr = rr[ind]
    sol = sol[ind]
    ana = potential_sphere(rr, sa, alpha/(4*pi), M=mass).squeeze()
    plt.figure(figsize=(8,5))
    plt.plot(rr, abs((sol-ana)/ana))
    fmt = lambda y, pos,: '{:.2E}'.format(y)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(fmt))
    plt.xlabel(r'$r$', fontsize=15)
    plt.ylabel('Relative Error', fontsize=15)
    plt.title("Interior domain", fontsize=17)
    plt.grid()
    plt.show()
    
    sol_ext = pb.solver.sols[1]
    eta = (pb.solver.weakforms[1].field.coors).squeeze()
    ind = eta.argsort()
    eta = eta[ind]
    sol_ext = sol_ext[ind]
    ana = potential_sphere(Rcut**2/eta, sa, alpha/(4*pi), M=mass).squeeze()
    ind_tronc = np.where(eta>1)[0][0]
    eta = eta[ind_tronc:]
    ana = ana[ind_tronc:]
    sol_ext = sol_ext[ind_tronc:]
    plt.figure(figsize=(8,5))
    plt.plot(eta, abs((sol_ext-ana)/ana))
    fmt = lambda y, pos,: '{:.2E}'.format(y)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(fmt))
    plt.xlabel(r'$\eta$', fontsize=15)
    plt.ylabel('Relative Error', fontsize=15)
    plt.title("Exterior domain", fontsize=17)
    plt.grid()
    plt.show()
    
if dimension == 2:
    field = pb.solver.weakforms[0].field
    sol = pb.solver.sols[0]

    if ana_comparison:
        nblevels = 500
        X = np.linspace(-2*sa, 2*sa, 200)
        Z = np.linspace(-2*sc, 2*sc, 200)
        x, z = np.meshgrid(X, Z)
        coors = np.array(list(zip(x.ravel(), z.ravel())))
        X = coors[:, 0] ; Z = coors[:, 1]
        coors_ana = np.insert(coors, 1, 0, axis=1)
        rr = sqrt(coors[:,0]**2 + coors[:,1]**2)
        if coorsys == 'polar':
            theta = arccos(Z/rr)
            coors_fem = np.ascontiguousarray(np.concatenate(
                (rr[:, np.newaxis], theta[:, np.newaxis]), axis=1))
        elif coorsys == 'cartesian':
            coors_fem = np.copy(np.ascontiguousarray(coors))
            if symmetry:
                coors_fem[:, 0] = abs(coors_fem[:, 0])
        fem_sol = np.squeeze(field.evaluate_at(coors_fem, sol[:, np.newaxis]))
        if sa == sc or ecc == 0:
            ana_sol = potential_sphere(rr, sa, alpha/(4*pi), rho=rho).squeeze()
        else:
            ana_sol = potential_ellipsoid(coors_ana, sa, alpha/(4*pi), sc=sc,
                                          rho=rho)
        cm = plt.get_cmap('viridis')
        vmin=min(fem_sol.min(), ana_sol.min())
        vmax=max(fem_sol.max(), ana_sol.max())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ctf1 = ax1.tricontourf(X, Z, ana_sol, nblevels, vmin=vmin, vmax=vmax,
                               cmap=cm)
        lvls = ctf1.levels[::int(nblevels/10)]
        ct1 = ax1.tricontour(X, Z, ana_sol, lvls, colors='k')
        ctf2 = ax2.tricontourf(X, Z, fem_sol, nblevels, vmin=vmin, vmax=vmax,
                               cmap=cm)
        ct2 = ax2.tricontour(X, Z, fem_sol, lvls, colors='k')
        ax1.axis('equal')
        ax2.axis('equal')
        plt.suptitle("Ellipsoid Potential", fontsize=15)
        ax1.set_title("Analytical Solution", fontsize=11)
        ax2.set_title("FEM Solution", fontsize=11)
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin,
                                                               vmax=vmax))
        fig.colorbar(sm, cax=cbar_ax)
        plt.show()
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ctf = ax.tricontourf(X, Z, abs((ana_sol-fem_sol)/ana_sol), 50, cmap=cm)
        ct = ax.tricontour(X, Z, (X/sa)**2+(Z/sc)**2, [1], colors='r')
        ax.axis('equal')
        ax.set_title('Relative error - {} - {}'.format(coorsys, conn),
                     fontsize=18)
        fmt = lambda x, pos: '{:.2E}'.format(x)
        plt.xlabel(r'$x$', fontsize=15)
        plt.ylabel(r'$y$', fontsize=15)
        fig.colorbar(ctf, format=FuncFormatter(fmt))
        plt.show()


if dimension == 3:

    meshfile = 'sphere.vtk'
    center_rf = [1, 1, 1, 1]
    MT = MeshingTools(dimension=3)
    s1 = MT.create_ellipsoid(sa, sa, sa)
    MT.create_subdomain(CellSizeMin=0.4, CellSizeMax=0.4)
    s2 = MT.create_ellipsoid(5, 5, 5)
    s12 = MT.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    MT.create_subdomain(CellSizeMin=0.4, CellSizeMax=0.6)
    out = MT.generate_mesh(meshfile, show_mesh=True, convert=True,
                           exterior=True, center_rf=center_rf,
                           ignoreTags=[200])

    densities_int = [rho, 0.0]
    pb = PoissonSplit(alpha, densities_int, None,
                      [Path(out[0]).stem, Path(out[1]).stem], coorsys=coorsys,
                      conn=conn, fem_order=fem_order, verbose=verbose)
    pb.solve()
    
    sol_3 = pb.solver.sols[0]
    field_3 = pb.solver.weakforms[0].field
    np.random.seed(503)
    X_rand = np.random.uniform(-Rcut, Rcut, 10000)
    np.random.seed(1165)
    Y_rand = np.random.uniform(-Rcut, Rcut, 10000)
    coors_cart_rand = np.concatenate(
        (X_rand[:, np.newaxis], Y_rand[:, np.newaxis]), axis=1)
    coors_cart_rand = coors_cart_rand[np.where(
        np.linalg.norm(coors_cart_rand, axis=1)<0.95*Rcut)]
    X_rand, Y_rand = coors_cart_rand[:, 0], coors_cart_rand[:, 1]
    coors_cart_2d = np.ascontiguousarray(coors_cart_rand)
    coors_cart_3d = np.ascontiguousarray(np.insert(coors_cart_rand, 1, 0, axis=1))
    rr = np.linalg.norm(coors_cart_2d, axis=1)
    sol_ana = potential_sphere(rr, 1.0, 1.0, M=mass).squeeze()
    fem_3 = field_3.evaluate_at(coors_cart_3d, sol_3[:, np.newaxis]).squeeze()
    print("mean pointwise error: {:.2E}".format(
        np.mean(abs((fem_3-sol_ana)/sol_ana))))


