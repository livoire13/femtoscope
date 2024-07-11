# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:31:23 2022

Compute the Newtonian potential inside and outside an ellipsoid of revolution.
The underlying Poisson equation is solved on a truncated domain. The exact
Dirichlet boundary condition is applied on the artificial border.

@author: hlevy
"""

import numpy as np
from numpy import sin, cos, pi, sqrt, arccos
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.inout.meshfactory import mesh_from_geo
from femtoscope.physics.poisson import PoissonBounded
from femtoscope.misc.analytical import potential_sphere, potential_ellipsoid
from matplotlib import pyplot as plt
import time

verbose = True # {False/0, True/1, 2}
compute_mesh = True

####################################
#       Simulation Parameters      #
####################################
Rcut = 5.0 # radius for artificial boundary
sa = 1.0
sc = 1.0
ecc = sqrt(1-(sc/sa)**2)
# mass = 10.0
# rho = 3*mass/(4*pi*sqrt(1-ecc**2)*sa**3)
rho = 1.0
alpha = 4*pi
fem_order = 3
coorsys = 'polar' # 'cartesian' or 'polar' or 'polar_mu'
dimension = 2
symmetry = True
print_kappa = True # print stiffness matrix condition number
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

# Define all BC functions
def dbc_cart2(ts, coors, **kwargs):
    r = np.linalg.norm(coors, axis=1)
    if sa == sc:
        return potential_sphere(r, sa, 1, rho=rho).reshape(-1, 1)
    else:
        coors_cart = np.insert(coors, 1, 0, axis=1)
        return potential_ellipsoid(coors_cart, sa, 1.0, sc=sc, rho=rho)

def dbc_pol2(ts, coors, **kwargs):
    r, theta = coors[:, 0], coors[:, 1]
    X, Y = r*cos(theta), r*sin(theta)
    coors = np.concatenate((X[:, np.newaxis], Y[:, np.newaxis]), axis=1)
    return dbc_cart2(ts, coors, **kwargs)

def dbc_cart3(ts, coors, **kwargs):
    r = np.linalg.norm(coors[:, 0:2], axis=1)
    coors = np.concatenate((r[:, np.newaxis], coors[:, 2][:, np.newaxis]),
                           axis=1)
    return dbc_cart2(ts, coors, **kwargs)


if coorsys == 'cartesian' and dimension == 2:

    Ngamma = 300
    meshfile = 'circle.vtk'
    if compute_mesh:
        MT = MeshingTools(dimension=2)
        s1 = MT.create_ellipse(sa, sc, xc=0, yc=0)
        MT.create_subdomain(CellSizeMin=0.05, CellSizeMax=0.2, DistMin=sa/5,
                            DistMax=3*sa)
        s2 = MT.create_disk_from_pts(Rcut, N=Ngamma) # impose vertices on gamma
        s12 = MT.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
        MT.create_subdomain(CellSizeMin=0.1, CellSizeMax=0.3, DistMax=0.0)
        MT.generate_mesh(meshfile, show_mesh=True, ignoreTags=[200],
                         symmetry=symmetry, embed_center=False)

    densities = [rho, 0.0]
    dirichlet_bc = [dbc_cart2]

    pb = PoissonBounded(alpha, densities, dirichlet_bc, meshfile,
                        coorsys=coorsys, fem_order=fem_order, verbose=verbose,
                        print_kappa=print_kappa)
    start_time = time.time()
    pb.solve()
    print("--- %s second(s) ---" % (time.time() - start_time))

    # compute the mean pointwise error
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    fem_2 = field_2.evaluate_at(coors_cart_2d, sol_2[:, np.newaxis]).squeeze()
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))

elif coorsys == 'cartesian' and dimension == 3:

    meshfile = 'sphere.vtk'
    if compute_mesh:
        MT = MeshingTools(dimension=3)
        s1 = MT.create_ellipsoid(sa, sa, sa)
        MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMin=sa/5,
                            DistMax=3*sa)
        s2 = MT.create_ellipsoid(Rcut, Rcut, Rcut)
        s12 = MT.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
        MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.7, DistMax=0.0)
        out = MT.generate_mesh(meshfile, show_mesh=True, convert=True,
                               ignoreTags=[200], embed_center=False)
    densities = [rho, 0]
    dirichlet_bc = [dbc_cart3]

    pb = PoissonBounded(alpha, densities, dirichlet_bc, meshfile,
                        coorsys=coorsys, fem_order=fem_order, verbose=verbose,
                        print_kappa=print_kappa)
    start_time = time.time()
    pb.solve()
    print("--- %s second(s) ---" % (time.time() - start_time))

    # compute the mean pointwise error
    sol_3 = pb.solver.sols[0]
    field_3 = pb.solver.weakforms[0].field
    fem_3 = field_3.evaluate_at(coors_cart_3d, sol_3[:, np.newaxis]).squeeze()
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_3-sol_ana)/sol_ana))))


elif coorsys == 'polar' and dimension == 1:
    assert sa == sc, "sa must be equal to sc in 1D!"
    # from sfepy.discrete.fem import Mesh
    from sfepy.examples.dg.example_dg_common import get_gen_1D_mesh_hook
    X1 = 0.0
    XN = Rcut
    n_nod = int(100*Rcut) + 1
    n_el = n_nod - 1
    mesh = get_gen_1D_mesh_hook(X1, XN, n_nod).read(None)

    def rho_func(r):
        return np.where(r<=sa, rho, 0.0)

    def bord_Rcut(r, domain=None):
        return np.where(r==Rcut)[0]

    ent_func = [(0, bord_Rcut)]
    densities = [rho_func]
    dirichlet_bc = [potential_sphere(Rcut, sa, 1.0, rho=rho)]
    pb = PoissonBounded(alpha, densities, dirichlet_bc, mesh, coorsys=coorsys,
                        entity_functions=ent_func, fem_order=fem_order,
                        verbose=verbose, print_kappa=print_kappa)
    pb.solve()
    sol_1 = pb.solver.sols[0]
    field_1 = pb.solver.weakforms[0].field
    rr = field_1.coors.squeeze()
    indsort = rr.argsort()
    sol_1 = sol_1[indsort]
    rr = rr[indsort]
    sol_ana = potential_sphere(rr, sa, 1.0, rho=rho)
    err_rel = abs((sol_ana-sol_1)/sol_ana)
    plt.figure()
    plt.plot(rr, err_rel)
    plt.yscale("log")
    plt.xlabel(r"$r$", fontsize=15)
    plt.ylabel("relative error", fontsize=15)
    plt.title(r"$N_{\mathrm{nodes}} = %d$ " %n_nod, fontsize=17)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5,4))
    plt.plot(rr, sol_ana, label='exact')
    plt.plot(rr, sol_1, label='FEM')
    plt.legend()
    plt.xlabel(r"$r$ (radial distance - dimensionless)", fontsize=13)
    plt.ylabel(r"$\Phi$ (Gravitational potential - dimensionless)", fontsize=13)
    plt.xlim([0.0, Rcut])
    plt.tight_layout()
    # plt.savefig("wrong_dbc1.png", dpi=500)
    plt.show()


elif coorsys == 'polar' and dimension == 2:

    meshfile = "mesh_theta.vtk"
    if compute_mesh:
        try:
            assert sa == sc
            mesh_from_geo('mesh_theta.geo', show_mesh=True,
                          param_dic={'size' : 0.1, 'Rc' : Rcut, 'sa' : sa})
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_theta.geo cannot be found!")
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(dx=Rcut, dy=pi, centered=False)
            MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202, 203])

    def rho_func(coors):
        value = np.zeros(coors.shape[0])
        norm2 = coors[:,0]**2
        theta = coors[:, 1]
        boolin = np.where(norm2*((sin(theta)/sa)**2+(cos(theta)/sc)**2)<1)[0]
        value[boolin] = rho
        return value

    densities = [rho_func]
    dirichlet_bc = [dbc_pol2]
    pb = PoissonBounded(alpha, densities, dirichlet_bc, meshfile,
                        coorsys=coorsys, fem_order=fem_order, verbose=verbose,
                        print_kappa=print_kappa)
    pb.solve()
    pb.save()
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    theta = arccos(Y_rand/rr)
    coors_eval = np.ascontiguousarray(
        np.concatenate((rr[:, np.newaxis], theta[:, np.newaxis]), axis=1))
    fem_2 = field_2.evaluate_at(coors_eval, sol_2[:, np.newaxis]).squeeze()
    print("#DOFs = {}".format(sol_2.shape[0]))
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))

elif coorsys == 'polar_mu' and dimension == 2:
    meshfile = "mesh_mu.vtk"
    if compute_mesh:
        try:
            assert sa == sc
            mesh_from_geo('mesh_mu.geo', show_mesh=True,
                          param_dic={'size' : 0.02, 'Rc' : Rcut, 'sa' : sa})
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_mu.geo cannot be found!")
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(xll=0, yll=-1, dx=Rcut, dy=2, centered=False)
            MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202, 203])

    def rho_func(coors):
        value = np.zeros(coors.shape[0])
        norm2 = coors[:,0]**2
        mu = coors[:, 1]
        sin_theta = sqrt(1-mu**2)
        boolin = np.where(norm2*((sin_theta/sa)**2+(mu/sc)**2)<1)[0]
        value[boolin] = rho
        return value

    densities = [rho_func]
    dirichlet_bc = [dbc_pol2]
    pb = PoissonBounded(alpha, densities, dirichlet_bc, meshfile,
                        coorsys=coorsys, fem_order=fem_order, verbose=verbose,
                        print_kappa=print_kappa)
    pb.solve()
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    theta = arccos(Y_rand/rr)
    mu = cos(theta)
    coors_eval = np.ascontiguousarray(
        np.concatenate((rr[:, np.newaxis], mu[:, np.newaxis]), axis=1))
    fem_2 = field_2.evaluate_at(coors_eval, sol_2[:, np.newaxis]).squeeze()
    print("#DOFs = {}".format(sol_2.shape[0]))
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))

else:
    raise ValueError("(coorsys, dim) = ({}, {}) is not implemented".format(
        coorsys, dimension))