# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:54:08 2022

@author: hlevy
"""

import numpy as np
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.inout.meshfactory import mesh_from_geo
from femtoscope.physics.poisson import PoissonCompact
from femtoscope.misc.analytical import potential_sphere, potential_ellipsoid
from numpy import sin, cos, pi, sqrt, arccos
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from pathlib import Path
from copy import deepcopy

verbose = True
compute_mesh = True

####################################
#       Simulation Parameters      #
####################################
Rcut = 5.0 # radius for artificial boundary
sa = 1.0
sc = 1.0
ecc = sqrt(1-(sc/sa)**2)
# mass = 10.0
rho = 1.0 #3*mass/(4*pi*sqrt(1-ecc**2)*sa**3)
alpha = 4*pi
fem_order = 2
dim = 2
coorsys = 'polar_mu' # 'polar' or 'polar_mu'

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
sol_ana = potential_sphere(rr, sa, 1.0, rho=rho).squeeze()

if coorsys == 'polar' and dim == 2:

    def rho_func(coors):
        value = np.zeros(coors.shape[0])
        norm2 = coors[:, 0]**2
        theta = coors[:, 1]
        boolin = np.where(norm2*((sin(theta)/sa)**2+(cos(theta)/sc)**2)<1)[0]
        value[boolin] = rho
        return value
    
    def rho_func_mod(coors):
        coors_mod = deepcopy(coors)
        coors_mod[:, 0] = coors_mod[:, 0] / (Rcut - coors_mod[:, 0])
        return rho_func(coors_mod)
    
    meshfile = "mesh_theta_compact.vtk"
    if compute_mesh:
        try:
            assert sa == sc
            mesh_from_geo('mesh_theta_compact.geo', show_mesh=True,
                          param_dic={'size' : 0.1, 'sa' : sa, 'Rc' : Rcut})
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_theta_compact.geo cannot be found!")
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(dx=Rcut, dy=pi, centered=False)
            MT.create_subdomain(CellSizeMin=0.05, CellSizeMax=0.05, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202, 203])
    
    pb_kwargs = {'coorsys' : coorsys,
                 'fem_order' : fem_order}
    pb = PoissonCompact(alpha, rho_func_mod, meshfile, **pb_kwargs)
    pb.solve()
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    theta = arccos(Y_rand/rr)
    eta = Rcut*rr/(1+rr)
    coors_eval = np.ascontiguousarray(
        np.concatenate((eta[:, np.newaxis], theta[:, np.newaxis]), axis=1))
    fem_2 = field_2.evaluate_at(coors_eval, sol_2[:, np.newaxis]).squeeze()
    print("#DOFs = {}".format(sol_2.shape[0]))
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))

elif coorsys == 'polar_mu' and dim == 2:

    def rho_func(coors):
        value = np.zeros(coors.shape[0])
        norm2 = coors[:, 0]**2
        mu = coors[:, 1]
        boolin = np.where((norm2*(1-mu**2)/sa**2+(mu/sc)**2)<1)[0]
        value[boolin] = rho
        return value
    
    def rho_func_mod(coors):
        coors_mod = deepcopy(coors)
        coors_mod[:, 0] = coors_mod[:, 0] / (Rcut - coors_mod[:, 0])
        return rho_func(coors_mod)
    
    meshfile = "mesh_mu_compact.vtk"
    if compute_mesh:
        try:
            assert sa == sc
            mesh_from_geo('mesh_mu_compact.geo', show_mesh=True,
                          param_dic={'size' : 0.008})
        except FileNotFoundError:
            print("femtoscope\data\mesh\geo\mesh_mu_compact.geo cannot be found!")
            MT = MeshingTools(dimension=2)
            MT.create_rectangle(xll=0, yll=-1, dx=Rcut, dy=2, centered=False)
            MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5, DistMax=0.0)
            MT.generate_mesh(meshfile, show_mesh=True, unique_boundary=False,
                             ignoreTags=[200, 202, 203])
    
    pb_kwargs = {'coorsys' : coorsys,
                 'fem_order' : fem_order}
    pb = PoissonCompact(alpha, rho_func_mod, meshfile, **pb_kwargs)
    pb.solve()
    sol_2 = pb.solver.sols[0]
    field_2 = pb.solver.weakforms[0].field
    theta = arccos(Y_rand/rr)
    mu = cos(theta)
    eta = Rcut*rr/(1+rr)
    coors_eval = np.ascontiguousarray(
        np.concatenate((eta[:, np.newaxis], mu[:, np.newaxis]), axis=1))
    fem_2 = field_2.evaluate_at(coors_eval, sol_2[:, np.newaxis]).squeeze()
    print("#DOFs = {}".format(sol_2.shape[0]))
    print("mean pointwise relative error: {:.2E}".format(
        np.mean(abs((fem_2-sol_ana)/sol_ana))))

else:
    raise NotImplementedError("({}, {}) unavailable".format(coorsys, dim))