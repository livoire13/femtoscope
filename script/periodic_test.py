# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:21:41 2024

Test of sfepy with periodic boundary conditions,
3D Poisson problem

@author: hlevy
"""

import numpy as np

import sys
sys.path.append('.')


from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Functions, Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
import sfepy.discrete.fem.periodic as per

from femtoscope.inout.meshfactory import generate_mesh_from_geo

# mesh = generate_mesh_from_geo('cube.geo', param_dict={'size': 0.5},
#                               show_mesh=True)

match_x_plane = Function('match_x_plane', per.match_x_plane)
match_y_plane = Function('match_y_plane', per.match_y_plane)
match_z_plane = Function('match_z_plane', per.match_z_plane)

mesh = Mesh.from_file("cube.vtk")
domain = FEDomain('domain', mesh)
omega = domain.create_region('Omega', 'all')
sphere = domain.create_region('Sphere', 'vertices of group 300', kind='cell')
gamma0 = domain.create_region('Gamma0', 'vertices in (y > 0.4999)', kind='facet')
gamma1 = domain.create_region('Gamma1', 'vertices in (y < -0.4999)', kind='facet')
gamma2 = domain.create_region('Gamma2', 'vertices in (z > 0.4999)', kind='facet')
gamma3 = domain.create_region('Gamma3', 'vertices in (z < -0.4999)', kind='facet')
gamma4 = domain.create_region('Gamma4', 'vertices in (x < -0.4999 )', kind='facet')
gamma5 = domain.create_region('Gamma5', 'vertices in (x > 0.4999)', kind='facet')

field = Field.from_args('fu', np.float64, 'scalar', omega, approx_order=2)
u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')
integral = Integral('i', order=4)
t1 = Term.new('dw_laplace(v, u)', integral, omega, v=v, u=u)
t2 = Term.new('dw_integrate(v)', integral, sphere, v=v)
eq = Equation('Poisson', t1 + 1e-5*t2)
eqs = Equations([eq])

periodic_x = PeriodicBC('periodic_x', [gamma4, gamma5], {'u.all': 'u.all'},
                        match='match_x_plane')
periodic_y = PeriodicBC('periodic_y', [gamma0, gamma1], {'u.all': 'u.all'},
                        match='match_y_plane')
periodic_z = PeriodicBC('periodic_z', [gamma2, gamma3], {'u.all': 'u.all'},
                        match='match_z_plane')

functions = Functions([match_x_plane, match_y_plane, match_z_plane])

epbcs = Conditions([periodic_x, periodic_y, periodic_z])

ls = ScipyDirect({})
nls = Newton({}, lin_solver=ls)
pb = Problem('Poisson', equations=eqs, functions=functions)
# pb.time_update(epbcs=epbcs, functions=functions)
pb.set_bcs(epbcs=epbcs)
pb.set_solver(nls)
variables = pb.solve()

pb.save_state('periodic-poisson.vtk', variables)

