# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:21:32 2024

Global tests on the Poisson class (from femtoscope.physics.physical_problems).

"""

from pathlib import Path

import pytest
import numpy as np
from numpy import pi

from femtoscope.physics.physical_problems import Poisson
from femtoscope.inout.meshfactory import (MeshingTools,
                                          generate_uniform_1d_mesh,
                                          generate_mesh_from_geo)
from femtoscope.misc.analytical import potential_sphere

sa = 1.0
sc = 1.0
Rc = 5.0
alpha = 4 * pi
meshint_name = 'mesh_test_sphere_int.vtk'
meshext_name = 'mesh_test_sphere_ext.vtk'


def get_radius(coors, dim, coorsys=None):
    if dim == 1:
        r = coors.squeeze()
    elif dim == 2:
        if coorsys == 'polar':
            r = coors[:, 0]
        elif coorsys == 'cylindrical':
            r = np.linalg.norm(coors, axis=1)
        else:
            raise ValueError(f"'{coorsys}' is not a valid coordinate system!")
    elif dim == 3:
        r = np.linalg.norm(coors, axis=1)
    else:
        raise ValueError(f"'{dim}' is not a valid dimension!")
    return r


def analytical_solution_int(coors, dim, coorsys=None):
    r = get_radius(coors, dim, coorsys=coorsys)
    return potential_sphere(r, sa, 1.0, rho=1.0)


def analytical_solution_ext(coors, dim, coorsys=None):
    eta = get_radius(coors, dim, coorsys=coorsys)
    r = Rc ** 2 / eta
    return potential_sphere(r, sa, 1.0, rho=1.0)


def get_pre_meshes(dim, coorsys=None):
    if dim == 1:
        mesh_int = mesh1d()
        mesh_ext = mesh1d()
    elif dim == 2:
        if coorsys == 'polar':
            mesh_int = mesh2dpol_int()
            mesh_ext = mesh2dpol_ext()
        elif coorsys == 'cylindrical':
            mesh_int = mesh2dcyl_int()
            mesh_ext = mesh2dcyl_ext()
        else:
            raise ValueError(f"'{coorsys}' is not a valid coordinate system!")
    elif dim == 3:
        pass
    else:
        raise ValueError(f"'{dim}' is not a valid dimension!")
    return (mesh_int, mesh_ext)


def get_poisson_problem(dim, coorsys=None):
    poisson = Poisson({'alpha': alpha}, dim, Rc=Rc, coorsys=coorsys)
    pre_mesh_int, pre_mesh_ext = get_pre_meshes(dim, coorsys=coorsys)
    partial_args_dict_int = {
        'dim': dim,
        'name': 'wf_int',
        'pre_mesh': pre_mesh_int,
        'fem_order': 2,
    }
    partial_args_dict_ext = {
        'dim': dim,
        'name': 'wf_ext',
        'pre_mesh': pre_mesh_ext,
        'fem_order': 2,
    }

    if dim == 1:
        def right_boundary(coors, domain=None):
            return np.where(coors.squeeze() == Rc)[0]

        def left_boundary(coors, domain=None):
            return np.where(coors.squeeze() == 0)[0]

        dim_func_entities = [(0, right_boundary, 0), (0, left_boundary, 1)]

        def density(r):
            return np.where(r <= 1, 1.0, 0.0)

        partial_args_dict_int['dim_func_entities'] = dim_func_entities
        partial_args_dict_ext['dim_func_entities'] = dim_func_entities
        partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 1): 0.0}
        poisson.set_wf_int(partial_args_dict_int, {('omega', -1): density})
        poisson.set_wf_ext(partial_args_dict_ext, density=None)
        poisson.set_default_solver(region_key_int=('vertex', 0),
                                   region_key_ext=('vertex', 0))

    elif dim == 2:
        if coorsys == 'polar':
            partial_args_dict_ext['pre_ebc_dict'] = {('facet', 203): 0.0}
            region_key_int = ('facet', 201)
            region_key_ext = ('facet', 201)
        elif coorsys == 'cylindrical':
            partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 0): 0.0}
            region_key_int = ('facet', 201)
            region_key_ext = ('facet', 200)
        else:
            raise ValueError(f"'{coorsys}' is not a valid coordinate system!")
        poisson.set_wf_int(partial_args_dict_int, {('subomega', 300): 1.0})
        poisson.set_wf_ext(partial_args_dict_ext, density=None)
        poisson.set_default_solver(region_key_int=region_key_int,
                                   region_key_ext=region_key_ext)

    elif dim == 3:
        raise NotImplementedError
    else:
        raise ValueError(f"'{dim}' is not a valid dimension!")

    return poisson


def mesh1d():
    return generate_uniform_1d_mesh(0, Rc, 501)


def mesh2dpol_int():
    param_dict = {'Rts': 0.0, 'Rc': Rc, 'sa': sa, 'size': 0.1, 'Ngamma': 40}
    return generate_mesh_from_geo('test_theta_int.geo', param_dict=param_dict,
                                  ignored_tags=[200, 202, 203], show_mesh=False,
                                  mesh_name=meshint_name)


def mesh2dpol_ext():
    param_dict = {'Rc': Rc, 'size': 0.1, 'Ngamma': 40}
    return generate_mesh_from_geo('test_theta_ext.geo', param_dict=param_dict,
                                  ignored_tags=[200, 202], show_mesh=False,
                                  mesh_name=meshext_name)


def mesh2dcyl_int():
    mt = MeshingTools(2)
    s1 = mt.create_ellipse(rx=sa, ry=sc)
    mt.create_subdomain(cell_size_min=0.05, cell_size_max=0.2,
                        dist_min=0.0, dist_max=4.0)
    s2 = mt.create_disk_from_pts(Rc, N=200)
    mt.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    return mt.generate_mesh(meshint_name, cylindrical_symmetry=True,
                            show_mesh=False, ignored_tags=[200])


def mesh2dcyl_ext():
    mt = MeshingTools(2)
    mt.create_disk_from_pts(Rc, N=200)
    mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    origin_rf = [0.07, 0.2, 0.1, 3.0]
    return mt.generate_mesh(
        meshext_name, cylindrical_symmetry=True, show_mesh=False,
        embed_origin=True, origin_rf=origin_rf)


@pytest.mark.parametrize(('dim', 'coorsys'), [(1, 'polar'), (2, 'polar'),
                                              (2, 'cylindrical')])
def test_poisson(dim, coorsys):
    # Setup
    poisson = get_poisson_problem(dim, coorsys=coorsys)
    solver = poisson.default_solver

    # Coordinates
    coors_int = solver.wf_int.field.coors
    coors_ext = solver.wf_ext.field.coors

    # Solve Poisson problem
    solver.solve()

    # FEM and analytical solutions
    sol_fem_int = solver.sol_int
    sol_fem_ext = solver.sol_ext
    sol_ana_int = analytical_solution_int(coors_int, dim, coorsys=coorsys)
    sol_ana_ext = analytical_solution_ext(coors_ext, dim, coorsys=coorsys)

    # Compute the relative (interior) and absolute (exterior) errors
    err_rel_int = abs(sol_fem_int - sol_ana_int) / abs(sol_ana_int)
    err_abs_ext = abs(sol_fem_ext - sol_ana_ext)

    # Test
    if dim == 1:
        assert err_rel_int.max() < 3e-10 and err_abs_ext.max() < 4e-11
    if dim == 2 and coorsys == 'polar':
        assert err_rel_int.max() < 2e-5 and err_abs_ext.max() < 2e-7
    if dim == 2 and coorsys == 'cylindrical':
        assert err_rel_int.max() < 6e-4 and err_abs_ext.max() < 7e-4

    # Remove meshfile
    if dim >= 2:
        pre_mesh_int = solver.wf_int.field.domain.mesh.name + '.vtk'
        pre_mesh_ext = solver.wf_ext.field.domain.mesh.name + '.vtk'
        Path(pre_mesh_int).unlink()
        Path(pre_mesh_ext).unlink()
