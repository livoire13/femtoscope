# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:21:32 2024

Global tests on the Chameleon class (from femtoscope.physics.physical_problems).

"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from femtoscope import DATA_DIR
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.inout.meshfactory import generate_mesh_from_geo
from femtoscope.inout.meshfactory import generate_uniform_1d_mesh
from femtoscope.physics.physical_problems import Chameleon

# Parameters (not be changed!!)
sa = 1.0
sc = 1.0
Rc = 5.0
alpha = 0.1
npot = 2
rho_min = 1.0
rho_max = 1e2
phi_min = rho_max ** (-1 / (npot + 1))
phi_max = rho_min ** (-1 / (npot + 1))
param_dict = {
    'alpha': alpha, 'npot': npot, 'rho_min': rho_min, 'rho_max': rho_max}

# Mesh names
meshint_name = 'mesh_test_sphere_int.vtk'
meshext_name = 'mesh_test_sphere_ext.vtk'

# Reference solution for the test
pkl_file = Path(DATA_DIR / "../femtoscope/tests/data/chameleon_radial_test.pkl")
with open(pkl_file, mode='rb') as f:
    ref_data = pickle.load(f)
rr_ref = ref_data['r']
eta_ref = ref_data['eta']
sol_int_ref = ref_data['sol_int']
sol_ext_ref = ref_data['sol_ext']


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


def get_chameleon_problem(dim, coorsys=None):
    chameleon = Chameleon(param_dict, dim, Rc=Rc, coorsys=coorsys)
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

        def density(coors):
            return np.where(coors.squeeze() <= 1, rho_max, rho_min)

        density_dict = {('omega', -1): density}

        partial_args_dict_int['dim_func_entities'] = dim_func_entities
        partial_args_dict_ext['dim_func_entities'] = dim_func_entities
        partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 1): phi_max}
        chameleon.set_wf_int(partial_args_dict_int, density_dict)
        chameleon.set_wf_residual(partial_args_dict_int, density_dict)
        chameleon.set_wf_ext(partial_args_dict_ext, density=rho_min)
        chameleon.set_default_solver(region_key_int=('vertex', 0),
                                     region_key_ext=('vertex', 0))

    elif dim == 2:
        if coorsys == 'polar':
            partial_args_dict_ext['pre_ebc_dict'] = {('facet', 203): phi_max}
            region_key_int = ('facet', 201)
            region_key_ext = ('facet', 201)
        elif coorsys == 'cylindrical':
            partial_args_dict_ext['pre_ebc_dict'] = {('vertex', 0): phi_max}
            region_key_int = ('facet', 201)
            region_key_ext = ('facet', 200)
        else:
            raise ValueError(f"'{coorsys}' is not a valid coordinate system!")

        density_dict = {('subomega', 300): rho_max, ('subomega', 301): rho_min}

        chameleon.set_wf_int(partial_args_dict_int, density_dict)
        chameleon.set_wf_residual(partial_args_dict_int, density_dict)
        chameleon.set_wf_ext(partial_args_dict_ext, density=rho_min)
        chameleon.set_default_solver(region_key_int=region_key_int,
                                     region_key_ext=region_key_ext)

    elif dim == 3:
        raise NotImplementedError
    else:
        raise ValueError(f"'{dim}' is not a valid dimension!")

    chameleon.set_default_monitor(10)

    return chameleon


def mesh1d():
    return generate_uniform_1d_mesh(0, Rc, 501)


def mesh2dpol_int():
    param_mesh = {'Rts': 0.0, 'Rc': Rc, 'sa': sa, 'size': 0.1,
                  'size_min': 0.03, 'Ngamma': 40}
    return generate_mesh_from_geo('test_theta_int.geo', param_dict=param_mesh,
                                  ignored_tags=[200, 202, 203], show_mesh=False,
                                  mesh_name=meshint_name)


def mesh2dpol_ext():
    param_mesh = {'Rc': Rc, 'size': 0.1, 'Ngamma': 40}
    return generate_mesh_from_geo('test_theta_ext.geo', param_dict=param_mesh,
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
def test_chameleon(dim, coorsys):
    # Setup
    chameleon = get_chameleon_problem(dim, coorsys=coorsys)
    solver = chameleon.default_solver
    monitor = solver.nonlinear_monitor

    # Solve Chameleon problem
    solver.solve(verbose=True)
    sol_int_test = solver.sol_int
    sol_ext_test = solver.sol_ext
    coors_int = solver.wf_int.field.coors
    coors_ext = solver.wf_ext.field.coors

    # First condition on the residual
    resnorm2 = monitor.criteria['ResidualVectorNorm2'].value
    if dim == 1:
        cond1 = resnorm2 < 3e-6
    elif dim == 2 and coorsys == 'polar':
        cond1 = resnorm2 < 3e-7
    elif dim == 2 and coorsys == 'cylindrical':
        cond1 = resnorm2 < 6e-5
    else:
        pass

    # Second condition: comparison with the reference solution
    r_test = get_radius(coors_int, dim, coorsys=coorsys)
    eta_test = get_radius(coors_ext, dim, coorsys=coorsys)
    ref_interp_int = np.interp(r_test, rr_ref, sol_int_ref)
    ref_interp_ext = np.interp(eta_test, eta_ref, sol_ext_ref)
    err_rel_int = abs(sol_int_test - ref_interp_int) / abs(ref_interp_int)
    err_rel_ext = abs(sol_ext_test - ref_interp_ext) / abs(ref_interp_ext)

    if dim == 1:
        cond2 = (err_rel_int.max() < 2e-3) and (err_rel_ext.max() < 6e-11)
    elif dim == 2 and coorsys == 'polar':
        cond2 = (err_rel_int.max() < 1.2e-2) and (err_rel_ext.max() < 8e-14)
    elif dim == 2 and coorsys == 'cylindrical':
        cond2 = (err_rel_int.max() < 4.5e-2) and (err_rel_ext.max() < 4e-13)
    else:
        pass

    # Remove meshfile
    if dim >= 2:
        pre_mesh_int = solver.wf_int.field.domain.mesh.name + '.vtk'
        pre_mesh_ext = solver.wf_ext.field.domain.mesh.name + '.vtk'
        Path(pre_mesh_int).unlink()
        Path(pre_mesh_ext).unlink()

    errors = []
    if not cond1:
        errors.append("The residual vector is too large!")
    if not cond2:
        errors.append("The relative error with respect to the reference "
                      "solution is too large!")

    # global assertion
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
