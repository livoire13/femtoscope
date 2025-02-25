# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:26:10 2024

Test of the `LinearSolver` class on unbounded domains (single weak form case)

Test cases:
-----------
Newtonian potential (which obeys a Poisson equation) for the oblate ellipsoid.
The analytical solution to this problem is computed in
`femtoscope.misc.analytical`

Notes:
------
The linear solver 'ScipyIterative' is not employed for the first test case as
it produces erroneous solutions. The reason for this undesired behavior is that
'ScipyIterative' uses the conjugate gradient method ('cg', by default), for
which the matrix is assumed positive-definite. However, the introduction of a
weight in the PDE's strong form results in non-symmetric stiffness matrices,
which is most probably why the iterative algorithm fails while more general
direct solvers produce accurate solution.

"""

from pathlib import Path

import numpy as np
import pytest
from numpy import pi, sqrt

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.solvers import LinearSolver
from femtoscope.core.weak_form import WeakForm
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.misc.analytical import potential_ellipsoid

sa = 2.0
sc = 1.0
Rcut = 5.0


@pytest.fixture(scope='module')
def wfs_ellipsoid_2d():
    """
    Create two instances of `WeakForm` (wf_int & wf_ext) to be used for
    computing the Newtonian potential created by an oblate ellipsoid with
    2d meshes
    """

    # Meshes creation
    mt_int = MeshingTools(2)
    s1 = mt_int.create_ellipse(rx=sa, ry=sc)
    mt_int.create_subdomain(cell_size_min=0.05, cell_size_max=0.2,
                            dist_min=0.0, dist_max=4.0)
    s2 = mt_int.create_disk_from_pts(Rcut, N=200)
    mt_int.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    mt_int.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    pre_mesh_int = mt_int.generate_mesh('mesh_test_ellipsoid_int.vtk',
                                        cylindrical_symmetry=True,
                                        show_mesh=False)

    mt_ext = MeshingTools(2)
    mt_ext.create_disk_from_pts(Rcut, N=200)
    mt_ext.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    origin_rf = [0.07, 0.2, 0.1, 3.0]
    pre_mesh_ext = mt_ext.generate_mesh('mesh_test_ellipsoid_ext.vtk',
                                        cylindrical_symmetry=True,
                                        embed_origin=True,
                                        origin_rf=origin_rf,
                                        show_mesh=False)

    # Materials
    def mat_laplacian_int(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x).reshape(coors.shape[0], 1, 1)
        return {'val': val}

    def mat_rho_int(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x).reshape(coors.shape[0], 1, 1)
        return {'val': val}

    def mat_laplacian1_ext(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val = norm2 ** 2 / Rcut ** 4 * abs(xi) * (7 - 6 * norm / Rcut)
        return {'val': val.reshape(coors.shape[0], 1, 1)}

    def mat_laplacian2_ext(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        val = np.zeros((coors.shape[0], 2, 1))
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val[:, 0, 0] = 42*norm2 * abs(xi) / Rcut ** 4 * (1 - norm / Rcut) * xi
        val[:, 1, 0] = 42*norm2 * abs(xi) / Rcut ** 4 * (1 - norm / Rcut) * eta
        return {'val': val}

    # Terms
    t1_int = PreTerm('dw_laplace', mat=mat_laplacian_int)
    t2_int = PreTerm('dw_integrate', mat=mat_rho_int, prefactor=4*pi,
                     region_key=('subomega', 300))
    t1_ext = PreTerm('dw_laplace', mat=mat_laplacian1_ext)
    t2_ext = PreTerm('dw_s_dot_mgrad_s', mat=mat_laplacian2_ext)

    # WeakForm creations
    args_dict_int = {
        'name': 'int',
        'dim': 2,
        'pre_mesh': pre_mesh_int,
        'fem_order': 2,
        'pre_terms': [t1_int, t2_int]
    }
    wf_int = WeakForm.from_scratch(args_dict_int)

    args_dict_ext = {
        'name': 'ext',
        'dim': 2,
        'pre_mesh': pre_mesh_ext,
        'fem_order': 2,
        'pre_terms': [t1_ext, t2_ext],
        'pre_ebc_dict': {('vertex', 0): 0.0}
    }
    wf_ext = WeakForm.from_scratch(args_dict_ext)

    yield wf_int, wf_ext

    Path(pre_mesh_int).unlink()
    Path(pre_mesh_ext).unlink()
    print(f"** Deleting meshes {pre_mesh_int} and {pre_mesh_int} **")


@pytest.fixture(params=['ScipyDirect'])
def solver_ellipsoid_2d(request, wfs_ellipsoid_2d):
    """Create `LinearSolver` instance."""

    wf_int, wf_ext = wfs_ellipsoid_2d
    wf_dict = {'wf_int': wf_int, 'wf_ext': wf_ext}

    ls_class = request.param
    if ls_class == 'ScipyDirect':
        ls_kwargs = {'eps_a': 1e-8, 'eps_r': 1e-8}
    elif ls_class == 'ScipyIterative':
        ls_kwargs = {'i_max': 1e4, 'method': 'cg',
                     'eps_a': 1e-12, 'eps_r': 1e-12}
    else:
        raise ValueError("'ls_class' not recognized!")

    region_key_int = ('facet', 201)
    region_key_ext = ('facet', 200)
    solver = LinearSolver(
        wf_dict, ls_class=ls_class, ls_kwargs=ls_kwargs,
        region_key_int=region_key_int, region_key_ext=region_key_ext)
    return solver


def test_ellipsoid_2d(solver_ellipsoid_2d):
    # Solver setup
    solver_ellipsoid_2d.solve(use_reduced_mtx_vec=True)

    # Coordinates
    coors_int = solver_ellipsoid_2d.wf_int.field.coors
    coors_ext = solver_ellipsoid_2d.wf_ext.field.coors
    aux_norm = ((Rcut/np.linalg.norm(coors_ext, axis=1))**2).reshape(-1, 1)
    coors_ext_inv = aux_norm * coors_ext
    idx_origin = np.where((coors_ext[:, 0] == 0) & (coors_ext[:, 1] == 0))[0][0]

    # FEM and Analytical solutions
    sol_fem_int = solver_ellipsoid_2d.sol_int
    sol_fem_ext = solver_ellipsoid_2d.sol_ext
    sol_ana_int = potential_ellipsoid(coors_int, sa, 1.0, sc=sc, rho=1.0)
    sol_ana_ext = potential_ellipsoid(coors_ext_inv, sa, 1.0, sc=sc, rho=1.0)

    # Remove the DOF associated with the origin in the inversed exterior domain
    sol_ana_ext = np.delete(sol_ana_ext, idx_origin)
    sol_fem_ext = np.delete(sol_fem_ext, idx_origin)

    # Compute the relative (interior) and absolute (exterior) errors
    err_rel_int = abs(sol_fem_int-sol_ana_int)/abs(sol_ana_int)
    err_abs_ext = abs(sol_fem_ext-sol_ana_ext)

    # Test
    assert err_rel_int.max() < 2e-3 and err_abs_ext.max() < 7e-3
