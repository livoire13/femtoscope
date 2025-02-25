# -*- coding: utf-8 -*-
r"""
Created on Fri Mar  1 09:00:42 2024
Test of the `LinearSolver` class on bounded domains (single weak form case)

Test cases:
-----------

Poisson 2D $ -\Delta u = f $, where the r.h.s. function $f$ is given by
$$ f(x, y) = 8 \frac{ax + by}{(1+\|\mathbf{x}\|^2)^2} \, , $$
for two real numbers $a$ and $b$.
The solution is given by
$$ u(x, y) = \frac{ax + by}{1+\|\mathbf{x}\|^2} \, . $$
and the Dirichlet boundary condition is chosen accordingly.

Helmholtz 1D $ -u''(x) + u(x) = 1 $ together with the boundary conditions:
- Dirichlet --> u(1) = 0
- Neumann --> u'(0) = 0
The solution on [0, 1] is given by
$$ u(x) = -\frac{e^{-x} (e^1 - e^x - e^{2+x} + e^{1+2x}) }{1+e^2} \, . $$

Newtonian potential (which obeys a Poisson equation) for the spherically
symmetric case. The analytical solution to that problem is computed in
`femtoscope.misc.analytical`

"""

import os
from pathlib import Path

import pytest
import numpy as np
from numpy import exp, pi

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.solvers import LinearSolver
from femtoscope.core.weak_form import WeakForm
from femtoscope.inout.meshfactory import MeshingTools, generate_uniform_1d_mesh
from femtoscope.misc.analytical import potential_sphere

a = 2
b = 1


def analytical_solution_poisson_2d(coors):
    x = coors[:, 0]
    y = coors[:, 1]
    return (a * x + b * y) / (1 + x ** 2 + y ** 2)


def analytical_solution_helmholtz_1d(coors):
    x = coors.squeeze()
    return -(exp(-x) * (exp(1) - exp(x) - exp(2 + x) + exp(1 + 2 * x))) \
        / (1 + exp(2))


@pytest.fixture(scope='module')
def wf_poisson_2d():
    """Create an instance of `WeakForm` to be used for solving the 2D Poisson
    problem on a disk."""

    # Mesh creation
    mt = MeshingTools(2)
    mt.create_ellipse(rx=1, ry=1)
    mt.create_subdomain(cell_size_min=0.02, cell_size_max=0.02)
    pre_mesh = mt.generate_mesh('mesh_test_poisson_2d.vtk', show_mesh=False)

    # Material
    def mat_rhs(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        y = coors[:, 1]
        val = 8 * (a * x + b * y) / (1 + x ** 2 + y ** 2) ** 3
        return {'val': val.reshape(coors.shape[0], 1, 1)}

    # Terms
    lhs = PreTerm('dw_laplace')
    rhs = PreTerm('dw_integrate', mat=mat_rhs, prefactor=-1)

    # EBC
    def bc_func(ts, coors, **kwargs):
        x = coors[:, 0]
        y = coors[:, 1]
        return (a * x + b * y) / 2

    # WeakForm creation
    args_dict = {'dim': 2, 'pre_mesh': pre_mesh,
                 'pre_ebc_dict': {('facet', 200): bc_func},
                 'pre_terms': [lhs, rhs],
                 'fem_order': 2,
                 }
    wf = WeakForm.from_scratch(args_dict)
    yield wf

    Path(pre_mesh).unlink()
    print(f"** Deleting mesh file {pre_mesh} **")


@pytest.fixture(params=['ScipyDirect', 'ScipyIterative'])
def solver_poisson_2d(request, wf_poisson_2d):
    """Create `LinearSolver` instance."""

    wf_dict = {'wf_int': wf_poisson_2d}

    ls_class = request.param
    if ls_class == 'ScipyDirect':
        ls_kwargs = {'eps_a': 1e-8, 'eps_r': 1e-8}
    elif ls_class == 'ScipyIterative':
        ls_kwargs = {'i_max': 500, 'method': 'cg', 'eps_a': 1e-8, 'eps_r': 1e-8}
    else:
        raise ValueError("'ls_class' not recognized!")

    solver = LinearSolver(wf_dict, ls_class=ls_class, ls_kwargs=ls_kwargs)
    return solver


@pytest.fixture(scope='module')
def wf_helmholtz_1d():
    """Create an instance of `WeakForm` to be used for solving an Helmholtz
    equation on the interval [0, 1]."""

    # Mesh creation
    pre_mesh = generate_uniform_1d_mesh(0, 1, 100, 'mesh_1d')

    # Terms
    t1 = PreTerm('dw_laplace')
    t2 = PreTerm('dw_dot')
    t3 = PreTerm('dw_integrate', prefactor=-1)

    # Vertex selection
    def right_boundary(coors, domain=None):
        return np.where(coors.squeeze() == 1.0)[0]

    dim_func_entities = [(0, right_boundary, 0)]

    # WeakForm creation
    args_dict = {'dim': 1, 'pre_mesh': pre_mesh,
                 'pre_ebc_dict': {('vertex', 0): 0.0},
                 'pre_terms': [t1, t2, t3],
                 'dim_func_entities': dim_func_entities,
                 'fem_order': 2
                 }
    wf = WeakForm.from_scratch(args_dict)
    return wf


@pytest.fixture(params=['ScipyDirect', 'ScipyIterative'])
def solver_helmholtz_1d(request, wf_helmholtz_1d):
    """Create a `LinearSolver` instance."""

    wf_dict = {'wf_int': wf_helmholtz_1d}

    ls_class = request.param
    if ls_class == 'ScipyDirect':
        ls_kwargs = {'eps_a': 1e-8, 'eps_r': 1e-8}
    elif ls_class == 'ScipyIterative':
        ls_kwargs = {'i_max': 500, 'method': 'cg', 'eps_a': 1e-8, 'eps_r': 1e-8}
    else:
        raise ValueError("'ls_class' not recognized!")

    solver = LinearSolver(wf_dict, ls_class=ls_class, ls_kwargs=ls_kwargs)
    return solver


@pytest.fixture(scope='module')
def wf_potential():
    """Create an instance of `WeakForm` to be used for computing the Newtonian
    potential created by a spherical Earth with a 2d mesh."""

    # Mesh creation
    mt = MeshingTools(2)
    s1 = mt.create_ellipse(rx=1, ry=1)
    mt.create_subdomain(cell_size_min=0.05, cell_size_max=0.2,
                        dist_min=0, dist_max=4)
    s2 = mt.create_ellipse(rx=5, ry=5)
    s12 = mt.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
    mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
    pre_mesh = mt.generate_mesh('mesh_test_potential.vtk', show_mesh=False,
                                cylindrical_symmetry=True)

    # Materials
    def mat_laplacian(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x).reshape(coors.shape[0], 1, 1)
        return {'val': val}

    def mat_rho(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x).reshape(coors.shape[0], 1, 1)
        return {'val': val}

    # Terms
    lhs = PreTerm('dw_laplace', mat=mat_laplacian)
    rhs = PreTerm('dw_integrate', mat=mat_rho, prefactor=4 * pi,
                  region_key=('subomega', 300))

    # EBC
    def bc_func(ts, coors, **kwargs):
        r = np.linalg.norm(coors, axis=1)
        return potential_sphere(r, 1.0, 1.0, rho=1.0)

    # WeakForm creation
    args_dict = {'dim': 2, 'pre_mesh': pre_mesh,
                 'pre_ebc_dict': {('facet', 201): bc_func},
                 'pre_terms': [lhs, rhs],
                 'fem_order': 2,
                 }

    wf = WeakForm.from_scratch(args_dict)
    yield wf

    Path(pre_mesh).unlink()
    print(f"** Deleting mesh file {pre_mesh} **")


@pytest.fixture(params=['ScipyDirect', 'ScipyIterative'])
def solver_potential(request, wf_potential):
    """Create `LinearSolver` instance."""

    wf_dict = {'wf_int': wf_potential}

    ls_class = request.param
    if ls_class == 'ScipyDirect':
        ls_kwargs = {'eps_a': 1e-8, 'eps_r': 1e-8}
    elif ls_class == 'ScipyIterative':
        ls_kwargs = {'i_max': 5000, 'method': 'cg',
                     'eps_a': 1e-12, 'eps_r': 1e-12}
    else:
        raise ValueError("'ls_class' not recognized!")

    solver = LinearSolver(wf_dict, ls_class=ls_class, ls_kwargs=ls_kwargs)
    return solver


@pytest.mark.parametrize("use_reduced_mtx_vec", [True, False])
def test_poisson_2d(solver_poisson_2d, use_reduced_mtx_vec):
    solver_poisson_2d.solve(use_reduced_mtx_vec=use_reduced_mtx_vec)
    fem_sol = solver_poisson_2d.sol
    ana_sol = analytical_solution_poisson_2d(solver_poisson_2d.wf.field.coors)
    err_abs = abs(fem_sol - ana_sol)
    assert err_abs.max() < 1e-4


@pytest.mark.parametrize("use_reduced_mtx_vec", [True, False])
def test_poisson_2d(solver_helmholtz_1d, use_reduced_mtx_vec):
    solver_helmholtz_1d.solve(use_reduced_mtx_vec=use_reduced_mtx_vec)
    fem_sol = solver_helmholtz_1d.sol
    ana_sol = analytical_solution_helmholtz_1d(
        solver_helmholtz_1d.wf.field.coors)
    err_abs = abs(fem_sol - ana_sol)
    assert err_abs.max() < 1e-11


@pytest.mark.parametrize("use_reduced_mtx_vec", [True, False])
def test_potential(solver_potential, use_reduced_mtx_vec):
    solver_potential.solve(use_reduced_mtx_vec=use_reduced_mtx_vec)
    fem_sol = solver_potential.sol
    r = np.linalg.norm(solver_potential.wf.field.coors, axis=1)
    ana_sol = potential_sphere(r, 1.0, 1.0, rho=1.0)
    err_rel = abs((fem_sol - ana_sol) / ana_sol)
    assert err_rel.max() < 5e-4
