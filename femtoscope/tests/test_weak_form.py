# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:43:28 2024
Testing of the `WeakForm` class.
"""

from pathlib import Path

from functools import partial
import numpy as np
import pytest

from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete.conditions import (Conditions, EssentialBC,)
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.base.base import IndexedStruct

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.weak_form import WeakForm
from femtoscope.inout.meshfactory import generate_mesh_from_geo


def weak_form(fem_order):
    """Create an instance of WeakForm to be used in the following tests."""

    def material(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        return {'val': abs(coors[:, 0]).reshape(coors.shape[0], 1, 1)}

    pre_mesh = generate_mesh_from_geo(
        'rectangle_test', param_dict={'size': 0.5})
    pre_term_300 = PreTerm('dw_laplace', region_key=('subomega', 300),
                           prefactor=5.0, mat=material)
    pre_term_301 = PreTerm('dw_laplace', region_key=('subomega', 301),
                           prefactor=2.0, mat=1.0)
    pre_term_omega = PreTerm('dw_integrate', region_key=('omega', -1),
                             prefactor=1.0)
    args_dict = {'dim': 2, 'pre_mesh': pre_mesh,
                 'pre_ebc_dict': {('facet', 200): 1.0},
                 'pre_terms': [pre_term_300, pre_term_301, pre_term_omega],
                 'fem_order': fem_order}

    wf = WeakForm.from_scratch(args_dict)
    wf.set_mtx_vec_no_bc()
    wf.set_mtx_vec_bc_reduced()
    wf.set_mtx_vec_bc_full()

    Path(pre_mesh).unlink()
    return wf


def sfepy_problem(fem_order):
    def matf(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        return {'val': abs(coors[:, 0]).reshape(coors.shape[0], 1, 1)}

    matf_func = Function('matf_func', matf)
    mat = Material('mat', kind='stationary', function=matf_func)
    pre_mesh = generate_mesh_from_geo(
        'rectangle_test', param_dict={'size': 0.5})
    mesh = Mesh.from_file(pre_mesh)
    domain = FEDomain('domain', mesh)
    omega = domain.create_region('Omega', 'all')
    subomega300 = domain.create_region('subomega300', 'cells of group 300',
                                       kind='cell')
    subomega301 = domain.create_region('subomega301', 'cells of group 301',
                                       kind='cell')
    gamma = domain.create_region('Gamma', 'vertices of group 200', kind='facet')
    field = Field.from_args('fu', np.float64, 'scalar', omega,
                            approx_order=fem_order)
    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    integral = Integral('i', order=2*fem_order+1)
    t1 = Term.new('dw_laplace(mat.val, v, u)', integral, subomega300, v=v, u=u,
                  mat=mat)
    t2 = Term.new('dw_laplace(v, u)', integral, subomega301, v=v, u=u)
    t3 = Term.new('dw_integrate(v)', integral, omega, v=v)
    eq = Equation('Poisson', 5*t1 + 2*t2 + t3)
    eqs = Equations([eq])
    dbc_gamma = EssentialBC('dbc_gamma', gamma, {'u.all': 1.0})
    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)
    pb = Problem('Poisson', equations=eqs, active_only=False)
    pb.set_bcs(ebcs=Conditions([dbc_gamma]))
    pb.set_solver(nls)
    return pb


@pytest.fixture(scope='module')
def weak_forms():
    """Mapping of possible weak forms (varying fem_order) to their
    constructor functions."""
    return {
        1: partial(weak_form, 1),
        2: partial(weak_form, 2),
        3: partial(weak_form, 3),
    }


@pytest.fixture(scope='module', params=[1, 2])
def wf(request, weak_forms):
    return weak_forms[request.param]()


@pytest.fixture(scope='module')
def sfepy_pbs():
    """Mapping of possible Sfepy `Problem` instances (varying fem_order) to
    their constructor functions."""
    return {
        1: partial(sfepy_problem, 1),
        2: partial(sfepy_problem, 2),
        3: partial(sfepy_problem, 3),
    }


@pytest.fixture(scope='module', params=[1, 2])
def sfepy_pb(request, sfepy_pbs):
    return sfepy_pbs[request.param]()


def test_wf_regions(wf):
    """Test region assigment through coordinates check."""
    errors = []
    region_dict = wf.region_dict

    # check that the number of region is correct
    if len(region_dict) != 3 + 2:
        errors.append("the number of regions is not consistent with the "
                      "number of physical groups")

    # check coordinates of each region
    field = wf.field
    idx_facet200 = field.get_dofs_in_region(region_dict[('facet', 200)])
    idx_subomega300 = field.get_dofs_in_region(region_dict[('subomega', 300)])
    idx_subomega301 = field.get_dofs_in_region(region_dict[('subomega', 301)])
    coors200 = field.coors[idx_facet200]
    coors300 = field.coors[idx_subomega300]
    coors301 = field.coors[idx_subomega301]
    bool200 = ((coors200[:, 0] == 0) | (coors200[:, 0] == 2) |
               (coors200[:, 1] == 0) | (coors200[:, 1] == 1)).all()
    bool300 = (((0.0 <= coors300[:, 0]) & (1.0 >= coors300[:, 0])) &
               ((0.0 <= coors300[:, 1]) & (1.0 >= coors300[:, 1]))).all()
    bool301 = (((1.0 <= coors301[:, 0]) & (2.0 >= coors301[:, 0])) &
               ((0.0 <= coors301[:, 1]) & (1.0 >= coors301[:, 1]))).all()
    if not bool200:
        errors.append("the boundary 'facet200' has wrong coordinates")
    if not bool300:
        errors.append("the cells of 'subomega300' have wrong coordinates")
    if not bool301:
        errors.append("the cells of 'subomega301' have wrong coordinates")

    # global assertion
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_mtx_vec_size(wf):
    """Check size of matrices and vectors"""
    errors = []
    field = wf.field
    region_dict = wf.region_dict
    fixed_dofs = field.get_dofs_in_region(region_dict[('facet', 200)])
    full_size = wf.field.coors.shape[0]
    reduced_size = full_size - fixed_dofs.shape[0]

    if wf.mtx_vec.mtx_no_bc.shape != (full_size, full_size):
        errors.append("'mtx_no_bc' has wrong shape, expected shape is "
                      "({}, {})".format(full_size, full_size))
    if wf.mtx_vec.vec_no_bc.shape[0] != full_size:
        errors.append("'vec_no_bc' has wrong length,"
                      "expected length is {}".format(full_size))

    if wf.mtx_vec.mtx_bc_reduced.shape != (reduced_size, reduced_size):
        errors.append("'mtx_bc_reduced' has wrong shape, expected shape is "
                      "({}, {})".format(reduced_size, reduced_size))
    if wf.mtx_vec.vec_bc_reduced.shape[0] != reduced_size:
        errors.append("'vec_bc_reduced' has wrong length,"
                      "expected length is {}".format(reduced_size))

    if wf.mtx_vec.mtx_bc_full_cst.shape != (full_size, full_size):
        errors.append("'mtx_bc_full_cst' has wrong shape, expected shape is "
                      "({}, {})".format(full_size, full_size))
    if wf.mtx_vec.vec_bc_full_cst.shape[0] != full_size:
        errors.append("'vec_bc_full_cst' has wrong length,"
                      "expected length is {}".format(full_size))

    # global assertion
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_active_bcs_switch_on(wf):
    wf.apply_bcs()
    assert wf.active_bcs['cst']


def test_active_bcs_switch_off(wf):
    wf.remove_bcs()
    assert not wf.active_bcs['cst']


def test_fill_region_dict_from_dim_func():
    def middle_line(coors, domain=None):
        return np.where(np.isclose(coors[:, 0], 1.0))[0]

    pre_mesh = generate_mesh_from_geo('rectangle_test', param_dict={'size': 0.2})
    pre_term = PreTerm('dw_laplace')
    dim_func_entities = [(1, middle_line, 201)]
    args_dict = {'dim': 2, 'pre_mesh': pre_mesh, 'pre_terms': [pre_term],
                 'fem_order': 1, 'dim_func_entities': dim_func_entities}
    wf = WeakForm.from_scratch(args_dict)
    idx_facet201 = wf.field.get_dofs_in_region(wf.region_dict[('facet', 201)])
    coors201 = wf.field.coors[idx_facet201]
    assert (np.isclose(coors201[:, 0], 1)).all()


@pytest.mark.parametrize("fem_order", [1, 2])
def test_mtx_vec_entries(fem_order):
    """Test the correct assembly of matrices/vectors by comparing femtoscope's
    outputs against the ones obtained with Sfepy's intended usage."""
    errors = []
    wf = weak_form(fem_order)
    sfepy_pb = sfepy_problem(fem_order)

    # Matrix check
    my_mtx = wf.mtx_vec.mtx_bc_full_cst
    tss = sfepy_pb.get_solver()
    sfepy_pb.equations.set_data(None, ignore_unknown=True)
    variables = sfepy_pb.get_initial_state(vec=None)
    sfepy_pb.time_update(tss.ts, is_matrix=(sfepy_pb.mtx_a is None))
    sfepy_pb.update_materials()
    ev = sfepy_pb.get_evaluator()
    sfepy_mtx = ev.eval_tangent_matrix(variables(), is_full=True)
    if not np.array_equal(my_mtx.todense(), sfepy_mtx.todense()):
        errors.append("bc_full matrices are not equal!")

    # Vector check
    my_vec = wf.mtx_vec.vec_bc_reduced
    gamma = sfepy_pb.domain.regions.find('Gamma')
    idx_gamma = sfepy_pb.fields['fu'].get_dofs_in_region(gamma)
    sfepy_vec = np.delete(-ev.eval_residual(variables()), idx_gamma)
    if not np.array_equal(my_vec, sfepy_vec):
        errors.append("reduced rhs vectors are not equal!")

    # global assertion
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("fem_order", [1, 2])
def test_assign_periodic_bc(fem_order):
    """Test the assignment of periodic boundary conditions."""

    # Boundary parts (selection by function)
    def left_boundary(coors, domain=None):
        return np.where(coors[:, 0] < 1e-6)[0]
    def right_boundary(coors, domain=None):
        return np.where(coors[:, 0] > 1.0 - 1e-6)[0]
    def bottom_boundary(coors, domain=None):
        return np.where(coors[:, 1] < 1e-6)[0]
    def top_boundary(coors, domain=None):
        return np.where(coors[:, 1] > 1.0 - 1e-6)[0]

    dim_func_entities = [(1, bottom_boundary, 200),
                         (1, right_boundary, 201),
                         (1, top_boundary, 202),
                         (1, left_boundary, 203)]

    pre_epbc_list = [[('facet', 200), ('facet', 202), 'match_x_line'],
                     [('facet', 201), ('facet', 203), 'match_y_line']]

    # Creation of the weak form with periodic boundary conditions
    pre_mesh = generate_mesh_from_geo(
        'square_periodic_test', param_dict={'size': 1.0}, show_mesh=False)
    pre_term = PreTerm('dw_laplace', region_key=('omega', -1))
    args_dict = {'dim': 2, 'pre_mesh': pre_mesh,
                 'dim_func_entities': dim_func_entities,
                 'pre_epbc_list': pre_epbc_list,
                 'pre_terms': [pre_term],
                 'fem_order': fem_order}
    wf = WeakForm.from_scratch(args_dict)  # Creation of the weak form

    wf.set_mtx_vec_no_bc()  # assembling the bc-free matrix/vector
    wf.set_mtx_vec_bc_reduced()  # assembling the reduced matrix/vector

    # The full matrix/vector cannot be assembled when epbc are used
    with pytest.raises(NotImplementedError):
        wf.set_mtx_vec_bc_full()
