# -*- coding: utf-8 -*-
r"""
Created on Tue Apr  9 16:42:18 2024

Test of the `NonLinearSolver` class on bounded domains (single weak form case)

Test cases:
-----------

1D Lane-Emden equation for n = 5
$$
\frac{1}{r^2} \frac{\mathrm{d}}{\mathrm{d}r}
\left( r^2 \frac{\mathrm{d}u}{\mathrm{d}r} \right) + u^5 = 0 \, , \quad
u'(0) = 0 \, , \quad
u(3) = 1/2 \, .
$$
This nonlinear problem has a simple solution $u(r) = 1/\sqrt{1+r^2/3}$.

1D Klein-Gordon equation governing the chameleon field in radial configurations
$$
\alpha \frac{\mathrm{d}}{\mathrm{d}r}
\left( r^2 \frac{\mathrm{d} u}{\mathrm{d}r} \right) = \rho - \phi^{-(n+1)} \, ,
\quad u'(0) = 0 \, ,
\quad u(R_c) = u_{\mathrm{vac}}
$$
This nonlinear problem does not have an analytical solution but only
approximations that can nonetheless serve as a benchmark.
"""

import os
import pytest
import numpy as np
from numpy import sqrt

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.weak_form import WeakForm
from femtoscope.core.solvers import NonLinearSolver
from femtoscope.core.nonlinear_monitoring import NonLinearMonitor
from femtoscope.inout.meshfactory import generate_uniform_1d_mesh


class TestLaneEmden:

    @staticmethod
    def analytical_solution(r):
        return 1 / sqrt(1 + r ** 2 / 3)

    @staticmethod
    @pytest.fixture
    def wf_int():
        """Create the linearized weak form (instance of `WeakForm`)."""

        # Mesh creation
        pre_mesh = generate_uniform_1d_mesh(0, 3, 300, 'mesh_1d')

        # Terms
        def mat1(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2
            return {'val': val.reshape(-1, 1, 1)}
        t1 = PreTerm('dw_laplace', mat=mat1, tag='cst')

        def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2 * vec_qp ** 4
            return {'val': val.reshape(-1, 1, 1)}
        t2 = PreTerm('dw_volume_dot', mat=mat2, tag='mod', prefactor=-5)

        def mat3(ts, coors, mode=None, vec_qp=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2 * vec_qp ** 5
            return {'val': val.reshape(-1, 1, 1)}
        t3 = PreTerm('dw_integrate', mat=mat3, tag='mod', prefactor=4)

        # Vertex selection
        def right_boundary(coors, domain=None):
            return np.where(coors.squeeze() == 3.0)[0]

        def left_boundary(coors, domain=None):
            return np.where(coors.squeeze() == 0.0)[0]

        dim_func_entities = [(0, left_boundary, 0), (0, right_boundary, 1)]

        # WeakForm creation
        args_dict = {
            'name': 'wf_linearized_LaneEmden',
            'dim': 1,
            'pre_mesh': pre_mesh,
            'pre_terms': [t1, t2, t3],
            'dim_func_entities': dim_func_entities,
            'fem_order': 1,
            'pre_ebc_dict': {('vertex', 0): 1.0, ('vertex', 1): 0.5}
        }
        wf = WeakForm.from_scratch(args_dict)
        return wf

    @staticmethod
    @pytest.fixture
    def wf_res():
        """Create the residual weak form (instance of `WeakForm`)."""
        # Mesh creation
        pre_mesh = generate_uniform_1d_mesh(0, 3, 300, 'mesh_1d')

        # Terms
        def mat1(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2
            return {'val': val.reshape(-1, 1, 1)}
        t1 = PreTerm('dw_laplace', mat=mat1, tag='cst')

        def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2 * vec_qp ** 5
            return {'val': val.reshape(-1, 1, 1)}
        t2 = PreTerm('dw_integrate', mat=mat2, tag='mod', prefactor=-1)

        # Vertex selection
        def right_boundary(coors, domain=None):
            return np.where(coors.squeeze() == 3.0)[0]

        def left_boundary(coors, domain=None):
            return np.where(coors.squeeze() == 0.0)[0]

        dim_func_entities = [(0, left_boundary, 0), (0, right_boundary, 1)]

        # WeakForm creation
        args_dict = {
            'name': 'wf_residual_LaneEmden',
            'dim': 1,
            'pre_mesh': pre_mesh,
            'pre_terms': [t1, t2],
            'dim_func_entities': dim_func_entities,
            'fem_order': 1,
            'pre_ebc_dict': {('vertex', 0): 0.0, ('vertex', 0): 0.0}
        }
        wf = WeakForm.from_scratch(args_dict)
        return wf

    @staticmethod
    @pytest.fixture
    def nonlinear_solver(wf_int, wf_res):
        wf_dict = {'wf_int': wf_int, 'wf_residual': wf_res}
        # initial_guess = TestLaneEmden.analytical_solution(wf_int.field.coors.squeeze())
        initial_guess_dict = {'int': 0.75*np.ones(wf_int.field.coors.shape[0])}
        solver = NonLinearSolver(wf_dict, initial_guess_dict)
        return solver

    @staticmethod
    @pytest.fixture
    def nonlinear_monitor(nonlinear_solver):
        criteria = (
            {
                'name': 'RelativeDeltaSolutionNorm2',
                'threshold': 1e-6,
                'look': True,
                'active': False
            },
            {
                'name': 'ResidualVector',
                'threshold': -1,
                'look': True,
                'active': False
            },
            {
                'name': 'ResidualVectorNorm2',
                'threshold': -1,
                'look': True,
                'active': False
            },
        )
        args_dict = {
            'minimum_iter_num': 5,
            'maximum_iter_num': 50,
            'criteria': criteria
        }
        monitor = NonLinearMonitor.from_scratch(args_dict)
        return monitor

    @staticmethod
    def test_lane_emden(nonlinear_solver, nonlinear_monitor):
        nonlinear_monitor.link_monitor_to_solver(nonlinear_solver)
        nonlinear_solver.solve(verbose=False)
        rr = nonlinear_solver.wf_int.field.coors.squeeze()
        sol = nonlinear_solver.sol
        print("")

class TestKleinGordon1D:

    rho_min = 1
    rho_max = 1e2
    alpha = 0.1
    Rcut = 6.0
    fem_order = 2

    @classmethod
    def create_wf_int(cls):
        """Create the linearized weak form (instance of `WeakForm`)."""

        # Mesh creation
        pre_mesh = generate_uniform_1d_mesh(0, cls.Rcut, 500 + 1, 'mesh_1d')

        # Terms
        def mat1(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2
            return {'val': val.reshape(-1, 1, 1)}

        t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=cls.alpha)

        def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2 * vec_qp ** (-4)
            return {'val': val.reshape(-1, 1, 1)}

        t2 = PreTerm('dw_volume_dot', mat=mat2, tag='mod', prefactor=3)

        def mat3(ts, coors, mode=None, vec_qp=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2 * vec_qp ** (-3)
            return {'val': val.reshape(-1, 1, 1)}

        t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='mod', prefactor=-4)

        def mat4(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            r = coors.squeeze()
            rho = np.where(r <= 1.0, cls.rho_max, cls.rho_min)
            val = r ** 2 * rho
            return {'val': val.reshape(-1, 1, 1)}

        t4 = PreTerm('dw_volume_integrate', mat=mat4, tag='cst', prefactor=1)

        # Vertex selection
        def right_boundary(coors, domain=None):
            return np.where(coors.squeeze() == cls.Rcut)[0]

        dim_func_entities = [(0, right_boundary, 0)]

        # WeakForm creation
        args_dict = {
            'name': 'wf_chameleon_1d',
            'dim': 1,
            'pre_mesh': pre_mesh,
            'pre_terms': [t1, t2, t3, t4],
            'dim_func_entities': dim_func_entities,
            'fem_order': cls.fem_order,
            'pre_ebc_dict': {('vertex', 0): cls.rho_min ** (-1 / 3)}
        }
        wf = WeakForm.from_scratch(args_dict)
        return wf

    @classmethod
    def create_wf_res(cls):
        """Create the residual weak form (instance of `WeakForm`)."""

        # Mesh creation
        pre_mesh = generate_uniform_1d_mesh(0, cls.Rcut, 500 + 1, 'mesh_1d')

        # Terms
        def mat1(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2
            return {'val': val.reshape(-1, 1, 1)}

        t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=cls.alpha)

        def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
            if mode != 'qp': return
            val = coors.squeeze() ** 2 * vec_qp ** (-3)
            return {'val': val.reshape(-1, 1, 1)}

        t2 = PreTerm('dw_volume_integrate', mat=mat2, tag='mod', prefactor=-1)

        def mat3(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            r = coors.squeeze()
            rho = np.where(r <= 1, cls.rho_max, cls.rho_min)
            val = r ** 2 * rho
            return {'val': val.reshape(-1, 1, 1)}

        t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='cst', prefactor=1)

        def mat4(ts, coors, mode=None, **kwargs):
            if mode != 'qp': return
            r = coors.squeeze()
            val = np.zeros((coors.shape[0], 1, 1))
            val[:, 0, 0] = r ** 2
            return {'val': val}

        t4 = PreTerm('dw_surface_flux', mat=mat4, tag='cst', prefactor=-cls.alpha,
                     region_key=('vertex', 0))

        # Vertex selection
        def right_boundary(coors, domain=None):
            return np.where(coors.squeeze() == cls.Rcut)[0]

        dim_func_entities = [(0, right_boundary, 0)]

        # WeakForm creation
        args_dict = {
            'name': 'wf_residual_1d',
            'dim': 1,
            'pre_mesh': pre_mesh,
            'pre_terms': [t1, t2, t3, t4],
            'dim_func_entities': dim_func_entities,
            'fem_order': cls.fem_order,
        }
        wf = WeakForm.from_scratch(args_dict)
        return wf

    @classmethod
    def create_nonlinear_solver(cls, wf_int, wf_res):
        wf_dict = {'wf_int': wf_int, 'wf_residual': wf_res}
        phi_min = cls.rho_max ** (-1 / 3)
        phi_max = cls.rho_min ** (-1 / 3)
        rr = wf_int.field.coors.squeeze()
        initial_guess = np.where(rr <= 1, phi_min, phi_max)
        initial_guess_dict = {'int': initial_guess}
        solver = NonLinearSolver(wf_dict, initial_guess_dict,
                                 sol_bounds=[phi_min, phi_max])
        return solver

    @classmethod
    def create_nonlinear_monitor(cls, nonlinear_solver):
        criteria = (
            {
                'name': 'RelativeDeltaSolutionNorm2',
                'threshold': 1e-6,
                'look': True,
                'active': False
            },
            {
                'name': 'ResidualVector',
                'threshold': -1,
                'look': True,
                'active': False
            },
            {
                'name': 'ResidualVectorNorm2',
                'threshold': -1,
                'look': True,
                'active': False
            },
            {
                'name': 'ResidualReductionFactor',
                'threshold': -1,
                'look': True,
                'active': False
            },
        )
        args_dict = {
            'minimum_iter_num': 5,
            'maximum_iter_num': 10,
            'criteria': criteria
        }
        monitor = NonLinearMonitor.from_scratch(args_dict)
        return monitor

    @classmethod
    def test_klein_gordon(cls):
        # Setup
        wf_int = cls.create_wf_int()
        wf_res = cls.create_wf_res()
        solver = cls.create_nonlinear_solver(wf_int, wf_res)
        monitor = cls.create_nonlinear_monitor(solver)
        monitor.link_monitor_to_solver(solver)

        # Solve Klein-Gordon equation
        solver.solve(verbose=False)

        # Tests
        cond1 = monitor.criteria['ResidualReductionFactor'].value < 7e-3
        assert cond1



