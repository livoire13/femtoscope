# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:57:09 2024

Solver engine for linear/nonlinear & bounded/unbounded problems.

Notes
-----
Documentation for Sfepy's linear solvers can be found at
https://sfepy.org/doc-devel/src/sfepy/solvers/ls.html#module-sfepy.solvers.ls
By default, femtoscope will use 'ScipyDirect' for solving linear systems.
"""

import warnings

from pathlib import Path
from typing import Union
import pickle

import numpy as np
from scipy.optimize import minimize_scalar, OptimizeResult

from sfepy.base.base import output
from sfepy.base.base import IndexedStruct
from sfepy.solvers.ls import (ScipyDirect, ScipyIterative, ScipySuperLU,
                              ScipyUmfpack, MUMPSSolver)

from femtoscope.core.weak_form import WeakForm
import femtoscope.core.nonlinear_monitoring as monitoring
from femtoscope.display.femdisplay import display_from_vtk
from femtoscope.inout.vtkfactory import (
    create_structured_vtk, make_absolute_vtk_name, create_connectivity_table)
from femtoscope.misc.util import get_date_string
from femtoscope import RESULT_DIR

output.set_output(quiet=True)  # Toggle for Sfepy console's messages


class LinearSolver:
    """
    Class for solving linear problems on bounded or unbounded domains.

    Attributes
    ----------
    ls_class_dict : dict
        Class attribute linking to some available linear system solver in
        Sfepy.
    default_eps_a : float
        Class attribute corresponding to the default absolute tolerance for
        the residual of the linear system. Relevant when using iterative
        linear system solvers.
    default_eps_r : float
        Class attribute corresponding to the default relative tolerance for
        the residual of the linear system. Relevant when using iterative
        linear system solvers.
    default_max_iter_ls : int
        Class attribute corresponding to the default maximum number of
        iterations performed by iterative linear system solvers.
    default_connection_method : str
        Class attribute corresponding to the default connection method for bi-
        domain computations.
    wf_dict : dict
        Dictionary comprising all weak forms that may be used in this class,
        including `wf_int` and `wf_ext` (see below).
    wf_int : WeakForm
        Interior weak form. If `wf_ext` is None, plays the role of the main
        weak form.
    wf_ext : WeakForm
        Exterior weak form.
    ls : LinearSolver
        Instance of a child class of `LinearSolver` listed in class attribute
        `ls_class_dict`.
    ls_kwargs : Union[dict, None]
            Input dictionary for the chosen linear system solver, with keys
            that may depend on the specific solver.
    sol_int : np.ndarray
        Solution vector in the interior domain.
    sol_ext : Union[np.ndarray, None]
        Solution vector in the exterior domain (when relevant).
    connection_method : str
            Selector of the method to be used for linking `wf_int` and `wf_ext`.
    region_key_int : tuple
        Key of the connecting region in the interior domain. Mandatory
        parameter for two-weak-form problems.
    region_key_ext : tuple
        Key of the connecting region in the exterior domain. Mandatory
        parameter for two-weak-form problems.
    result_int_name : Union[None, str]
        Name of the file containing the results of the FEM computation in the
        interior domain. This attribute gets set when the method
        `save_results_to_vtk` is called.
    result_ext_name : Union[None, str]
        Same as `result_int_name` but for the exterior domain.

    """
    ls_class_dict = {'ScipyDirect': ScipyDirect,
                     'ScipyIterative': ScipyIterative,
                     'ScipySuperLU': ScipySuperLU,
                     'ScipyUmfpack': ScipyUmfpack,
                     'MUMPSSolver': MUMPSSolver}

    default_eps_a = 1e-10
    default_eps_r = 1e-10
    default_max_iter_ls = 500
    default_connection_method = 'ifem'

    def __init__(self, wf_dict: dict, ls_class='ScipyDirect',
                 ls_kwargs=None, **kwargs):
        """
        Parameters
        ----------
        wf_dict : dict
            Dictionary comprising all weak forms that may be used in this class
            instance, including `wf_int` and `wf_ext`.
        ls_class : str, optional
            Key from class attribute `ls_class_dict`.
            The default is 'ScipyDirect'.
        ls_kwargs : dict, optional
            Input dictionary for the chosen linear system solver, with keys
            that may depend on the specific solver. The default is None.

        Other Parameters
        ----------------
        connection_method : str, optional
            Selector of the method to be used for linking `wf_int` and `wf_ext`.
            The default is class attribute `default_connection_method`.
        region_key_int : tuple
            Key of the connecting region in the interior domain. Mandatory
            parameter for two-weak-form problems.
        region_key_ext : tuple
            Key of the connecting region in the exterior domain. Mandatory
            parameter for two-weak-form problems.

        """

        # Get keyword arguments
        connection_method = kwargs.get(
            'connection_method', self.default_connection_method)
        region_key_int = kwargs.get('region_key_int')
        region_key_ext = kwargs.get('region_key_ext')

        # Perform checks on the inputs
        self._check_arguments(wf_dict, ls_class, region_key_int, region_key_ext)

        # Set weak form attributes
        self.wf_dict = wf_dict
        self.wf_int = wf_dict['wf_int']
        self.wf_ext = wf_dict.get('wf_ext', None)

        # Set linear solver attributes
        self.ls = LinearSolver.ls_class_dict[ls_class](ls_kwargs)
        ls_kwargs = {} if ls_kwargs is None else ls_kwargs
        self.ls_kwargs = ls_kwargs

        # Set solution vector attributes
        self.sol_int = np.empty(len(self.wf_int.field.coors), dtype=np.float64)
        if self.wf_ext is not None:
            self.sol_ext = np.empty(
                len(self.wf_ext.field.coors), dtype=np.float64)
        else:
            self.sol_ext = None

        # Set bi-domain attributes
        self.connection_method = connection_method
        self.region_key_int = region_key_int
        self.region_key_ext = region_key_ext

        # Names for saving results
        self.result_int_name = None
        self.result_ext_name = None

    @property
    def wf(self):
        if self.wf_ext is None:
            return self.wf_int
        else:
            print("There is not a single weak form for this problem but two, "
                  "returning 'None'")

    @property
    def sol(self):
        if self.wf_ext is None:
            return self.sol_int
        else:
            print("There is not a single weak form for this problem but two, "
                  "returning 'None'")

    @property
    def is_linear(self):
        return self.__class__.__name__ == 'LinearSolver'

    def solve(self, use_reduced_mtx_vec=True, use_buffer=False, **kwargs):
        """
        Solve the linear problem.

        Parameters
        ----------
        use_reduced_mtx_vec : bool, optional
            Use reduced matrix and rhs vector for the linear system solving
            stage. The default is True.
        use_buffer : bool, optional
            Store the solution of the linear system in a buffer at the end of
            the linear system solving stage. This option is enabled when solving
            nonlinear problems with Newton's iterations.
            The default is False.

        Other Parameters
        ----------------
        verbose : bool
            Display user information. The default is False.

        """

        verbose = kwargs.get('verbose', False)

        if self.wf_ext is None:  # Single weak form case
            self._solve_single_wf(use_reduced_mtx_vec, use_buffer)

        else:  # Two weak forms case

            if self.connection_method == 'ifem':
                if not use_reduced_mtx_vec:
                    warnings.warn(f"Cannot use full mtx/vec with method "
                                  f"'{self.connection_method}'")
                self._solve_two_wf_ifem(
                    use_buffer, self.region_key_int, self.region_key_ext)

            elif self.connection_method == 'a-ifem':
                self._solve_two_wf_a_ifem(
                    use_buffer, self.region_key_int, self.region_key_ext)

            else:
                raise NotImplementedError(
                    f"connection method '{self.connection_method}' "
                    f"is not implemented!")

    def save_results(self, name: Union[None, str] = None):
        """
        Save FEM results to a VTK file together with a .pkl file containing
        metadata for offline post-processing. The results are saved in a
        subdirectory of `RESULT_DIR`.

        Parameters
        ----------
        name : Union[None, str], optional
            Name of the directory containing the results. The default is None,
            in which case the current date-time is used.

        See Also
        --------
        save_results_to_vtk : Creation of the VTK file
        WeakForm.get_pickable_args_dict : metadata dictionary

        """

        # Setup
        if name is None:
            name = get_date_string()
        dir_path = Path(RESULT_DIR / name)
        dir_path.mkdir()

        # Create VTK and metadata files
        self.save_results_to_vtk(str(Path(dir_path / name)))
        self._save_metadata(name)

    def save_results_to_vtk(self, name: Union[None, str] = None):
        """
        Save the FEM results and set the attributes `result_int_name` and
        `result_ext_name` (when relevant) using `create_structured_vtk`.
        When `sol_ext` is not None, two separate files are created.

        Parameters
        ----------
        name : Union[None, str], optional
            Name of the .vtk result file. If `sol_ext` is not None, the suffix
            '_int' / '_ext' is appended to the provided name in order to
            distinguish between the two created files. The default is None, in
            which case the current date-time is used.

        """

        if self.wf_int.dim == 1:  # handling of the one-dimensional case
            print("Results of 1D problems cannot be saved to VTK.")
            return
        aux_name = make_absolute_vtk_name(name)
        aux_name = str(Path(aux_name).with_suffix(''))
        for suffix in ('_int', '_ext'):

            if self.wf_ext is None and suffix == '_ext':
                continue  # no exterior solution

            # Prepare data for VTK creation
            if self.wf_ext is not None:
                path_name = aux_name + suffix
            else:
                path_name = aux_name
            wf = getattr(self, 'wf' + suffix)
            coors = wf.field.coors
            cells = create_connectivity_table(coors)
            vars_dict = {'sol' + suffix: getattr(self, 'sol' + suffix)}

            # Residual vector & parts (when relevant!)
            if (suffix == '_int') and (not self.is_linear):
                monitor = self.nonlinear_monitor
                if monitor.residual_vetor is not None:
                    vars_dict['residual'] = monitor.residual_vetor
                if 'ResidualVectorParts' in monitor.criteria:
                    res_parts = monitor.criteria['ResidualVectorParts'].value
                    vars_dict['res_mtx'] = res_parts['mtx_term']
                    vars_dict['res_rhs_cst'] = res_parts['rhs_cst_term']
                    vars_dict['res_rhs_mod'] = res_parts['rhs_mod_term']

            # VTK creation
            create_structured_vtk(coors, vars_dict, cells, path_name=path_name)
            setattr(self, 'result{}_name'.format(suffix), path_name + '.vtk')

    def display_results(self):
        """Display solver results. Proxy for `display_from_vtk` method from
        femtoscope.display.femdisplay module."""

        # Save results in TMP_DIR
        path_name = make_absolute_vtk_name(None)
        self.save_results_to_vtk(name=path_name)

        # display
        display_from_vtk(self.result_int_name)
        display_from_vtk(self.result_ext_name)

        # Delete the VTK file
        Path(self.result_int_name).unlink()
        if self.result_ext_name is not None:
            Path(self.result_ext_name).unlink(missing_ok=True)

    def _solve_single_wf(self, use_reduced_mtx_vec: bool, use_buffer: bool):
        """Solve the linear problem consisting of a single weak form."""

        # Get matrix and rhs vector
        if use_reduced_mtx_vec:
            if self.wf.mtx_vec.mtx_bc_reduced is None:
                self.wf.set_mtx_vec_bc_reduced()
            mtx = self.wf.mtx_vec.mtx_bc_reduced
            rhs = self.wf.mtx_vec.vec_bc_reduced
        else:
            if self.wf.mtx_vec.get_mtx_bc_full(self.wf.pb_cst) is None:
                self.wf.set_mtx_vec_bc_full()
            mtx = self.wf.mtx_vec.get_mtx_bc_full(self.wf.pb_cst)
            rhs = self.wf.mtx_vec.get_vec_bc_full(self.wf.pb_cst)

        # Solve linear system
        vec_x = self._solve_linear_system(mtx, rhs)

        # Reconstruct full solution satisfying E(P)BCs
        if use_reduced_mtx_vec:
            vec_x = self.wf.make_full_vec(vec_x)

        # Store solution
        if use_buffer:
            self.buffer_int[:] = vec_x
        else:
            self.sol_int[:] = vec_x

    def _solve_two_wf_ifem(self, use_buffer: bool,
                           region_key_int: tuple, region_key_ext: tuple):
        """Solve the linear problem consisting of two weak forms using the
        so-called 'inverted finite element method'."""

        # Create new weak form with LCBCs
        wf_int = self.wf_int
        wf_ext = self.wf_ext
        wf = WeakForm.from_two_weak_forms(
            wf_int, wf_ext, region_key_int, region_key_ext)

        # Get matrix and rhs vector
        if wf.mtx_vec.mtx_bc_reduced is None:
            wf.set_mtx_vec_bc_reduced()
        mtx = wf.mtx_vec.mtx_bc_reduced
        rhs = wf.mtx_vec.vec_bc_reduced

        # Solve linear system
        vec_x = self._solve_linear_system(mtx, rhs)

        # Reconstruct interior and exterior solutions
        variables_cst = wf.pb_cst.get_variables()
        variables_cst.set_state(vec_x, reduced=True, force=True)
        sol_int = variables_cst.get(wf_int.get_unknown_name('cst'))()
        sol_ext = variables_cst.get(wf_ext.get_unknown_name('cst'))()

        # Store solution
        if use_buffer:
            self.buffer_int = sol_int
            self.buffer_ext = sol_ext
        else:
            self.sol_int[:] = sol_int
            self.sol_ext[:] = sol_ext

    def _solve_two_wf_a_ifem(self, use_buffer: bool,
                             region_key_int: tuple, region_key_ext: tuple):
        """Solve the linear problem consisting of two weak forms using the
        so-called 'alternate inverted finite element method'."""
        raise NotImplementedError("'a-ifem' method is not implemented yet!")

    def _solve_linear_system(self, mtx, rhs):
        status = IndexedStruct()
        eps_a = self.ls_kwargs.get('eps_a', LinearSolver.default_eps_a)
        eps_r = self.ls_kwargs.get('eps_r', LinearSolver.default_eps_r)
        max_iter_ls = self.ls_kwargs.get('i_max',
                                         LinearSolver.default_max_iter_ls)
        vec_x = self.ls(rhs, x0=None, eps_a=eps_a, eps_r=eps_r,
                        i_max=max_iter_ls, mtx=mtx, status=status)
        return vec_x

    def _save_metadata(self, name):
        """
        Pickle save weak forms metadata for future offline postprocessing.
        The location of such files is necessarily a subdirectory of `RESULT_DIR`
        with name `name`.

        Notes
        -----
        This method is private and should therefore only be called within the
        `save_results` method.

        """

        dir_path = Path(RESULT_DIR / name)

        for suffix in ['_int', '_ext']:
            path_name = str(Path(dir_path / name))
            if suffix == '_ext' and self.wf_ext is None:
                continue
            if self.wf_ext is not None:
                path_name += suffix
            wf = getattr(self, 'wf' + suffix)
            args_dict = wf.get_pickable_args_dict()
            with open(path_name + '.pkl', 'wb') as f:
                pickle.dump(args_dict, f)

    def _check_arguments(self, wf_dict: dict, ls_class: str,
                         region_key_int: tuple, region_key_ext: tuple):
        self._check_wf_dict(wf_dict)
        self._check_ls_class(ls_class)
        self._check_region_keys(wf_dict, region_key_int, region_key_ext)

    @staticmethod
    def _check_wf_dict(wf_dict: dict):
        """
        Perform several sanity checks on the provided weak forms at solver
        instantiation.

        Parameters
        ----------
        wf_dict : dict
            Dictionary containing `WeakForm` instances passed to the current
            class constructor.
        """

        # Check types
        for wf in wf_dict.values():
            if not isinstance(wf, WeakForm):
                raise TypeError(f"'wf_dict' should only contain 'WeakForm'"
                                f"instances, but '{wf}' is of type {type(wf)}")

        if 'wf_ext' not in wf_dict:
            for wf in wf_dict.values():
                if wf._is_exterior:
                    raise ValueError("weak forms should all be defined on the"
                                     "interior domain!")
                wf._is_exterior = False

        else:
            wf_dict['wf_int']._is_exterior = False
            wf_dict['wf_ext']._is_exterior = True
            for key, wf in wf_dict.items():
                if key in ['wf_int', 'wf_ext']:
                    continue
                if wf._is_exterior is None:
                    raise ValueError(f"User must specify whether weak form"
                                     f"'{wf.name}' is interior or exterior.")

    @staticmethod
    def _check_ls_class(ls_class: str):
        """Check the `ls_class` parameter at solver instantiation."""
        ls_class_dict = LinearSolver.ls_class_dict
        if ls_class not in ls_class_dict:
            raise KeyError(f"'ls_class' must be in {ls_class_dict}")

    @staticmethod
    def _check_region_keys(
            wf_dict: dict, region_key_int: tuple, region_key_ext: tuple):
        if wf_dict.get('wf_ext') is None:
            return
        if region_key_int is None or region_key_ext is None:
            raise ValueError("User must provide region keys of the shared "
                             "interface when solving a two-weak-form problem!")


class NonLinearSolver(LinearSolver):
    """
    Class for solving nonlinear problems on bounded or unbounded domains.
    It inherits from class `LinearSolver`.

    Inherited Attributes
    --------------------
    ls_class_dict : dict
        Class attribute linking to some available linear system solver in
        Sfepy.
    default_eps_a : float
        Class attribute corresponding to the default absolute tolerance for
        the residual of the linear system. Relevant when using iterative
        linear system solvers.
    default_eps_r : float
        Class attribute corresponding to the default relative tolerance for
        the residual of the linear system. Relevant when using iterative
        linear system solvers.
    default_max_iter_ls : int
        Class attribute corresponding to the default maximum number of
        iterations performed by iterative linear system solvers.
    wf_dict : dict
        Dictionary comprising all weak forms that may be used in this class,
        including `wf_int` and `wf_ext` (see below).
    wf_int : WeakForm
        Interior weak form. If `wf_ext` is None, plays the role of the main
        weak form.
    wf_ext : WeakForm
        Exterior weak form.
    ls : LinearSolver
        Instance of a child class of `LinearSolver` listed in class attribute
        `ls_class_dict`.
    ls_kwargs : Union[dict, None]
            Input dictionary for the chosen linear system solver, with keys
            that may depend on the specific solver.
    sol_int : np.ndarray
        Solution vector in the interior domain.
    sol_ext : Union[np.ndarray, None]
        Solution vector in the exterior domain (when relevant).

    New Attributes
    --------------
    default_relax_param : float
        Class attribute, default relaxation parameter to be used if the user
        does not provide one.
    implemented_relax_methods : Tuple[str]
        Currently available relaxation methods
    old_int : np.ndarray
        Solution vector in the interior domain at the previous iteration.
    old_ext : np.ndarray
        Solution vector in the exterior domain at the previous iteration (when
        relevant).
    buffer_int : np.ndarray
        Buffer space for the solution vector in the interior domain.
    buffer_ext : np.ndarray
        Buffer space for the solution vector in the exterior domain (when
        relevant).
    initial_guess_dict : dict
        Dictionary with keys ('int', 'ext') containing the initial guess as
        numpy 1d-arrays.
    sol_bounds : tuple
        Pair of two real values (sol_min, sol_max) that (theoretically) bound
        the solution. For all iterations, it is used to 'saturate' entries of
        the solution vector that fall off bounds.
    nonlinear_monitor : monitoring.NonLinearMonitor
        Instance of custom class `NonLinearMonitor` for monitoring the iterative
        process: stopping criteria, convergence, residual vector, ...
    relax_param : float
        Relaxation parameter. Can be iteration-dependent depending on the chosen
        relaxation method (see right below).
    relax_method : str
        Relaxation method. Accepted strings are
            - 'constant' for a fixed parameter throughout all the iterations
            - 'line-search' for employing the line-search algorithm
            - 'dwindle' for forcing the parameter to decrease if convergence
            does not occur 'naturally'

    """

    default_relax_param = 0.8
    implemented_relax_methods = ('constant', 'line-search', 'dwindle')

    class Decorators:
        """Inner class for defining decorators."""

        @staticmethod
        def handle_wf_keys(func):
            def wrapper(*args, **kwargs):
                self = args[0]
                wf_keys = args[1:]
                if len(wf_keys) == 1 and wf_keys[0] == 'all':
                    wf_keys = tuple(self.wf_dict.keys())
                args = (self,) + wf_keys
                return func(*args, **kwargs)

            return wrapper

    def __init__(self, wf_dict: dict, initial_guess_dict: dict, **kwargs):
        r"""
        Extend constructor from parent class `LinearSolver`.

        Parameters
        ----------
        wf_dict : dict
            Dictionary comprising all weak forms that may be used in this class
            instance, including `wf_int` and `wf_ext`.
        initial_guess_dict : dict
            Dictionary with keys ('int', 'ext') containing the initial guess as
            numpy 1d-arrays.

        Other Parameters
        ----------------
        ls_class : str, optional
            Key from class attribute `ls_class_dict`.
            The default is 'ScipyDirect'.
        ls_kwargs : dict, optional
            Input dictionary for the chosen linear system solver, with keys
            that may depend on the specific solver. The default is None.
        relax_param : float, optional
            Relaxation parameter to be used. The default is specified by the
            class attribute `default_relax_param`.
        relax_method : str, optional
            Relaxation method to be used, either:
                - 'constant' (same relaxation parameter across all iterations).
                - 'line-search' ($w = \arg \min \| R \|_2^2$). Needs the
                residual weak form to be specified.
                - 'dwindle' (manually decreasing the relaxation parameter when
                convergence is not happening).
        connection_method : str, optional
            Selector of the method to be used for linking `wf_int` and `wf_ext`.
            The default is class attribute `default_connection_method`.
        region_key_int : tuple
            Key of the connecting region in the interior domain. Mandatory
            parameter for two-weak-form problems.
        region_key_ext : tuple
            Key of the connecting region in the exterior domain. Mandatory
            parameter for two-weak-form problems.

        """

        # Retrieve keyword arguments
        relax_param = kwargs.get('relax_param', self.default_relax_param)
        relax_method = kwargs.get('relax_method', 'constant')

        # Set parent class attributes (LinearSolver)
        super().__init__(wf_dict, **kwargs)

        # Perform checks on the inputs
        self._check_initial_guess_dict(initial_guess_dict)
        self._check_relax_method(relax_method, wf_dict)
        self._check_is_nonlinear()  # make sure that the problem is not linear

        # Set interior-domain attributes
        self.initial_guess_dict = initial_guess_dict
        self.sol_int[:] = initial_guess_dict['int']
        self.old_int = np.copy(initial_guess_dict['int'])
        self.buffer_int = np.empty_like(self.sol_int)

        # When relevant, set exterior-domain attributes
        if self.sol_ext is not None:
            self.sol_ext[:] = initial_guess_dict['ext']
            self.old_ext = np.copy(initial_guess_dict['ext'])
            self.buffer_ext = np.empty_like(self.sol_ext)
        else:
            self.old_ext = None
            self.buffer_ext = None

        # Miscellaneous attributes
        self.sol_bounds = kwargs.get('sol_bounds', None)
        self.nonlinear_monitor = None
        self.relax_param = relax_param
        self.relax_method = relax_method

    def solve(self, verbose=True, **kwargs):
        """
        Solve the nonlinear problem.
        Override `solve` method from parent class `LinearSolver`.

        Parameters
        ----------
        verbose : bool, optional
            Get information about the solving process. The default is True.

        Other Parameters
        ----------------
        first_call : bool, optional
            Whether this method is called for the first time or it resumes.
            The default is False.
        pause_iter_num : int, optional
            If specified, pause the iterations at the corresponding iteration
            number. The default is -1.
        connection_method : str
            Selector of the method to be used for linking `wf_int` and `wf_ext`.
            The default is 'ifem'.
        region_key_int : tuple
            Key of the connecting region in the interior domain. Mandatory
            parameter for two-weak-form problems.
        region_key_ext : tuple
            Key of the connecting region in the exterior domain. Mandatory
            parameter for two-weak-form problems.

        """

        # Retrieve keyword arguments
        first_call = kwargs.get('first_call', True)
        pause_iter_num = kwargs.get('pause_iter_num', -1)

        # Set up before entering main loop
        monitor = self.nonlinear_monitor
        if first_call:
            self._check_solver_monitor_link()
            self.update_nonlinear_materials('all')
            monitor.current_iter_num = 0
            monitor.evaluate_all_criteria()
        if verbose:
            monitor.display_criteria_info('look', 'active', 'threshold')

        # Main 'while' loop
        while not monitor.stop:
            monitor.current_iter_num += 1

            # Solve the linearized problem, the solution is put in the buffer
            LinearSolver.solve(
                self, use_reduced_mtx_vec=True, use_buffer=True, **kwargs)
            self.bound_solution(buffer=True)

            # Relaxation step
            self.perform_relaxation_step()
            self.bound_solution(buffer=False)

            # Update matrices and vectors with new solution
            self.update_nonlinear_mtx_vec('all')

            # Monitor the current iteration
            monitor.evaluate_all_criteria()
            monitor.update_status()
            if verbose:
                monitor.display_criteria_info('value', 'threshold')

            # Update 'old' solution
            self.old_int[:] = self.sol_int
            if self.sol_ext is not None:
                self.old_ext[:] = self.sol_ext

            # Pause?
            if monitor.current_iter_num == pause_iter_num:
                self._write_pause_message(pause_iter_num)
                return

        monitor.write_termination_status()
        monitor.display_iterations_report()

    def resume(self, force=False, new_maximum_iter_num=None,
               extra_iter_num=None, **kwargs):
        """
        Resume the iterations by calling `solve` method with `first_call=False`.

        Parameters
        ----------
        force : bool, optional
            If True, set all stopping criteria as inactive except the maximum
            number of iterations 'MaximumIterations' which remains active.
        new_maximum_iter_num : int, optional
            The new threshold for criterion 'MaximumIterations'.
            The default is None.
        extra_iter_num : int, optional
            Number of additional iterations to undertake. The default is None.
        kwargs : dict
            Keyword arguments passed to the 'solve' method.
        """

        # Retrieve keyword arguments and handle exception
        verbose = kwargs.get('verbose', True)
        if (new_maximum_iter_num is not None) and (extra_iter_num is not None):
            raise Warning("It is not possible to set 'new_maximum_iter_num' "
                          "and 'extra_iter_num' simultaneously, aborting.")
            return

        # Current data
        monitor = self.nonlinear_monitor
        old_maximum_iter_num = monitor.maximum_iter_num

        # Set a new maximum number of iterations
        if new_maximum_iter_num is not None:
            if new_maximum_iter_num <= old_maximum_iter_num:
                if verbose:
                    print("No additional iterations to perform!")
                return
            monitor.set_maximum_iter_num(new_maximum_iter_num)
        if extra_iter_num is not None:
            monitor.set_maximum_iter_num(old_maximum_iter_num + extra_iter_num)

        if force:  # force the undertaking of iterations by disabling criteria
            monitor.disable_all_criteria()
            monitor.stop = False

        self.solve(first_call=False, **kwargs)

    def bound_solution(self, buffer=False):
        """
        Bound the solution between `sol_bounds[0]` and `sol_bounds[1]` (when
        bounds have been provided to the solver). The `buffer` toggle (bool)
        indicates whether the correction is to be made on the buffer or on the
        current solution. Lower and/or upper bounds can be None, in which case
        the truncation is not performed.
        """

        if self.sol_bounds is None:  # No bounds provided
            return

        sol_min = self.sol_bounds[0]
        sol_max = self.sol_bounds[1]

        if buffer:  # Use the buffer site
            sol_list = [self.buffer_int, self.buffer_ext]
        else:  # Use the solution site
            sol_list = [self.sol_int, self.sol_ext]

        for sol in sol_list:
            if sol is None:  # When there is no exterior domain
                continue
            if sol_min is not None:
                sol[np.where(sol < sol_min)[0]] = sol_min  # Apply lower bound
            if sol_max is not None:
                sol[np.where(sol > sol_max)[0]] = sol_max  # Apply upper bound

    def perform_relaxation_step(self):
        """
        Wrapper function for performing the relaxation step. The actual
        implementation depends on `self.relax_method`.
        """
        if self.relax_method == 'constant':
            self._constant_relaxation()
        elif self.relax_method == 'line-search':
            self._line_search_relaxation()
        elif self.relax_method == 'dwindle':
            self._dwindle_relaxation()
        else:
            raise NotImplementedError(
                f"Method {self.relax_method} is not a valid relaxation method")

    def link_solver_to_monitor(self, monitor: monitoring.NonLinearMonitor):
        """
        Link the current `NonLinearSolver` instance to a given
        `NonLinearMonitor` instance. This link is two-way in the sense that the
        two class instances hold themselves as attributes.

        Parameters
        ----------
        monitor : monitoring.NonLinearMonitor
            Instance of class `NonLinearMonitor` for monitoring the nonlinear
            solver.

        Notes
        -----
        One can either link with the present function, or equivalently with
        `NonLinearMonitor.link_monitor_to_solver`.
        """
        self.nonlinear_monitor = monitor
        monitor.nonlinear_solver = self

    def evaluate_residual_vector(self, vec: Union[None, np.ndarray] = None,
                                 force_reassemble=False) -> np.ndarray:
        """
        Evaluate the residual vector using the user-provided WeakForm instance
        `wf_residual`.

        Parameters
        ----------
        vec : Union[None, np.ndarray], optional
            If specified, `vec` is the solution vector used to compute the
            residual vector. The default is None.
        force_reassemble : bool
            Force the reassembly of matrices and vectors associated with terms
            tagged 'mod'. Is relevant only when `vec` is None.
            The default is False.

        Returns
        -------
        residual_vector : np.ndarray
            The residual vector.

        See Also
        --------
        evaluate_residual_vector_parts

        """

        # Reassemble matrices and vectors
        if vec is not None:
            self.update_nonlinear_mtx_vec('wf_residual', vec=vec)
        else:
            vec = self.sol_int
            if force_reassemble:
                self.update_nonlinear_mtx_vec('wf_residual')

        # Compute the residual vector using 'no_bc' mtx/vec
        wf_res = self.wf_dict['wf_residual']
        if wf_res.mtx_vec.mtx_no_bc is None:
            wf_res.set_mtx_vec_no_bc()
        mtx_res = wf_res.mtx_vec.mtx_no_bc
        rhs_res = wf_res.mtx_vec.vec_no_bc
        residual_vector = mtx_res.dot(vec) - rhs_res

        return residual_vector

    def evaluate_residual_vector_parts(self,
                                       vec: Union[None, np.ndarray] = None,
                                       force_reassemble=False) -> dict:
        r"""
        Evaluate the residual vector decomposed into
            $$ R = A u_k - rhs_{\text{cst}} - rhs_{\text{mod}} $$

        Returns
        -------
        residual_vector_parts : dict
            Dictionary with keys ('mtx_term', 'rhs_cst_term', 'rhs_mod_term')
            and values of type np.ndarray.

        See Also
        --------
        evaluate_residual_vector
        """

        # Reassemble matrices and vectors
        if vec is not None:
            self.update_nonlinear_mtx_vec('wf_residual', vec=vec)
        else:
            vec = self.sol_int
            if force_reassemble:
                self.update_nonlinear_mtx_vec('wf_residual')

        # Compute the residual vector using 'no_bc' mtx/vec
        wf_res = self.wf_dict['wf_residual']
        if wf_res.mtx_vec.mtx_no_bc is None:
            wf_res.set_mtx_vec_no_bc()
        mtx_term = wf_res.mtx_vec.mtx_no_bc.dot(vec)
        rhs_cst_term = wf_res.mtx_vec.vec_no_bc_cst
        rhs_mod_term = wf_res.mtx_vec.vec_no_bc_mod

        residual_vector_parts = {
            'mtx_term': mtx_term,
            'rhs_cst_term': rhs_cst_term,
            'rhs_mod_term': rhs_mod_term
        }

        return residual_vector_parts

    @Decorators.handle_wf_keys
    def update_nonlinear_mtx_vec(self, *wf_keys: str,
                                 vec: Union[None, np.ndarray] = None):
        """
        Re-assemble nonlinear matrices/vectors (those associated with pre_terms
        tagged 'mod') belonging to the weak forms provided through their keys
        `wf_keys`. Calls `update_nonlinear_materials` first (see below).

        Parameters
        ----------
        wf_keys : str
            Keys of the weak forms to update.
        vec : Union[None, np.ndarray], optional
            Vector to be used to used for the update. If not set, use `sol_int`
            (and `sol_ext` when relevant). The default is None.

        Notes
        -----
        Provided keys must be keys of `wf_dict`. String 'all' is only exception,
        for which all available weak forms will be updated. This exception is
        handled with the decorator `handle_wf_keys`.
        """

        # Update corresponding nonlinear materials
        self.update_nonlinear_materials(*wf_keys, vec=vec)

        # Looping through all given weak forms and re-assemble matrices and
        # vectors tagged 'mod'
        for key in wf_keys:
            wf = self.wf_dict[key]
            if wf.mtx_vec.mtx_no_bc is not None:
                wf.set_mtx_vec_no_bc(tag='mod')
            if wf.mtx_vec.mtx_bc_reduced is not None:
                wf.set_mtx_vec_bc_reduced(tag='mod')
            if wf.mtx_vec.get_mtx_bc_full(wf.pb_cst) is not None:
                wf.set_mtx_vec_bc_full(tag='mod')

    @Decorators.handle_wf_keys
    def update_nonlinear_materials(self, *wf_keys: str,
                                   vec: Union[None, np.ndarray] = None):
        """
        Update the nonlinear materials (those associated with pre_terms tagged
        'mod').

        Parameters
        ----------
        wf_keys : str
            Keys of the weak forms to update.
        vec : Union[None, np.ndarray], optional
            Vector to be used to used for the update. If not set, use `sol_int`
            (and `sol_ext` when relevant). The default is None.

        Notes
        -----
        Provided keys must be keys of `wf_dict`. String 'all' is only exception,
        for which all 'mod' materials will be updated. This exception is handled
        with the decorator `handle_wf_keys`.

        See Also
        --------
        _set_vec_qp_in_materials_extra_args : setting material's data in qps.
        """

        # Set data in quadrature points
        self._set_vec_qp_in_materials_extra_args(*wf_keys, vec=vec)

        # Looping through all given weak forms and update their 'mod' materials
        for key in wf_keys:
            wf = self.wf_dict[key]
            wf.update_materials(tag='mod', force=True)

    @Decorators.handle_wf_keys
    def _set_vec_qp_in_materials_extra_args(
            self, *wf_keys: str, vec: Union[None, np.ndarray] = None):
        """
        Set the `vec_qp` data inside material's extra_args for terms tagged
        'mod'. The actual data is evaluated at the quadrature points of the
        term's region using the current solution vector (`sol_int` or
        `sol_ext`). The remaining step consists in updating the material, where
        the material's function is evaluated with its new `vec_qp` data
        (see method `update_nonlinear_materials` above).

        Parameters
        ----------
        wf_keys : str
            Keys of the weak forms to update.
        vec : Union[None, np.ndarray], optional
            Vector to be used to used for the update. If not set, use `sol_int`
            (and `sol_ext` when relevant). The default is None.

        Notes
        -----
        Provided keys must be keys of `wf_dict`. String 'all' is only exception,
        for which all 'mod' materials will be considered. This exception is
        handled with the decorator `handle_wf_keys`.
        As a reminder, a material associated with a pre_term tagged 'mod' must
        have the following signature
            ``def mat(ts, coors, mode=None, vec_qp=None, **kwargs)``

        """

        # Looping through all given weak forms
        for key in wf_keys:

            wf = self.wf_dict[key]

            # Get the vector to be evaluated at quadrature points
            if vec is None:
                if wf._is_exterior:
                    vec_dof = self.sol_ext  # exterior solution
                else:
                    vec_dof = self.sol_int  # interior solution
            else:
                vec_dof = vec

            # Looping through all weak form's terms
            for pre_term in wf.pre_terms:
                if pre_term.tag == 'mod':  # only update 'mod' materials
                    qps_coors = wf.get_qps_coors_in_region(pre_term.region_key)
                    vec_qp = wf.field.evaluate_at(
                        qps_coors, vec_dof[:, np.newaxis]).squeeze()
                    sfepy_material = pre_term.sfepy_material
                    extra_args = sfepy_material.extra_args
                    extra_args['vec_qp'] = vec_qp
                    sfepy_material.set_extra_args(**extra_args)

    def _constant_relaxation(self, w=None):
        r"""
        Use a constant relaxation parameter:
        $$ U_{k+1} \gets \omega U_{k+\omega} + (1-\omega) U_k $$

        Parameters
        ----------
        w : Union[None, float], optional
            Relaxation parameter to be used. The default is None, is which case
            `self.relax_param` is used.
        """
        w = self.relax_param if w is None else w
        self.sol_int = w * self.buffer_int + (1 - w) * self.old_int
        if self.sol_ext is not None:
            self.sol_ext = w * self.buffer_ext + (1 - w) * self.old_ext

    def _line_search_relaxation(self):
        """
        Use the line-search method to find an optimal relaxation parameter (by
        minimizing the 2-norm of the strong residual squared). The current
        implementation uses Scipy's `minimize_scalar` bounded method.
        """

        old_sol = self.old_int
        new_sol = self.buffer_int

        def func(w: float):
            """Scalar function to be minimized"""
            vec = w * new_sol + (1 - w) * old_sol
            res_vec = self.evaluate_residual_vector(vec)
            return np.sum(res_vec**2)

        options = {'maxiter': 10, 'xatol': 5e-2}
        optim_res = minimize_scalar(func, bounds=(0, 2), method='bounded',
                                    options=options)
        w = optim_res.x if optim_res.success else None
        self._constant_relaxation(w=w)
        self._write_linesearch_message(optim_res)

    def _dwindle_relaxation(self):
        pass

    def _write_pause_message(self, pause_iter_num):
        """Write pause message in 'solve' method."""
        pause_str = " PAUSING AT ITERATION NO {} "
        pause_str = pause_str.format(pause_iter_num).center(20, '*')
        print(pause_str)

    def _write_linesearch_message(self, optim_res: OptimizeResult):
        """Write message following line-search algorithm."""
        it = self.nonlinear_monitor.current_iter_num
        if optim_res.success:
            message = f"line-search for iteration no {it}: found w_opt=" \
                      f"{optim_res.x:.2f} in {optim_res.nit} iterations"
        else:
            message = f"line-search for iteration no {it} failed: using " \
                      f"default w={self.default_relax_param}"
        print(message)

    def _check_is_nonlinear(self):
        """
        Check if the inputted interior weak form actually corresponds to a
        nonlinear problem.
        """
        is_linear = True
        for pre_term in self.wf_int.pre_terms:
            if pre_term.tag == 'mod':
                is_linear = False
                break
        if is_linear:
            raise TypeError("Problem is actually linear. Consider using class "
                            "'LinearSolver' instead of 'NonLinearSolver'.")

    def _check_initial_guess_dict(self, initial_guess_dict: dict):
        """
        Check if the provided `initial_guess_dict` complies with the fixed
        convention.
        """

        if type(initial_guess_dict) is not dict:
            raise TypeError("'initial_guess_dict' must be a dictionary!")
        if 'int' not in initial_guess_dict.keys():
            raise KeyError("'initial_guess_dict' must have key 'int'.")
        if type(initial_guess_dict['int']) is not np.ndarray:
            raise TypeError("'initial_guess_dict['int']' must be a numpy array")
        if initial_guess_dict['int'].shape != self.sol_int.shape:
            raise ValueError(f"'initial_guess_dict['int']' has shape"
                             f"{initial_guess_dict['int'].shape}, but should be"
                             f"{self.sol_int.shape}.")

        if self.sol_ext is not None:
            if 'ext' not in initial_guess_dict.keys():
                raise KeyError("'initial_guess_dict' must have key 'ext'.")
            if type(initial_guess_dict['ext']) is not np.ndarray:
                raise TypeError(
                    "'initial_guess_dict['ext']' must be a numpy array")
            if initial_guess_dict['ext'].shape != self.sol_ext.shape:
                raise ValueError(f"'initial_guess_dict['ext']' has shape"
                                 f"{initial_guess_dict['ext'].shape}, but"
                                 f"should be {self.sol_ext.shape}.")
        else:
            initial_guess_dict['ext'] = None

    def _check_solver_monitor_link(self):
        """Check that solver and monitor have been properly linked together."""
        if self.nonlinear_monitor is None:
            raise ValueError(
                "solver's 'nonlinear_monitor' attribute has not been set!"
                "consider calling 'link_solver_to_monitor' first.")
        if self.nonlinear_monitor.nonlinear_solver is None:
            raise ValueError(
                "monitor's 'nonlinear_solver' attribute has not been set!"
                "Consider calling 'link_solver_to_monitor' first.")

        assert self.nonlinear_monitor.nonlinear_solver == self

    @staticmethod
    def _check_relax_method(relax_method: str, wf_dict: dict):
        """
        Check that relaxation step can be performed with `relax_method` and
        `wf_dict`.
        """
        if relax_method not in NonLinearSolver.implemented_relax_methods:
            raise ValueError(
                f"{relax_method} is not a valid relaxation method. Implemented"
                f"methods are: {NonLinearSolver.implemented_relax_methods}")
        if relax_method == 'line-search':
            if not ('wf_residual' in wf_dict):
                raise KeyError("'wf_residual' and 'wf_jacobian' must be keys of"
                               "'wf_dict' when using line-search algorithm!")
