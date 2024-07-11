# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:32:37 2022

Simulation engine (bounded or unbounded domains, linear or nonlinear PDEs...).
Reminder for getting matrix / rhs vector:
https://github.com/sfepy/sfepy/issues/565

05/2022 update note
-------------------
In the past, femtoscope used to re-assemble all matrices and vectors at each
iteration of the Newton method (and even at each ping-pong iteration!), which
is computationally inefficient. This problem can be circumvented by re-
assembling only the iteration-dependent terms. An important portion of the
program was edited in order to make this performance enhancement.

@author: hlevy
"""

import os

# from sfepy.solvers.auto_fallback import AutoDirect
from sfepy.solvers.ls import ScipyDirect #, ScipyIterative
from sfepy.solvers.nls import Newton
from sfepy.base.base import IndexedStruct
from sfepy.discrete import Equations, Problem, Function
from sfepy.discrete.conditions import (Conditions, EssentialBC,
                                       LinearCombinationBC)
from femtoscope.misc.util import concatenate, copy_list_of_arrays
import sfepy.discrete.fem.periodic as per
from sfepy.discrete.common.mappings import get_physical_qps # get_normals
import numpy as np
from sfepy.base.base import output

output.set_output(quiet=True) # Toggle for Sfepy console's messages


class Solver:
    """
    Class for performing FEM simulations.

    Attributes
    ----------
    eps_dic : dict
        Dictionary with keys 'eps_a' and 'eps_r' corresponding to the absolute
        and relative tolerance settings respectively.
        The default is {'eps_a' : 1e-10, 'eps_r' : 1.0}
    ls : `sfepy.solvers.ls.ScipyDirect` by default.
        Linear solver to be used. The default is ScipyDirect.
    nls : `sfepy.solvers.nls.Newton`
        Nonlinear solver to be used. The default is Newton.
    wfs_dic : dict
        Dictionary of all weakforms used in the simulation. Item with key
        'weakforms' represents the weakforms associated with the PDE to be
        solved (1 wf for bounded domains, 2 wfs for unbounded ones). Other
        weakforms are for supplementary purposes (evaluation of the strong
        residual, line-search...).
    nwf : int
        Number of weak forms provided to the solver.
    islinear : bool
        Whether or not the PDE to be solved is linear.
    is_bounded : bool
        Whether or not the domain is bounded.
    iter : int, optional
        Maximum number of newton/picard iterations (defined if islinear is
        False).
    initial_guess : list, optional
        List of initial guesses (one per weak form) used to initialize
        nonlinear solver (defined if islinear is False).The default is a scalar
        field set to zero everywhere.
    relax : float, optional
        Relaxation parameter for newton/picard method (defined if islinear is
        False). The default is 0.9.
    criteria : `Criteria` instance, optional
        Criteria instance gathering the directives for when to stop nonlinear
        iterations (defined if islinear is False). The default is to iterate
        20 times.
    has_converged : bool
        False until the Criteria instance decrees convergence has been reached
        (defined if islinear is False).
    sols : list
        List of the solution(s) (one per weak form).
    cogammas : list
        List of two `sfepy.discrete.common.region.Region` instances
        corresponding to the interior and exterior domain common frontier
        (defined if nwf = 2).
    conn : str
        Method for linking the interior domain with the exterior domain
        ['connected' , 'ping-pong']. The default is 'connected'.
    max_iter_pp : int
        Maximum number of iterations for the ping-pong method (defined if
        conn = 'ping-pong'). The default is 7.
    relax_pp :  float
        Relaxation parameter for ping-pong iterations (between 0 and 1).
        The default is 0.7.
    bounds : list
        List of the min & max field values (can be crucial to prevent
        divergence in the solution for nonlinear problems).
    oldsols : list
        List of the solution(s) at the previous iteration (one per weak form).
    buffersols : list
        Temporary storage of the solution(s).
    outfiles : list
        List of the exported .vtk files containing simulation results.
    save_all_newton : bool
        Save all Newton's iterations (for nonlinear problems only).
        The default is False.
    verbose : bool
        Display user's information. The default is False.

    """

    forbidden_nl_arg = ['update_nl', 'rho', 'rho_vac', 'Rcut', 'solver']
    """Class variable `forbidden_nl_arg` is a list of material extra-arguments
    (strings) which are not to be interpreted as non-linear terms.
    Consequently, material's data will not be updated when calling the function
    `update_material` unless the material's function has at least one extra-
    argument not contained in that list."""

    def __init__(self, wfs_dic, islinear=True, **kwargs):
        """
        Construct a Solver instance.

        Parameters
        ----------
        weakforms : list
            List of one or two WeakForm instances.
        islinear : bool, optional
            Whether or not the PDE to be solved is linear. The default is True.
        kwargs
            *Cf class documentation*

        """

        self.verbose = kwargs.get('verbose', False)
        if self.verbose >= 3:
            output.set_output(quiet=False) # Display Sfepy console's messages
        self.print_kappa =  kwargs.get('print_kappa', False)

        # Solver(s) & precision
        self.eps_dic = kwargs.get('eps_dic', {'eps_a' : 1e-15,
                                              'eps_r' : 1e-15,
                                              'i_max' : 5})
        # self.ls = kwargs.get('ls', AutoDirect({}))
        self.ls = ScipyDirect({})
        self.nls = kwargs.get('nls', Newton({'is_linear': False},
                                            lin_solver=self.ls,
                                            status=IndexedStruct()))
        # Weak Forms
        self.wfs_dic = wfs_dic
        weakforms = wfs_dic['weakforms']
        self.weakforms = weakforms if isinstance(weakforms, list) \
                                   else [weakforms]
        self.nwf = len(self.weakforms)
        self.is_bounded = kwargs.get('is_bounded')

        # TODO: pre-allocate solutions' arrays
        self.oldsols = [None] * self.nwf
        self.buffersols = [None] * self.nwf
        self.sols = [None] * self.nwf
        self.bounds = kwargs.get('bounds', [-1e15, 1e15])

        # Non-linear problems
        self.islinear = islinear
        if not islinear:
            wf_res = self.wfs_dic.get('wf_res')
            wf_G = self.wfs_dic.get('wf_G')
            self.iter = 0
            initial_guess = kwargs.get('initial_guess', [])
            if len(initial_guess) == 0:
                for wf in self.weakforms:
                    initial_guess.append(np.zeros(wf.field.n_nod))
            elif (self.nwf == 1) and (not isinstance(initial_guess, list)):
                initial_guess = [initial_guess]
            self.initial_guess = initial_guess
            self.oldsols = copy_list_of_arrays(self.initial_guess)
            self.relax = kwargs.get('relax', 0.9)
            self.criteria = kwargs.get('criteria', StopCriteria(20))
            self.has_converged = False
            self.save_all_newton = kwargs.get('save_all_newton', False)

            # Set materials' extra-arguements
            for k, (wf,oldsol) in enumerate(zip(self.weakforms,self.oldsols)):
                oldsol_qps = evaluate_at_qps(wf, oldsol)
                oldsol_qps[oldsol_qps < self.bounds[0]] = self.bounds[0]
                oldsol_qps[oldsol_qps < self.bounds[1]] = self.bounds[1]
                set_mat_extra_arg(wf, oldsol_qps, update=False)
                if k==0 and wf_res is not None:
                    set_mat_extra_arg(wf_res, oldsol_qps, update=False)

            # line-search related weak form
            self.line_search_bool = (wf_G is not None) \
                and kwargs.get('line_search_bool', False)

        # Unbounded domains
        self.sols = [None] * self.nwf
        if self.nwf >= 2:
            self.cogammas = kwargs.get('cogammas', None)
            self.conn = kwargs.get('conn', 'connected') # or 'ping-pong'
            if self.conn == 'ping-pong':
                self.max_iter_pp = kwargs.get('max_iter_pp', 7)
                self.relax_pp = kwargs.get('relax_pp', 0.7)
            elif self.conn == 'connected':
                pass
            else:
                raise Exception("Connection keyword not recognized")
        else:
            self.conn = None

        # Additional initializations
        self.init_mtxvec_dic()
        self.initialized = False
        self.outfiles = None

        if self.verbose:
            print(self.__str__())

    def init_mtxvec_dic(self):
        """Initialize the dictionary of all matrices & vectors. The keys
        depend on the nature of the simulation (linear or nonlinear, connection
        technique, etc.)"""
        mtxvec_dic = {}
        wf1 = self.weakforms[0]
        nwf = self.nwf
        if nwf == 1 or (nwf == 2 and self.conn == 'connected'):
            mtxvec_dic['mtx'] = None
            mtxvec_dic['rhs'] = None
            mtxvec_dic['mtx_cst'] = None
            mtxvec_dic['rhs_cst'] = None
            if wf1.eqn_mod is not None:
                mtxvec_dic['mtx_mod'] = None
                mtxvec_dic['rhs_mod'] = None
        elif nwf == 2 and self.conn == 'ping-pong':
            mtxvec_dic['mtx_pp1'] = None
            mtxvec_dic['mtx_pp2'] = None
            mtxvec_dic['rhs_pp1'] = None
            mtxvec_dic['rhs_pp2'] = None
            mtxvec_dic['mtx_cst_pp1'] = None
            mtxvec_dic['mtx_cst_pp2'] = None
            mtxvec_dic['rhs_cst_pp1'] = None
            mtxvec_dic['rhs_cst_pp2'] = None
            mtxvec_dic['mtx_mod_pp2'] = None
            mtxvec_dic['rhs_mod_pp2'] = None
            if wf1.eqn_mod is not None:
                mtxvec_dic['mtx_mod_pp1'] = None
                mtxvec_dic['rhs_mod_pp1'] = None
        else:
            raise Exception()

        wf_res = self.wfs_dic.get('wf_res')
        if wf_res is not None:
            mtxvec_dic['mtx_res'] = None
            mtxvec_dic['rhs_res'] = None
            mtxvec_dic['mtx_cst_res'] = None
            mtxvec_dic['rhs_cst_res'] = None
            if wf_res.eqn_mod is not None:
                mtxvec_dic['mtx_mod_res'] = None
                mtxvec_dic['rhs_mod_res'] = None

        self.mtxvec_dic = mtxvec_dic

    def init_problems(self):
        """Initialize problem instances. The keys of the `pbs_dic` depend on
        the nature of the simulation (linear or nonlinear, connection
        technique, etc.)"""

        self.pbs_dic = {}
        wf1 = self.weakforms[0]

        if self.nwf == 1:
            ebcs = Conditions(wf1.ebcs)
            epbcs = Conditions(wf1.epbcs)
            eqs_cst = Equations([wf1.eqn_cst])
            pb_cst = Problem('pb_cst', equations=eqs_cst, active_only=True)
            pb_cst.set_bcs(ebcs=ebcs, epbcs=epbcs)
            pb_cst.set_solver(self.nls)
            self.pbs_dic['pb_cst'] = pb_cst
            if wf1.eqn_mod is not None:
                eqs_mod = Equations([wf1.eqn_mod])
                pb_mod = Problem('pb_mod', equations=eqs_mod, active_only=True)
                pb_mod.set_bcs(ebcs=ebcs, epbcs=epbcs)
                pb_mod.set_solver(self.nls)
                self.pbs_dic['pb_mod'] = pb_mod

        elif self.nwf == 2:
            wf2 = self.weakforms[1]

            if self.conn == 'connected':
                def zero_shift(ts, coors, region):
                    return np.zeros_like(coors[:, 0], dtype=np.float64)
                match_coors = Function('match_coors', per.match_coors)
                shift_fun = Function('shift_fun', zero_shift)
                lcbcs = Conditions(
                    [LinearCombinationBC('lc12', self.cogammas,
                    {'%s.all' %wf1.unknown_name : '%s.all' %wf2.unknown_name},
                    match_coors, 'shifted_periodic', arguments=(shift_fun,))])
                ebcs = Conditions(concatenate([wf1.ebcs, wf2.ebcs]))
                epbcs = Conditions(concatenate([wf1.epbcs, wf2.epbcs]))
                eqs_cst = Equations([wf1.eqn_cst, wf2.eqn_cst])
                pb_cst = Problem('pb_cst', equations=eqs_cst, active_only=True)
                pb_cst.set_bcs(ebcs=ebcs, epbcs=epbcs, lcbcs=lcbcs)
                pb_cst.set_solver(self.nls)
                self.pbs_dic['pb_cst'] = pb_cst
                if wf2.eqn_mod is not None:
                    eqs_mod = Equations([wf1.eqn_mod, wf2.eqn_mod])
                    pb_mod = Problem('pb_mod', equations=eqs_mod,
                                     active_only=True)
                    pb_mod.set_bcs(ebcs=ebcs, epbcs=epbcs, lcbcs=lcbcs)
                    pb_mod.set_solver(self.nls)
                    self.pbs_dic['pb_mod'] = pb_mod

            elif self.conn == 'ping-pong':
                pb_cst_pp1 = Problem('pb_cst_pp1',
                                     equations=Equations([wf1.eqn_cst]),
                                     active_only=True)
                pb_cst_pp1.set_bcs(ebcs=Conditions(wf1.ebcs),
                                   epbcs=Conditions(wf1.epbcs))
                pb_cst_pp1.set_solver(self.nls)
                self.pbs_dic['pb_cst_pp1'] = pb_cst_pp1

                pb_cst_pp2 = Problem('pb_cst_pp2',
                                     equations=Equations([wf2.eqn_cst]),
                                     active_only=True)
                pb_cst_pp2.set_bcs(ebcs=Conditions(wf2.ebcs),
                                   epbcs=Conditions(wf2.epbcs))
                pb_cst_pp2.set_solver(self.nls)
                self.pbs_dic['pb_cst_pp2'] = pb_cst_pp2

                pb_mod_pp2 = Problem('pb_mod_pp2',
                                    equations=Equations([wf2.eqn_mod]),
                                    active_only=True)
                pb_mod_pp2.set_bcs(ebcs=Conditions(wf2.ebcs),
                                   epbcs=Conditions(wf2.epbcs))
                pb_mod_pp2.set_solver(self.nls)
                self.pbs_dic['pb_mod_pp2'] = pb_mod_pp2

                if wf1.eqn_mod is not None:
                    pb_mod_pp1 = Problem('pb_mod_pp1',
                                        equations=Equations([wf1.eqn_mod]),
                                        active_only=True)
                    pb_mod_pp1.set_bcs(ebcs=Conditions(wf1.ebcs),
                                       epbcs=Conditions(wf1.epbcs))
                    pb_mod_pp1.set_solver(self.nls)
                    self.pbs_dic['pb_mod_pp1'] = pb_mod_pp1

            else:
                raise Exception("Connection keyword not recognized")

        wf_res = self.wfs_dic.get('wf_res')
        if wf_res is not None:
            pb_cst_res = Problem('pb_cst_res',
                                 equations=Equations([wf_res.eqn_cst]),
                                 active_only=False)
            pb_cst_res.set_bcs(ebcs=Conditions(wf_res.ebcs),
                               epbcs=Conditions(wf_res.epbcs))
            pb_cst_res.set_solver(self.nls)
            self.pbs_dic['pb_cst_res'] = pb_cst_res

            pb_mod_res = Problem('pb_mod_res',
                                 equations=Equations([wf_res.eqn_mod]),
                                 active_only=False)
            pb_mod_res.set_bcs(ebcs=Conditions(wf_res.ebcs),
                               epbcs=Conditions(wf_res.epbcs))
            pb_mod_res.set_solver(self.nls)
            self.pbs_dic['pb_mod_res'] = pb_mod_res

        wf_G = self.wfs_dic.get('wf_G')
        if wf_G is not None:
            pb_cst_G = Problem('pb_cst_G',
                                equations=Equations([wf_G.eqn_cst]),
                                active_only=False)
            pb_cst_G.set_bcs(ebcs=Conditions(wf_G.ebcs),
                              epbcs=Conditions(wf_G.epbcs))
            pb_cst_G.set_solver(self.nls)
            self.pbs_dic['pb_cst_G'] = pb_cst_G

            pb_mod_G = Problem('pb_mod_G',
                                equations=Equations([wf_G.eqn_mod]),
                                active_only=False)
            pb_mod_G.set_bcs(ebcs=Conditions(wf_G.ebcs),
                              epbcs=Conditions(wf_G.epbcs))
            pb_mod_G.set_solver(self.nls)
            self.pbs_dic['pb_mod_G'] = pb_mod_G

    def init_mtxvec(self):
        """Assemble all matrices and vectors. These objects are then accessible
        through the `self.mtxvec_dic` dictionary."""
        pb_list = list(self.pbs_dic.values())
        for pb in list(pb_list):
            if pb.name=='pb_cst_G' or pb.name=='pb_mod_G':
                pb_list.remove(pb)
        assemble_mtxvec(self, *pb_list)

    def update_mtxvec(self):
        """Re-assemble matrices and vectors associated with terms featured in
        `eqn_mod` equations from all weak forms."""
        pb_list = []
        for pb in list(self.pbs_dic.values()):
            pb_name = pb.name
            if pb_name.split('_')[1]=='mod' and \
                (pb_name!='pb_cst_G' or pb_name!='pb_mod_G'):
                pb_list.append(pb)
        assemble_mtxvec(self, *pb_list)

    def solve(self, **kwargs):
        """Launch the FEM computation!"""
        if self.islinear:
            lsolve(self)
        else:
            nlsolve(self, **kwargs)

    def save(self, grad=False, **kwargs):
        """
        Save the results to vtk. This routine is a wrapper for
        `femtoscope.inout.vtkfactory.createStructuredVTK`, which itself rely
        on meshio.
        Warning: the connectivity is reconstructed from scratch

        Parameters
        ----------
        grad : bool, optional
            Save the gradient of the scalar field. The default is False.
        name : str, optional
            Name of the output file (prefix).
            
        Returns
        -------
        out : list
            Result file names.

        """

        from femtoscope.inout.vtkfactory import createStructuredVTK
        from femtoscope.misc.util import merge_dicts
        from pathlib import Path
        val_renorm = kwargs.get('val_renorm', 1.0)
        grad_renorm = kwargs.get('grad_renorm', 1.0)
        name = Path((kwargs.get('name', '')))
        if name.name:
            name = str(name.with_suffix(''))
        else:
            name = ''
        dim = self.weakforms[0].dimension
        out = []
        for kk, (wf, sol) in enumerate(zip(self.weakforms, self.sols)):
            vars_dic = {}
            # Retrieve coordinates and connectivity
            pos = wf.field.get_coor() # also works with pos = field.coors
            X = pos[:, 0]
            Y = pos[:, 1]
            Z = pos[:, 2] if pos.shape[1]==3 else None
            if dim == 2:
                from matplotlib.tri import Triangulation
                cells = Triangulation(X, Y).triangles
            elif dim == 3:
                from scipy.spatial import Delaunay
                cells = Delaunay(pos).vertices
            vars_dic[wf.unknown_name] = val_renorm * sol
            if grad:
                gradsol = grad_renorm * wf.field.evaluate_at(
                    pos, sol[:, np.newaxis], mode='grad').squeeze()
                normgradsol = np.linalg.norm(gradsol, axis=1)
                vars_dic['norm_grad_'+wf.unknown_name] = normgradsol
                vars_dic['gradX_'+wf.unknown_name] = gradsol[:, 0]
                vars_dic['gradY_'+wf.unknown_name] = gradsol[:, 1]
                if dim == 3:
                    vars_dic['gradZ_'+wf.unknown_name] = gradsol[:, 2]
            if (kk==0) and (not self.islinear):
                vars_dic['residual'] = self.criteria.res_vec.data
                merge_dicts(vars_dic, self.criteria.res_vec_parts)
            out.append(createStructuredVTK(
                X, Y, Z, vars_dic, cells, name=name+'_'+wf.name))
        self.outfiles = out
        return out
    
    def plot_residual(self, log=False):
        """
        Plot the residual vector.
        Since 22/05/23, it is no longer necessary to remove ebc-DOFs from the
        coordinate array. The residual vector is not cropped anymore but
        contains NaN values instead.

        Parameters
        ----------
        log : bool, optional
            If True, plot the log of the absolute value of the residual vector.
            The default is False.

        Returns
        -------
        ax : Axes
            Figure axe.

        """
        from matplotlib import pyplot as plt
        dim = self.weakforms[0].dimension
        res_vec = np.copy(self.criteria.res_vec)
        coors = self.weakforms[0].field.coors
        if log:
            res_vec = np.log(abs(res_vec))
        
        plt.figure(figsize=(6,5))
        if dim == 1:
            plt.scatter(coors.squeeze(), res_vec, s=1)
            plt.xlabel(r"$r$", fontsize=15)
            plt.ylabel("Residual", fontsize=15)
        elif dim == 2:
            plt.scatter(coors[:, 0], coors[:, 1], c=res_vec, s=1)
            plt.colorbar(label="Residual")
        else:
            raise ValueError("Cannot plot residual for dimension %d" %dim)
        ax = plt.gca()
        plt.show()
        return ax

    def display(self, grad=False, **kwargs):
        """
        Display FEM results. The `solve` method must have been called on the
        Solver instance beforehand.

        Parameters
        ----------
        grad : bool, optional
            Save the gradient of the scalar field. The default is False.
        kwargs :
            Potential extra arguments (to be added in the future).

        Returns
        -------
        None.

        """
        from femtoscope.display.fea_plot import plot_from_vtk
        if self.outfiles is None:
            self.save(grad=grad)
            delete = True
        for out in self.outfiles:
            plot_from_vtk(out, **kwargs)
            if delete:
                os.remove(out)

    def __str__(self):
        from femtoscope.misc.util import remove_empty_lines
        string = f"""
        Solver:
            # weak-form(s): {self.nwf}
        """

        if self.is_bounded:
            string += f"""
            bounded domain
            """
        else:
            string += f"""
            unbounded domain
            """

        if self.conn == 'ping-pong':
            string += f"""
            ping-pong max iter: {self.max_iter_pp}
            """

        if self.islinear:
            string += f"""
            is linear?: YES
            """
        else:
            string += f"""
            is linear?: NO
            relax parameter: {self.relax}
            """
            string += self.criteria.__str__()

        return remove_empty_lines(string)


class StopCriteria:
    """
    Class defining stopping conditions for nonlinear problems.

    Attributes
    ----------
    min_iter : int
        Minimum number of Newton iterations. The default is 8.
    max_iter : int
        Maximum number of Newton iterations.
    sol_delta_tol : float
        Threshold on the minimum relative distance (in 2-norm) between two
        successive iterations.
    active_dic : dict
        Dictionary for knowing which criteria are active and which are not.
    verbose : bool
        Display user's information. The default is False.
    stop : bool
        False until some stopping criterion is met.
    res_vec : ndarray
        Current strong residual vector.
    res_vec_parts : dic
        Dictionary with keys ('mtx', 'rhs_cst', 'rhs_mod') and values each term
        of the strong residual vector.
    res : float
        Current value of the strong residual (L2-norm).
    sol_delta : float
        Current value of the relative distance between two successive
        iterations.
    history : list of dicts
        List of dictionaries containing all the above parameters for monitoring
        convergence.
    last_relax_update : List of int
        Last iteration number for which the relaxation parameter of the Newton
        method was decreased.

    """

    def __init__(self, max_iter, min_iter=8, **kwargs):
        self.max_iter = max_iter if max_iter is not None else 50
        self.min_iter = min_iter if min_iter is not None else 8
        self.res_delta_tol = None
        self.sol_delta_tol = None
        self.stop = False
        self.res_vec = None
        self.res_vec_parts = None
        self.res = None
        self.sol_delta = None
        self.history = []
        self.last_relax_update = [0]

        # Dictionary of active stopping criteria
        active_dic = {}
        active_dic['res_delta_tol'] = ('res_delta_tol' in kwargs)
        if active_dic['res_delta_tol']:
            self.res_delta_tol = kwargs.get('res_delta_tol')
        active_dic['sol_delta_tol'] = ('sol_delta_tol' in kwargs)
        if active_dic['sol_delta_tol']:
            self.sol_delta_tol = kwargs.get('sol_delta_tol')
        self.active_dic = active_dic

        self.verbose = kwargs.get('verbose', True)

    def __str__(self):
        return f"""
        Stop Criteria:
            max iter: {self.max_iter}
            relative delta_sol tolerance: {self.sol_delta_tol}
            relative delta_res tolerance: {self.res_delta_tol}
        """

    def evaluate(self, solver):
        """
        Evaluate all the criteria set active. Some additional indicators are
        computed but are intended for information purposes only.

        Parameters
        ----------
        solver : `Solver` instance
            The solver instance.

        Returns
        -------
        bool
            True if one of the stopping criterion has been met, False otherwise.

        """

        # Set to None
        sol_delta = None
        res_delta = None
        res_L2 = None
        res_mean = None
        res_inf = None

        # strong residual evaluation
        res_vec, res_vec_parts = strong_residual(solver, solver.sols[0])
        res_L2 = np.sqrt((res_vec**2).sum())
        res_mean = np.mean(abs(res_vec))
        res_inf = abs(res_vec).max()
        if self.res_vec is not None:
            res_delta = np.sqrt(((res_vec-self.res_vec)**2).sum()) / self.res
        else:
            res_delta = np.nan

        # distance between two successive iterations
        if solver.iter > 0:
            delta = 0.0
            denominator = 0.0
            for oldsol, newsol in zip(solver.oldsols, solver.sols):
                if (oldsol is not None) and (newsol is not None):
                    delta += np.linalg.norm(newsol-oldsol)
                    denominator += np.linalg.norm(oldsol)
            sol_delta = delta / denominator
        else:
            sol_delta = np.nan

        self.sol_delta = sol_delta
        self.res = res_L2
        self.res_vec = res_vec
        self.res_vec_parts = res_vec_parts

        if solver.verbose:
            print("{0: <17} {1:.3E}".format("Sol rel delta", sol_delta))
            print("{0: <17} {1:.3E}".format("Residual L2", res_L2))
            print("{0: <17} {1:.3E}".format("Residual mean", res_mean))
            print("{0: <17} {1:.3E}".format("Residual inf", res_inf))
            print("{0: <17} {1:.3E}\n".format("Residual rel delta", res_delta))

        self.fill_history(solver.iter, sol_delta, res_delta, res_L2, res_mean,
                          res_inf)

        # Active criteria
        if self.active_dic['sol_delta_tol'] and solver.iter > self.min_iter:
            if sol_delta < self.sol_delta_tol:
                self.stop = True
                solver.has_converged = True
                if self.verbose:
                    print("STOP: Delta in successive iterations is small")
                return True

        if self.active_dic['res_delta_tol'] and solver.iter > self.min_iter:
            if res_delta < self.res_delta_tol:
                self.stop = True
                solver.has_converged = True
                if self.verbose:
                    print("STOP: Residual variation is small")
                return True

        if solver.iter >= self.max_iter:
            self.stop = True
            if self.verbose:
                print("STOP: Maximum number of iteration has been reached")
            return True

        return False

    def fill_history(self, niter, sol_delta, res_delta, res_L2, res_mean,
                     res_inf):
        iter_dic = {
            'iter': niter,
            'sol delta' : sol_delta,
            'res delta' : res_delta,
            'res L2' : res_L2,
            'res mean' : res_mean,
            'res inf' : res_inf
            }
        if self.history == []:
            iter_dic['res_k / res_0 (%)'] = 100
        else:
            iter_dic['res_k / res_0 (%)'] = res_L2/self.history[0]['res L2']*100
        self.history.append(iter_dic)

    def adjust_relax(self, solver, history_size=2, factor=0.8):
        """
        Based on the analyis of the current convergence-history of the on-going
        nonlinear equation solving, the relaxation parameter manually decreased
        to achieve convergence of Newton's iterations. The two following
        conditions must be satisfied simultaneously:
            - the relative delta has stopped decreasing
            - the residual in 2-norm has stopped decreasing

        Parameters
        ----------
        solver : `Solver` instance
            The solver instance.
        history_size : int, optional
            Number of past iteration used to check the two stalling conditions.
            The default is 3.
        factor : float, optional
            Factor by which the relaxation parameter is multiplied (should be
            strictly smaller than 1). The default value is 0.8.
        """
        
        assert factor < 1 and factor > 0
        if solver.iter > history_size + self.last_relax_update[-1] - 1:
            res_L2 = np.array([self.history[-k]['res L2'] for k in range(history_size, 0, -1)])
            sol_delta = np.array([self.history[-k]['sol delta'] for k in range(history_size, 0, -1)])
            cond1 = (np.diff(sol_delta)/sol_delta.max() > -1e-2).any()
            cond2 = (np.diff(res_L2)/np.mean(res_L2) > -1e-2).any()
            if cond1 and cond2:
                if len(self.last_relax_update) > 10:
                    factor *= 1e-2
                elif len(self.last_relax_update) > 7:
                    factor *= 0.25
                elif len(self.last_relax_update) > 4:
                    factor *= 0.5
                solver.relax *= factor
                self.last_relax_update.append(solver.iter)
                if solver.verbose:
                    print("! Decreasing relax. param. to {:.3f} !\n".format(
                        solver.relax))
        

def lsolve(solver, buffer=False):
    """
    Linear solve function.

    Parameters
    ----------
    solver : `Solver` instance
        The solver instance.
    buffer : bool, optional
        If True, the computed solution is kept in *solver.buffersols*. This is
        only relevant to nonlinear problems. The way the buffer is unload into
        *solver.sols* depends on the nonlinear method implementation.
        The default is False.

    """

    # Retrieve tolerance settings
    eps_a = solver.eps_dic['eps_a']
    eps_r = solver.eps_dic['eps_r']
    i_max = solver.eps_dic['i_max']
    mtxvec_dic = solver.mtxvec_dic

    # Setup problem instances & assemble matrices / vectors
    if not solver.initialized:
        solver.init_problems()
        solver.init_mtxvec()
        solver.initialized = True

    nwf = solver.nwf
    status = IndexedStruct()
    update_global_mtxvec(solver)
    ls = solver.nls.lin_solver

    # Single domain
    if nwf == 1:
        pb = solver.pbs_dic['pb_cst']
        mtx = mtxvec_dic['mtx']
        rhs = mtxvec_dic['rhs']
        vec_x = ls(rhs, x0=None, eps_a=eps_a, eps_r=eps_r, mtx=mtx,
                   status=status)
        sol = reconstruct_full_sol(pb, vec_x)
        solver.buffersols[0] = sol

    # Double domain
    elif nwf == 2:
        wf1 = solver.weakforms[0]
        wf2 = solver.weakforms[1]

        # Virtual connection of DOFs at the boundary
        if solver.conn == 'connected':
            pb = solver.pbs_dic['pb_cst']
            mtx = mtxvec_dic['mtx']
            rhs = mtxvec_dic['rhs']
            vec_x = ls(rhs, x0=None, eps_a=eps_a, eps_r=eps_r, mtx=mtx,
                       i_max=i_max, status=status)
            sol = reconstruct_full_sol(pb, vec_x)
            solver.buffersols[0] = sol[:wf1.field.n_nod]
            solver.buffersols[1] = sol[wf1.field.n_nod:]

        # DtN / NtD iterations ('ping-pong')
        elif solver.conn == 'ping-pong':
            theta = solver.relax_pp
            convergence_flag = False
            old_pp1 = np.empty(solver.weakforms[0].field.n_nod)
            old_pp2 = np.empty(solver.weakforms[1].field.n_nod)
            pb_cst_pp1 = solver.pbs_dic['pb_cst_pp1']
            pb_cst_pp2 = solver.pbs_dic['pb_cst_pp2']
            
            for k in range(solver.max_iter_pp):

                # Dirichlet BC on interior domain
                mtx_pp1 = mtxvec_dic['mtx_pp1']
                rhs_pp1 = mtxvec_dic['rhs_pp1']
                vec_x_pp1 = ls(rhs_pp1, x0=None, eps_a=eps_a, eps_r=eps_r,
                               mtx=mtx_pp1, status=status)
                sol_pp1 = reconstruct_full_sol(pb_cst_pp1, vec_x_pp1)
                solver.sols[0] = sol_pp1

                # Neumann BC on inversed exterior domain
                update_neumann(wf2, solver.cogammas[1])
                assemble_mtxvec(solver, solver.pbs_dic['pb_mod_pp2'],
                                assemble_mtx=False) # do not rebuild matrices!

                mtx_mod_pp2 = mtxvec_dic.get('mtx_mod_pp2')
                if mtx_mod_pp2 is not None:
                    mtx_pp2 = mtxvec_dic['mtx_cst_pp2'] + mtx_mod_pp2
                else:
                    mtx_pp2 = mtxvec_dic['mtx_cst_pp2']

                rhs_mod_pp2 = mtxvec_dic.get('rhs_mod_pp2')
                if rhs_mod_pp2 is not None:
                    rhs_pp2 = mtxvec_dic['rhs_cst_pp2'] + rhs_mod_pp2
                else:
                    rhs_pp2 = mtxvec_dic['rhs_cst_pp2']

                vec_x_pp2 = ls(rhs_pp2, x0=None, eps_a=eps_a, eps_r=eps_r,
                               mtx=mtx_pp2, status=status)
                sol_pp2 = reconstruct_full_sol(pb_cst_pp2, vec_x_pp2)

                # Reset Dirichlet BC on the interior domain using the solution
                # on the inversed exterior domain that we just calculated...
                if k >= 1:
                    sol_relax = theta*sol_pp2 + (1-theta)*old_pp2
                else:
                    sol_relax = sol_pp2
                reset_dirichlet(sol_relax, wf1, wf2, solver.cogammas)
                pb_cst_pp1.set_bcs(ebcs=Conditions(wf1.ebcs),
                                   epbcs=Conditions(wf1.epbcs))
                pb_cst_pp1.equations.variables.apply_ebc()
                pb_mod_pp1 = solver.pbs_dic.get('pb_mod_pp1')
                if pb_mod_pp1 is not None:
                    pb_mod_pp1.set_bcs(ebcs=Conditions(wf1.ebcs),
                                       epbcs=Conditions(wf1.epbcs))
                    pb_mod_pp1.equations.variables.apply_ebc()

                # ... which involves re-assembling rhs vectors
                assemble_mtxvec(solver, pb_cst_pp1, assemble_mtx=False)
                assemble_mtxvec(solver, pb_mod_pp1)
                update_global_mtxvec(solver)
                
                # check if convergence has been reached
                if k >= 1:
                    norm_delta1 = np.linalg.norm(sol_pp1-old_pp1)
                    norm_delta2 = np.linalg.norm(sol_pp2-old_pp2)
                    norm_old1 = np.linalg.norm(old_pp1)
                    norm_old2 = np.linalg.norm(old_pp2)
                    rel_delta = norm_delta1/norm_old1 + norm_delta2/norm_old2
                    convergence_flag = rel_delta < 1e-8
                
                # update auxiliary array
                old_pp1 = sol_pp1.copy()
                old_pp2 = sol_pp2.copy()
                
                if convergence_flag:
                    if solver.verbose>1 or (solver.verbose and solver.islinear):
                        print("ping-pong convergence in {} iter".format(k+1))
                    break

            solver.buffersols[0] = sol_pp1
            solver.buffersols[1] = sol_pp2

        else:
            raise Exception("%s is not implemented" %solver.conn)

    else:
        raise Exception("femtoscope cannot deal with more than 2 domains")

    if not buffer:
        solver.sols = [x for x in solver.buffersols]
        
    # compute the condition number of the stifness matrix
    if solver.print_kappa:
        from scipy.sparse.linalg import eigsh
        if solver.conn == 'ping-pong':
            mtx = mtxvec_dic['mtx_pp1']
        max_eig = eigsh(mtx, return_eigenvectors=False, which='LM', k=1)[0]
        sigma = 1e-8
        min_eig = eigsh(mtx, return_eigenvectors=False, sigma=sigma).min()
        counter = 0
        while min_eig < sigma and counter < 10:
            sigma /= 100
            min_eig = eigsh(mtx, return_eigenvectors=False, sigma=sigma).min()
            counter += 1
        if counter == 10:
            import warnings
            warnings.warn("Could not compute the condition number!")
        else:
            kappa = abs(max_eig/min_eig)
            print("Matrix condition number = {:.2e}".format(kappa))


def nlsolve(solver, **kwargs):
    """
    Nonlinear solve function based on Newton iterations. Each iteration is
    associated with a linear problem that is solved by calling the `solve()`
    function.

    Parameters
    ----------
    solver : `Solver` instance
        The solver instance.
        
    Other Parameters
    ----------------
    save_all_newton : bool, optional
        If true, save all Newton's iterations (solution, residual...).
        The default is False.

    Returns
    -------
    None.

    """

    import pandas as pd
    save_all_newton = kwargs.get('save_all_newton', False)
    solver.save_all_newton = save_all_newton
    if save_all_newton:
        from pathlib import Path
        from femtoscope import RESULT_DIR
        from femtoscope.misc.util import mkdir_p, date_string
        dirname = "newton-it_" + date_string()
        dirpath = str(Path(RESULT_DIR/dirname))
        mkdir_p(dirpath)
    stop_criteria = solver.criteria
    sol_min = solver.bounds[0]
    sol_max = solver.bounds[1]
    
    if solver.verbose:
        print(' Iteration no {} '.format(0).center(40,'*'))

    for _ in range(stop_criteria.max_iter+1):
        lsolve(solver, buffer=True)
        
        # fill history for 0-th iteration
        if solver.iter == 0:
            for k in range(len(solver.sols)):
                solver.sols[k] = np.copy(solver.oldsols[k])
            stop_criteria.evaluate(solver)
            if save_all_newton:
                solver.save(name=dirname+'/iter-{:03d}'.format(0))

        solver.iter += 1                
        if solver.verbose:
            print(' Iteration no {} '.format(solver.iter).center(40,'*'))

        # Crop the (buffered) solution to fit interval requirement
        for k, sol in enumerate(solver.buffersols):
            solver.buffersols[k][np.where(sol<sol_min)[0]] = sol_min
            solver.buffersols[k][np.where(sol>sol_max)[0]] = sol_max

        w = solver.relax
        for k in range(solver.nwf):
            solver.sols[k] = w*solver.buffersols[k] + (1-w)*solver.oldsols[k]

        # Crop the (new iteration) solution to fit interval requirement
        for k, sol in enumerate(solver.sols):
            solver.sols[k][np.where(sol<sol_min)[0]] = sol_min
            solver.sols[k][np.where(sol>sol_max)[0]] = sol_max

        # Update material's data (linearized nonlinear terms)
        wf_res = solver.wfs_dic.get('wf_res')
        for k in range(len(solver.sols)):
            sol_qps = evaluate_at_qps(solver.weakforms[k], solver.sols[k])
            sol_qps[np.where(sol_qps<sol_min)[0]] = sol_min
            sol_qps[np.where(sol_qps>sol_max)[0]] = sol_max
            set_mat_extra_arg(solver.weakforms[k], sol_qps, update=True)
            if k==0 and wf_res is not None:
                set_mat_extra_arg(wf_res, sol_qps, update=True)

        # Re-assemble matrices & vectors and update the dictionary
        solver.update_mtxvec()
        update_global_mtxvec(solver)
        
        # if solver.weakforms[0].dimension == 2:
        #     hook_writer(solver.weakforms[0].field.coors, solver.sols[0],
        #                 solver.criteria.res_vec, solver.iter)

        # Stop criteria: do we do one more iteration?
        if stop_criteria.evaluate(solver): # NO
            stop_criteria.history = pd.DataFrame(stop_criteria.history)
            break
        else: # YES
            for k in range(len(solver.sols)):
                solver.oldsols[k] = np.copy(solver.sols[k])
                
        # Stop criteria: res stagnation --> decrease the relaxation parameter
        stop_criteria.adjust_relax(solver)

        # Save current Newton iteration
        if save_all_newton:
            solver.save(name=dirname+'/iter-{:03d}'.format(solver.iter))
            
        # to_pickle.append(stop_criteria.strong_res_vec)

    # Save last Newton iteration
    if save_all_newton:
        solver.save(name=dirname+'/iter-{:03d}'.format(solver.iter))
        
    if solver.verbose:
        if solver.has_converged:
            print("CONVERGENCE in {} iterations\n".format(solver.iter))
        else:
            print("NO CONVERGENCE after {} iterations\n".format(solver.iter))
        print(" Recap of all Iterations ".center(88, '-'))
        print(solver.criteria.history.to_string(index=False))

def hook_writer(coors, sol, res, no_iter):
    import pickle
    data_dic = {}
    x1 = 1.059
    x2 = 1.111
    x3 = 1.314
    x4 = 4.645
    x5 = 6.617
    XX = [x1, x2, x3, x4, x5]
    XX_str = ['x1', 'x2', 'x3', 'x4', 'x5']
    for xx, xx_str in zip(XX, XX_str):
        ind_x = np.where(coors[:, 0] == xx)[0]
        theta_x = coors[ind_x, 1]
        sol_x = sol[ind_x]
        ind_sort = theta_x.argsort()
        theta_x = theta_x[ind_sort]
        sol_x = sol_x[ind_sort]
        
        if no_iter > 1:
            res_x = res[ind_x]
            res_x = res_x[ind_sort]
            data = np.concatenate(
                (theta_x[:, np.newaxis], sol_x[:, np.newaxis],
                 res_x[:, np.newaxis]), axis=1)
        else:
            data = np.concatenate((theta_x[:, np.newaxis],
                                   sol_x[:, np.newaxis]), axis=1)
        data_dic[xx_str] = data
    filename = 'iter\iter{}.pkl'.format(no_iter)
    with open(filename, 'wb') as handle:
        pickle.dump(data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


def line_search(solver, delta):
    r"""
    WARNING -- EXPERIMENTAL!!
    Find the real number w such that $u_{k+1} = w u^* + (1-w) u_{k}$ minimizes
    the L² norm of the strong residual.

    Parameters
    ----------
    solver : `Solver` instance
        The solver instance.
    delta : ndarray
        Local direction of descent $\delta u = u^* - u_k$.

    Returns
    -------
    float
        The real number w resulting from the line-search algorithm.

    """

    def fun(solver, delta, w):
        sol_w = w*solver.buffersols[0] + (1-w)*solver.oldsols[0]
        str_res = strong_residual(solver, sol_w)
        G = vec_G(solver, sol_w, delta)
        return np.dot(str_res, G)

    a = 0
    b = 1
    fa = fun(solver, delta, a)
    fb = fun(solver, delta, b)
    if fa*fb > 0:
        c = (a*fb - b*fa) / (fb - fa)
    else:
        for _ in range(5):
            c = (a*fb - b*fa) / (fb - fa)
            if abs(a-c)<1e-3 or abs(b-c)<1e-3:
                break
            fc = fun(solver, delta, c)
            if fc*fa < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
    if solver.verbose:
        print("Line search result w = {}".format(c))
    return c

def set_mat_extra_arg(wf, *args, keys=None, update=False):
    """
    Utilitary function for setting material extra arg.

    Parameters
    ----------
    wf : `WeakForm` instance
        Weak form containing the materials of interest.
    arg : *any type*
        Value of the argument to be set.
    update : bool, optional
        If True, the material data will be re-computed. Typically, it has to be
        set to True when called after Sfepy's Problem instanciation.
        The default is False.

    Returns
    -------
    None.

    """

    if len(args) > 1:
        assert keys is not None and len(keys)==len(args)
        args_dic = dict(zip(keys, args))
    else:
        arg = args[0]

    for mat, term in zip(wf.eqn.collect_materials(), wf.terms):
        modified = False
        if hasattr(mat, 'extra_args') and mat.extra_args != {}:
            extra_args = mat.extra_args # fetch material's extra-args dict
            for key in list(extra_args.keys()):
                if key not in Solver.forbidden_nl_arg:
                    if len(args)==1:
                        extra_args[key] = arg
                    else:
                        extra_args[key] = args_dic[key]
                    modified = True
            mat.set_extra_args(**extra_args) # reset material's extra-args

            # recompute material's data with its new extra-arg
            if update and modified:
                update_material(mat, term.region, wf.integral)


def evaluate_at_qps(wf, sol):
    """
    Compute an extrapolation of the solution at the domain's quadrature points.

    Parameters
    ----------
    wf : `WeakForm` instance
        Weak form associated with the solution field `sol`.
    sol : ndarray
        1D array containing the solution at DOFs.

    Returns
    -------
    sol_qps : ndarray
        1D array containing the values of the solution field at quadrature
        points.

    """
    qps = get_physical_qps(wf.omega, wf.integral)
    sol_qps = wf.field.evaluate_at(qps.values, sol[:, np.newaxis]).squeeze()
    return sol_qps

def update_neumann(wf, gamma):
    """
    Utility function for updating the Neumann boundary condition term on the
    exterior domain. Note that the neumann material function should take the
    solver instance as one of its extra-arguments in order to be able to
    actually compute the flux of the solution across the interior domain
    boundary. This function is only called when dealing with unbounded problems
    with `conn` set to 'ping-pong'.

    Parameters
    ----------
    wf : `WeakForm` instance
        Weak form whose Neumann boundary condition term is updated.
    gamma : `Region` instance
        Sub region of dimension D-1 on which the Neumann term is defined.

    Returns
    -------
    None.

    """
    term_idx = None
    for k, term in enumerate(wf.terms):
        # check if term is the neumann term
        if term.name == 'dw_surface_integrate' and term.region==gamma:
            term_idx = k
            break
    # get neumann term and update its material
    mat_neumann = wf.eqn.collect_materials()[term_idx]
    update_material(mat_neumann, gamma, wf.integral)


def update_material(mat, region, integral):
    """
    Utility function for updating a material's data at the region quadrature
    points.

    Parameters
    ----------
    mat : `Material` instance
        The material to be updated.
    region : `Region` instance
        Region over which the material is defined.
    integral : `Integral` instance
        Sfepy's Integral class instance (wrapper class around quadratures).

    Returns
    -------
    None.

    """
    from sfepy.solvers.ts import TimeStepper
    dummy_ts = TimeStepper(0, 100) # ficticious time-stepper
    qps = get_physical_qps(region, integral)
    coors = qps.values
    data = mat.function(dummy_ts, coors, mode='qp', **mat.extra_args)
    key = (region.name, integral.order)
    mat.set_data(key, qps, data)


def update_material_all(wf):
    """
    Utility function for updating all materials' data of a given weak form.

    Parameters
    ----------
    wf : `WeakForm` instance
        Weak form whose materials' data are updated.

    Returns
    -------
    None.

    """
    for mat, term in zip(wf.eqn.collect_materials(), wf.terms):
        update_material(mat, term.region, term.integral)


def reset_dirichlet(data, wf1, wf2, cogammas):
    """
    Utility function for setting new Dirichlet boundary condition on the outer
    boundary of the interior domain. This function is only called when dealing
    with unbounded problems with `conn` set to 'ping-pong'.

    Parameters
    ----------
    data : ndarray
        1D numpy array containing the solution field at the exterior domain
        DOFs.
    wf1 : `WeakForm` instance
        Weak form associated with the interior domain.
    wf2 : `WeakForm` instance
        Weak form associated with the exterior domain.
    cogammas : list
        List of two `sfepy.discrete.common.region.Region` instances
        corresponding to the interior and exterior domain common frontier.

    """

    def dbc(ts, coor, **kwargs):
        return wf2.field.evaluate_at(coor, data[:, np.newaxis]).squeeze()

    d_func = Function('d_func', dbc)
    new_ebc = EssentialBC('dbc_pp', cogammas[0],
                          {'%s.all' %wf1.unknown_name : d_func})

    for k, ebc in enumerate(wf1.ebcs):
        if ebc.region == cogammas[0]:
            wf1.ebcs[k] = new_ebc
            break


def strong_residual(solver, sol_w):
    """
    Compute the strong residual (for nonlinear problems only). Note that the
    strong residual might be evaluated on both the interior and exterior
    if two nonlinear WeakForms instances are provided at the creation of
    `solver` (unbounded problems only).

    Parameters
    ----------
    solver : `Solver` instance
        The solver instance.
    sol_w : array
        Current solution (vector).

    Returns
    -------
    res_vec : ndarray
        Residual vector.

    """

    # Fetch solution vector and matrix & rhs associated with the residual
    wf_res = solver.wfs_dic['wf_res']
    cg = solver.cogammas[0] if hasattr(solver, 'cogammas') else None
    idx_bc = np.array(get_ebc_dofs(wf_res, gamma=cg))
    mtx_res = solver.mtxvec_dic['mtx_res']
    rhs_res = solver.mtxvec_dic['rhs_res']
    
    # Set coeff associated with EBC to zero
    # mtx_res[idx_bc, :] = 0.0
    # mtx_res[:, idx_bc] = 0.0
    # rhs_res[idx_bc] = 0.0
    
    # Evaluate the residual vector
    Ax = mtx_res.dot(sol_w)
    res_vec = Ax - rhs_res
    res_vec[idx_bc] = np.NaN
    res_vec = np.ma.masked_invalid(res_vec)
    res_vec_parts = {'mtx' : Ax,
                     'rhs_cst' : solver.mtxvec_dic['rhs_cst_res'],
                     'rhs_mod' : solver.mtxvec_dic['rhs_mod_res']}

    return res_vec, res_vec_parts


def vec_G(solver, sol_w, delta):
    """Utility function for the line-search algorithm"""
    wf1 = solver.weakforms[0]
    wf_G = solver.wfs_dic['wf_G']
    sol_qps = evaluate_at_qps(wf1, sol_w)
    dphi_qps = evaluate_at_qps(wf1, delta)
    set_mat_extra_arg(wf_G, sol_qps, dphi_qps,
                      keys=('phi', 'dphi'), update=True)
    is_not_assembled = solver.mtxvec_dic.get('mtx_cst_G') is None
    if is_not_assembled:
        assemble_mtxvec(solver, solver.pbs_dic['pb_cst_G'], assemble_rhs=False)
    assemble_mtxvec(solver, solver.pbs_dic['pb_mod_G'], assemble_mtx=False)
    mtx = solver.mtxvec_dic['mtx_cst_G']
    rhs = solver.mtxvec_dic['rhs_mod_G']
    vec = mtx.dot(delta) - rhs
    cg = solver.cogammas[0] if hasattr(solver, 'cogammas') else None
    idx_bc = get_ebc_dofs(wf_G, gamma=cg)
    vec = np.delete(vec, idx_bc)
    return vec

def delete_ebc_dofs(vec, wf, gamma=None):
    """
    Remove vector elements that correspond to DOFs subject to essential
    boundary conditions or DOFs that are part of a certain facet region
    specified by the keyword argument `gamma`.

    Parameters
    ----------
    vec : ndarray
        1D numpy array defined over DOFs.
    wf : `WeakForm` instance
        Weak form instance containing the DOFs info as well as the facet(s)
        subject to essential boundary conditions.
    gamma : `Region` instance, optional
        Extra boundary where `vec` elements should be deleted (a priori not
        concerned by ebc). The default is None.

    Returns
    -------
    idx : ndarray
        Indices of the DOFs deleted.
    ndarray
        The amputed version of `vec`.

    """
    vertices_list = []
    for k in range(len(wf.ebcs)):
        vertices_list.append(wf.field.get_dofs_in_region(wf.ebcs[k].region))
    if gamma is not None:
        vertices_list.append(wf.field.get_dofs_in_region(gamma))
    idx = np.unique(np.concatenate(vertices_list))
    return idx, np.delete(vec, idx)

def get_ebc_dofs(wf, gamma=None):
    """Same as `delete_ebc_dofs` but only returns the indices of the DOFs
    associated with Dirichlet boundary conditions or belong to facet gamma."""
    vertices_list = []
    for k in range(len(wf.ebcs)):
        vertices_list.append(wf.field.get_dofs_in_region(wf.ebcs[k].region))
    if gamma is not None:
        vertices_list.append(wf.field.get_dofs_in_region(gamma))
    idx = np.unique(np.concatenate(vertices_list))
    return idx

def assemble_mtxvec(solver, *pbs, assemble_mtx=True, assemble_rhs=True):
    """
    Assemble the terms (matrices and vectors) of `Problem` instances (Cf sfepy
    API) provided by the user. The assembled matrices and vectors are then
    accessible through the solver's `mtxvec_dic` dictionary.

    Parameters
    ----------
    solver : `Solver` instance
        The solver instance.
    *pbs : list or tupple
        List of `Problem` instances.
    assemble_mtx : bool, optional
        Whether matrices are assembled or not. The default is True.
    assemble_rhs : bool, optional
        Whether vectors are assembled or not. The default is True.

    Returns
    -------
    None.

    """

    saved_assemble_mtx = assemble_mtx
    saved_assemble_rhs = assemble_rhs

    for pb in pbs:

        if pb is None:
            continue

        # name handling
        pb_name = pb.name
        sname = pb_name.split('_')
        sname.pop(0)
        mtx_key = 'mtx_' + '_'.join(sname)
        rhs_key = 'rhs_' + '_'.join(sname)

        # handling exceptions
        if ((solver.islinear and pb.name=='pb_mod_pp2') or 
            (not solver.islinear and pb.name=='pb_mod_res')):
            assemble_mtx = False
        if not solver.initialized and pb.name=='pb_mod_pp2':
            assemble_rhs = False
        if not(assemble_mtx or assemble_rhs):
            continue

        # assembling matrix and rhs vector
        tss = pb.get_solver()
        pb.equations.set_data(None, ignore_unknown=True)
        variables = pb.get_initial_state(vec=None)
        variables.adof_conns = {}
        pb.time_update(tss.ts, is_matrix=(pb.mtx_a is None))
        variables.apply_ebc(force_values=None)
        pb.update_materials()
        ev = pb.get_evaluator()
        if assemble_mtx:
            mtx = ev.eval_tangent_matrix(variables(), is_full=True)
            solver.mtxvec_dic[mtx_key] = mtx
        if assemble_rhs:
            rhs = -ev.eval_residual(variables(), is_full=True)
            solver.mtxvec_dic[rhs_key] = rhs
            # alternative way / syntax
            # rhs = pb.equations.create_reduced_vec()
            # rhs = pb.equations.evaluate(mode='weak', dw_mode='vector',
            #                             asm_obj=rhs)
            # solver.mtxvec_dic[rhs_key] = -rhs

        # reset assembling boolean options
        assemble_mtx = saved_assemble_mtx
        assemble_rhs = saved_assemble_rhs


def update_global_mtxvec(solver):
    """Update the global stiffness matrix & rhs vector (those subsequently
    passed to the linear solver) which are only sums of pre-computed matrices
    & vectors."""
    mtxvec_dic = solver.mtxvec_dic
    wf1 = solver.weakforms[0]
    if solver.nwf==1 or (solver.nwf==2 and solver.conn=='connected'):
        if wf1.eqn_mod is None:
            mtxvec_dic['mtx'] = mtxvec_dic['mtx_cst']
            mtxvec_dic['rhs'] = mtxvec_dic['rhs_cst']
        else:
            mtxvec_dic['mtx'] = mtxvec_dic['mtx_cst'] + mtxvec_dic['mtx_mod']
            mtxvec_dic['rhs'] = mtxvec_dic['rhs_cst'] + mtxvec_dic['rhs_mod']
    elif solver.nwf==2 and solver.conn=='ping-pong':
        if wf1.eqn_mod is None:
            mtxvec_dic['mtx_pp1'] = mtxvec_dic['mtx_cst_pp1']
            mtxvec_dic['rhs_pp1'] = mtxvec_dic['rhs_cst_pp1']
            mtxvec_dic['mtx_pp2'] = mtxvec_dic['mtx_cst_pp2']
            mtxvec_dic['rhs_pp2'] = mtxvec_dic['rhs_cst_pp2']
        else:
            mtxvec_dic['mtx_pp1'] = mtxvec_dic['mtx_cst_pp1'] \
                + mtxvec_dic['mtx_mod_pp1']
            mtxvec_dic['rhs_pp1'] = mtxvec_dic['rhs_cst_pp1'] \
                + mtxvec_dic['rhs_mod_pp1']

    wf_res = solver.wfs_dic.get('wf_res')
    if wf_res is not None:
        if wf_res.eqn_mod is None:
            mtxvec_dic['mtx_res'] = mtxvec_dic['mtx_cst_res']
            mtxvec_dic['rhs_res'] = mtxvec_dic['rhs_cst_res']
        else:
            mtxvec_dic['mtx_res'] = mtxvec_dic['mtx_cst_res'] #+ mtxvec_dic['mtx_mod_res']
            mtxvec_dic['rhs_res'] = mtxvec_dic['rhs_cst_res'] \
                + mtxvec_dic['rhs_mod_res']


def reconstruct_full_sol(pb, sol_reduced):
    """
    Reconstruct the full solution vector from the reduced solution vector
    (defined only at active DOFs i.e. DOFs that are not constrained by
    essential boundary conditions).

    Parameters
    ----------
    pb : `Problem` instance
        Problem instance associated with the solution to reconstruct.
    sol_reduced : ndarray
        Reduced solution vector.

    Returns
    -------
    sol_full : ndarray
        Solution vector defined on all DOFs.

    """
    pb.equations.variables.set_state(sol_reduced, reduced=True)
    sol_full = pb.equations.variables()
    return sol_full