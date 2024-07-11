# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:47:57 2024

Module dedicated to the implementation of specific PDE problems coming from
physics.

"""

from typing import List, Union, Callable

from abc import ABC, abstractmethod

import numpy as np
from sfepy.discrete.fem.mesh import Mesh

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.weak_form import WeakForm
from femtoscope.core.solvers import LinearSolver, NonLinearSolver
from femtoscope.core.nonlinear_monitoring import NonLinearMonitor
from femtoscope.physics.materials_library import (
    DensityMaterials, LaplacianMaterials, LapAdvectionMaterials,
    LaplacianMaterialsVacuum, LapAdvectionMaterialsVacuum, NonLinearMaterials,
    LapSurfaceMaterials)
from femtoscope.inout.meshfactory import read_physical_groups_from_mesh_file


class AbstractLinear(ABC):

    def __init__(self, param_dict: dict, dim: int, coorsys=None, Rc=None):
        self.wf_int = None
        self.wf_ext = None
        self.wf_residual = None
        self.param_dict = param_dict
        self.dim = dim
        self.coorsys = coorsys
        self.Rc = Rc
        self.subomegas_number = None
        self.vacuum_exterior = None
        self.default_solver = None

    @property
    def wf_dict(self):
        weakform_dict = {'wf_int': self.wf_int}
        if self.wf_ext is not None:
            weakform_dict['wf_ext'] = self.wf_ext
        if self.wf_residual is not None:
            weakform_dict['wf_residual'] = self.wf_residual
        return weakform_dict

    def set_wf_int(self, partial_args_dict: dict, density_dict: dict):
        self._set_subomegas_number(partial_args_dict['pre_mesh'])
        partial_args_dict['dim'] = self.dim
        pre_terms = self._create_pre_terms_int(density_dict)
        partial_args_dict['pre_terms'] = pre_terms
        wf = WeakForm.from_scratch(partial_args_dict)
        self.wf_int = wf

    def set_wf_ext(self, partial_args_dict: dict, density=None):
        self.vacuum_exterior = (density is None) or (density == 0)
        partial_args_dict['dim'] = self.dim
        pre_terms = self._create_pre_terms_ext(density)
        partial_args_dict['pre_terms'] = pre_terms
        partial_args_dict['is_exterior'] = True
        wf = WeakForm.from_scratch(partial_args_dict)
        self.wf_ext = wf

    def set_default_solver(self, region_key_int=None, region_key_ext=None):
        ls_class = 'ScipyDirect'
        ls_kwargs = {'eps_a': 1e-8, 'eps_r': 1e-8}
        solver = LinearSolver(self.wf_dict, ls_class=ls_class,
                              ls_kwargs=ls_kwargs,
                              region_key_int=region_key_int,
                              region_key_ext=region_key_ext)
        self.default_solver = solver

    @abstractmethod
    def _create_pre_terms_int(self, density_dict: dict) -> List[PreTerm]:
        raise NotImplementedError

    @abstractmethod
    def _create_pre_terms_ext(
            self, density: Union[None, Callable, float]) -> List[PreTerm]:
        raise NotImplementedError

    def _set_subomegas_number(self, pre_mesh: Union[str, Mesh]):
        if self.subomegas_number is not None: return
        if isinstance(pre_mesh, str):
            self.subomegas_number = len(
                read_physical_groups_from_mesh_file(pre_mesh))
        else:
            self.subomegas_number = 1

    def _append_density_pre_terms(self, pre_terms: List[PreTerm],
                                  density_dict: dict, prefactor: float):
        """
        Append density terms to the `pre_terms` list.

        Parameters
        ----------
        pre_terms : List[PreTerm]
            List of already registered `PreTerm` instances, to which density
            pre_terms are appended.
        density_dict : dict
            Specification of the density using region keys, e.g.
            ('subomega', 300), ('subomega', 301), ('omega', -1), etc.
        prefactor : float
            Constant weighting the density terms.

        Notes
        -----
        The tag in `get_material` is set to 'int' because exterior domains
        cannot have multiple subomega regions. For non-constant density in the
        exterior domain, specify it with a function of the coordinates instead.

        """
        for region_key, rho in density_dict.items():
            if (rho is None) or (rho == 0):
                continue
            rhomat = DensityMaterials()
            mat_kwargs = {'rho': rho, 'Rc': self.Rc}
            mat = rhomat.get_material(self.dim, coorsys=self.coorsys, tag='int')
            t = PreTerm('dw_volume_integrate', tag='cst', prefactor=prefactor,
                        region_key=region_key, mat=mat, mat_kwargs=mat_kwargs)
            pre_terms.append(t)


class AbstractNonLinear(AbstractLinear, ABC):

    def __init__(self, param_dict: dict, dim: int, coorsys=None, Rc=None):
        super().__init__(param_dict, dim, coorsys=coorsys, Rc=Rc)
        self.wf_residual = None
        self.default_monitor = None

    @property
    def wf_dict(self):
        weakform_dict = {'wf_int': self.wf_int, 'wf_residual': self.wf_residual}
        if self.wf_ext is not None:
            weakform_dict['wf_ext'] = self.wf_ext
        return weakform_dict

    def set_default_monitor(self, max_iter: int, min_iter: int = 0):
        """
        Set the `default_monitor` attribute with (commonly used) hard coded
        criteria.

        Parameters
        ----------
        max_iter : int
            Minimum number of iterations to be completed (lower threshold).
        min_iter : int
            Maximum number of iterations to be completed (upper threshold).

        """

        criteria = (
            {
                'name': 'RelativeDeltaSolutionNorm2',
                'threshold': 1e-8,
                'look': True,
                'active': True
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
            {
                'name': 'ResidualVectorParts',
                'threshold': -1,
                'look': True,
                'active': False
            },
        )

        args_dict = {
            'minimum_iter_num': min_iter,
            'maximum_iter_num': max_iter,
            'criteria': criteria
        }

        monitor = NonLinearMonitor.from_scratch(args_dict)

        # Link with nonlinear_solver
        if self.default_solver is not None:
            self.default_solver.link_solver_to_monitor(monitor)

        self.default_monitor = monitor

    def set_wf_residual(self, partial_args_dict: dict, density_dict: dict):
        self._set_subomegas_number(partial_args_dict['pre_mesh'])
        partial_args_dict['dim'] = self.dim
        pre_terms = self._create_pre_terms_res(density_dict)
        partial_args_dict['pre_terms'] = pre_terms
        partial_args_dict['is_exterior'] = False
        wf = WeakForm.from_scratch(partial_args_dict)
        self.wf_residual = wf

    @abstractmethod
    def set_default_solver(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _create_pre_terms_res(self, density_dict: dict) -> List[PreTerm]:
        raise NotImplementedError


class Poisson(AbstractLinear):
    r"""
    Poisson problem reading
    $$ \Delta u = \alpha \rho(x) $$
    """

    def __init__(self, param_dict: dict, dim: int, coorsys=None, Rc=None):
        super().__init__(param_dict, dim, coorsys=coorsys, Rc=Rc)
        self.vacuum_exterior = None

    def _create_pre_terms_int(self, density_dict: dict) -> List[PreTerm]:
        """
        Create the list of pre-terms associated with the Poisson problem in the
        interior domain.

        Parameters
        ----------
        density_dict : dict
            Specification of the density using region keys, e.g.
            ('subomega', 300), ('subomega', 301), ('omega', -1), etc.

        Returns
        -------
        pre_terms : List[PreTerm]
            List of pre-terms further used to instantiate the interior weakform.

        """
        alpha = self.param_dict['alpha']

        lapmat = LaplacianMaterials()
        mat1 = lapmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
        t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=1.0)

        pre_terms = [t1]
        self._append_density_pre_terms(pre_terms, density_dict, alpha)

        return pre_terms

    def _create_pre_terms_ext(
            self, density: Union[None, Callable, float]) -> List[PreTerm]:
        """
        Create the list of pre-terms associated with the Poisson problem in the
        exterior domain.

        Parameters
        ----------
        density : Union[None, Callable, float]
            Specification of the density in the exterior domain. Cases are
            - None or 0: vacuum, no density term;
            - float (different from zero): constant density;
            - Callable: density specified by a function.

        Returns
        -------
        pre_terms : List[PreTerm]
            List of pre-terms further used to instantiate the exterior weakform.

        """

        alpha = self.param_dict['alpha']
        coorsys = self.coorsys

        if self.vacuum_exterior:  # No density term
            lapfac = self.Rc ** 2 if self.dim == 1 else 1.0
            lapmat = LaplacianMaterialsVacuum()
            mat1 = lapmat.get_material(self.dim, coorsys=coorsys, tag='ext')
            t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
            t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=lapfac,
                         mat_kwargs={'Rc': self.Rc})
            pre_terms = [t1]

            admat = LapAdvectionMaterialsVacuum()
            mat2 = admat.get_material(self.dim, coorsys=coorsys, tag='ext')
            if mat2 is not None:
                t2 = PreTerm('dw_s_dot_mgrad_s', mat=mat2, tag='cst',
                             prefactor=1.0, mat_kwargs={'Rc': self.Rc})
                pre_terms.append(t2)

        else:  # One density
            lapmat = LaplacianMaterials()
            mat1 = lapmat.get_material(self.dim, coorsys=coorsys, tag='ext')
            t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
            t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=1.0,
                         mat_kwargs={'Rc': self.Rc})

            admat = LapAdvectionMaterials()
            mat2 = admat.get_material(self.dim, coorsys=coorsys, tag='ext')
            t2 = PreTerm('dw_s_dot_mgrad_s', mat=mat2, tag='cst', prefactor=1.0,
                         mat_kwargs={'Rc': self.Rc})
            pre_terms = [t1, t2]

            rhomat = DensityMaterials()
            mat3 = rhomat.get_material(self.dim, coorsys=coorsys, tag='ext')
            t3 = PreTerm('dw_volume_integrate', tag='cst', prefactor=alpha,
                         mat=mat3, mat_kwargs={'Rc': self.Rc, 'rho': density})
            pre_terms.append(t3)

        return pre_terms


class Yukawa(AbstractLinear):
    r"""
    Linear Klein-Gordon equation reading
    $$ \gamma \Delta u = u + \rho $$
    """

    def __init__(self, param_dict: dict, dim: int, coorsys=None, Rc=None):
        super().__init__(param_dict, dim, coorsys=coorsys, Rc=Rc)

    def _create_pre_terms_int(self, density_dict: dict) -> List[PreTerm]:
        """
        Create the list of pre-terms associated with the Yukawa problem in the
        interior domain.

        Parameters
        ----------
        density_dict : dict
            Specification of the density using region keys, e.g.
            ('subomega', 300), ('subomega', 301), ('omega', -1), etc.

        Returns
        -------
        pre_terms : List[PreTerm]
            List of pre-terms further used to instantiate the interior weakform.

        """
        gamma = self.param_dict['gamma']

        lapmat = LaplacianMaterials()
        mat1 = lapmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
        t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=gamma)

        massmat = DensityMaterials()
        mat2 = massmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t2 = PreTerm('dw_volume_dot', mat=mat2, tag='cst', prefactor=1,
                     mat_kwargs={'rho': 1.0})

        pre_terms = [t1, t2]
        self._append_density_pre_terms(pre_terms, density_dict, 1)

        return pre_terms

    def _create_pre_terms_ext(
            self, density: Union[None, Callable, float]) -> List[PreTerm]:
        """
        Create the list of pre-terms associated with the Yukawa problem in the
        exterior domain.

        Parameters
        ----------
        density : Union[None, Callable, float]
            Specification of the density in the exterior domain. Cases are
            - None or 0: vacuum, no density term;
            - float (different from zero): constant density;
            - Callable: density specified by a function.

        Returns
        -------
        pre_terms : List[PreTerm]
            List of pre-terms further used to instantiate the exterior weakform.

        """
        alpha = self.param_dict['alpha']
        coorsys = self.coorsys

        if self.vacuum_exterior:
            density = 0

        lapmat = LaplacianMaterials()
        mat1 = lapmat.get_material(self.dim, coorsys=coorsys, tag='ext')
        t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
        t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=alpha,
                     mat_kwargs={'Rc': self.Rc})

        admat = LapAdvectionMaterials()
        mat2 = admat.get_material(self.dim, coorsys=coorsys, tag='ext')
        t2 = PreTerm('dw_s_dot_mgrad_s', mat=mat2, tag='cst', prefactor=alpha,
                     mat_kwargs={'Rc': self.Rc})

        massmat = DensityMaterials()
        mat3 = massmat.get_material(self.dim, coorsys=coorsys, tag='ext')
        t3 = PreTerm('dw_volume_dot', tag='cst', prefactor=1, mat=mat3,
                     mat_kwargs={'Rc': self.Rc, 'rho': 1})

        rhomat = DensityMaterials()
        mat4 = rhomat.get_material(self.dim, coorsys=coorsys, tag='ext')
        t4 = PreTerm('dw_volume_integrate', tag='cst', prefactor=1,
                     mat=mat4, mat_kwargs={'Rc': self.Rc, 'rho': density})

        pre_terms = [t1, t2, t3, t4]

        return pre_terms


class Chameleon(AbstractNonLinear):
    r"""
    Chameleon field equation reading
    $$ \alpha \Delta u = \rho(x) - u^{-(n + 1)} $$
    """

    def set_default_solver(self, relax_param=0.8, guess='min_pot',
                           region_key_int=None, region_key_ext=None):
        """
        Set `default_solver` attribute of a chameleon problem by creating a
        NonLinearSolver instance based on user provided input and hard coded
        default parameters.

        Parameters
        ----------
        relax_param : float, optional
            Relaxation parameter. The default is 0.8.
        guess : Union[str, dict]
            Ways of specifying the initial guess
            - 'min_pot' (str, default) constructs the initial guess from the
            density profile (minimization of the chameleon effective potential).
            - initial_guess_dict (dict) is ready to be passed to NonLinearSolver
            constructor. Dictionary with keys ('int', 'ext') containing the
            initial guess as numpy 1d-arrays.
        region_key_int : tuple
            Key of the connecting region in the interior domain. Mandatory
            parameter for two-weak-form problems.
        region_key_ext : tuple
            Key of the connecting region in the exterior domain. Mandatory
            parameter for two-weak-form problems.

        """

        # Get chameleon parameters
        npot = self.param_dict['npot']
        rho_min = self.param_dict.get('rho_min', None)
        rho_max = self.param_dict.get('rho_max', None)

        # Theoretical lower / upper bounds
        phi_min = rho_max ** (-1 / (npot + 1)) if rho_max is not None else None
        phi_max = rho_min ** (-1 / (npot + 1)) if rho_min is not None else None
        if (phi_min is None) and (phi_max is None):
            sol_bounds = None
        else:
            sol_bounds = [phi_min, phi_max]

        initial_guess_dict = self._create_initial_guess_dict(guess)

        solver = NonLinearSolver(self.wf_dict, initial_guess_dict,
                                 sol_bounds=sol_bounds, relax_method='constant',
                                 relax_param=relax_param,
                                 region_key_int=region_key_int,
                                 region_key_ext=region_key_ext)

        # Link with nonlinear_monitor
        if self.default_monitor is not None:
            self.default_monitor.link_monitor_to_solver(solver)

        self.default_solver = solver

    def _create_initial_guess_dict(self, guess: Union[str, dict]) -> dict:
        """
        Create the `initial_guess_dict` argument needed for NonLinearSolver
        constructor (see `set_default_solver` docstring).

        Returns
        -------
        dict
            The initial_guess_dict argument.
        """

        if isinstance(guess, dict):
            return guess
        if guess == 'min_pot':
            return self._initial_guess_min_pot()
        else:
            raise NotImplementedError(f"Method '{guess}' for setting the "
                                      f"initial guess is not implemented!")

    def _initial_guess_min_pot(self) -> dict:
        r"""
        Initialize the initial guess dictionary by minimizing the chameleon
        effective potential in the numerical domain(s). In practice, this is
        done by setting $\phi(x) = \rho(x)^{-1/(npot+1)}$.

        Returns
        -------
        initial_guess_dict : dict
            The initial_guess_dict argument set using the density profile.
        """

        # Get chameleon parameter & initialize dictionary
        npot = self.param_dict['npot']
        initial_guess_dict = {}

        # Fill in the dictionary
        for key, wf in zip(('int', 'ext'), (self.wf_int, self.wf_ext)):
            if wf is None: continue

            coors = wf.field.coors
            rho_vec = np.nan * np.ones(coors.shape[0], dtype=np.float64)

            for pre_term in wf.pre_terms:  # Looping over all pre_terms
                if 'rho' in pre_term.mat_kwargs:  # select those with density
                    rho = pre_term.mat_kwargs['rho']
                    reg_key = pre_term.region_key
                    dofs = wf.field.get_dofs_in_region(wf.region_dict[reg_key])
                    if callable(rho):
                        rho_vec[dofs] = (rho(coors))[dofs]
                    else:
                        rho_vec[dofs] = rho * np.ones(len(dofs))

            if np.isnan(rho_vec).any():  # Sanity check
                raise ValueError("Initial guess is not correctly set!")

            # Compute the initial guess from the density vector
            initial_guess_dict[key] = rho_vec ** (-1 / (npot + 1))

        return initial_guess_dict

    def _create_pre_terms_int(self, density_dict: dict) -> List[PreTerm]:

        # Get chameleon parameters
        alpha = self.param_dict['alpha']
        npot = self.param_dict['npot']

        lapmat = LaplacianMaterials()
        mat1 = lapmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
        t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=alpha)

        nlmat = NonLinearMaterials()
        matnl = nlmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t2 = PreTerm('dw_volume_dot', mat=matnl, tag='mod', prefactor=npot + 1,
                     mat_kwargs={'nl_fun': lambda x: x ** (-(npot + 2))})
        t3 = PreTerm(
            'dw_volume_integrate', mat=matnl, tag='mod', prefactor=-(npot + 2),
            mat_kwargs={'nl_fun': lambda x: x ** (-(npot + 1))})

        pre_terms = [t1, t2, t3]
        self._append_density_pre_terms(pre_terms, density_dict, 1.0)

        return pre_terms

    def _create_pre_terms_ext(
            self, density: Union[None, Callable, float]) -> List[PreTerm]:

        # Get chameleon parameters
        alpha = self.param_dict['alpha']
        npot = self.param_dict['npot']
        Rc = self.Rc

        lapmat = LaplacianMaterials()
        mat1 = lapmat.get_material(self.dim, coorsys=self.coorsys, tag='ext')
        t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
        t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=alpha,
                     mat_kwargs={'Rc': Rc})

        advmat = LapAdvectionMaterials()
        mat2 = advmat.get_material(self.dim, coorsys=self.coorsys, tag='ext')
        t2 = PreTerm('dw_s_dot_mgrad_s', mat=mat2, tag='cst', prefactor=alpha,
                     mat_kwargs={'Rc': Rc})

        nlmat = NonLinearMaterials()
        matnl = nlmat.get_material(self.dim, coorsys=self.coorsys, tag='ext')
        t3 = PreTerm('dw_volume_dot', mat=matnl, tag='mod', prefactor=npot + 1,
                     mat_kwargs={'Rc': Rc,
                                 'nl_fun': lambda x: x ** (-(npot + 2))})
        t4 = PreTerm('dw_volume_integrate', mat=matnl, tag='mod',
                     prefactor=-(npot + 2),
                     mat_kwargs={'Rc': Rc,
                                 'nl_fun': lambda x: x ** (-(npot + 1))})

        pre_terms = [t1, t2, t3, t4]

        if not self.vacuum_exterior:  # One density term
            rhomat = DensityMaterials()
            mat5 = rhomat.get_material(
                self.dim, coorsys=self.coorsys, tag='ext')
            t5 = PreTerm('dw_volume_integrate', tag='cst', prefactor=1.0,
                         mat=mat5, mat_kwargs={'Rc': Rc, 'rho': density})
            pre_terms.append(t5)

        return pre_terms

    def _create_pre_terms_res(self, density_dict: dict) -> List[PreTerm]:

        # Get chameleon parameters
        alpha = self.param_dict['alpha']
        npot = self.param_dict['npot']

        lapmat = LaplacianMaterials()
        mat1 = lapmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t1_name = _get_diffusion_term_name(self.dim, self.coorsys)
        t1 = PreTerm(t1_name, mat=mat1, tag='cst', prefactor=alpha)

        nlmat = NonLinearMaterials()
        matnl = nlmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t2 = PreTerm('dw_volume_integrate', mat=matnl, tag='mod', prefactor=-1,
                     mat_kwargs={'nl_fun': lambda x: x ** (-(npot + 1))})

        surfmat = LapSurfaceMaterials()
        mat3 = surfmat.get_material(self.dim, coorsys=self.coorsys, tag='int')
        t3 = PreTerm('dw_surface_flux', mat=mat3, tag='cst', prefactor=-alpha,
                     region_key=('gamma', -1))

        pre_terms = [t1, t2, t3]

        self._append_density_pre_terms(pre_terms, density_dict, 1.0)

        return pre_terms


class Symmetron(AbstractNonLinear):
    r"""
    Chameleon field equation reading
    $$ \alpha \Delta u = (\rho(x)-\beta) u + u^3 $$
    """

    def set_default_solver(self, *args, **kwargs):
        pass

    def _create_pre_terms_res(self, density_dict: dict) -> List[PreTerm]:
        pass

    def _create_pre_terms_int(self, density_dict: dict) -> List[PreTerm]:
        pass

    def _create_pre_terms_ext(
            self, density: Union[None, Callable, float]) -> List[PreTerm]:
        pass


def _get_diffusion_term_name(dim, coorsys):
    if dim == 2 and coorsys == 'polar':
        return 'dw_diffusion'
    else:
        return 'dw_laplace'
