# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:24:52 2024

Definition of a 'weak-formulation' class, mainly gathering Sfepy's objects.
The current implementation relies partially on the assumption that, even when
nonlinear, the PDE at stake has at least one linear term.
"""

from __future__ import annotations

import warnings
from numbers import Number
from typing import List, Union
from copy import copy

import numpy as np

from sfepy.base.base import Container
import sfepy.discrete.fem
from sfepy.discrete import (FieldVariable, Function, Functions, Integral,
                            Materials, Variables)
from sfepy.discrete.conditions import (Conditions, EssentialBC,
                                       LinearCombinationBC, PeriodicBC)
from sfepy.discrete.equations import Equation, Equations
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.discrete.problem import Problem
from sfepy.terms.terms import Terms
import sfepy.discrete.fem.periodic as per
from sfepy.discrete.common.mappings import get_physical_qps

from femtoscope.core import valid_topological_kinds_ebc, valid_matching_epbc
from femtoscope.core.matrix_vector import MatrixVector
from femtoscope.core.pre_term import PreTerm, is_matrix_term
from femtoscope.inout.meshfactory import (make_absolute_mesh_name,
                                          read_physical_groups_from_mesh_file)


class WeakForm:
    """
    Class gathering objects linked to the weak formulation of some PDE.

    Attributes
    ----------
    name : str
        Name of the weak form.
    dim : int
        Dimension of the problem: 1, 2 or 3.
    region_dict : dict
        Dictionary of regions with
        keys: pairs topological kind - tag, e.g. ('subomega', 301).
        values: corresponding Sfepy `Region` instances.
    ebc_dict : dict
        Dictionary of essential boundary conditions with
        keys: same as `region_dict`.
        values: dictionary with keys ('cst', 'mod') containing corresponding
        Sfepy's `EssentialBC` instances.
    epbc_dict : dict
        Dictionary of periodic boundary conditions with
        keys: 'cst' and 'mod'.
        values: list of `PeriodicBC` instances.
    pre_terms : list[PreTerm]
        List of `PreTerm` instances. Each 'pre_term' contains extra information
        with respect to Sfepy's `Term` instance.
    fem_order : int
        Finite Element approximation order.
    pb_cst : Problem
        Sfepy's Problem instance associated with the linear part of the PDE.
    pb_mod : Problem
        Sfepy's Problem instance associated with the nonlinear part of the PDE.
    mtx_vec : MatrixVector
        Instance of class `MatrixVector` for storing assembled matrices and
        vectors associated with the current weak form.
    _active_bcs : dict
        Private attribute indicating whether boundary conditions are set active
        on `pb_cst` and `pb_mod`.

    Properties
    ----------
    active_bcs : dict
        Proxy for `_active_bcs` in read-only mode.
    field : Field
        Sfepy Field instance.
    integral : Integral
        Sfepy Integral instance.
    ndofs : int
        Total number of degrees of freedom.
    vec_bc : np.ndarray
        Full vector filled with zeros except at fixed DOFs where Dirichlet
        BC values are used.

    """

    match_coors = Function('match_coors', per.match_coors)
    match_x_line = Function('match_x_line', per.match_x_line)
    match_y_line = Function('match_y_line', per.match_y_line)
    match_z_line = Function('match_z_line', per.match_z_line)
    match_x_plane = Function('match_x_plane', per.match_x_plane)
    match_y_plane = Function('match_y_plane', per.match_y_plane)
    match_z_plane = Function('match_z_plane', per.match_z_plane)
    functions = Functions(
        [match_coors, match_x_line, match_y_line, match_z_line,
         match_x_plane, match_y_plane, match_z_plane])

    class Decorators:
        """Inner class for defining decorators."""

        @staticmethod
        def check_lcbcs(func):
            def wrapper(*args, **kwargs):
                self = args[0]
                try:
                    if self.pb_cst.equations.variables.has_lcbc:
                        raise ValueError(
                            f"Method '{func.__name__}' cannot be called on weak"
                            f" form '{self.name}' because it has lcbcs!")
                except AttributeError:
                    warnings.warn(f"weak form {self.name} is not fully set up!")
                return func(*args, **kwargs)
            return wrapper

    def __init__(self):
        self.name = None
        self.dim = None
        self.region_dict = None
        self.ebc_dict = None
        self.epbc_dict = None
        self.pre_terms = None
        self.fem_order = None
        self.pb_cst = None
        self.pb_mod = None
        self.mtx_vec = MatrixVector()
        self._active_bcs = {'cst': None, 'mod': None}
        self._is_exterior = None

    @property
    def active_bcs(self):
        return self._active_bcs

    @property
    def field(self):
        return self.get_field()

    @property
    def integral(self):
        return self.pb_cst.equations[0].terms[0].integral

    @property
    def ndofs(self):
        return self.field.coors.shape[0]

    @property
    def vec_bc(self):
        """Full vector filled with zeros except at fixed DOFs where Dirichlet
        BC values are used."""
        self.apply_bcs()
        return self.pb_cst.get_variables().get_state(reduced=False)

    @classmethod
    def from_attr_dict(cls, attr_dict: dict, verbose=False) -> WeakForm:
        """Create an instance of `WeakForm`, set its attributes from the
        `attr_dict` dictionary and return it."""
        wf_instance = cls()
        for key, val in attr_dict.items():
            if not isinstance(key, str):
                raise ValueError("Dictionary keys must be strings")
            if key not in vars(wf_instance):
                raise KeyError(f"'{key}' is not a WeakForm attribute name")
            setattr(wf_instance, key, val)
        if len(attr_dict) < len(vars(wf_instance)) and verbose:
            warnings.warn("Some attributes have not been set")
        return wf_instance

    @classmethod
    def from_scratch(cls, args_dict: dict) -> WeakForm:
        """
        Create an instance of `WeakForm` and set its attributes by creating the
        relevant Sfepy's objects step by step. The dictionary `args_dict`
        passed to the function contains the following items:

        Parameters
        ----------
        dim : int
            Dimension of the problem: 1, 2 or 3.
        pre_mesh : str or `sfepy.discrete.fem.mesh.Mesh`
            In 1D: already existing Sfepy's `Mesh` instance.
            In 2D and 3D: name of the .vtk mesh file.
        name : str, optional
            Name of the weak form. The default is 'wf'.
        dim_func_entities : List[tuple]
            List of triplets (entity dimension, function selection, tag).
        pre_ebc_dict : dict
            Dictionary that is used to assign some essential boundary conditions
            to some topological entities. Its keys are a subset of `region_dict`
            ones and its values are either numbers or functions.
        pre_epbc_list : List[list]
            List containing pairs of region keys and a matching keyword to form
            periodic boundary conditions.
        fem_order : int, optional
            Finite Element approximation order. The default is 2.
        pre_terms : List[PreTerm]
            List of `PreTerm` instances associated with the PDE to be solved.
        is_exterior : Union[bool, None]
            Wheter the weak form to be created shall be associated with the
            exterior domain. The default is None.

        Returns
        -------
        wf_instance : WeakForm
            The created weak form.

        """

        # Get arguments and default values
        dim = args_dict['dim']
        pre_mesh = args_dict['pre_mesh']
        name = args_dict.get('name', 'wf')
        dim_func_entities = args_dict.get('dim_func_entities', [])
        pre_ebc_dict = args_dict.get('pre_ebc_dict', {})
        pre_epbc_list = args_dict.get('pre_epbc_list', [])
        fem_order = args_dict.get('fem_order', 2)
        pre_terms = args_dict.get('pre_terms', [])
        is_exterior = args_dict.get('is_exterior')

        # Create empty class instance
        wf_instance = cls()

        # Name of the weak form, is_exterior?
        wf_instance.name = name
        wf_instance._is_exterior = is_exterior

        # Dimension
        wf_instance.dim = dim

        # Region dictionary
        mesh = wf_instance.create_sfepy_internal_mesh(pre_mesh)
        domain = FEDomain('domain', mesh)
        if dim == 1:
            physical_group_ids = []
        else:
            physical_group_ids = read_physical_groups_from_mesh_file(pre_mesh)
        wf_instance.region_dict = {}
        wf_instance.fill_region_dict(domain, physical_group_ids,
                                     dim_func_entities)

        # Essential boundary conditions dictionary
        cls._check_pre_ebc_dict_format(pre_ebc_dict)
        wf_instance.ebc_dict = {}
        wf_instance.fill_ebc_dict(pre_ebc_dict)

        # Periodic boundary conditions list
        cls._check_pre_epbc_list_format(pre_epbc_list)
        wf_instance.epbc_dict = {}
        wf_instance.fill_epbc_dict(pre_epbc_list)

        # FEM order
        wf_instance.fem_order = fem_order

        # Problem instances
        wf_instance.pre_terms = pre_terms
        wf_instance._fix_pb_without_state_variable()
        wf_instance.set_problems(pre_terms)
        wf_instance.apply_bcs(tag='both')

        return wf_instance

    @classmethod
    def from_two_weak_forms(cls, wf_int: WeakForm, wf_ext: WeakForm,
                            region_key_int: tuple, region_key_ext: tuple,
                            verbose=False) -> WeakForm:
        """
        Create an instance of `WeakForm` from two existing weak forms,
        corresponding to the interior and exterior domains respectively.
        This class method is to be used when solving problems on unbounded
        domains with the inverted finite element method.

        Parameters
        ----------
        wf_int : WeakForm
            Interior weak form.
        wf_ext : WeakForm
            Exterior weak form.
        region_key_int : tuple
            Key of the connecting region in the interior domain.
        region_key_ext : tuple
            Key of the connecting region in the exterior domain.
        verbose : bool, optional
            The default is False.

        Returns
        -------
        wf_instance : WeakForm
            The created weak form.

        Notes
        -----
        The following attributes of the new WeakForm instance will be disabled
        (i.e. set to None):
        - region_dict
        - ebc_dict
        - pre_terms
        Moreover, the boundary conditions are necessarily set active, meaning
        that several methods & functionalities will not work with the newly
        created weak form.

        """
        # Safety checks
        cls._check_args_before_merging(wf_int, wf_ext,
                                       region_key_int, region_key_ext)
        # Fill basic attributes
        attr_dict = {}
        attr_dict['name'] = "combine_{}_&_{}".format(wf_int.name, wf_ext.name)
        attr_dict['dim'] = wf_int.dim
        attr_dict['region_dict'] = None
        attr_dict['ebc_dict'] = None
        attr_dict['pre_terms'] = None
        attr_dict['fem_order'] = wf_int.fem_order

        # Preparation of LCBCs
        connecting_regions = [wf_int.region_dict[region_key_int],
                              wf_ext.region_dict[region_key_ext]]

        # Construct new Problem instance(s)
        if wf_int.pb_mod is None:
            tag_list = ['cst']
            attr_dict['pb_mod'] = None
        else:
            tag_list = ['cst', 'mod']

        for tag in tag_list:
            # Set up ebcs & names
            pb_key = 'pb_{}'.format(tag)
            ebcs_int = [dico[tag] for dico in list(wf_int.ebc_dict.values())]
            ebcs_ext = [dico[tag] for dico in list(wf_ext.ebc_dict.values())]
            ebcs_new = Conditions(ebcs_int + ebcs_ext)
            u_int_name = wf_int.get_unknown_name(tag)
            u_ext_name = wf_ext.get_unknown_name(tag)

            # Linear Combination Boundary Conditions
            lcbc = LinearCombinationBC(
                'lc_cst', connecting_regions,
                {'{}.all'.format(u_int_name) : '{}.all'.format(u_ext_name)},
                cls.match_coors, 'match_dofs')
            lcbcs = Conditions([lcbc])

            # Create new Problem instance
            eqs = _EquationsFix(getattr(wf_int, pb_key).equations._objs +
                                getattr(wf_ext, pb_key).equations._objs)
            pb_new = Problem(pb_key, equations=eqs, active_only=True)
            pb_new.set_bcs(ebcs=ebcs_new, lcbcs=lcbcs)

            # Apply ebc to variables
            variables = pb_new.get_initial_state()
            pb_new.time_update()
            variables.apply_ebc()
            attr_dict[pb_key] = pb_new

        # Create WeakForm instance from 'attr_dict' and mark bcs as active
        wf_instance = cls.from_attr_dict(attr_dict, verbose=verbose)
        wf_instance._active_bcs['cst'] = True
        wf_instance._active_bcs['mod'] = True

        return wf_instance

    def set_problems(self, pre_terms: List[PreTerm]):
        """
        Create Problem instance(s) and set corresponding attribute(s). As a
        side effect, the `term` attribute of each `PreTerm` instance is set.

        Parameters
        ----------
        pre_terms : list[PreTerm]
            List of `PreTerm` instances

        Notes
        -----
        If none of the `PreTerm` instances has tag 'mod', `pb_mod` attribute is
        left to None.
        Some parameters are hardcoded in this process, e.g.
            test function space = 'H1'
            basis of the FE space = 'lagrange'
            order of the polynomials to integrate = 2*fem_order + 1

        """

        has_nonlinear_term = self._has_nonlinear_term(pre_terms)

        # Define field and variables
        field_name = "field_{}".format(self.name)
        field = Field.from_args(field_name, np.float64, 'scalar',
                                self.region_dict[('omega', -1)],
                                approx_order=self.fem_order, space='H1',
                                poly_space_basis='lagrange')

        u_cst = FieldVariable(self.get_unknown_name('cst'), 'unknown', field)
        v_cst = FieldVariable(self.get_test_name('cst'), 'test', field,
                              primary_var_name=u_cst.name)
        u_mod = FieldVariable(self.get_unknown_name('mod'), 'unknown', field)
        v_mod = FieldVariable(self.get_test_name('mod'), 'test', field,
                              primary_var_name=u_mod.name)

        # Define integral
        integral = Integral('i', order=2 * self.fem_order + 1)

        # Construct terms from the list of pre_term
        terms_cst = []
        terms_mod = []
        for pre_term in pre_terms:
            if pre_term.tag == 'cst':
                pre_term.set_term(integral, self.region_dict, u=u_cst, v=v_cst)
                terms_cst.append(pre_term.term)
            else:
                pre_term.set_term(integral, self.region_dict, u=u_mod, v=v_mod)
                terms_mod.append(pre_term.term)
        terms_cst = Terms(terms_cst)
        terms_mod = Terms(terms_mod)

        # Define equations
        eq_cst_name = "eq_{}_cst".format(self.name)
        eq_cst = Equation(eq_cst_name, terms_cst)
        eq_mod_name = "eq_{}_mod".format(self.name)
        eq_mod = Equation(eq_mod_name, terms_mod)

        # Set problem(s)
        pb_cst_name = "pb_{}_cst".format(self.name)
        self.pb_cst = Problem(pb_cst_name, equations=Equations([eq_cst]),
                              active_only=True, functions=self.functions)
        if has_nonlinear_term:
            pb_mod_name = "pb_{}_mod".format(self.name)
            self.pb_mod = Problem(pb_mod_name, equations=Equations([eq_mod]),
                                  active_only=True, functions=self.functions)

        # Evaluate materials in physical quadrature points for the first time
        self.update_materials(tag='cst')

    @Decorators.check_lcbcs
    def apply_bcs(self, tag='both'):
        """
        Apply boundary conditions to the weak form `Problem` instance(s).
        The 'tag' argument is used to select either `pb_cst` or `pb_mod`. Use
        'tag'='both' to select both (default behavior).
        """

        self._check_pb_tag(tag)  # Safety checks
        tag_list, pb_list = self._get_tag_pb_lists(tag)

        # Setting up bcs following Sfepy's API
        for tag, pb in zip(tag_list, pb_list):
            ebcs = Conditions(
                [pair_ebc[tag] for pair_ebc in list(self.ebc_dict.values())])
            epbcs = Conditions(self.epbc_dict[tag])
            pb.set_bcs(ebcs=ebcs, epbcs=epbcs)
            variables = pb.get_initial_state()
            pb.time_update()
            variables.apply_ebc()
            self._active_bcs[tag] = True  # update dictionary of active bcs

    @Decorators.check_lcbcs
    def remove_bcs(self, tag='both'):
        """
        Remove boundary conditions to the weak form `Problem` instance(s).
        The 'tag' argument is used to select either `pb_cst` or `pb_mod`. Use
        'tag'='both' to select both (default behavior).
        """

        self._check_pb_tag(tag)  # Safety checks
        tag_list, pb_list = self._get_tag_pb_lists(tag)

        # Removing bcs following Sfepy's API
        for tag, pb in zip(['cst', 'mod'], pb_list):
            pb.remove_bcs()
            pb.get_initial_state()
            self._active_bcs[tag] = False  # update dictionary of active bcs

    def update_materials(self, tag='both', force=False):
        """
        Wrapper for `update_materials` from Sfepy `Problem` class.

        Parameters
        ----------
        tag : str
            'cst', 'mod' or 'both'
        force : bool
            If True, set materials data to None before calling
            `pb.update_materials`, which has the same effect as setting
            'mode=force' in `Material.time_update` method.

        """
        self._check_pb_tag(tag)
        tag_list, pb_list = self._get_tag_pb_lists(tag)
        for tag, pb in zip(tag_list, pb_list):
            if force:
                self._deplete_mat_datas(pb)
            pb.update_materials(verbose=False)

    def make_full_vec(self, reduced_vec: np.ndarray):
        """
        Reconstruct a full DOF vector satisfying E(P)BCs from a reduced DOF
        vector.
        """
        if not self.active_bcs['cst']:
            self.apply_bcs(tag='cst')
        variables = self.pb_cst.get_variables()
        full_vec = variables.make_full_vec(reduced_vec)
        return full_vec

    def make_reduced_vec(self, full_vec: np.ndarray):
        """
        Get the reduced DOF vector (with EBC DOFs removed) from a full vector.
        """
        if not self.active_bcs['cst']:
            self.apply_bcs(tag='cst')
        variables = self.pb_cst.get_variables()
        reduced_vec = variables.reduce_vec(full_vec)
        return reduced_vec

    @Decorators.check_lcbcs
    def set_mtx_vec_no_bc(self, tag='both'):
        """
        Set the 'no_bc' attributes of 'mtx_vec' (which is an instance of the
        `MatrixVector` class). 'tag' must be either 'cst', 'mod' or 'both'.

        todo: handle the case 'mod' terms are only vectors.
        """
        self._check_pb_tag(tag)
        tag_list, pb_list = self._get_tag_pb_lists(tag)
        mtx_vec = self.mtx_vec
        for tag, pb in zip(tag_list, pb_list):
            key_mtx = 'mtx_no_bc_{}'.format(tag)
            key_vec = 'vec_no_bc_{}'.format(tag)
            if not hasattr(mtx_vec, key_mtx) or not hasattr(mtx_vec, key_vec):
                raise ValueError(
                    f"'mtx_vec' does not have attr. ({key_mtx}, {key_vec})")
            if self.active_bcs[tag]:
                self.remove_bcs(tag=tag)
            ev = pb.get_evaluator()
            vec_r = pb.get_variables().get_state(reduced=False)
            mtx = ev.eval_tangent_matrix(vec_r, mtx=None, is_full=True).copy()
            rhs = -ev.eval_residual(vec_r, is_full=True).copy()
            setattr(mtx_vec, key_mtx, mtx)
            setattr(mtx_vec, key_vec, rhs)

    def set_mtx_vec_bc_reduced(self, tag='both'):
        """
        Set the 'bc_reduced' attributes of 'mtx_vec' (which is an instance of
        the `MatrixVector` class). 'tag' must be either 'cst', 'mod' or 'both'.

        todo: handle the case 'mod' terms are only vectors.
        """
        self._check_tag(tag)
        tag_list, pb_list = self._get_tag_pb_lists(tag)
        mtx_vec = self.mtx_vec
        for tag, pb in zip(tag_list, pb_list):
            key_mtx = 'mtx_bc_reduced_{}'.format(tag)
            key_vec = 'vec_bc_reduced_{}'.format(tag)
            if not hasattr(mtx_vec, key_mtx) or not hasattr(mtx_vec, key_vec):
                raise ValueError(
                    f"'mtx_vec' does not have attr. ({key_mtx}, {key_vec})")
            if not self.active_bcs[tag]:
                self.apply_bcs(tag=tag)
            ev = pb.get_evaluator()
            vec_r = pb.get_variables().get_state(reduced=True, force=True)
            mtx = ev.eval_tangent_matrix(vec_r, mtx=None, is_full=False).copy()
            rhs = -ev.eval_residual(vec_r, is_full=False).copy()
            setattr(mtx_vec, key_mtx, mtx)
            setattr(mtx_vec, key_vec, rhs)

    @Decorators.check_lcbcs
    def set_mtx_vec_bc_full(self, tag='both'):
        """
        Set the 'bc_full' attributes of 'mtx_vec' (which is an instance of the
        `MatrixVector` class). 'tag' must be either 'cst', 'mod' or 'both'.

        todo: handle the case 'mod' terms are only vectors.
        """
        self._check_tag(tag)
        tag_list, pb_list = self._get_tag_pb_lists(tag)
        mtx_vec = self.mtx_vec
        for tag, pb in zip(tag_list, pb_list):
            key_mtx = 'mtx_bc_full_{}'.format(tag)
            key_vec = 'vec_bc_full_{}'.format(tag)
            if not hasattr(mtx_vec, key_mtx) or not hasattr(mtx_vec, key_vec):
                raise ValueError(
                    f"'mtx_vec' does not have attr. ({key_mtx}, {key_vec})")
            if getattr(mtx_vec, 'mtx_no_bc_{}'.format(tag)) is None:
                self.set_mtx_vec_no_bc(tag=tag)
            mtx_no_bc = getattr(mtx_vec, 'mtx_no_bc_{}'.format(tag))
            vec_no_bc = getattr(mtx_vec, 'vec_no_bc_{}'.format(tag))
            if not self.active_bcs[tag]:
                self.apply_bcs(tag=tag)
            mtx = mtx_vec._make_mtx_bc_full_from_mtx_no_bc(mtx_no_bc, pb)
            vec = mtx_vec._make_vec_bc_full_from_mtx_vec_no_bc(
                mtx_no_bc, vec_no_bc, pb)
            setattr(mtx_vec, key_mtx, mtx)
            setattr(mtx_vec, key_vec, vec)

    def get_field(self):
        if self.pb_cst is None:
            raise ValueError("Sfepy 'Field' has not yet been instantiated.")
        field_name = "field_{}".format(self.name)
        return self.pb_cst.fields.get(field_name)

    def get_unknown_name(self, tag: str) -> str:
        if tag not in ['cst', 'mod']:
            raise ValueError(f"'{tag}' is not a valid tag, must be either "
                             f"'cst' or 'mod'.")
        return 'u_{}_{}'.format(self.name, tag)

    def get_test_name(self, tag: str) -> str:
        if tag not in ['cst', 'mod']:
            raise ValueError(f"'{tag}' is not a valid tag, must be either "
                             f"'cst' or 'mod'.")
        return 'v_{}_{}'.format(self.name, tag)

    def create_sfepy_internal_mesh(self, pre_mesh: Union[str, Mesh]) -> Mesh:
        """
        Create a Sfepy `Mesh` instance and return it. The input `pre_mesh` must
        already be an instance of `sfepy.discrete.fem.mesh.Mesh` in 1D. For the
        2D or 3D case, it is the name of an already existing .vtk mesh file.
        """

        if self.dim == 1:
            if not isinstance(pre_mesh, Mesh):
                raise TypeError(f"In the one-dimensional case, 'mesh' should "
                                f"be an instance of {Mesh}")
            sfepy_mesh = pre_mesh

        else:
            if not isinstance(pre_mesh, str):
                raise TypeError("In the two- and three-dimensional case, "
                                "'mesh' should be a string")
            mesh_path_name = make_absolute_mesh_name(pre_mesh)
            sfepy_mesh = Mesh.from_file(mesh_path_name)

        return sfepy_mesh

    def fill_ebc_dict(self, pre_ebc_dict: dict):
        """
        Fill the 'ebc_dict' from user specified Dirichlet boundary conditions.

        Parameters
        ----------
        pre_ebc_dict : dict
            Dictionary that is used to assign some essential boundary conditions
            to some topological entities. Its keys are a subset of
            `region_dict`'s ones and its values are either numbers or functions.

        Notes
        -----
        In order to match Sfepy's syntax, function signatures should be
            def func(ts, coors, **kwargs)

        In view of Sfepy's handling of essential boundary conditions, it is
        necessary to define pairs of ebcs (for 'u_cst' and 'u_mod').

        """

        if pre_ebc_dict is None:
            pre_ebc_dict = {}

        # Fill ebc_dict
        for key, val in pre_ebc_dict.items():
            ebc_pair = {'cst': None, 'mod': None}
            for tag in ['cst', 'mod']:
                ebc_name = 'ebc_{}{}_{}'.format(key[0], key[1], tag)
                unknown_name = self.get_unknown_name(tag)

                if isinstance(val, Number):  # val is a single number
                    ebc = EssentialBC(ebc_name, self.region_dict[key],
                                      {'%s.all' % unknown_name: val})

                else:  # val is a function
                    ebc_func_name = 'ebc_func_{}{}'.format(key[0], key[1])
                    ebc_func = Function(ebc_func_name, val)
                    ebc = EssentialBC(ebc_name, self.region_dict[key],
                                      {'%s.all' % unknown_name: ebc_func})

                ebc_pair[tag] = ebc

            self.ebc_dict[key] = ebc_pair

    def fill_epbc_dict(self, pre_epbc_list: List[list]):
        """
        Fill the 'epbc_dict' form user specified periodic boundary conditions.

        Parameters
        ----------
        pre_epbc_list : List[list]
            List containing pairs of region keys to form periodic boundary
            conditions.
        """

        if pre_epbc_list is None:
            pre_epbc_list = []

        region_dict = self.region_dict

        # Fill epbc_dict
        for tag in ['cst', 'mod']:
            epbc_list = []
            for pre_epbc in pre_epbc_list:  # loop over all pairs of regions
                epbc_name = 'epbc_{}{}-{}{}_{}'.format(pre_epbc[0][0],
                                                       pre_epbc[0][1],
                                                       pre_epbc[1][0],
                                                       pre_epbc[1][1],
                                                       tag)
                unknown_name = self.get_unknown_name(tag)
                regions = [region_dict[pre_epbc[0]], region_dict[pre_epbc[1]]]
                epbc_list.append(PeriodicBC(
                    epbc_name, regions,
                    {'%s.all' % unknown_name: '%s.all' % unknown_name},
                    match=pre_epbc[2]))

            self.epbc_dict[tag] = epbc_list

    def fill_region_dict(self, domain: FEDomain, physical_group_ids: List[int],
                         dim_func_entities: List[tuple]):
        """
        Fill the 'region_dict' based on:
        1) the physical groups present in the mesh;
        2) the topological entities selected via a function of the coordinates.
        Each entry of 'region_dict' is then an instance of the class
        `sfepy.discrete.common.region.Region`.

        Parameters
        ----------
        domain : FEDomain
            Sfepy Domain instance.
        physical_group_ids : list
            List of all the physical groups (int) readable from the mesh.
        dim_func_entities : list
            List of triplets (entity dimension, function selection, tag).

        """

        # Omega and Gamma regions
        self.region_dict[('omega', -1)] = domain.create_region('omega', 'all')
        self.region_dict[('gamma', -1)] = domain.create_region(
            'gamma', 'vertices of surface', 'facet')

        topological_entities_counter = {'vertices': 0,
                                        'edges': 0,
                                        'facets': 0,
                                        'subomegas': 0}

        for group_id in physical_group_ids:
            self._fill_region_dict_from_group_id(domain, group_id,
                                                 topological_entities_counter)

        for entity_dimension, function_selection, tag in dim_func_entities:
            self._fill_region_dict_from_dim_func(domain, entity_dimension,
                                                 function_selection, tag,
                                                 topological_entities_counter)

    def get_qps_coors_in_region(self, region_key=('omega', -1)):
        """
        Fetch quadrature points coordinates in given region.

        Parameters
        ----------
        region_key : tuple, optional
            Region where to fetch the quadrature points. The default is
            ('omega', -1), which corresponds to the whole domain.

        Returns
        -------
        qps_coors : np.ndarray
            Numpy array containing the coordinates of the qps in the specified
            region.

        """
        region = self.region_dict[region_key]
        qps = get_physical_qps(region, self.integral)
        qps_coors = qps.values
        return qps_coors

    def get_pickable_args_dict(self):
        """
        Construct a dictionary that
        (i) can be used to create a new weak form with `from_scratch` method;
        (ii) can be serialized and thus save using pickle.
        The second requirement implies that user defined functions are not
        registered in the process. We get rid of all 'pre_term's and replace
        them by a dummy 'dw_laplace' term.

        Returns
        -------
        args_dict : dict

        Notes
        -----
        This function is a wrapper for `_reconstruct_args_dict` and
        `_remove_functions_from_args_dict`.

        """
        args_dict = self._reconstruct_args_dict()
        _remove_functions_from_args_dict(args_dict)
        args_dict['pre_terms'] = [PreTerm('dw_laplace')]
        return args_dict

    def _reconstruct_args_dict(self):
        """
        Mirror of `from_scratch` method, i.e. recreate the `args_dict`
        parameter from the instantiated weak form. This method does not apply to
        one-dimensional weak forms.

        Returns
        -------
        args_dict : dict
            Dictionary that was used to create to present instance of WeakForm.

        See Also
        --------
        from_scratch : create weak form instance from an `args_dict`.

        Notes
        -----
        Some items cannot be recovered (e.g. 'dim_func_entities') or can only be
        partially reconstructed (e.g. 'pre_ebc_dict' when the dirichlet boundary
        condition is given by a function).
        """

        dim = self.dim
        if dim == 1:
            raise ValueError(
                "Cannot call 'reconstruct_args_dict' on 1D weak forms!")
        pre_mesh = self.field.domain.mesh.name
        name = self.name
        dim_func_entities = []  # Cannot be recovered
        pre_ebc_dict = {}
        for region_key in self.ebc_dict:
            ebc = self.ebc_dict[region_key]['cst']
            val = list(ebc.dofs.values())[0]
            pre_ebc_dict[region_key] = val
        fem_order = self.fem_order
        pre_terms = []
        for pre_term in self.pre_terms:
            pre_terms.append(copy(pre_term))
            pre_terms[-1].term = None
        is_exterior = self._is_exterior

        args_dict = {
            'dim': dim,
            'pre_mesh': pre_mesh,
            'name': name,
            'dim_func_entities': dim_func_entities,
            'pre_ebc_dict': pre_ebc_dict,
            'fem_order': fem_order,
            'pre_terms': pre_terms,
            'is_exterior': is_exterior
        }

        return args_dict

    def _fill_region_dict_from_group_id(self, domain: FEDomain, group_id: int,
                                        topological_entities_counter: dict):
        """
        Fill the `region_dict` dictionary with a single additional item based
        on the provided physical group id.

        Parameters
        ----------
        domain : `sfepy.discrete.fem.domain.FEDomain`
            Sfepy Domain instance.
        group_id : int
            Physical group id.
        topological_entities_counter : dict
            Dictionary containing the current number of regions with a certain
            topology created.

        """

        if 0 <= group_id < 100:  # vertex
            vertex = domain.create_region('vertex%d' % group_id,
                                          'vertices of group %d' % group_id,
                                          kind='vertex')
            self.region_dict[('vertex', group_id)] = vertex
            topological_entities_counter['vertices'] += 1

        elif 100 <= group_id < 200:  # edge
            edge = domain.create_region('edge%d' % group_id,
                                        'vertices of group %d' % group_id,
                                        kind='edge')
            self.region_dict[('edge', group_id)] = edge
            topological_entities_counter['edges'] += 1

        elif 200 <= group_id < 300:  # facet
            facet = domain.create_region('facet%d' % group_id,
                                         'vertices of group %d' % group_id,
                                         kind='facet')
            self.region_dict[('facet', group_id)] = facet
            topological_entities_counter['facets'] += 1

        elif group_id >= 300:  # subomega
            subomega = domain.create_region('subomega%d' % group_id,
                                            'cells of group %d' % group_id,
                                            kind='cell')
            self.region_dict[('subomega', group_id)] = subomega
            topological_entities_counter['subomegas'] += 1

        else:
            raise ValueError("'group_id' must be a positive integer.")

    def _fill_region_dict_from_dim_func(self, domain: FEDomain, ent_dim: int,
                                        func: callable, tag: int,
                                        topological_entities_counter: dict):
        """
        Fill the `region_dict` dictionary with a single additional item based
        on the provided (entity dimension, function selection) pair. This
        method is to be called after `_fill_region_dict_from_group_id`, because
        it checks whether keys are already listed in the 'region_dict'.

        Parameters
        ----------
        domain : `sfepy.discrete.fem.domain.FEDomain`
            Sfepy Domain instance.
        ent_dim : int
            Dimension of the selected topological entity.
        func : function
            Function specifying the geometry to be selected, e.g.
            def func(coors, domain=None):
                return np.where(np.linalg.norm(coors, axis=1) < 1)[0]
        tag : int
            Tag assigned to the topological entity.
        topological_entities_counter : dict
            Dictionary containing the current number of regions with a certain
            kind created.
        """

        func_name = func.__name__
        for key in self.region_dict.keys():
            if tag == key[1]:
                raise ValueError("tag {} is already assigned to some "
                                 "topological entity".format(tag))

        if ent_dim == 0:  # vertex
            assert 0 <= tag < 100
            key = ('vertex', tag)
            topological_entities_counter['vertices'] += 1
            region = domain.create_region('vertex%d' % tag,
                                          'vertices by %s' % func_name,
                                          kind='vertex',
                                          functions={func_name: func})
        elif ent_dim == 1 and self.dim == 3:  # edge
            assert 100 <= tag < 200
            key = ('edge', tag)
            topological_entities_counter['edges'] += 1
            region = domain.create_region('edge%d' % tag,
                                          'vertices by %s' % func_name,
                                          kind='edge',
                                          functions={func_name: func})
        elif self.dim - ent_dim == 1:  # facet
            assert 200 <= tag < 300
            key = ('facet', tag)
            topological_entities_counter['facets'] += 1
            region = domain.create_region('facet%d' % tag,
                                          'vertices by %s' % func_name,
                                          kind='facet',
                                          functions={func_name: func})
        elif self.dim == ent_dim:  # subomega
            assert tag >= 300
            key = ('subomega', tag)
            topological_entities_counter['subomegas'] += 1
            region = domain.create_region('subomega%d' % tag,
                                          'cells by %s' % func_name,
                                          kind='cell',
                                          functions={func_name: func})
        else:
            raise ValueError(
                "Invalid dimension specified in 'dim_func_entities'.")

        if key in self.region_dict:
            raise ValueError("Overwritten key of 'region_dict'!")

        self.region_dict[key] = region

    def _get_tag_pb_lists(self, tag: str):
        """
        Create a list of tags and a corresponding list of `Problem`
        instances for convenience.
        """
        pb_cst = self.pb_cst
        pb_mod = self.pb_mod
        if tag == 'cst':
            pb_list = [pb_cst]
            tag_list = [tag]
        elif tag == 'mod':
            pb_list = [pb_mod]
            tag_list = [tag]
        else:  # 'tag' is then expected to be set to 'both'
            pb_list = [pb_cst, pb_mod] if pb_mod is not None else [pb_cst]
            tag_list = ['cst', 'mod']
        return tag_list, pb_list

    @classmethod
    def _check_pre_ebc_dict_format(cls, pre_ebc_dict: dict):
        """Check that `pre_ebc_dict` has the right format."""
        if pre_ebc_dict is None:
            pre_ebc_dict = {}
        for key, val in pre_ebc_dict.items():
            if not isinstance(key, tuple):
                raise ValueError("Keys of 'pre_ebc_dict' must be pairs "
                                 "specifying the topological entity kind and "
                                 "tag")
            if key[0] not in valid_topological_kinds_ebc:
                raise ValueError(f"'{key[0]}' is not a valid kind for ebc")
            if not isinstance(val, Number) and not callable(val):
                raise ValueError("Values of 'pre_ebc_dict' must either be a "
                                 "single number or a function")

    @classmethod
    def _check_pre_epbc_list_format(cls, pre_epbc_list: list):
        """Check that `pre_epbc_list` has the right format."""
        if pre_epbc_list is None:
            pre_epbc_list = []
        for pre_epbc in pre_epbc_list:
            if not isinstance(pre_epbc, list):
                raise ValueError("Elements of 'pre_epbc_list' must be lists")
            if len(pre_epbc) != 3:
                raise ValueError("Elements of 'pre_epbc_list' must of length 2")
            for region_key in pre_epbc[:-1]:
                if len(region_key) != 2:
                    raise ValueError("region keys must be tuple of length 2")
                if region_key[0] not in valid_topological_kinds_ebc:
                    raise ValueError(
                        f"{region_key[0]} is not a valid kind for epbc")
            if pre_epbc[-1] not in valid_matching_epbc:
                raise ValueError(f"{pre_epbc[-1]} is not a valid matching key, "
                                 f"must be in {valid_matching_epbc}")

    @staticmethod
    def _deplete_mat_datas(pb):
        for mat in pb.get_materials():
            mat.datas = {}

    @staticmethod
    def _has_nonlinear_term(pre_terms: Union[List[PreTerm], None] = None):
        """Check if `pre_terms` list contains at least one pre_term with
        tag 'mod'."""
        for pre_term in pre_terms:
            if pre_term.tag == 'mod':
                return True
        return False

    def _check_pb_tag(self, tag: str):
        """Check that 'tag' and current `Problem` attributes comply."""
        self._check_tag(tag)
        if self.pb_cst is None:
            raise ValueError("Problem instances  have not been created yet. "
                             "Consider calling 'set_problems' beforehand.")
        if tag == 'mod' and self.pb_mod is None:
            raise ValueError("'pb_mod' is not set")

    @staticmethod
    def _check_tag(tag: str):
        """Check that 'tag' is either 'cst', 'mod' or 'both'."""
        valid_tags = ['cst', 'mod', 'both']
        if tag not in valid_tags:
            raise ValueError(f"Invalid 'tag' (valid tags are {valid_tags})")

    @staticmethod
    def _check_args_before_merging(wf_int: WeakForm,
                                   wf_ext: WeakForm,
                                   region_key_int: tuple,
                                   region_key_ext: tuple):
        """Check that arguments of `from_two_weak_forms` are compliant for
        proceeding to the merging of the two weak forms."""

        # Check dim
        if wf_int.dim != wf_ext.dim:
            raise ValueError("The two weak forms have different dimensions!")

        # Check fem_order
        if wf_int.fem_order != wf_ext.fem_order:
            raise ValueError(
                "The two weak forms have different 'fem_order' attributes!")

        # Check if specified regions exist
        if region_key_int not in wf_int.region_dict:
            raise KeyError(f"Region with key '{region_key_int}' is not included"
                           f"in the interior weak form 'region_dict' with keys "
                           f"{list(wf_int.region_dict.keys())}")
        if region_key_ext not in wf_ext.region_dict:
            raise KeyError(f"Region with key '{region_key_ext}' is not included"
                           f" in the exterior weak form 'region_dict' with keys"
                           f" {list(wf_ext.region_dict.keys())}")

        # Check that both regions have the same number of DOFs
        n_dofs_int = wf_int.region_dict[region_key_int].vertices.shape[0]
        n_dofs_ext = wf_ext.region_dict[region_key_ext].vertices.shape[0]
        if n_dofs_int != n_dofs_ext:
            raise ValueError(
                "Connecting regions do not have the same number of DOFs!")

        # Check that both weak forms have the same (non)linearity settings
        if (wf_int.pb_mod is None) is not (wf_ext.pb_mod is None):
            raise ValueError(
                "One weak form is nonlinear while the other one is linear!")

        # Check that the weak forms have different names
        if wf_int.name == wf_ext.name:
            raise NameError(f"The two provided weak forms have the same name "
                            f"('{wf_int.name}'): abort combination.")

    def _fix_pb_without_state_variable(self):
        """
        For tag = ('cst', 'mod'), add a zero dummy `PreTerm` instance to the
        list `self.pre_terms` if it does not feature any 'matrix term'.

        Notes
        -----
        This is a (dirty) fix to the issue encountered when trying to assemble
        rhs vector from `Problem` instances with no matrix terms. Adding a
        zero matrix term solves the issue. So far, there is no other fix
        available.

        """

        for tag in ['cst', 'mod']:
            has_matrix_term = False
            has_tag = False
            for pre_term in self.pre_terms:
                if pre_term.tag != tag:
                    continue
                has_tag = True
                if is_matrix_term(pre_term.name):
                    has_matrix_term = True
                    break

            if not has_matrix_term and has_tag:
                def mat_zero(ts, coors, mode=None, vec_qp=None, **kwargs):
                    if mode != 'qp': return
                    val = np.zeros(coors.shape[0])
                    return {'val': val.reshape(-1, 1, 1)}
                dummy_term = PreTerm(
                    'dw_laplace', mat=mat_zero, prefactor=0, tag=tag)
                self.pre_terms.append(dummy_term)


def _remove_functions_from_args_dict(args_dict: dict):
    """
    Remove functions references in dictionary `args_dict` in order to make it
    'pickable'. This operation is done in-place.

    Parameters
    ----------
    args_dict : dict
            Dictionary that was used to create to present instance of WeakForm.
    """

    # Replace Dirichlet BC functions by np.nan
    for key, val in args_dict['pre_ebc_dict'].items():
        if callable(val):
            args_dict['pre_ebc_dict'][key] = np.nan

    # Replace material functions by None
    for term in args_dict['pre_terms']:
        term.mat = None
        term.tag = 'cst'
        term.term = None


class _EquationsFix(Equations):
    """
    Class inheriting from Sfepy `Equations`, where the __init__ method is
    overriden in order to sort the variables ('int' goes first, 'ext' goes
    second). Without this fix, the order chosen by Sfepy is random. This caused
    spurious bugs when 'cst' and 'mod' variables end up in different orders.
    """
    def __init__(self, equations):
        Container.__init__(self, equations)
        self.variables = Variables(self.collect_variables())

        # Fix ordered_state & ordered_virtual
        if 'ext' in self.variables.ordered_state[0].split('_'):
            self.variables.ordered_state.reverse()
            self.variables.ordered_virtual.reverse()

        self.materials = Materials(self.collect_materials())
        self.domain = self.get_domain()
        self.active_bcs = set()
        self.collect_conn_info()
