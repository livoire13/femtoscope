# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:43:02 2024

Utility of 'weak_form' module.
"""

from numbers import Number
from typing import Union

from sfepy.discrete import Function, Material, Integral
from sfepy.terms import term_table
from sfepy.terms.terms import Term

from femtoscope.core import valid_topological_kinds
from femtoscope.misc.util import get_function_signature


class PreTerm:
    """
    User-provided data intended to later construct an instance of the
    `sfepy.terms.terms.Term` class, which is then stored as an attribute.

    Attributes
    ----------
    name : str
        Name of the Sfepy term. To see the full list, one can either check:
        - `sfepy.terms.term_table` variable
        - https://sfepy.org/doc-devel/terms_overview.html
    region_key : tuple
        Pair consisting of a topological kind (str) and a tag (int), e.g.
        ('facet', 200).
    tag : str
        If the PDE to be solved is linear:
            tag = 'cst'
        If the PDE to be solved is nonlinear and the term is not to be
        re-assembled at each Newton's iteration:
            tag = 'cst'
        If the PDE to be solved is nonlinear and the term is to be
        re-assembled at each Newton's iteration:
            tag = 'mod'
    prefactor : float
        Constant coefficient in front of the term.
    mat : function or Number
        Material associated to the term. See e.g.
        https://sfepy.org/doc-devel/users_guide.html#materials
    mat_kwargs : dict
        Dictionary for setting up the material's function keyword arguments.
        Only used if `mat` is a function and has default keywords arguments.
    term : Term
        Instance of Sfepy's `Term` class.

    Notes
    -----
    In order to match Sfepy's syntax, function signatures for `mat` should be
        def mat(ts, coors, mode=None, **kwargs)
    """

    def __init__(self, name: str, region_key=('omega', -1), tag='cst',
                 prefactor=1.0, mat=None, mat_kwargs=None):
        check_arguments(name, region_key, tag, prefactor, mat)
        self.name = name
        self.region_key = region_key
        self.tag = tag
        self.prefactor = prefactor
        self.mat = mat
        self.mat_kwargs = mat_kwargs if mat_kwargs is not None else {}
        self.term = None

    @property
    def sfepy_material(self):
        return self.term.get_materials(join=True)[0]

    def set_term(self, integral: Integral, region_dict: dict, u=None, v=None):
        """
        Construct an instance of the Sfepy `Term` class and set it as the
        'term' attribute.

        Parameters
        ----------
        integral : Integral
            Sfepy `Integral` instance.
        region_dict : dict
            Dictionary of regions with
            keys: pairs topological kind - tag, e.g. ('subomega', 301).
            values: corresponding Sfepy `Region` instances.
        u : sfepy.discrete.FieldVariable, optional
            Unknown function of the problem. The default is None.
        v : sfepy.discrete.FieldVariable, optional
            Test function of the problem. The default is None.

        """
        arg_types = correct_arg_types(term_table.get(self.name).arg_types)
        use_state = 'state' in arg_types
        use_material = 'material' in arg_types or \
                       ('opt_material' in arg_types and callable(self.mat))
        term_extra_args = {u.name: u, v.name: v}

        if use_material:
            mat_func = Function('mat_func', self.mat)
            mat_name = "mat_{}_{}{}".format(self.name, *self.region_key)
            mat_name = mat_name.split('-')[0]  # handling the 'omega' exception
            mat = Material(mat_name, kind='stationary', function=mat_func)
            mat.set_extra_args(**self.mat_kwargs)
            term_extra_args[mat_name] = mat

            if use_state:
                term_str = "{}({}.val, {}, {})".format(self.name, mat_name,
                                                       v.name, u.name)
                t = Term.new(term_str, integral, region_dict[self.region_key],
                             **term_extra_args)

            else:
                term_str = "{}({}.val, {})".format(self.name, mat_name, v.name)
                t = Term.new(term_str, integral, region_dict[self.region_key],
                             **term_extra_args)
        else:
            if isinstance(self.mat, Number):
                self.prefactor *= self.mat

            if use_state:
                term_str = "{}({}, {})".format(self.name, v.name, u.name)
                t = Term.new(term_str, integral, region_dict[self.region_key],
                             **term_extra_args)

            else:
                term_str = "{}({})".format(self.name, v.name)
                t = Term.new(term_str, integral, region_dict[self.region_key],
                             **term_extra_args)

        t.sign = self.prefactor
        self.term = t


def check_arguments(name: str, region_key: tuple, tag: str, prefactor: float,
                    mat: Union[Number, callable, None]):
    """Check that arguments provided to `PreTerm` constructor comply."""
    _check_name(name)
    _check_region_key(region_key)
    _check_tag(tag)
    _check_prefactor(prefactor)
    _check_mat(mat, tag)


def correct_arg_types(arg_types: tuple):
    """From a given term `arg_types`, return the tuple containing 'virtual'."""
    if isinstance(arg_types[0], str):
        return arg_types
    else:
        for tup in arg_types:
            if 'virtual' in tup:
                return tup


def is_matrix_term(name: str):
    """From a given term's name, tell if it corresponds to a matrix term."""
    _check_name(name)
    arg_types = correct_arg_types(term_table.get(name).arg_types)
    if 'state' in arg_types:
        return True
    else:
        return False


def _check_name(name: str):
    if name not in term_table:
        raise NameError(
            f"Name {name} is not a valid Sfepy term name: {term_table.keys()}")
    if name.split('_')[0] != 'dw':
        raise NameError("Term name should start with 'dw_'")


def _check_region_key(region_key: tuple):
    if not isinstance(region_key, tuple):
        raise ValueError("'region_key' must be a tuple")
    if len(region_key) != 2:
        raise ValueError("'region_key' must be a tuple of length 2")
    if not (isinstance(region_key[0], str) or isinstance(region_key[1], int)):
        raise ValueError("Wrong types for 'region_key'")
    if region_key[0] not in valid_topological_kinds:
        raise ValueError(f"Invalid topological kind. "
                         f"Must be in {valid_topological_kinds}")


def _check_tag(tag: str):
    if tag not in ['cst', 'mod']:
        raise ValueError("'tag' must be either 'cst' or 'mod'")


def _check_prefactor(prefactor: float):
    if not isinstance(prefactor, Number):
        raise ValueError("'prefactor' must be a number")


def _check_mat(mat: Union[Number, callable, None], tag: str):
    if not callable(mat) and not isinstance(mat, Number) and mat is not None:
        raise ValueError("'mat' must be either a function, a number or None")
    if tag == 'mod':
        if not callable(mat):
            raise ValueError("Term with tag 'mod' should have a callable "
                             "material in order to perform the iteration "
                             "update.")
        args_dict = get_function_signature(mat)
        if 'vec_qp' not in args_dict:
            raise KeyError(f"'vec_qp' must appear in the list of default "
                           f"arguments of function {mat}")
