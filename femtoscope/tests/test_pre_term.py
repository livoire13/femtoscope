# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:50:12 2024
Testing of `PreTerm` class.
"""

import pytest
from femtoscope.core.pre_term import PreTerm


@pytest.fixture()
def input_args_dict():
    return {'name': 'dw_laplace',
            'region_key': ('subomega', 300),
            'tag': 'cst',
            'prefactor': 2.0,
            'mat': None}


@pytest.mark.parametrize('name', ['dw_femtoscope', 'ev_grad'])
def test_check_name(input_args_dict, name):
    input_args_dict['name'] = name
    with pytest.raises(NameError):
        PreTerm(**input_args_dict)


@pytest.mark.parametrize('region_key', [('node', 1), 'facet200', (1, 'vertex')])
def test_check_region_key(input_args_dict, region_key):
    input_args_dict['region_key'] = region_key
    with pytest.raises(ValueError):
        PreTerm(**input_args_dict)


def test_check_tag(input_args_dict):
    input_args_dict['tag'] = 'foo'
    with pytest.raises(ValueError):
        PreTerm(**input_args_dict)


def test_check_prefactor(input_args_dict):
    input_args_dict['prefactor'] = [1, 2]
    with pytest.raises(ValueError):
        PreTerm(**input_args_dict)


def test_check_mat1(input_args_dict):
    input_args_dict['mat'] = 'material'
    with pytest.raises(ValueError):
        PreTerm(**input_args_dict)


def test_check_mat2(input_args_dict):
    def material(ts, coors, mode=None, dummy=None):
        return None
    input_args_dict['mat'] = material
    input_args_dict['tag'] = 'mod'
    with pytest.raises(KeyError):
        PreTerm(**input_args_dict)


def test_check_mat_kwargs(input_args_dict):
    def material(ts, coors, mode=None, dummy=None):
        return None
    input_args_dict['mat'] = material
    input_args_dict['mat_kwargs'] = {'mock': None}
    input_args_dict['tag'] = 'mod'
    with pytest.raises(KeyError):
        PreTerm(**input_args_dict)
