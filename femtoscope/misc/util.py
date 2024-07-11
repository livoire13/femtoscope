# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:02:02 2024
Collection of heteroclite utility functions.
"""

import inspect
from datetime import datetime
from typing import Union, Callable, Tuple

import numpy as np


def listit(t) -> list:
    """
    Convert nested tuples/lists into nested lists only.

    Parameters
    ----------
    t : Nested tuples/lists
        The object to convert (not in place).

    Returns
    -------
    Nested lists
        The initial object, but with lists instead of tuples.

    """
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def numpyit(t: Union[list, tuple]) -> np.ndarray:
    """Convert nested tuples/lists into numpy array."""
    return np.array(listit(t))


def concatenate_lists(list_of_lists) -> list:
    """Concatenate several lists into one and return it"""
    new_list = []
    for i in list_of_lists:
        new_list.extend(i)
    return new_list


def get_function_default_args(function: Callable) -> list:
    """
    Return a list of the default arguments in a function signature. Fails
    when the function is decorated.

    See Also
    --------
    get_function_signature
    """
    default_args_dict = {}
    args, varargs, varkw, defaults = inspect.getfullargspec(function)[:4]
    if defaults:
        defargs = args[-len(defaults):]
        for k in range(len(defaults)):
            default_args_dict[defargs[k]] = defaults[k]
    return default_args_dict


def get_function_signature(function: Callable) -> Tuple[str]:
    """
    Return a tuple containing the parameters appearing in the signature of a
    given function as strings. This works even when the function is decorated,
    provided one uses `functools.wraps` in the custom defined decorator as done
    in e.g. `femtoscope.physics.materials_library`

    See Also
    --------
    get_function_default_args
    femtoscope.physics.materials_library.check_mode_qp

    """

    func_signature = inspect.signature(function, follow_wrapped=True)
    return tuple(func_signature.parameters.keys())


def get_date_string() -> str:
    """Return the current date-time up to the second as a string."""
    return datetime.today().strftime('%Y-%m-%d_%H-%M-%S')