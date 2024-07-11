# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:33:57 2022

Collection of heteroclite utility functions.

@author: hlevy
"""

import os, os.path
import errno
import inspect
from inspect import signature
import numpy as np
from matplotlib.legend_handler import HandlerLine2D


def concatenate(list_of_lists):
    """Concatenate several lists into one and return it"""
    new_list = []
    for i in list_of_lists:
        new_list.extend(i)
    return new_list

def get_function_kwargs(function):
    """Return a list of the keyword arguments in a function signature"""
    names = []
    sig = signature(function)
    for param in sig.parameters.values():
        if (param.default is not param.empty) and (param.name != 'mode'):
            names.append(param.name)
    return names

def get_default_args(function):
    """Return a list of the default arguments in a function signature"""
    dic = {}
    args, varargs, varkw, defaults = inspect.getargspec(function)
    if defaults:
        defargs = args[-len(defaults):]
        for k in range(len(defaults)):
            if defargs[k] != 'mode':
                dic[defargs[k]] = defaults[k]
    return dic

def none_dic_from_names(names):
    """Initialize a dictionary with `names` as keys and None as values."""
    none_dic = {}
    for name in names:
        none_dic[name] = None
    return none_dic

def is_sorted(a):
    return np.all(a[:-1] <= a[1:])

def listit(t):
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

def moving_average(x, w, mode='same'):
    return np.convolve(x, np.ones(w), mode) / w

def numpyit(t):
    """Convert nested tuples/lists into numpy array."""
    return np.array(listit(t))

def copy_list_of_arrays(list_of_arrays):
    """Copy a list of arrays and return the copy-list."""
    new_list = []
    for array in list_of_arrays:
        new_list.append(array.copy())
    return new_list

def remove_empty_lines(string_with_empty_lines):
    """Remove empty lines from a string."""
    lines = string_with_empty_lines.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_without_empty_lines = ""
    for line in non_empty_lines:
        string_without_empty_lines += line + "\n"
    return string_without_empty_lines

def date_string():
    """Return the current date-time up to the second"""
    from datetime import datetime
    return datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

def latex_float(f, nd):
    """Return scientific notation of a float `f` with `nd` decimals."""
    from math import floor, log10
    exponent = floor(log10(f))
    base = f / 10**exponent
    str_to_format = "{:.%df}" %nd
    if nd > 0:
        base_mod = float(str_to_format.format(base))
    else:
        base_mod = int(str_to_format.format(base))
    if exponent == 0:
        return r"{0}".format(base_mod)
    else:
        return r"{0} \times 10^{{{1}}}".format(base_mod, exponent)

class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle,xdescent,
                       ydescent, width,height, fontsize, trans):
        xx = 0.6*height
        return super(SymHandler, self).create_artists(
            legend, orig_handle,xdescent, xx, width, height, fontsize, trans)

def write_geo_params(geofile, param_dic, geo_dir=None):
    """
    Modify a .geo file according to the key/value pairs specified by the user
    in the `param_dic` dictionary.

    Parameters
    ----------
    geofile : str
        Name of the .geo file.
    param_dic : dict
        Dictionary containing key/value pairs to be modified.
    geo_dir : str, optional
        Directory where the .geo file is located. The default is None, in which
        case the file will be sought in the MESH_DIR/geo directory.

    Returns
    -------
    fullFileName : str
        Absolute pathname of the .geo file.

    """
    from pathlib import Path
    from femtoscope import MESH_DIR

    if geo_dir is not None:
        geo_dir = Path(geo_dir)
        fullFileName = geo_dir / geofile
    else:
        fullFileName = MESH_DIR / 'geo' / geofile
    fullFileName = str(fullFileName.with_suffix('.geo'))

    param_keys = list(param_dic.keys())
    separator = ' '

    # Write parameters to .geo file
    old_file = open(fullFileName, 'r')
    lines = old_file.readlines()
    old_file.close()
    new_file = open(fullFileName, 'w')

    for line in lines:
        curr_str = line.strip("\n")
        if curr_str and curr_str.split()[0] in param_keys:
            newline = curr_str.split()
            newline[2] = str(param_dic[curr_str.split()[0]]) + ';'
            newline = separator.join(newline) + '\n'
            new_file.write(newline)
        else:
            new_file.write(line)

    new_file.close()
    return fullFileName

def merge_dicts(dict1, dict2):
    """In-place merging of dict2 into dict1. If the key already exists, keep
    the corresponding val from dict1."""
    keys1 = list(dict1.keys())
    for key in list(dict2.keys()):
        if key not in keys1:
            dict1[key] = dict2[key]

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
