# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:50:31 2024

Module dedicated to the creation of VTK/VTU files.

Notes
-----
The handling of meshio version to get function 'vtk_to_meshio_type' is used in
https://github.com/tianyikillua/paraview-meshio/blob/master/meshioPlugin.py

"""

from pathlib import Path
from typing import Union

import meshio
import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay

try:
    from pyevtk.hl import pointsToVTK, unstructuredGridToVTK
except ImportError:
    from evtk.hl import pointsToVTK, unstructuredGridToVTK

if float(meshio.__version__[:3]) < 3.3:  # Handling meshio version
    vtk_to_meshio_type = meshio._vtk.vtk_to_meshio_type
else:
    vtk_to_meshio_type = meshio.vtk._vtk.vtk_to_meshio_type

from femtoscope import TMP_DIR
from femtoscope.misc.util import get_date_string

meshio_to_vtk_type = {v: k for k, v in vtk_to_meshio_type.items()}


def create_unstructured_vtu(coors: np.ndarray, vars_dict: dict,
                            path_name: Union[None, str] = None):
    """
    Create a VTU file containing point coordinates together with scalar fields.
    The data is represented on an unstructured grid.

    Parameters
    ----------
    coors : np.ndarray
        Point coordinates in dimension 2 or 3.
    vars_dict : dict
        Dictionary containing the scalar fields to be saved.
    path_name : str, optional
        Absolute path name of the file to be created. No need to provide the
        file extension. The default is None, in which the file is created in
        femtoscope's `TMP_DIR` directory.

    See Also
    --------
    create_structured_vtk
    """

    path_name = make_absolute_vtk_name(path_name)
    path_name = str(Path(path_name).stem)  # 'pointsToVTK' takes no extension
    perform_input_checks(coors, vars_dict)

    x = np.ascontiguousarray(coors[:, 0])
    y = np.ascontiguousarray(coors[:, 1])
    z = np.ascontiguousarray(coors[:, 2]) if coors.shape[1] == 2 \
        else np.zeros_like(x)

    pointsToVTK(path_name, x, y, z, data=vars_dict)


def create_structured_vtk(coors: np.ndarray, vars_dict: dict, cells: np.ndarray,
                          path_name: Union[None, str] = None):
    """
    Create a VTK file containing point coordinates together with scalar fields.
    The data is represented on a structured grid.

    Parameters
    ----------
    coors : np.ndarray
        Point coordinates in dimension 2 or 3.
    vars_dict : dict
        Dictionary containing the scalar fields to be saved.
    cells : np.ndarray
        Connectivity table of mesh used (which nodes belong to which cells).
    path_name : str
        Absolute path name of the file to be created. No need to provide the
        file extension. The default is None, in which the file is created in
        femtoscope's `TMP_DIR` directory.

    Notes
    -----
    At the time this function was written, Sfepy's equivalent function did not
    include data points associated with higher-order DOFs.

    See Also
    --------
    create_unstructured_vtu
    """

    path_name = make_absolute_vtk_name(path_name)
    perform_input_checks(coors, vars_dict)

    # Handling dimension
    if coors.shape[1] == 2:
        z = np.zeros((coors.shape[0], 1))
        coors = np.concatenate((coors, z), axis=1)
        cell_type = 'triangle'
    else:
        cell_type = 'tetra'

    # Fill in all required fields
    cells_dict = {cell_type: cells}
    cell_idx = meshio_to_vtk_type[cell_type]
    point_data = vars_dict.copy()
    size_data = coors.shape[0]
    node_groups = np.zeros(size_data)
    point_data['node_groups'] = node_groups
    cell_data_array = cell_idx * np.ones(cells.shape[0])
    cell_data = {'mat_id': [cell_data_array]}
    field_data = None
    point_sets_array = np.arange(0, coors.shape[0], dtype=np.float64)
    point_sets = {'0': point_sets_array}
    cell_sets_array = np.arange(0, cells.shape[0], dtype=np.uint32)
    cell_sets = {str(cell_idx): [cell_sets_array]}
    file_format = 'vtk'

    # Create VTK file using meshio
    meshio._helpers.write_points_cells(
        path_name, coors, cells_dict, point_data=point_data,
        cell_data=cell_data, field_data=field_data, point_sets=point_sets,
        cell_sets=cell_sets, file_format=file_format)


def create_connectivity_table(coors: np.ndarray):
    """Create connectivity table from `coors` in 2D or 3D."""
    _check_coors_shape(coors)
    if coors.shape[1] == 2:  # Dimension = 2
        cells = Triangulation(coors[:, 0], coors[:, 1]).triangles
    else:  # Dimension = 3
        cells = Delaunay(coors).simplices
    return cells


def make_absolute_vtk_name(path_name: Union[None, str]):
    """
    Return an absolute path name as a string.
    If `path_name` is not an absolute path name, use `TMP_DIR` as the default
    location. If `path_name` is left as None, the file name will be the current
    date-time.
    """
    if path_name is None:
        path_name = get_date_string()
    else:
        assert isinstance(path_name, str), "'path_name' must be a string"
    path_name = Path(path_name)
    if not path_name.is_absolute():
        path_name = Path(TMP_DIR / path_name)
    return str(path_name.with_suffix('.vtk'))


def perform_input_checks(coors: np.ndarray, vars_dict: dict):
    """Launch all checks """
    _check_coors_shape(coors)
    _check_vars_dict(coors, vars_dict)


def _check_coors_shape(coors: np.ndarray):
    """Check the compliance of `coors` array for methods in this module."""
    if 1 in coors.shape:
        coors = coors.squeeze()
    if len(coors.shape) != 2:
        raise ValueError(
            f"'coors' array must have 2 dimensions (not {len(coors.shape)})")
    dim = coors.shape[1]
    if dim not in (2, 3):
        raise ValueError(f"'coors' must have 2 or 3 coordinates (not {dim}).")


def _check_vars_dict(coors: np.ndarray, vars_dict: dict):
    """Check that values of `vars_dict` have the same length as coordinates
    array."""
    expected_length = coors.shape[0]
    for key, var in vars_dict.items():
        if not isinstance(var, np.ndarray):
            raise TypeError(f"Entry with key '{key}' is not a numpy array.")
        if len(var.shape) != 1:
            raise ValueError(f"Entry with key '{key}' is not one-dimensional.")
        if var.shape[0] != expected_length:
            raise ValueError(f"Entry with key '{key}' has wrong length.")
