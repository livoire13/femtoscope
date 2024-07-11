# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:11:19 2022

Tools to create and read VTK files using meshio or (py)evtk.

@author: hlevy
"""

import numpy as np
try:
    from pyevtk.hl import pointsToVTK, unstructuredGridToVTK
except ImportError:
    from evtk.hl import pointsToVTK, unstructuredGridToVTK
from femtoscope import RESULT_DIR
from femtoscope.misc.util import date_string


def createUnstructuredVTK(x, y, z, vars_dic, **kwargs):
    """
    Create a vtk file containing point coordinates and the values of one or
    several scalar fields at those points.

    Parameters
    ----------
    x : 1d numpy array
        x-coordinates of the points.
    y : 1d numpy array
        y-coordinates of the points.
    z : 1d numpy array
        z-coordinates of the points.
    vars_dic : dictionary
        Fields to be saved in the VTK file.
        For example fields = {1_Xelec' : Ex , '2_Yelec' : Ey}, where V, Ex, Ey
        are 1D numpy arrays with same length as x, y, z. Quote from the
        'points.py' example: "keys are sorted before exporting, hence it is
        useful to prefix a number to determine an order".

    Other Parameters
    ----------------
    name : String
        Name of the vtkfile to be saved. The default is the current date-time.

    Returns
    -------
    fullFileName : String
        Absolute pathname of the saved vtk.

    """

    # Keyword arguments handling
    name = kwargs.get('name', '')

    if not name:
        name = date_string()
    fullFileName = RESULT_DIR / name
    fullFileName = str(fullFileName.with_suffix(''))

    if z is None:
        z = np.zeros(x.shape)

    # make coordinate-arrays 'C-contiguous'
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)

    # keys are sorted before exporting, hence it is useful to prefix
    # a number to determine an order
    pointsToVTK(fullFileName, x, y, z, data=vars_dic)

    return fullFileName + '.vtu'


# Meshio compatibility version hack inspired from:
# https://github.com/tianyikillua/paraview-meshio/blob/master/meshioPlugin.py
def createStructuredVTK(x, y, z, fields, cells, **kwargs):
    """
    Create a .vtk file using the meshio module (it is part of femtoscope
    dependencies). The idea is to save FEM outputs the exact same way as sfepy,
    but with more flexibility. In particular, we want to be able to save the
    complete solution of the FEM computation instead of only its values at the
    finite element vertices (for higher order approximation).

    Parameters
    ----------
    x : 1d numpy array
        x-coordinates of the points.
    y : 1d numpy array
        y-coordinates of the points.
    z : 1d numpy array
        z-coordinates of the points. Can be None.
    fields : dictionary
        Fields to be saved in the VTK file.
        For example fields = {1_Xelec' : Ex , '2_Yelec' : Ey}, where V, Ex, Ey
        are 1D numpy arrays with same length as x, y, z. Quote from the
        'points.py' example: "keys are sorted before exporting, hence it is
        useful to prefix a number to determine an order".
    cells : numpy array
        Connectivity table for cells (nodes belonging to each cell).

    Other Parameters
    ----------------
    name : String
        Name of the vtkfile to be saved. The default is the current date-time.

    Returns
    -------
    fullFileName : String
        Absolute pathname of the saved vtk.

    """

    import meshio
    # handling meshio version
    if float(meshio.__version__[:3]) < 3.3:
        vtk_to_meshio_type = meshio._vtk.vtk_to_meshio_type
    else:
        vtk_to_meshio_type = meshio.vtk._vtk.vtk_to_meshio_type

    # dictionary reverse
    meshio_to_vtk_type = {v: k for k, v in vtk_to_meshio_type.items()}

    # Keyword arguments handling
    name = kwargs.get('name', '')
    if not name:
        name = date_string()
    fullFileName = RESULT_DIR / name
    fullFileName = str(fullFileName) + '.vtk'

    # points
    if z is None:
        z = np.zeros(x.shape)
    x = x[:, np.newaxis] if len(x.shape)==1 else x
    y = y[:, np.newaxis] if len(y.shape)==1 else y
    z = z[:, np.newaxis] if len(z.shape)==1 else z
    points = np.concatenate((x, y, z), axis=1)
    cell_type = 'triangle' if (z==0).all() else 'tetra'
    cells_dic = {cell_type : cells}
    cell_idx = meshio_to_vtk_type[cell_type]

    # point_data
    point_data = fields.copy()
    size_data = len(list(point_data.values())[0])
    node_groups = np.zeros(size_data)
    point_data["node_groups"] = node_groups

    # cell data
    cell_data_array = cell_idx * np.ones(cells.shape[0])
    cell_data = {'mat_id' : [cell_data_array]}

    # field data
    field_data = None

    # point sets
    point_sets_array = np.arange(0, x.shape[0], dtype=np.int64)
    point_sets = {'0' : point_sets_array}

    # cell sets
    cell_sets_array = np.arange(0, cells.shape[0], dtype=np.uint32)
    cell_sets = {str(cell_idx) : [cell_sets_array]}

    # file format
    file_format = 'vtk'

    # call the meshio built-in vtk writer
    meshio._helpers.write_points_cells(
        fullFileName, points, cells_dic, point_data=point_data,
        cell_data=cell_data, field_data=field_data, point_sets=point_sets,
        cell_sets=cell_sets, file_format=file_format)

    return fullFileName

# def createStructuredVTK(x, y, z, fields, conn, dim, **kwargs):
#     """
#     Create a .vtk file using the meshio module (it is part of femtoscope
#     dependencies). The idea is to save FEM outputs the exact same way as sfepy,
#     but with more flexibility. In particular, we want to be able to save the
#     complete solution of the FEM computation instead of only its values at the
#     finite element vertices (for higher order approximation).

#     Parameters
#     ----------
#     x : 1d numpy array
#         x-coordinates of the points.
#     y : 1d numpy array
#         y-coordinates of the points.
#     z : 1d numpy array
#         z-coordinates of the points. Can be None.
#     fields : dictionary
#         Fields to be saved in the VTK file.
#         For example fields = {1_Xelec' : Ex , '2_Yelec' : Ey}, where V, Ex, Ey
#         are 1D numpy arrays with same length as x, y, z. Quote from the
#         'points.py' example: "keys are sorted before exporting, hence it is
#         useful to prefix a number to determine an order".
#     conn : numpy array
#         Connectivity table for cells (nodes belonging to each cell).
#     dim : int
#         Dimension (2 or 3).

#     Other Parameters
#     ----------------
#     name : String
#         Name of the vtkfile to be saved. The default is the current date-time.

#     Returns
#     -------
#     fullFileName : String
#         Absolute pathname of the saved vtk.

#     """

#     import meshio
#     # handling meshio version
#     if float(meshio.__version__[:3]) < 3.3:
#         vtk_to_meshio_type = meshio._vtk.vtk_to_meshio_type
#     else:
#         vtk_to_meshio_type = meshio.vtk._vtk.vtk_to_meshio_type

#     # dictionary reverse
#     meshio_to_vtk_type = {v: k for k, v in vtk_to_meshio_type.items()}

#     # Keyword arguments handling
#     name = kwargs.get('name', '')
#     if not name:
#         name = date_string()
#     fullFileName = RESULT_DIR / name
#     fullFileName = str(fullFileName) + '.vtk'

#     # points
#     if z is None:
#         z = np.zeros(x.shape)
#     x = x[:, np.newaxis] if len(x.shape)==1 else x
#     y = y[:, np.newaxis] if len(y.shape)==1 else y
#     z = z[:, np.newaxis] if len(z.shape)==1 else z
#     points = np.concatenate((x, y, z), axis=1)
#     cell_type = get_meshio_type(conn, dim)
#     cells_dic = {cell_type : conn}
#     cell_idx = meshio_to_vtk_type[cell_type]

#     # point_data
#     point_data = fields.copy()
#     size_data = len(list(point_data.values())[0])
#     node_groups = np.zeros(size_data)
#     point_data["node_groups"] = node_groups

#     # cell data
#     cell_data_array = cell_idx * np.ones(conn.shape[0])
#     cell_data = {'mat_id' : [cell_data_array]}

#     # field data
#     field_data = None

#     # point sets
#     point_sets_array = np.arange(0, x.shape[0], dtype=np.int64)
#     point_sets = {'0' : point_sets_array}

#     # cell sets
#     cell_sets_array = np.arange(0, conn.shape[0], dtype=np.uint32)
#     cell_sets = {str(cell_idx) : [cell_sets_array]}

#     # file format
#     file_format = 'vtk'

#     # call the meshio built-in vtk writer
#     meshio._helpers.write_points_cells(
#         fullFileName, points, cells_dic, point_data=point_data,
#         cell_data=cell_data, field_data=field_data, point_sets=point_sets,
#         cell_sets=cell_sets, file_format=file_format)

#     return fullFileName

# def get_meshio_type(conn, dim):
#     """Infer meshio cell type from the number of nodes per element (using the
#     connectivity table `conn` and the dimension `dim`). Visit
#     https://github.com/nschloe/meshio/blob/main/doc/cell_types.tex for the
#     nomenclature and element schematics."""
#     nconn  = conn.shape[1]
#     if dim == 2:
#         if nconn == 3:
#             cell_type = 'triangle'
#         elif nconn == 4:
#             cell_type = 'quad'
#         elif nconn == 6:
#             cell_type = 'triangle6'
#         elif nconn == 7:
#             cell_type = 'triangle7'
#         elif nconn == 8:
#             cell_type == 'quad8'
#         elif nconn == 9:
#             cell_type = 'quad9'
#         else:
#             raise ValueError("Cannot assign a 2D meshio cell type!")
#     elif dim == 3:
#         if nconn == 4:
#             cell_type = 'tetra'
#         elif nconn == 5:
#             cell_type = 'pyramid'
#         elif nconn == 6:
#             cell_type = 'wedge'
#         elif nconn == 8:
#             cell_type = 'hexadron'
#         elif nconn == 10:
#             cell_type = 'tetra10'
#         elif nconn == 12:
#             cell_type = 'wedge12'
#         elif nconn == 13:
#             cell_type = 'pyramid13'
#         elif nconn == 14:
#             cell_type = 'pyramid14'
#         elif nconn == 15:
#             cell_type = 'wedge15'
#         elif nconn == 20:
#             cell_type = 'hexahedron20'
#         elif nconn == 24:
#             cell_type = 'hexahedron24'
#         elif nconn == 27:
#             cell_type = 'hexahedron27'
#         else:
#             raise ValueError("Cannot assign a 3D meshio cell type!")
#     else:
#         raise ValueError("dimension can only be 2 or 3")
#     return cell_type