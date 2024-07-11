# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:45:06 2024

Utilities for displaying FEM simulation results.

Useful information can be found at:
https://stackoverflow.com/questions/56401123/reading-and-plotting-vtk-file-data-structure-with-python
"""

import subprocess
from pathlib import Path

try:
    from mayavi import mlab
    from mayavi.modules.surface import Surface
except ImportError:
    pass
import pyvista as pv
import sfepy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from femtoscope.inout.vtkfactory import make_absolute_vtk_name


def handle_vtk_name(func):
    """Decorator to handle the `vtk_name` parameter."""
    def wrapper(vtk_name):
        path_name = make_absolute_vtk_name(vtk_name)
        return func(path_name)
    return wrapper


def display_from_vtk(vtk_name: str, engine='pyvista'):
    """
    Display the FEM results saved in a VTK file.

    Parameters
    ----------
    vtk_name : str
        Name of the VTK file containing the results.
    engine : str, optional
        Display engine. The default is 'pyvista'.

    See Also
    --------
    """

    if vtk_name is None:
        return

    available_engines = {
        'mayavi': display_with_mayavi,
        'pyvista': display_with_pyvista,
        'sfepy': display_with_sfepy,
        'resview': display_with_resview,
    }

    if engine not in available_engines:
        raise NameError(f"'{engine}' is not recognized as a display engine. "
                        f"Available engines are: {available_engines}")

    available_engines[engine](vtk_name)


def display_from_data(X, Y, scalar_field, name, triang=None, **kwargs):
    """
    Plots 2d FEM results from raw data, using matplotlib built-in routines.

    Parameters
    ----------
    X : 1d numpy array
        X-coordinates of the nodes constituting the mesh.
    Y : 1d numpy array
        Y-coordinates of the nodes constituting the mesh.
    scalar_field : 1d numpy array
        FEM-computed solution of PDE at nodes.
    name : String
        Becomes the title of the figure if keyword argument 'title' is not
        filled out.
    triang : numpy array of integers
        Connectivity table for triangles (nodes belonging to each triangle).
        The default is None.

    Other Parameters
    ----------------
    title : String
        Title to be given to the figure.
        The default is the positional argument [name]
    xlabel : String
        Name of the x-axis. The default is ''
    ylabel : String
        Name of the y-axis. The default is ''
    unit : String
        Unit of the scalar field being displayed. The default is ''
    colormap : String
        See https://matplotlib.org/stable/tutorials/colors/colormaps.html for
        colormap examples. The default is 'viridis'.

    """

    # Keyword Arguments handling
    title = kwargs.get('title', name)
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    unit = kwargs.get('unit', '')
    colormap = kwargs.get('colormap', 'viridis')

    scalar_field = np.squeeze(scalar_field)  # remove possible useless dimension

    # getting the min & max values
    levels = np.linspace(np.min(scalar_field), np.max(scalar_field), 500)

    # Creating figure
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    ax.margins(0.07)

    # Plot triangles in the background
    if triang is not None:
        ax.triplot(X, Y, triang, color='grey', lw=0.4)

    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Creating plot & colorbar
    tricont = ax.tricontourf(X, Y, scalar_field,
                             levels=levels, cmap=cmap)
    cbar = fig.colorbar(tricont, ax=ax, shrink=1, aspect=15)
    cbar.ax.set_ylabel(unit)

    # Adding labels & title
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=15)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=15)
    ax.set_title(title, fontweight='bold', fontsize=20)

    # Show plot
    plt.show()


@handle_vtk_name
def display_with_mayavi(vtk_name: str):
    """Display using mayavi"""
    # create a new figure, grab the engine that's created with it
    mlab.figure()
    engine = mlab.get_engine()

    # open the vtk file, let mayavi figure it all out
    vtk_file_reader = engine.open(vtk_name)

    # plot surface corresponding to the data
    surface = Surface()
    engine.add_filter(surface, vtk_file_reader)

    # block until figure is closed
    mlab.options.offscreen = True
    mlab.show()


@handle_vtk_name
def display_with_pyvista(vtk_name: str):
    """Display using pyvista."""

    pv.set_plot_theme("document")  # for white background
    data = pv.read(vtk_name)

    for name in data.array_names:
        if name in ('mat_id', 'node_groups'):
            continue
        plotter = pv.Plotter()
        plotter.add_mesh(data, scalars=name, show_scalar_bar=True,
                         cmap='plasma')
        plotter.add_text(name)
        plotter.view_xy()
        plotter.show()


@handle_vtk_name
def display_with_sfepy(vtk_name: str):
    """Display using sfepy built-in post process functions."""

    # run the post-processing script with subprocess.run() function
    # and sfepy command-wrapper
    try:
        out = subprocess.run(["sfepy-run", "postproc", vtk_name])
        failed = (out.returncode != 0)
    except FileNotFoundError:
        failed = True
        pass
    if failed:  # command-wrapper failed
        print("""sfepy-run command failed, trying with sfepy absolute
                  path...""")
    pp_path = Path(sfepy.__file__).parent.absolute() / 'script' / 'postproc'
    pp_path = str(pp_path.with_suffix('.py'))
    subprocess.run(["python", pp_path, vtk_name], shell=True)


@handle_vtk_name
def display_with_resview(vtk_name: str):
    """Display using resview."""
    pp_path = Path(sfepy.__file__).parent.absolute() / 'script' / 'resview'
    pp_path = str(pp_path.with_suffix('.py'))
    subprocess.run(["python", pp_path, vtk_name], shell=True)
