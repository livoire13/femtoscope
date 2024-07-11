# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:29:27 2022

Utilities for displaying simulation results.

@author: hlevy
"""

import numpy as np
from femtoscope import RESULT_DIR
from pathlib import Path


def plot_from_vtk(vtkfile, result_dir='', plotfunc='pyvista', **kwargs):
    """
    Wrapper for all functions that generate plots straight from .vtk file.

    Parameters
    ----------
    vtkfile : str
        Name of the .vtk file containing FEA results.
    result_dir : str, optional
        Path to directory where the meshfile is located. If empty string,
        the program looks for the mesh in global `RESULT_DIR` directory.
        The default is ''.
    plotfunc : str
        Plot function to be used. The default is 'mayavi'.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.

    Returns
    -------
    None.

    """

    verbose = kwargs.get("verbose", False)

    assert type(vtkfile)==str, "First argument should be a string"

    # getting vtkfile full-file-name
    if not result_dir: # search file in result directory (default behaviour)
        fullFileName = RESULT_DIR / vtkfile
    else:
        result_dir = Path(result_dir)
        fullFileName = result_dir / vtkfile
    fullFileName = str(fullFileName.with_suffix('.vtk'))

    if verbose:
        print("Reading file at: \n")
        print(fullFileName)

    plot_methods = ['mayavi', 'pyvista', 'sfepy', 'resview', 'matplotlib']
    if plotfunc=='mayavi':
        mayavi_plot(fullFileName, **kwargs)
    elif plotfunc=='pyvista':
        pyvista_plot(fullFileName, **kwargs)
    elif plotfunc=='sfepy':
        sfepy_postproc_plot(fullFileName, **kwargs)
    elif plotfunc=='resview':
        sfepy_resview_plot(fullFileName, **kwargs)
    else:
        print("not recognized 'plotfunc' argument \n")
        print(plot_methods)


def mayavi_plot(fullFileName, **kwargs):
    """
    Plot the data contained within a .vtk or .vtu data file using the
    mayavi.mlab module, which is a higher-level module than bare vtk module.
    More information regarding the function's code can be found at
    https://stackoverflow.com/questions/56401123/reading-and-plotting-vtk-file-data-structure-with-python

    Parameters
    ----------
    fullFileName : str
        Absolute path of the .vtk file to be displayed.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.


    """

    from mayavi import mlab
    from mayavi.modules.surface import Surface

    # verbose
    verbose = kwargs.get('verbose', False)
    print("** Producing mayavi plot **") if verbose else None

    # create a new figure, grab the engine that's created with it
    fig = mlab.figure()
    engine = mlab.get_engine()

    # open the vtk file, let mayavi figure it all out
    vtk_file_reader = engine.open(fullFileName)

    # plot surface corresponding to the data
    surface = Surface()
    engine.add_filter(surface, vtk_file_reader)

    # block until figure is closed
    #mlab.options.offscreen = True
    mlab.show()


def pyvista_plot(fullFileName, **kwargs):
    """
    Plot the data contained within a .vtk or .vtu data file using the pyvista
    module, which is a higher-level module than bare vtk module. More
    information regarding the function's code can be found at
    https://stackoverflow.com/questions/56401123/reading-and-plotting-vtk-file-data-structure-with-python

    Parameters
    ----------
    fullFileName : str
        Absolute path of the .vtk file to be displayed.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.

    """

    import pyvista as pv
    pv.set_plot_theme("document") # for white background
    sargs = dict(height=0.5, vertical=True, position_x=0.0, position_y=0.25,
                  fmt="%.2f", n_labels=5, title='', label_font_size=50)

    # verbose
    verbose = kwargs.get('verbose', False)
    print("** Producing pyvista plot **") if verbose else None
    save = kwargs.get('save', False)

    # read the data
    try:
        data = pv.read(fullFileName)
    except FileNotFoundError:
        fullFileName = str(Path(fullFileName).with_suffix('.vtu'))
        data = pv.read(fullFileName)

    # plot the data with an automatically created Plotter
    if save:
        for name in data.array_names:
            if name != 'mat_id' and name != 'node_groups':
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(data, scalars=name, show_scalar_bar=True,
                                 scalar_bar_args=sargs, show_edges=False,
                                 cmap="CET_L8")
                plotter.add_text(name)
                plotter.view_xy()
                file = (Path(RESULT_DIR) / 'plot' / name).with_suffix('.png')
                plotter.window_size = 4000, 4000
                plotter.ren_win.SetOffScreenRendering(1)
                plotter.screenshot(file)
    else:
        for name in data.array_names:
            if name != 'mat_id' and name != 'node_groups':
                plotter = pv.Plotter()
                plotter.add_mesh(data, scalars=name, show_scalar_bar=True,
                                 cmap="CET_L8")
                plotter.add_text(name)
                plotter.view_xy()
                plotter.show()


def sfepy_postproc_plot(fullFileName, **kwargs):
    """
    Use sfepy built-in post-processing routines, in particular 'postproc.py'

    Parameters
    ----------
    fullFileName : str
        Absolute path of the .vtk file to be displayed.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.

    """

    import subprocess
    import sfepy

    # verbose
    verbose = kwargs.get('verbose', False)
    print("** Producing sfepy postproc plot **") if verbose else None

    # run the post-processing script with subprocess.run() function
    # and sfepy command-wrapper
    try:
        out = subprocess.run(["sfepy-run", "postproc", fullFileName])
        failed = (out.returncode != 0)
        print(out) if verbose>1 else None
    except FileNotFoundError:
        failed = True
        pass
    if failed: # command-wrapper failed
        if verbose:
            print("""sfepy-run command failed, trying with sfepy absolute
                  path...""")
        pp_path = Path(sfepy.__file__).parent.absolute()/'script'/'postproc'
        pp_path = str(pp_path.with_suffix('.py'))
        outbis = subprocess.run(["python", pp_path, fullFileName], shell=True)
        print(outbis) if verbose else None


def sfepy_resview_plot(fullFileName, **kwargs):
    """
    Use sfepy built-in post-processing routines, in particular 'resview.py'.

    Parameters
    ----------
    fullFileName : str
        Absolute path of the .vtk file to be displayed.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.

    """

    import subprocess
    import sfepy

    # verbose
    verbose = kwargs.get('verbose', False)
    print("** Producing sfepy resview plot **") if verbose else None

    pp_path = Path(sfepy.__file__).parent.absolute() / 'script' / 'resview'
    pp_path = str(pp_path.with_suffix('.py'))

    out = subprocess.run(["python", pp_path, fullFileName], shell=True)
    print(out) if verbose else None


def plot_from_data(X, Y, scalar_field, name, triang=None, **kwargs):
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

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Keyword Arguments handling
    title = kwargs.get('title', name)
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    unit = kwargs.get('unit', '')
    colormap = kwargs.get('colormap', 'viridis')

    scalar_field = np.squeeze(scalar_field) # remove possible useless dimension

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
    ax.set_xlabel(xlabel, fontweight ='bold', fontsize=15)
    ax.set_ylabel(ylabel, fontweight ='bold', fontsize=15)
    ax.set_title(title, fontweight='bold', fontsize=20)

    # Show plot
    plt.show()