# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:51:10 2022

Utilities for displaying Meshes & IGA domains.

@author: hlevy
"""

from femtoscope import MESH_DIR
import meshio


def plot_iga(igafile, mesh_dir='', selector='matplotlib', **kwargs):
    """
    Display IGA domain (extension '.iga') created with igakit.

    Parameters
    ----------
    igafile : str
        Name of the .iga file.
    mesh_dir : String, optional
        Path to directory where the meshfile is located. If empty string,
        the program looks for the mesh in global MESH_DIR directory.
        The default is ''.
    selector : str, optional
        Plot function selector {'mayavi' , 'matplotlib'}.
        The default is 'matplotlib'.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.

    """

    from igakit.plot import plt as iplt
    from sfepy.discrete.iga.domain import IGDomain
    from pathlib import Path
    from sfepy.discrete.iga import plot_nurbs
    import matplotlib.pyplot as plt

    verbose = kwargs.get("verbose", False)

    # getting igafile full-file-name
    if not mesh_dir: # search file in the meshes directory (default behaviour)
        fullFileName = MESH_DIR / igafile
    else:
        mesh_dir = Path(mesh_dir)
        fullFileName = mesh_dir / igafile
    filename_iga = str(fullFileName.with_suffix('.iga'))

    # Setting a Domain instance
    domain = IGDomain.from_file(filename_iga)
    nurbs_sfepy = domain.nurbs
    nurbs_igakit = nurbs_sfepy._to_igakit()

    if selector=='matplotlib':
        iplt.use('matplotlib') # plt.use('matplotlib') or plt.use('mayavi')
        iplt.figure(1)
        iplt.plot(nurbs_igakit)
        iplt.figure(2)
        iplt.cplot(nurbs_igakit)
        iplt.kplot(nurbs_igakit)
        iplt.show()

    elif selector=='mayavi':
        iplt.use('mayavi') # plt.use('matplotlib') or plt.use('mayavi')
        iplt.figure(1)
        iplt.plot(nurbs_igakit)
        iplt.figure(2)
        iplt.cplot(nurbs_igakit)
        iplt.kplot(nurbs_igakit)
        iplt.show()

    elif verbose:
        print("Invalid 'selector' argument! Choose {'mayavi', 'matplotlib'}\n")
        print("Cannot view IGA domain")

    # NURBS, Bézier, etc. plots brought to you by sfepy
    figsize = (12.0, 7.0)
    label = True
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    plot_nurbs.plot_parametric_mesh(axs[0, 0], nurbs_sfepy.knots)
    plot_nurbs.plot_control_mesh(axs[0, 1], nurbs_sfepy.cps, label=label)
    plot_nurbs.plot_bezier_mesh(axs[1, 0], nurbs_sfepy.cps, nurbs_sfepy.conn,
                                nurbs_sfepy.degrees, label=label)
    plot_nurbs.plot_iso_lines(axs[1, 1], nurbs_igakit, color='b', n_points=100)
    for i in range(2):
        for j in range(2):
            axs[i, j].set_aspect('equal', 'box')
    axs[0, 0].set_title("Parametric mesh of NURBS given by its knots")
    axs[0, 1].set_title("Control mesh of NURBS given by its control points")
    axs[1, 0].set_title(
        "Bezier mesh of NURBS given by its control points and connectivity")
    axs[1, 1].set_title("Iso-lines in Greville abscissae coordinates")

    if nurbs_sfepy.dim==1:
        figsize = (12.0, 7.0)
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        plot_nurbs.plot_parametric_mesh(axs[0], nurbs_sfepy.knots)
        plot_nurbs.plot_nurbs_basis_1d(axs[1], nurbs_igakit,n_points=500,
                                       x_axis='parametric')
        axs[0].set_title("(a) Parametric mesh of NURBS given by its knots")
        axs[1].set_title("(b) NURBS basis functions in 1D")
        fig.tight_layout(pad=2.0)


def plot_mesh(meshfile, mesh_dir='', selector='gmsh', **kwargs):
    """
    Wrapper for mesh-plot functions plot_mesh_gmsh and plot_mesh_matplotlib.

    Parameters
    ----------
    meshfile : String
        Name of the mesh file to read (can omit '.mesh') without full path.
    mesh_dir : String, optional
        Path to directory where the meshfile is located. If empty string,
        the program looks for the mesh in global `MESH_DIR` directory.
        The default is ''.
    selector : String
        Plot function selector {'gmsh' , 'matplotlib'}. The default is 'gmsh'.

    Other Parameters
    ----------------
    verbose : bool
        Display user's information. The default is False.
    save  : bool
        Whether or not the plots are saved. The default is False.

    """

    verbose = kwargs.get("verbose", False)

    mpl_fail = False
    if selector=='matplotlib':
        coors = _mesh2py(meshfile, mesh_dir=mesh_dir)[0]
        if (coors.shape[1]==3) and not ((coors[:, 2]==0).all()):
            print("Cannot plot 3D mesh with matplotlib") if verbose else None
            mpl_fail = True
        else:
            plot_mesh_matplotlib(meshfile, mesh_dir=mesh_dir, **kwargs)

    elif selector=='gmsh' or mpl_fail:
        plot_mesh_gmsh(meshfile, mesh_dir=mesh_dir, **kwargs)

    elif verbose:
        print("Invalid 'selector' argument! Choose {'gmsh', 'matplotlib'} \n")
        print("Cannot view mesh")


def plot_mesh_gmsh(meshfile, mesh_dir=''):
    """
    Plots 2D and 3D mesh using Gmsh GUI.

    Parameters
    ----------
    meshfile : String
        Name of the mesh file to read (can omit '.vtk') without full path.
    mesh_dir : String, optional
        Path to directory where the meshfile is located. If empty string,
        the program looks for the mesh in global `DATA_DIR` directory.
        The default is ''.

    """

    from pathlib import Path
    import gmsh

    # getting meshfile full-file-name
    if not mesh_dir: # search file in the meshes directory (default behaviour)
        fullFileName = MESH_DIR / meshfile
    else:
        mesh_dir = Path(mesh_dir)
        fullFileName = mesh_dir / meshfile
    fullFileName = str(fullFileName.with_suffix('.vtk'))

    gmsh.initialize()
    gmsh.open(fullFileName)
    gmsh.fltk.run()
    gmsh.finalize()


def plot_mesh_matplotlib(meshfile, mesh_dir='', **kwargs):
    """
    Plots 2D mesh using matplotlib built-in routines.

    Parameters
    ----------
    meshfile : String
        Name of the mesh file to read (can omit '.vtk') without full path.
    data_dir : String, optional
        Path to directory where the meshfile is located. If empty string,
        the program looks for the mesh in global `DATA_DIR` directory.
        The default is ''.

    Other Parameters
    ----------------
    title : str
        Title to be given to the figure.
        The default is the positional argument [name]
    xlabel : str
        Name of the x-axis. The default is ''
    ylabel : str
        Name of the y-axis. The default is ''

    """
    import matplotlib.pyplot as plt

    # Keyword Arguments handling
    title = kwargs.get('title', meshfile)
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')

    # Get mesh data
    coors, triang = _mesh2py(meshfile, mesh_dir=mesh_dir)
    X, Y = coors[:, 0], coors[:, 1]

    # Creating figure
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    ax.margins(0.07)

    # Plot triangles in the background
    ax.triplot(X, Y, triang, color='black', lw=0.4)

    # Adding labels & title
    ax.set_xlabel(xlabel, fontweight ='bold', fontsize=15)
    ax.set_ylabel(ylabel, fontweight ='bold', fontsize=15)
    ax.set_title(title, fontweight='bold', fontsize=20)

    # Show plot
    plt.show()


def _mesh2py(meshfile, mesh_dir=''):
    """Utility for reading a mesh file"""
    from femtoscope.inout.meshfactory import get_meshsource
    fullfilename = get_meshsource(meshfile, mesh_dir=mesh_dir)
    reader = meshio.read(fullfilename)
    coors = reader.points
    triangles = reader.cells_dict['triangle']
    return coors, triangles
