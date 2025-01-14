# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:59:06 2024
Module dedicated to the creation and manipulation of Gmsh meshes (& more!).
"""

import os
from copy import deepcopy
from inspect import cleandoc
from pathlib import Path

import gmsh
import meshio
import numpy as np
from numpy import cos, sin, sqrt, pi
from sfepy.base.base import output
from sfepy.discrete.fem import Mesh

from femtoscope import MESH_DIR
from femtoscope.misc.util import numpyit


class MeshingTools:
    """
    Construct and manipulate user-defined meshes through the Gmsh Python API.
    Gmsh Python API is initialized when the class is instantiated.
    Following Gmsh's terminology, a dimTag is a pair of integers (dim, tag) and
    a dimTags is a list of several dimTag.

    Attributes
    ----------
    dim : int
        Dimension of the mesh, 2 or 3.
    Rcut : float
        Truncation radius of the mesh.
    boundaries : list
        List of dimTags representing boundary shapes of dimension `dim`-1.
    sources : list
        List of dimTags representing source shapes of dimension `dim`.
    refinement_settings : list
        List of mesh refinement settings (see `create_subdomain` method), one
        per source shape.
    number_sources : int
        Number of source shapes currently defined in the Gmsh model.
    number_boundaries : int
        Number of boundary shapes currently defined in the Gmsh model.
    min_length : float
        Minimum distance between neighbouring points.
    geom : class gmsh.model.occ
        Shortcut.

    """

    class Decorators:
        """Inner class for defining decorators."""

        @staticmethod
        def check_dimension(dim):
            """Decorator for checking that the current `MeshingTools` instance
            has its dimension set to `dim`."""
            def decorator(func):
                def wrapper(self, *args, **kwargs):
                    if self.dim != dim:
                        self.close_gmsh()
                        raise ValueError(cleandoc("""Current instance of '{}'
                        has dim={} while method '{}' requires dim={}""".format(
                            type(self).__name__, self.dim, func.__name__, dim)))
                    return func(self, *args, **kwargs)
                return wrapper
            return decorator

    def __init__(self, dimension: int, Rcut=None):
        """
        Parameters
        ----------
        dimension : int
            Dimension of the mesh, 2 or 3.
        Rcut : float, optional
            Truncation radius of the mesh. The default is None.
        """

        # Check dimension
        assert dimension in [2, 3], "Dimension should be 2 or 3."

        # Set attributes
        self.dim = dimension
        self.Rcut = Rcut
        self.boundaries = []
        self.sources = []
        self.refinement_settings = []
        self.number_sources = 0
        self.number_boundaries = 0
        self.min_length = 1.3e-6
        self.geom = gmsh.model.occ

        # open gmsh
        self.open_gmsh()

    @staticmethod
    def open_gmsh():
        """Initialize the Gmsh Python API."""
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)

    @staticmethod
    def close_gmsh():
        """To be called when one is done using the Gmsh Python API."""
        gmsh.clear()
        gmsh.finalize()

    def generate_mesh(self, mesh_name: str, show_mesh=False, **kwargs) -> str:
        """
        Generate the mesh and save it.

        Parameters
        ----------
        mesh_name : str
            Name of the mesh file to be created. If not specified as an absolute
            path name, will be stored in the `MESH_DIR` directory.
        show_mesh : bool, optional
            Open the Gmsh graphical user interface to inspect the mesh.
            The default is False.

        Other Parameters
        ----------------
        ignored_tags : list, optional
            List of physical groups not to be included in the final .vtk file.
            The default is [].
        convert : bool, optional
            Convert the .vtk mesh file created by Gmsh into a .vtk file readable
            by Sfepy. The default is True.
        embed_origin : bool, optional
            Force the embedding of the origin (i.e. point of coordinates x=0,
            y=0, [z=0]) into the mesh. The default is False.
        origin_rf : list, optional
            Settings for refinement around the origin, namely
            [size_min, size_max, dist_min, dist_max].
            The default is [1e-2, 1e-1, 0.1, 1.0].
        unique_boundary_tags : bool, optional
            Assign a single physical group will be created for all elementary
            entities that are parts of a shape's boundary. The default is True.
        cylindrical_symmetry : bool, optional
            Whether the 2D mesh corresponds to a cylindrically symmetric
            problem. The default is False.
        verbose : bool, optional
            The default is False.

        Returns
        -------
        mesh_path_name : str
            Absolute path name of the created mesh file.

        """

        # Get keyword arguments
        ignored_tags = kwargs.get('ignored_tags', [])
        convert = kwargs.get('convert', True)
        embed_origin = kwargs.get('embed_origin', False)
        origin_rf = kwargs.get('origin_rf', [1e-2, 1e-1, 0.1, 1.0])
        unique_boundary_tags = kwargs.get('unique_boundary_tags', True)
        cylindrical_symmetry = kwargs.get('cylindrical_symmetry', False)
        verbose = kwargs.get('verbose', False)
        origin_tag = None

        self.create_subdomain()
        self.geom.synchronize()

        if cylindrical_symmetry:
            self._apply_cylindrical_symmetry()
        if embed_origin:
            origin_tag = self.geom.addPoint(0, 0, 0)
            self.geom.synchronize()

        # Refinement settings
        self._apply_refinement_settings(embed_origin, origin_tag, origin_rf)

        if embed_origin:  # Force the embedding of the origin to the mesh
            is_on_curve = self.dim == 2 and cylindrical_symmetry
            self._embed_origin_to_mesh(origin_tag, is_on_curve)

        # Assign physical groups
        self._assign_physical_groups(embed_origin, origin_tag,
                                     unique_boundary_tags)

        # Actual mesh generation
        gmsh.model.mesh.generate(dim=self.dim)
        gmsh.model.mesh.removeDuplicateNodes()

        if verbose > 1:  # Inspect the log:
            log = gmsh.logger.get()
            print("Logger has recorded {} lines".format(len(log)))
            gmsh.logger.stop()

        if show_mesh:  # open Gmsh graphical interface
            gmsh.fltk.run()
        mesh_path_name = self._save(mesh_name, convert, ignored_tags, verbose)
        self.close_gmsh()

        return mesh_path_name

    def create_subdomain(self,
                         cell_size_min=0.1, cell_size_max=0.1,
                         dist_min=0.0, dist_max=1.0,
                         number_pts_per_curve=1000):
        """
        Creates a subdomain from the shapes currently in an open gmsh window.
        Shapes already present in previous subdomains will not be added to the
        new one. This subdomain will be labeled by an index value corresponding
        to the next available integer value.
        The size of mesh cells at distances less than 'DistMin' from the
        boundary of this subdomain will be 'SizeMin', while at distances
        greater than 'DistMax' cell size is 'SizeMax'. Between 'DistMin'
        and 'DistMax' cell size will increase linearly.

        Parameters
        ----------
        cell_size_min : float, optional
            Minimum size of the mesh cells. The default is 0.1.
        cell_size_max : float, optional
            Maximum size of the mesh cells. The default is 0.1.
        dist_min : float, optional
            At distances less than this value, the cell size is set to its
            minimum. The default is 0.0.
        dist_max : float, optional
            At distances greater than this value, the cell size is set to its
            maximum. The default is 1.0.
        number_pts_per_curve : int, optional
            Number of points used to define each curve. The default is 1000.

        """

        # Save sources, remove duplicates, and update the number of sources.
        self.sources.append(self.geom.getEntities(dim=self.dim))
        del self.sources[-1][:self.number_sources]
        self.number_sources += len(self.sources[-1])

        if self.sources[-1]:

            # Save boundary information
            self.boundaries.append(self.geom.getEntities(dim=self.dim - 1))
            del self.boundaries[-1][:self.number_boundaries]
            self.number_boundaries += len(self.boundaries[-1])

            # Record refinement settings for this subdomain.
            self.refinement_settings.append([cell_size_min, cell_size_max,
                                             dist_min, dist_max,
                                             number_pts_per_curve])

        else:  # new entry is empty
            del self.sources[-1]

    @Decorators.check_dimension(3)
    def create_inversed_exterior_domain_3d(self):
        """
        Create a solid sphere whose surface DOFs match the last created shape
        boundary. Because the exterior mesh has to be spatially separated from
        all other geometries, its center of mass is set three radii away from
        the interior domain along the x-axis.

        Returns
        -------
        inversed_exterior_domain_3d : list
            List containing tuple representing the inversed exterior domain
            (which is a sphere).

        """

        Rcut = self.Rcut
        if Rcut is None:
            # Retrieve Rc assuming the interior domain is spherical
            bb = numpyit(self.geom.getBoundingBox(3, self.sources[-1][0][1]))
            Rcut = abs(bb).min()
            Rcut = np.float64(round(Rcut))
            self.Rcut = Rcut

        # Retrieve master boundary
        master_tags = [self.boundaries[-1][0][1]]

        # create new shape
        inversed_exterior_domain_3d = [(3, self.geom.addSphere(
            xc=3 * Rcut, yc=0, zc=0, radius=Rcut))]
        self.geom.synchronize()
        srf_tags = [gmsh.model.getBoundary(inversed_exterior_domain_3d)[0][1]]
        translation = [1, 0, 0, 3 * Rcut, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(2, srf_tags, master_tags, translation)

        return inversed_exterior_domain_3d

    @Decorators.check_dimension(2)
    def create_ellipse(self, rx=0.1, ry=0.1, xc=0.0, yc=0.0) -> list:
        """
        Create an ellipse with given semi-major axis / semi-minor axis and
        center position.

        Parameters
        ----------
        rx : float, optional
            Ellipse radial size along x-axis. The default is 0.1.
        ry : float, optional
            Ellipse radial size along y-axis. The default is 0.1.
        xc : float, optional
            Ellipse center along x-axis. The default is 0.0.
        yc : float, optional
            Ellipse center along y-axis. The defautl is 0.0.

        Returns
        -------
        ellipse : list
            List containing the ellipse dimTag.

        """

        rx = max(self.min_length, abs(rx))
        ry = max(self.min_length, abs(ry))

        if rx >= ry:
            ellipse = [(2, self.geom.addDisk(xc=xc, yc=yc, zc=0,
                                             rx=rx, ry=ry))]
        else:
            ellipse = [(2, self.geom.addDisk(xc=xc, yc=yc, zc=0,
                                             rx=ry, ry=rx))]
            self.geom.rotate(ellipse, x=xc, y=yc, z=0,
                             ax=0, ay=0, az=1, angle=pi/2)
        return ellipse

    @Decorators.check_dimension(2)
    def create_rectangle(self, dx=1.0, dy=1.0, xll=0.0, yll=0.0, centered=False,
                         vperiodic=False, hperiodic=False) -> list:
        """
        Create a rectangle specified by its side lengths and position.

        Parameters
        ----------
        dx : float, optional
            Length of rectangle along x-axis. The default is 1.0.
        dy : float, optional
            Length of rectangle along y-axis. The default is 1.0.
        xll : float, optional
            Low left corner x-coordinate. The default is 0.0.
        yll : float, optional
            Low left corner y-coordinate. The default is 0.0.
        centered : bool, optional
            Whether the rectangle is origin-centered. If True, ignores keyword
            parameters `xll` and `yll`. The default is False.
        vperiodic : bool, optional
            Whether the mesh is periodic in the vertical direction.
            The default is False.
        hperiodic : bool, optional
            Whether the mesh is periodic in the horizontal direction.
            The default is False.

        Returns
        -------
        rectangle : list
            List containing the rectangle dimTag.

        """

        dx = max(self.min_length, abs(dx))
        dy = max(self.min_length, abs(dy))

        if centered:
            rectangle = [(2, self.geom.addRectangle(x=-dx / 2, y=-dy / 2, z=0,
                                                    dx=dx, dy=dy))]
        else:
            rectangle = [(2, self.geom.addRectangle(x=xll, y=yll, z=0,
                                                    dx=dx, dy=dy))]
        self.geom.synchronize()

        tags = MeshingTools._extract_tags(self.geom.getEntities(1)[-4:])
        if vperiodic:
            vtranslation = [1, 0, 0, 0, 0, 1, 0, dy, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [tags[0]], [tags[2]], vtranslation)

        if hperiodic:
            htranslation = [1, 0, 0, dx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [tags[3]], [tags[1]], htranslation)

        return rectangle

    @Decorators.check_dimension(2)
    def create_disk_from_pts(self, radius: float, N=100) -> list:
        """
        @author: Hugo Lévy
        Generates a disk with a given number of evenly spaced points across
        the circular boundary.

        Parameters
        ----------
        radius : float
            Radius of the disk.
        N : int, optional
            Number of points defining the boundary. The default is 100.

        Returns
        -------
        disk : list
            List containing the disk dimTag.

        """
        angle = (np.linspace(0, 2 * pi, N))[:-1]
        X = (radius * cos(angle))[:, np.newaxis]
        Y = (radius * sin(angle))[:, np.newaxis]
        Z = np.zeros_like(X)
        coors = np.concatenate((X, Y, Z), axis=1)
        disk = [self.points_to_surface(coors)]
        return disk

    @Decorators.check_dimension(2)
    def points_to_surface(self, points_list) -> tuple:
        """
        Create a closed surface whose boundary is defined by a list of points.

        Parameters
        ----------
        points_list : list or ndarray
            List containing the points which define the surface boundary.
            Each element of the list is a triplet (x, y, z) corresponding to
            the coordinates of the point it represents.

        Returns
        -------
        surface_dimTag : tuple
            Tuple containing the dimension and tag (i.e. dimTag) of the
            created surface.

        """

        if len(points_list) < 3:
            raise Exception("'points_list' requires a minimum of 3 points.")

        Pl = []
        Ll = []

        # Set points.
        for p in points_list:
            Pl.append(self.geom.addPoint(p[0], p[1], p[2]))

        # Join points as lines.
        for i, _ in enumerate(points_list):
            Ll.append(self.geom.addLine(Pl[i - 1], Pl[i]))

        # Join lines as a closed loop and surface.
        sf = self.geom.addCurveLoop(Ll)
        surface_dimTag = (2, self.geom.addPlaneSurface([sf]))

        return surface_dimTag

    @Decorators.check_dimension(3)
    def create_ellipsoid(self, rx=0.1, ry=0.1, rz=0.1,
                         xc=0.0, yc=0.0, zc=0.0) -> list:
        """
        Create an ellipsoid specified by the length of its semi-axes.

        Parameters
        ----------
        rx, ry, rz : float, optional
            Ellipsoid radial size along (x, y, z)-axis. The default is 0.1.
        xc, yc, zc : float, optional
            Ellipsoid center along (x, y, z)-axis. The default is 0.0.

        Returns
        -------
        ellipsoid : list
            List containing the ellipsoid dimTag.

        """

        rx = max(self.min_length, abs(rx))
        ry = max(self.min_length, abs(ry))
        rz = max(self.min_length, abs(rz))

        ellipsoid = [(3, self.geom.addSphere(xc=xc, yc=yc, zc=zc, radius=1))]
        self.geom.dilate(ellipsoid, x=xc, y=yc, z=zc, a=rx, b=ry, c=rz)

        return ellipsoid

    @Decorators.check_dimension(3)
    def create_box(self, dx=0.2, dy=0.2, dz=0.2) -> list:
        """
        Create a box specified by its side lengths and centered at the origin.

        Parameters
        ----------
        dx, dy, dz : float, optional
            Length of box along (x, y, z)-axis. The default is 0.2.


        Returns
        -------
        box : list tuple
            List containing the box dimTag.

        """

        dx = max(self.min_length, abs(dx))
        dy = max(self.min_length, abs(dy))
        dz = max(self.min_length, abs(dz))

        box = [(3, self.geom.addBox(x=-dx / 2, y=-dy / 2, z=-dz / 2,
                                    dx=dx, dy=dy, dz=dz))]

        return box

    @Decorators.check_dimension(3)
    def create_cylinder(self, height=0.1, radius=0.1):
        """
        @author: Chad Briddon
        Create a cylinder

        Parameters
        ----------
        height : float, optional
            Length of cylinder. The default is 0.1.
        radius : float, optional
            Radial size of cylinder. The default is 0.1.

        Returns
        -------
        cylinder : list tuple
            List containing the cylinder dimTag.

        """

        h = max(self.min_length, abs(height))
        r = max(self.min_length, abs(radius))

        cylinder = [(3, self.geom.addCylinder(x=0, y=0, z=-h / 2,
                                              dx=0, dy=0, dz=h, r=r))]

        return cylinder

    @Decorators.check_dimension(3)
    def create_torus(self, r_hole=0.1, r_tube=0.1) -> list:
        """
        @author: Chad Briddon
        Generates a torus in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        r_hole : float, optional
            Radius of hole through centre of the torus. The default is 0.1.
        r_tube : float, optional
            Radius of the torus tube. The default is 0.1.

        Returns
        -------
        torus : list tuple
            List containing the torus dimTag.

        """

        r_hole = max(self.min_length, abs(r_hole))
        r_tube = max(self.min_length, abs(r_tube))

        torus = [(3, self.geom.addTorus(x=0, y=0, z=0,
                                        r1=r_hole+r_tube, r2=r_tube))]
        return torus

    @Decorators.check_dimension(3)
    def points_to_volume(self, contour_list: list) -> tuple:
        """
        Creates a closed volume whose boundary is defined by list of contours.

        Parameters
        ----------
        contour_list : list
            List containing the contours which define the volume boundary. The
            contours are themselves a list whose elements are lists, each
            containing the x, y, and z coordinates of the point it represents.

        Returns
        -------
        volume_dimTag : tuple
            Tuple containing the dimension and tag of the generated volume.

        """

        for points_list in contour_list:
            if len(points_list) < 3:
                raise Exception(
                    "One or more contours does not have enough points. (min 3)"
                )

        L_list = []
        for points_list in contour_list:
            # Create data lists.
            Pl = []
            Ll = []

            # Set points.
            for p in points_list:
                Pl.append(self.geom.addPoint(p[0], p[1], p[2]))

            # Join points as lines.
            for i, _ in enumerate(points_list):
                Ll.append(self.geom.addLine(Pl[i - 1], Pl[i]))

            # Join lines as a closed loop and surface.
            L_list.append(self.geom.addCurveLoop(Ll))

        volume_dimTag = self.geom.addThruSections(L_list)

        # Delete contour lines.
        self.geom.remove(self.geom.getEntities(dim=1), recursive=True)

        return volume_dimTag

    @Decorators.check_dimension(2)
    def rectangle_cutout(self, shapes_dimTags, dx, dy, xll, yll):
        """
        Intersect the provided shapes with a rectangle (truncation).

        Parameters
        ----------
        shapes_dimTags : list
            List of dimTags associated with the shapes to be cutout.
        dx : float
            Length of rectangle along x-axis.
        dy : float
            Length of rectangle along y-axis.
        xll : float
            Low left corner x-coordinate.
        yll : float
            Low left corner y-coordinate.

        """

        concatenated_shapes = [y for x in shapes_dimTags for y in x]
        tool_tag = self.geom.addRectangle(xll, yll, 0, dx, dy)
        tool_dimTags = [(2, tool_tag)]
        self.geom.synchronize()
        self.geom.intersect(objectDimTags=concatenated_shapes,
                            toolDimTags=tool_dimTags)

    def radial_cutoff(self, shapes_dimTags: list, cutoff_radius=1.0):
        """
        Intersect the provided shapes with a disk (truncation).

        Parameters
        ----------
        shapes_dimTags : list
            List of dimTags associated with the shapes to be cutout.
        cutoff_radius : float, optional
            The radial size of the cutoff. Any part of a shape that is further
            away from the origin than this radius will be erased.
            The default is 1.0.

        """
        dim = max([x[0] for x in shapes_dimTags])  # dimension of shapes

        if dim == 3:
            cutoff = [(3, self.geom.addSphere(xc=0, yc=0, zc=0,
                                              radius=cutoff_radius))]
        elif dim == 2:
            cutoff = [(2, self.geom.addDisk(0, 0, 0, cutoff_radius,
                                            cutoff_radius))]
        else:
            raise Exception("No cutoff for 1D shapes!")

        self.geom.synchronize()
        self.geom.intersect(objectDimTags=shapes_dimTags, toolDimTags=cutoff)

    def add_shapes(self, shapes_1: list, shapes_2: list) -> list:
        """
        Fuse together elements of `shapes_1` and `shapes_2` to form new group
        of shapes.

        Parameters
        ----------
        shapes_1, shapes_2 : list
            List of tuples representing a group of shapes. Each tuple contains
            the dimension and tag (i.e. dimTag) of its corresponding shape.

        Returns
        -------
        new_shapes : list
            List of tuples representing the new group of shapes.

        """

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.fuse(shapes_1, shapes_2,
                                           removeObject=False,
                                           removeTool=False)
            # Get rid of unneeded shapes
            for shape in shapes_1:
                if shape not in new_shapes:
                    self.geom.remove([shape], recursive=True)
            for shape in shapes_2:
                if shape not in new_shapes:
                    self.geom.remove([shape], recursive=True)
        else:
            new_shapes = shapes_1 + shapes_2

        return new_shapes

    def subtract_shapes(self, shapes_1: list, shapes_2: list,
                        removeObject=False, removeTool=False) -> list:
        """
        Subtract elements of `shapes_2` from `shapes_1` to form new group of
        shapes.

        Parameters
        ----------
        shapes_1, shapes_2 : list
            List of tuples representing a group of shapes. Each tuple contains
            the dimension and tag (i.e. dimTag) of its corresponding shape.
        removeObject : bool, optional
            Delete the object shape if set to True. The default is False.
        removeTool : bool, optional
            Delete the tool shape if set to True. The default is False.

        Returns
        -------
        new_shapes : list
            List of tuples representing the new group of shapes.

        """

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.cut(shapes_1,
                                          shapes_2,
                                          removeObject=removeObject,
                                          removeTool=removeTool)
        else:
            new_shapes = shapes_1
            self.geom.remove(shapes_2, recursive=True)

        return new_shapes

    def intersect_shapes(self, shapes_1: list, shapes_2: list) -> list:
        """
        Create group of shapes consisting of the intersection of elements
        from `shapes_1` and `shapes_2`.

        Parameters
        ----------
        shapes_1, shapes_2 : list
            List of tuples representing a group of shapes. Each tuple contains
            the dimension and tag (i.e. dimTag) of its corresponding shape.

        Returns
        -------
        new_shapes : list
            List of tuples representing the new group of shapes.

        """

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.intersect(shapes_1, shapes_2)
        else:
            self.geom.remove(shapes_1 + shapes_2, recursive=True)
            new_shapes = []

        return new_shapes

    def non_intersect_shapes(self, shapes_1: list, shapes_2: list) -> list:
        """
        Create group of shapes consisting of the non-intersection of elements
        from `shapes_1` and `shapes_2`.

        Parameters
        ----------
        shapes_1, shapes_2 : list
            List of tuples representing a group of shapes. Each tuple contains
            the dimension and tag (i.e. dimTag) of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples representing the new group of shapes.

        """

        if shapes_1 and shapes_2:
            _, fragment_map = self.geom.fragment(shapes_1, shapes_2)

            shape_fragments = []
            for s in fragment_map:
                shape_fragments += s

            to_remove = []
            new_shapes = []
            while shape_fragments:
                in_overlap = False
                for i, s in enumerate(shape_fragments[1:]):
                    if shape_fragments[0] == s:
                        to_remove.append(shape_fragments.pop(i + 1))
                        in_overlap = True

                if in_overlap:
                    shape_fragments.pop(0)
                else:
                    new_shapes.append(shape_fragments.pop(0))

            self.geom.remove(to_remove, recursive=True)

        else:
            self.geom.remove(shapes_1 + shapes_2, recursive=True)
            new_shapes = []

        return new_shapes

    def rotate(self, shapes: list, angle: float, axis: str) -> list:
        """
        Rotates group of shapes around the x-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a group of shapes. Each tuple contains
            the dimension and tag (i.e. dimTag) of its corresponding shape.
        angle : float
            Angle of rotation in radians.
        axis : str
            Axis of rotation {'x', 'y', 'z'}.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """
        if axis not in ['x', 'y', 'z']:
            raise ValueError("axis should be either 'x' or 'y' or 'z'")
        self.geom.rotate(shapes, x=0, y=0, z=0,
                         ax=int(axis == 'x'),
                         ay=int(axis == 'y'),
                         az=int(axis == 'z'),
                         angle=angle)
        return shapes

    def translate(self, shapes: list, dx=0, dy=0, dz=0) -> list:
        """
        @author: Chad Briddon
        Translates group of shapes in the x-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dx, dy, dz : float
            Amount the group of shapes is to be translated by in the positive
            (x, y, z)-direction. If negative then translation will be in the
            negative (x, y, z)-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.translate(shapes, dx=dx, dy=dy, dz=dz)
        return shapes

    def _save(self, mesh_name, convert, ignored_tags, verbose) -> str:
        """
        Save the mesh generated by Gmsh.

        Parameters
        ----------
        mesh_name : str
            Name of the mesh file to be saved. If not specified as an absolute
            path name, will be stored in the `MESH_DIR` directory.
        convert : bool
            Convert the .vtk mesh file created by Gmsh into a .vtk file readable
            by Sfepy.
        ignored_tags : list
            List of physical groups not to be included in the final .vtk file.
        verbose : bool

        Returns
        -------
        mesh_path_name : str
            Absolute path name of the saved mesh file.
        """
        mesh_path_name = make_absolute_mesh_name(mesh_name)
        gmsh.write(fileName=mesh_path_name)
        if convert:
            convert_mesh_for_sfepy(mesh_path_name, ignored_tags=ignored_tags,
                                   verbose=verbose)
        return mesh_path_name

    def _apply_cylindrical_symmetry(self):
        """
        If the problem to be solved has cylindrical symmetry, it is possible
        to reduce the dimension of the problem from 3 (r, theta, z) to 2 (r, z).
        In that case, only keep the half-plane r>0 of the 2D mesh.

        """

        # Determination of a suitable cutout distance
        distance_cutout = 0.0
        for source in self.sources:
            for dimTag in source:
                bbox = gmsh.model.getBoundingBox(*dimTag)
                xur, yur = bbox[3], bbox[4]  # upper-right corner
                dist = sqrt(xur ** 2 + yur ** 2)
                if dist > distance_cutout:
                    distance_cutout = dist
        distance_cutout += 1  # safety margin

        # Apply cutout
        params = [distance_cutout, 2 * distance_cutout, 0, -distance_cutout]
        self.rectangle_cutout(self.sources, *params)
        self.geom.synchronize()

        # Repair boundaries
        self.boundaries = []
        for dimTags in self.sources:
            blist = []
            for dimTag in dimTags:
                boundary = gmsh.model.getBoundary([dimTag])
                for curve in boundary:
                    if curve[1] < 0:
                        continue
                    bbox = gmsh.model.getBoundingBox(curve[0], curve[1])
                    xmin, xmax = bbox[0], bbox[3]
                    if abs(xmin) > 1e-5 or abs(xmax) > 1e-5:
                        blist.append(curve)
            self.boundaries.append(blist)

    def _embed_origin_to_mesh(self, origin_tag, is_on_curve):
        """
        Force the point of coordinates x=0, y=0, [z=0] to be part of the mesh
        vertices.

        Parameters
        ----------
        origin_tag : int
            Tag of the origin point.
        is_on_curve : bool
            Whether the origin lies on a curve. This happens e.g. when
            `cylindrical_symmetry` is set to True.
        """

        # Gmsh hack to embed a point on a curve
        # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/1591
        if is_on_curve and self.dim == 2:
            self.geom.fragment([(0, origin_tag)], self.geom.getEntities())

        # Gmsh traditional way to embed a point in surface / volume
        else:
            for source in self.sources:
                for tag in [s[1] for s in source]:
                    if gmsh.model.isInside(self.dim, tag, (0, 0, 0)):
                        self.geom.synchronize()
                        gmsh.model.mesh.embed(0, [origin_tag], self.dim, tag)
                        break

        self.geom.synchronize()

    def _apply_refinement_settings(self, embed_origin, origin_tag, origin_rf):
        """Apply the refinement settings to the mesh being created."""

        # Get boundary type
        if self.dim == 2:
            boundary_type = "CurvesList"
        elif self.dim == 3:
            boundary_type = "SurfacesList"
        else:
            raise ValueError("Gmsh can only deal with 2d or 3d meshes!")

        # Group boundaries together and define distance fields
        i = 0
        for boundary, rf in zip(self.boundaries, self.refinement_settings):
            i += 1
            gmsh.model.mesh.field.add("Distance", i)
            gmsh.model.mesh.field.setNumbers(i, boundary_type,
                                             [b[1] for b in boundary])
            gmsh.model.mesh.field.setNumber(i, "NumPointsPerCurve", rf[4])

        if embed_origin:
            i += 1
            gmsh.model.mesh.field.add("Distance", i)
            gmsh.model.mesh.field.setNumbers(i, "PointsList", [origin_tag])

        # Define threshold fields.
        j = 0
        for rf in self.refinement_settings:
            j += 1
            gmsh.model.mesh.field.add("Threshold", i + j)
            gmsh.model.mesh.field.setNumber(i + j, "InField", j)
            gmsh.model.mesh.field.setNumber(i + j, "SizeMin", rf[0])
            gmsh.model.mesh.field.setNumber(i + j, "SizeMax", rf[1])
            gmsh.model.mesh.field.setNumber(i + j, "DistMin", rf[2])
            gmsh.model.mesh.field.setNumber(i + j, "DistMax", rf[3])

        if embed_origin:
            j += 1
            gmsh.model.mesh.field.add("Threshold", i + j)
            gmsh.model.mesh.field.setNumber(i + j, "InField", j)
            gmsh.model.mesh.field.setNumber(i + j, "SizeMin", origin_rf[0])
            gmsh.model.mesh.field.setNumber(i + j, "SizeMax", origin_rf[1])
            gmsh.model.mesh.field.setNumber(i + j, "DistMin", origin_rf[2])
            gmsh.model.mesh.field.setNumber(i + j, "DistMax", origin_rf[3])

        # Set mesh resolution.
        gmsh.model.mesh.field.add("Min", i + j + 1)
        gmsh.model.mesh.field.setNumbers(i + j + 1, "FieldsList",
                                         list(range(i + 1, i + j + 1)))
        gmsh.model.mesh.field.setAsBackgroundMesh(i + j + 1)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def _assign_physical_groups(self, embed_origin, origin_tag,
                                unique_boundary_tags):
        """Assign physical groups to geometrical entities of the mesh."""

        # 0-dimensional entities
        if embed_origin:
            gmsh.model.addPhysicalGroup(0, [origin_tag], tag=0)

        # (D-1)-dimensional entities
        for i, boundary in enumerate(self.boundaries):
            if unique_boundary_tags:  # same tag for all parts of a boundary
                gmsh.model.addPhysicalGroup(dim=self.dim - 1,
                                            tags=[b[1] for b in boundary],
                                            tag=200 + i)
            else:  # separate tag for each part of a boundary
                inc = max([len(x) for x in self.boundaries])
                for m, dimTag in enumerate(boundary):
                    gmsh.model.addPhysicalGroup(dim=self.dim - 1,
                                                tags=[dimTag[1]],
                                                tag=200 + (inc * i) + m)

        # D-dimensional entities
        for i, source in enumerate(self.sources):
            gmsh.model.addPhysicalGroup(dim=self.dim,
                                        tags=[s[1] for s in source],
                                        tag=300 + i)

    @staticmethod
    def _extract_tags(dimTags: list) -> list:
        """Return list of tags from a list of dimTag."""
        out = []
        for pair in dimTags:
            out.append(pair[1])
        return out


def generate_mesh_from_geo(geo_name: str, param_dict=None, show_mesh=False,
                           ignored_tags=None, convert=True, **kwargs):
    """
    Generate a mesh from an existing .geo file with Gmsh.

    Parameters
    ----------
    geo_name : str
        Name of the .geo file.
    param_dict : dict
        Dictionary containing key/value pairs to be modified.
        The default is None.
    show_mesh : bool, optional
        If True, will open a window to allow viewing of the generated mesh.
        The default is False.
    ignored_tags : list, optional
            List of physical groups not to be included in the final .vtk file.
            The default is None.
    convert : bool, optional
        Convert the .vtk mesh file created by Gmsh into a .vtk file readable
        by Sfepy. The default is True.

    Other Parameters
    ----------------
    mesh_name : str, optional
        Output file name. Will be the same as the .geo file if not specified.
        The default is None.
    verbose : bool, optional
        Display user information. The default is False.

    Returns
    -------
    mesh_path_name : str
        Absolute path name of the created mesh file.

    """

    # Get keyword arguments
    if param_dict is None:
        param_dict = {}
    mesh_name = kwargs.get('mesh_name', None)
    verbose = kwargs.get('verbose', False)

    # Write parameters to the existing .geo file
    geo_path_name = make_absolute_geo_name(geo_name)
    _write_geo_params(geo_path_name, param_dict)

    if mesh_name is None:
        mesh_name = str(Path(geo_name).stem)
    mesh_path_name = make_absolute_mesh_name(mesh_name)

    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.open(geo_path_name)
    gmsh.model.geo.synchronize()
    dim = gmsh.model.getDimension()

    if verbose:
        print('Model' + gmsh.model.getCurrent() + ' (' + str(dim) + 'D)')

    gmsh.model.mesh.generate(dim)

    if verbose > 1:  # Inspect the log
        log = gmsh.logger.get()
        print("Logger has recorded {} lines".format(len(log)))
        gmsh.logger.stop()

    if show_mesh:
        gmsh.fltk.run()

    # save the generated mesh
    gmsh.write(fileName=mesh_path_name)
    gmsh.clear()
    gmsh.finalize()

    # convert the mesh to a .vtk file readable by Sfepy
    if convert:
        convert_mesh_for_sfepy(mesh_name,
                               ignored_tags=ignored_tags,
                               verbose=verbose)
    return mesh_path_name


def convert_mesh_for_sfepy(mesh_name: str, ignored_tags=None, verbose=False):
    """
    Convert a .vtk mesh file written by Gmsh into a new .vtk file which
    facilitates topological entity selection with Sfepy. Vertices are given a
    unique group id following the convention:
    [0 - 99]    --> vertex tag (i.e. entity of dimension 0)
    [100 - 199] --> edge tag (i.e. entity of dimension 1)
    [200 - 299] --> facet tag (i.e. entity of dimension D-1)
    [300 - xxx] --> cell tag (i.e. entity of dimension D)
    If one node belongs to several topological entities, it will be tagged with
    the lowest dimension group id. For instance, a vertex belonging to a facet
    of tag 200 and a subvolume of tag 300 will be tagged 200. This is
    problematic for subvolume selection. This difficulty is overcome by taking
    advantage of the 'mat_ids' field that is readable by Sfepy.

    Parameters
    ----------
    mesh_name : str
        Name of the mesh file to be created. If not specified as an absolute
        path name, will be stored in the `MESH_DIR` directory.
    ignored_tags : list, optional
        List of tags to be ignored in the new .vtk mesh. The default is None.
    verbose : bool, optional
        Toggle for Sfepy automatic console messages. The default is False.
    """

    output.set_output(quiet=not verbose)

    if ignored_tags is None:
        ignored_tags = []
    mesh_path_name = make_absolute_mesh_name(mesh_name)

    mesh = Mesh.from_file(mesh_path_name)
    cell_dim = mesh.dim

    # All the necessary & sufficient information for defining a mesh
    data = list(mesh._get_io_data(cell_dim_only=cell_dim))

    # Managing vertex groups
    ngroups = np.array([None for _ in range(mesh.n_nod)])
    reader = meshio.read(mesh_path_name)
    for key in reader.cells_dict.keys():
        conn = reader.cells_dict[key]
        former_tags = list(reader.cell_data_dict.values())[0][key]
        for k, tag in enumerate(former_tags):
            if tag not in ignored_tags:
                for idx in conn[k]:
                    if ngroups[idx] is None:
                        ngroups[idx] = int(tag)
    ngroups[np.where(ngroups is None)] = 400  # default marker for untagged
    data[1] = ngroups.astype(dtype=np.int32)

    # Managing cell groups
    conns = list(reader.cells_dict.values())[-1]  # entities of highest dim
    mat_ids = np.max(ngroups[conns], axis=1)
    data[3] = [mat_ids.astype(dtype=np.int32)]

    # Overwrite the former mesh
    mesh = Mesh.from_data(mesh.name, *data)
    mesh.write(mesh_path_name, None, binary=False)


def adjust_boundary_nodes(mesh_name_ref, mesh_name_mod,
                          boundary_tag_ref, boundary_tag_mod, verbose=False):
    """
    Adjust the coordinates of the nodes belonging to the boundary tagged
    `boundary_tag_mod` of mesh `mesh_name_mod` in order to match those of
    `mesh_name_ref` with tag `boundary_tag_ref`.

    Parameters
    ----------
    mesh_name_ref : str
        Name of the reference mesh file (read only). If not specified as an
        absolute path name, will be searched for in the `MESH_DIR` directory.
    mesh_name_mod : str
        Name of the mesh file to be modified if its boundary nodes do not match
        those of the reference mesh file. If not specified as an absolute path
        name, will be searched for in the `MESH_DIR` directory.
    boundary_tag_ref : int
        Physical group of the reference boundary nodes.
    boundary_tag_mod : int
        Physical group of the boundary nodes to be adjusted.

    Other Parameters
    ----------------
    verbose : bool, optional
        The default is False.

    """

    mesh_path_name_ref = make_absolute_mesh_name(mesh_name_ref)
    mesh_path_name_mod = make_absolute_mesh_name(mesh_name_mod)

    # Read the meshes, get boundary indices and associated coordinates
    mesh_ref = meshio.read(mesh_path_name_ref)
    mesh_mod = meshio.read(mesh_path_name_mod)

    ind_ref, ind_mod = (
        np.where(mesh_ref.point_data['node_groups'] == boundary_tag_ref)[0],
        np.where(mesh_mod.point_data['node_groups'] == boundary_tag_mod)[0]
    )
    assert len(ind_ref) == len(ind_mod), "Number of DOFs does not match!"

    pts_ref = mesh_ref.points[ind_ref]
    pts_mod = mesh_mod.points[ind_mod]

    # Check for non-matching nodes between mesh_ref and mesh_mod
    actually_modified = []
    for kk, ptmod in enumerate(pts_mod):
        if not (pts_ref - ptmod == [0, 0, 0]).prod(axis=1).any():
            ind = abs(pts_ref - ptmod).sum(axis=1).argmin()
            actually_modified.append(ind_mod[kk])
            ptmod = np.copy(pts_ref[ind])  # in place modification
            mesh_mod.points[ind_mod[kk]] = ptmod

    # If applicable, overwrite mesh_mod with adjusted boundary nodes
    if len(actually_modified) != 0:
        if verbose:
            print("Adjusting the position of some boundary nodes...\n")
        mesh_overwrite = Mesh.from_file(mesh_path_name_mod)
        coors_mod = mesh_mod.points[:, :mesh_overwrite.dim]
        mesh_overwrite.coors[actually_modified] = coors_mod[actually_modified]
        mesh_overwrite.write(mesh_path_name_mod, None, binary=False)


def split_spheres_from_mesh(mesh_name, Rcut, verbose=False):
    """
    Split two meshed spheres that are geometrically separable by a hyperplane
    into two distinct mesh files. The original mesh is then deleted. The 
    original mesh must have gone through the `convert_mesh_for_sfepy` routine
    before being processed by `split_spheres_from_mesh`!

    Parameters
    ----------
    mesh_name : str
        Name of the mesh file to be read. If not specified as an absolute
        path name, will be searched for in the `MESH_DIR` directory.
    Rcut : float
        Truncation radius.

    Other Parameters
    ----------------
    verbose : bool, optional
        Toggle for Sfepy automatic console messages. The default is False.

    Returns
    -------
    out_files : list
        List of absolute path of output mesh files.

    """

    output.set_output(quiet=not verbose)

    # Retrieve data
    mesh_path_name = make_absolute_mesh_name(mesh_name)
    mesh = Mesh.from_file(mesh_path_name)
    data = list(mesh._get_io_data(cell_dim_only=mesh.dim))
    data_int = deepcopy(data)
    data_ext = deepcopy(data)

    # Classify nodes belonging to sphere_int and sphere_ext
    norms = np.linalg.norm(data[0], axis=1)
    idx_node_int = np.where(norms < 1.5 * Rcut)[0]
    idx_node_ext = np.where(norms > 1.5 * Rcut)[0]

    # Sort out data structures according to node class
    offset = np.array([3 * Rcut, 0, 0]).reshape(1, 3)
    data_int[0] = data_int[0][idx_node_int]
    data_int[1] = data_int[1][idx_node_int]
    data_ext[0] = data_ext[0][idx_node_ext] - offset
    data_ext[1] = data_ext[1][idx_node_ext]
    extTag = np.max(data[3][0])
    idx_cell_ext = np.where(data[3][0] == extTag)[0]
    data_int[2] = [np.delete(data_int[2][0], idx_cell_ext, axis=0)]
    data_int[3] = [np.delete(data_int[3][0], idx_cell_ext)]
    data_ext[2] = [data_ext[2][0][idx_cell_ext]]
    data_ext[3] = [data_ext[3][0][idx_cell_ext]]

    # Repair connectivity because it has been broken!
    conn_int = deepcopy(data_int[2][0])
    conn_ext = deepcopy(data_ext[2][0])
    min_int = conn_int.min()
    conn_int -= min_int
    data_int[2][0] -= min_int
    idx_node_int -= min_int
    min_ext = conn_ext.min()
    conn_ext -= min_ext
    data_ext[2][0] -= min_ext
    idx_node_ext -= min_ext
    gaps_int = np.diff(idx_node_int) - 1
    gaps_ext = np.diff(idx_node_ext) - 1
    for k, gap in enumerate(gaps_int):
        if gap > 0:
            conn_int[data_int[2][0] > idx_node_int[k]] -= gap
    for k, gap in enumerate(gaps_ext):
        if gap > 0:
            conn_ext[data_ext[2][0] > idx_node_ext[k]] -= gap
    data_int[2] = [conn_int]
    data_ext[2] = [conn_ext]

    # I/O stuff
    name = str(Path(mesh.name).name)
    name_int = Path(mesh.name).with_name(name + '_int')
    name_ext = Path(mesh.name).with_name(name + '_ext')
    mesh_int = Mesh.from_data(str(name_int), *data_int)
    mesh_int.write(str(name_int.with_suffix('.vtk')), None, binary=False)
    mesh_ext = Mesh.from_data(str(name_ext), *data_ext)
    mesh_ext.write(str(name_ext.with_suffix('.vtk')), None, binary=False)
    os.remove(mesh.name + '.vtk')

    # Return new file names
    out_files = [str(name_int.with_suffix('.vtk')),
                 str(name_ext.with_suffix('.vtk'))]

    return out_files


def make_absolute_mesh_name(mesh_name: str) -> str:
    """Return an absolute path name from a mesh name, with .vtk extension.
    The default path is `MESH_DIR`."""

    if not isinstance(mesh_name, str):
        raise TypeError(
            "mesh_name is of type '{}' but should be a string instead".format(
                type(mesh_name).__name__))

    mesh_path_name = Path(mesh_name).with_suffix('.vtk')
    if not mesh_path_name.is_absolute():
        mesh_path_name = Path(MESH_DIR / mesh_path_name)

    return str(mesh_path_name)


def make_absolute_geo_name(geo_name: str) -> str:
    """Return an absolute path name from a mesh name, with .geo extension.
    The default path is `MESH_DIR`/geo."""

    if not isinstance(geo_name, str):
        raise TypeError(
            "geo_name is of type '{}' but should be a string instead".format(
                type(geo_name).__name__))

    geo_name_path = Path(geo_name).with_suffix('.geo')
    if not geo_name_path.is_absolute():
        geo_name_path = Path(MESH_DIR / "geo" / geo_name_path)

    return str(geo_name_path)


def read_Rcut_from_mesh_file(mesh_name, coorsys='cartesian'):
    """
    Read truncation radius Rc from existing .vtk mesh file.

    Parameters
    ----------
    mesh_name : str
        Name of the mesh file to be read. If not specified as an absolute
        path name, will be searched for in the `MESH_DIR` directory.
    coorsys : str
        Coordinate system to be used in accordance with the mesh.

    Returns
    -------
    Rc : float
        Truncation radius.

    """
    mesh_path_name = make_absolute_mesh_name(mesh_name)
    reader = meshio.read(mesh_path_name)
    coors = reader.points
    if coorsys == 'cartesian':
        Rcut = float(np.max(np.linalg.norm(coors, axis=1)))
    elif coorsys in ['polar', 'polar_mu']:
        Rcut = float(np.max(coors[:, 0]))
    else:
        raise Exception("Not implemented coordinates system: '%s'" % coorsys)
    return Rcut


def read_dimension_from_mesh_file(mesh_name: str):
    """Read dimension from existing .vtk mesh file."""
    mesh_path_name = make_absolute_mesh_name(mesh_name)
    reader = meshio.read(mesh_path_name)
    Z = reader.points[:, 2]
    if (Z == 0.0).all():
        return 2
    else:
        return 3


def read_physical_groups_from_mesh_file(mesh_name: str) -> list:
    """
    Read the physical groups present in the mesh and return them as a
    list.
    """
    mesh_path_name = make_absolute_mesh_name(mesh_name)
    reader = meshio.read(mesh_path_name)
    return list(np.unique(reader.point_data["node_groups"]))


def generate_uniform_1d_mesh(start: float, stop: float, number_nodes: int,
                             mesh_name='mesh1d') -> Mesh:
    """
    Generate uniform one-dimensional mesh on the interval [`start`, `stop`]
    using `number_nodes` nodes. Unlike higher dimensional meshes, the resulting
    mesh is not saved as a .vtk file. Instead, it is directly returned as an
    instance of the class `sfepy.discrete.fem.mesh.Mesh`.

    Parameters
    ----------
    start : float
        Lower bound of the interval to be meshed.
    stop : float
        Upper bound of the interval to be meshed.
    number_nodes : int
        Number of nodes evenly  spaced across the interval (including
        endpoints).
    mesh_name : str, optional
        Name of the mesh. The default is 'mesh1d'.

    Returns
    -------
    mesh : `sfepy.discrete.fem.mesh.Mesh`
        Uniform one-dimensional mesh (instance of Sfepy's class `Mesh`).

    """
    number_nodes = int(number_nodes)
    r = np.linspace(start, stop, number_nodes)
    return generate_1d_mesh_from_array(r, mesh_name=mesh_name)


def generate_1d_mesh_from_array(coors_array: np.ndarray,
                                mesh_name='mesh1d') -> Mesh:
    """
    Generate a one-dimensional mesh from a 1d Numpy array. Unlike higher
    dimensional meshes, the resulting mesh is not saved as a .vtk file.
    Instead, it is directly returned as an instance of the class
    `sfepy.discrete.fem.mesh.Mesh`.

    Parameters
    ----------
    coors_array : np.ndarray
        Coordinates of the mesh vertices (1st-order nodes only).
    mesh_name : str, optional
        Name of the mesh. The default is 'mesh1d'.

    Returns
    -------
    mesh : `sfepy.discrete.fem.mesh.Mesh`
        Uniform one-dimensional mesh (instance of Sfepy's class `Mesh`).

    """
    r = coors_array.reshape(-1, 1)
    conn = np.arange(r.shape[0], dtype=np.int32).repeat(2)[1:-1].reshape(-1, 2)
    mat_ids = np.zeros(r.shape[0]-1, dtype=np.int32)
    mesh = Mesh.from_data(mesh_name, r, None, [conn], [mat_ids], ['1_2'])
    return mesh


def _write_geo_params(geo_name, param_dic):
    """
    Modify a .geo file according to the key/value pairs specified by the user
    in the `param_dict` dictionary.

    Parameters
    ----------
    geo_name : str
        Name of the .geo file.
    param_dic : dict
        Dictionary containing key/value pairs to be modified.

    """

    geo_path_name = make_absolute_geo_name(geo_name)

    param_keys = list(param_dic.keys())
    separator = ' '

    # Read .geo file
    old_file = open(geo_path_name, 'r')
    lines = old_file.readlines()
    old_file.close()

    # Write to a new .geo file
    new_file = open(geo_path_name, 'w')
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
