#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:06:42 2021

@author: Chad Briddon (& Hugo Lévy)
Visit https://github.com/C-Briddon/SELCIE/blob/main/SELCIE/MeshingTools.py

Tools to produce and modify meshes to be used in simulations.
"""

import math
import gmsh
import meshio
import numpy as np

from femtoscope import MESH_DIR
from pathlib import Path


class MeshingTools():
    """
    Class used to construct user-defined meshes. Creates an open gmsh
    window when class is called.

     Attributes
    ----------
    dim : int
        Dimension of the mesh (2 or 3).
    boundaries : list
        List of shapes boundaries represented via their dimTag i.e. tuples
        (dim, tag). The dimTags are gathered by shape and encapsulated in
        sublists for this purpose. Those entities are of dimension D-1.
    source : list
        List of shapes represented via their dimTag. The dimTags are gathered
        by shape and encapsulated in sublists for this purpose. Those entities
        are of dimension D.
    refinement_settings : list
        List of lists, each sublist containing mesh refinement settings of a
        source element.
    source_number : int
        Number of sources currently defined in the Gmsh model.
    boundary_number : int
        Number of boundaries currently defined in the Gmsh model.
    Min_length : float
        Minimum distance between neighbouring points.
    geom :
        Shortcut for *gmsh.model.occ*.
    Rcut : float
        Radius that delimits the interior domain (disk in 2D / sphere in 3D).
        Only relevant when dealing with exterior mesh generation.
    center_ext : tuple
        Coordinates (x, y, z) of the geometrical center of the exterior domain.
        Only relevant when dealing with exterior mesh generation.
    outfiles : list of str
        List of generated mesh files (absolute path).

    """

    def __init__(self, dimension):
        """
        Construct a MeshingTools instance

        Parameters
        ----------
        dimension : int
            The dimension of the mesh being constructed. Currently works for
            2D and 3D.

        """

        # Set attributes
        self.dim = dimension
        self.boundaries = []
        self.source = []
        self.refinement_settings = []
        self.source_number = 0
        self.boundary_number = 0
        self.Min_length = 1.3e-6
        self.geom = gmsh.model.occ
        self.Rcut = None
        self.center_ext = (0, 0, 0)
        self.outfiles = []

        # Open GMSH window.
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)

        return None

    def constrain_distance(self, points_list):
        """
        @author: Chad Briddon
        Removes points from list so that the distance between neighbouring
        points is greater than the minimum gmsh line length. Is performed such
        that the last point in the list will not be removed.

        Parameters
        ----------
        points_list : list of list
            List containing the points. Each element of the list is a list
            containing the x, y, and z coordinate of the point it represents.

        Returns
        -------
        points_new : list of list
            Copy of inputted list with points removed so that all neighbouring
            points are seperated by the minimum allowed distance.

        """

        index = []
        p_pre = points_list[-1]
        for p in points_list[:-1]:
            if math.dist(p, p_pre) < self.Min_length:
                index.append(False)
            else:
                index.append(True)
                p_pre = p

        points_new = [q for q, I in zip(points_list, index) if I]

        # Reinclude last point from original list.
        if len(points_new) > 0:
            if math.dist(points_list[-1], points_new[-1]) < self.Min_length:
                points_new[-1] = points_list[-1]
            else:
                points_new.append(points_list[-1])
        else:
            points_new.append(points_list[-1])

        return points_new

    def points_to_surface(self, points_list):
        """
        @author: Chad Briddon
        Generates closed surface whose boundary is defined by a list of points.

        Parameters
        ----------
        points_list : list of list
            List containing the points which define the exterior of the
            surface. Each element of the list is a list containing the x, y,
            and z coordinate of the point it represents.

        Returns
        -------
        SurfaceDimTag : tuple
            Tuple containing the dimension and tag of the generated surface.

        """

        if len(points_list) < 3:
            raise Exception(
                "'points_list' requires a minimum of 3 points.")

        Pl = []
        Ll = []

        # Set points.
        for p in points_list:
            Pl.append(self.geom.addPoint(p[0], p[1], p[2]))

        # Join points as lines.
        for i, _ in enumerate(points_list):
            Ll.append(self.geom.addLine(Pl[i-1], Pl[i]))

        # Join lines as a closed loop and surface.
        sf = self.geom.addCurveLoop(Ll)
        SurfaceDimTag = (2, self.geom.addPlaneSurface([sf]))

        return SurfaceDimTag

    def points_to_volume(self, contour_list):
        """
        @author: Chad Briddon
        Generates closed volume whose boundary is defined by list of contours.

        Parameters
        ----------
        contour_list : list of list of list
            List containing the contours which define the exterior of the
            volume. The contours are themselves a list whose elements are
            lists, each containing the x, y, and z coordinate of the point
            it represents.

        Returns
        -------
        VolumeDimTag : tuple
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
                Ll.append(self.geom.addLine(Pl[i-1], Pl[i]))

            # Join lines as a closed loop and surface.
            L_list.append(self.geom.addCurveLoop(Ll))

        VolumeDimTag = self.geom.addThruSections(L_list)

        # Delete contour lines.
        self.geom.remove(self.geom.getEntities(dim=1), recursive=True)

        return VolumeDimTag

    def radial_cutoff(self, shape_DimTags, cutoff_radius=1.0):
        """
        @author: Chad Briddon
        Applies a radial cutoff to all shapes in open gmsh window.

        Parameters
        ----------
        cutoff_radius : float, optional
            The radial size of the cutoff. Any part of a shape that is
            further away from the origin than this radius will be erased.
            The default is 1.0.

        Returns
        -------
        None.

        """

        # Check for 3D interecting spheres.
        dim = max([x[0] for x in shape_DimTags])
        if dim == 3:
            cutoff = [(3, self.geom.addSphere(xc=0, yc=0, zc=0,
                                              radius=cutoff_radius))]
        elif dim == 2:
            cutoff = [(2, self.geom.addDisk(0, 0, 0, cutoff_radius, cutoff_radius))]
        else:
            raise Exception("No cutoff for 1D meshes!")

        self.geom.synchronize()
        self.geom.intersect(objectDimTags=shape_DimTags, toolDimTags=cutoff)

        return None

    def rectangle_cutoff(self, shape_DimTags, R):
        """
        @author: Hugo Lévy
        Applies a radial cutoff to all shapes in open gmsh window.

        Parameters
        ----------
        cutoff_radius : float, optional
            The radial size of the cutoff. Any part of a shape that is
            further away from the origin than this radius will be erased.
            The default is 1.0.

        Returns
        -------
        None.

        """

        # Check for 3D interecting spheres.
        cutoff = [(2, self.geom.addRectangle(0, -R, 0, R, 2*R))]
        self.geom.synchronize()
        self.geom.intersect(objectDimTags=shape_DimTags, toolDimTags=cutoff)

        return None

    def create_subdomain(self, CellSizeMin=0.1, CellSizeMax=0.1, DistMin=0.0,
                         DistMax=1.0, NumPointsPerCurve=1000):
        r"""
        @author: Chad Briddon
        Creates a subdomain from the shapes currently in an open gmsh window.
        Shapes already present in previous subdomains will not be added to the
        new one. This subdomain will be labeled by an index value corresponding
        to the next available integer value.

        The size of mesh cells at distances less than 'DistMin' from the
        boundary of this subdomain will be 'SizeMin', while at distances
        greater than 'DistMax' cell size is 'SizeMax'. Between 'DistMin'
        and 'DistMax' cell size will increase linearly as illustrated below.


                           DistMax
                              |
        SizeMax-             /--------
                            /
                           /
                          /
        SizeMin-    o----/
                         |
                      DistMin


        Parameters
        ----------
        CellSizeMin : float, optional
            Minimum size of the mesh cells. The default is 0.1.
        CellSizeMax : float, optional
            Maximum size of the mesh cells. The default is 0.1.
        DistMin : float, optional
            At distances less than this value the cell size is set to its
            minimum. The default is 0.0.
        DistMax : float, optional
            At distances greater than this value the cell size is set to its
            maximum. The default is 1.0.
        NumPointsPerCurve : int, optional
            Number of points used to define each curve. The default is 1000.

        Returns
        -------
        None.

        """

        # Save sources, remove duplicates, and update source number.
        self.source.append(self.geom.getEntities(dim=self.dim))
        del self.source[-1][:self.source_number]
        self.source_number += len(self.source[-1])

        # Check if new entry is empty.
        if self.source[-1]:
            # Save boundary information
            self.boundaries.append(self.geom.getEntities(dim=self.dim-1))
            del self.boundaries[-1][:self.boundary_number]
            self.boundary_number += len(self.boundaries[-1])

            # Record refinement settings for this subdomain.
            self.refinement_settings.append([CellSizeMin, CellSizeMax, DistMin,
                                             DistMax, NumPointsPerCurve])
        else:
            del self.source[-1]

        return None

    def create_background_mesh(self, CellSizeMin=0.1, CellSizeMax=0.1,
                               DistMin=0.0, DistMax=1.0,
                               NumPointsPerCurve=1000, background_radius=1.0,
                               wall_thickness=None,
                               refine_outer_wall_boundary=False):
        r"""
        @author: Chad Briddon
        Generates a backgound mesh filling the space between shapes in the
        open gmsh window and a circular/spherical shell.

        The size of mesh cells at distances less than 'DistMin' from the
        boundary of this background will be 'SizeMin', while at distances
        greater than 'DistMax' cell size is 'SizeMax'. Between 'DistMin'
        and 'DistMax' cell size will increase linearly as illustrated below.


                           DistMax
                              |
        SizeMax-             /--------
                            /
                           /
                          /
        SizeMin-    o----/
                         |
                      DistMin


        Parameters
        ----------
        CellSizeMin : float, optional
            Minimum size of the mesh cells. The default is 0.1.
        CellSizeMax : float, optional
            Maximum size of the mesh cells. The default is 0.1.
        DistMin : float, optional
            At distances less than this value the cell size is set to its
            minimum. The default is 0.0.
        DistMax : float, optional
            At distances greater than this value the cell size is set to its
            maximum. The default is 1.0.
        NumPointsPerCurve : int, optional
            Number of points used to define each curve. The default is 1000.
        background_radius : float, optional
            Radius of the circular/spherical shell used to define the
            background mesh. The default is 1.0.
        wall_thickness : None or float, optional
            If not None generates a boundary wall around the background mesh
            with specified thickness. The default is None.
        refine_outer_wall_boundary : bool, optional
            If True will also apply refinement to the exterior boundary of the
            outer wall (if exists). The default is False.

        Returns
        -------
        None.

        """

        # Get source information.
        self.create_subdomain(CellSizeMin, CellSizeMax, DistMin, DistMax,
                              NumPointsPerCurve)

        # Define vacuum and inner wall boundary.
        source_sum = self.geom.getEntities(dim=self.dim)

        if self.dim == 2:
            background_0 = [(2, self.geom.addDisk(xc=0, yc=0, zc=0,
                                                  rx=background_radius,
                                                  ry=background_radius))]
        elif self.dim == 3:
            background_0 = [(3, self.geom.addSphere(xc=0, yc=0, zc=0,
                                                    radius=background_radius))]

        if self.source:
            self.geom.cut(objectDimTags=background_0, toolDimTags=source_sum,
                          removeObject=True, removeTool=False)

        # Record background as new subdomain.
        self.create_subdomain(CellSizeMin, CellSizeMax, DistMin, DistMax,
                              NumPointsPerCurve)

        # Define wall and outer wall boundary.
        if wall_thickness:
            source_sum = self.geom.getEntities(dim=self.dim)

            if self.dim == 2:
                wall_0 = [(2, self.geom.addDisk(
                    xc=0, yc=0, zc=0, rx=background_radius+wall_thickness,
                    ry=background_radius+wall_thickness))]

            elif self.dim == 3:
                wall_0 = [(3, self.geom.addSphere(
                    xc=0, yc=0, zc=0,
                    radius=background_radius+wall_thickness))]

            self.geom.cut(objectDimTags=wall_0, toolDimTags=source_sum,
                          removeObject=True, removeTool=False)

            if refine_outer_wall_boundary:
                self.create_subdomain(CellSizeMin, CellSizeMax, DistMin,
                                      DistMax, NumPointsPerCurve)
            else:
                self.create_subdomain()

        return None

    def generate_mesh(self, filename=None, show_mesh=False,
                      embed_center=False, symmetry=False, unique_boundary=True,
                      ignoreTags=[], convert=True, exterior=False, **kwargs):
        """
        @author: Chad Briddon & Hugo Lévy
        Generate and save mesh.

        Parameters
        ----------
        filename : str, optional
            If not None then saves mesh as 'data/mesh/filename.vtk'. If
            directory 'Saved Meshes' does not exist in current directory then
            one is created. The default is None.
        show_mesh : bool, optional
            If True will open a window to allow viewing of the generated mesh.
            The default is False.
        embed_center : bool, optional
            If True, point of coordinates (0,0,0) is added to the mesh (useful
            for inverse domains in cartesian coordinates).
            The default is False.
        unique_boundary : bool, optional
            If True, a single physical group will be created for all
            elementary entities that are parts of a shape's boundary.
            The default is True.
        ignoreTags : list, optional
            List of physical groups not to be included in the final .vtk file.
            The default is [].
        convert : bool, optional
            If True, the mesh file exported by Gmsh will be converted so as to
            be readable by Sfepy. The default is True.
        exterior : bool, optional
            If True, construct a second mesh matching the main one on a
            spherical boundary (for 3D domain-splitting techniques).
            The default is False.

        Other Parameters
        ----------------
        verbose : bool
            Display user's information. The default is False.
        center_rf : list
            Settings for refinement around the origin, namely
            [SizeMin, SizeMax, DistMin, DistMax].
            The default is [1e-2, 1e-1, 0.1, 1.0].

        Returns
        -------
        outfiles : list of str
            List of absolute path of output file(s).

        """

        self.create_subdomain()
        self.geom.synchronize()
        verbose = kwargs.get('verbose', False)

        if exterior:
            self.create_exterior_mesh()
            self.create_subdomain()
            self.refinement_settings[-1] = self.refinement_settings[-2]
            embed_center = True
            self.center_ext = (3*self.Rcut, 0, 0)
            self.geom.synchronize()

        if symmetry:
            # apply symmetry
            shapes = [y for x in self.source for y in x]
            self.rectangle_cutoff(shapes, 10)
            self.geom.synchronize()
            # repair boundaries
            self.boundaries = []
            for dimTags in self.source:
                blist = []
                for dimTag in dimTags:
                    boundary = gmsh.model.getBoundary([dimTag])
                    for curve in boundary:
                        if curve[1] < 0:
                            continue
                        bbox = gmsh.model.getBoundingBox(curve[0], curve[1])
                        xmin, xmax = bbox[0], bbox[3]
                        if abs(xmin)>1e-5 or abs(xmax)>1e-5:
                            blist.append(curve)
                self.boundaries.append(blist)

        if embed_center:
            centerTag = self.geom.addPoint(*self.center_ext)
            self.geom.synchronize()


        # If no refinement settings have been imputted then use default.
        if self.refinement_settings:

            # Get boundary_type.
            if self.dim == 2:
                boundary_type = "CurvesList"
            elif self.dim == 3:
                boundary_type = "SurfacesList"

            # Group boundaries together and define distance fields.
            i = 0
            for boundary, rf in zip(self.boundaries, self.refinement_settings):
                i += 1
                gmsh.model.mesh.field.add("Distance", i)
                gmsh.model.mesh.field.setNumbers(i, boundary_type,
                                                  [b[1] for b in boundary])
                gmsh.model.mesh.field.setNumber(i, "NumPointsPerCurve", rf[4])

            if embed_center:
                i += 1
                gmsh.model.mesh.field.add("Distance", i)
                gmsh.model.mesh.field.setNumbers(i, "PointsList", [centerTag])

            # Define threshold fields.
            j = 0
            for rf in self.refinement_settings:
                j += 1
                gmsh.model.mesh.field.add("Threshold", i+j)
                gmsh.model.mesh.field.setNumber(i+j, "InField", j)
                gmsh.model.mesh.field.setNumber(i+j, "SizeMin", rf[0])
                gmsh.model.mesh.field.setNumber(i+j, "SizeMax", rf[1])
                gmsh.model.mesh.field.setNumber(i+j, "DistMin", rf[2])
                gmsh.model.mesh.field.setNumber(i+j, "DistMax", rf[3])

            if embed_center:
                j += 1
                center_rf = kwargs.get('center_rf', [1e-2, 1e-1, 0.1, 1.0])
                gmsh.model.mesh.field.add("Threshold", i+j)
                gmsh.model.mesh.field.setNumber(i+j, "InField", j)
                gmsh.model.mesh.field.setNumber(i+j, "SizeMin", center_rf[0])
                gmsh.model.mesh.field.setNumber(i+j, "SizeMax", center_rf[1])
                gmsh.model.mesh.field.setNumber(i+j, "DistMin", center_rf[2])
                gmsh.model.mesh.field.setNumber(i+j, "DistMax", center_rf[3])

            # Set mesh resolution.
            gmsh.model.mesh.field.add("Min", i+j+1)
            gmsh.model.mesh.field.setNumbers(i+j+1, "FieldsList",
                                              list(range(i+1, i+j+1)))
            gmsh.model.mesh.field.setAsBackgroundMesh(i+j+1)

            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            # add center point to the mesh
            if embed_center:

                # Gmsh hack to embed a point on a curve
                # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/1591
                if symmetry and self.dim==2:
                    self.geom.fragment([(0, centerTag)],
                                       self.geom.getEntities())

                # Gmsh traditional way to embed a point in surface / volume
                else:
                    for source in self.source:
                        for tag in [s[1] for s in source]:
                            if gmsh.model.isInside(self.dim, tag,
                                                   self.center_ext):
                                self.geom.synchronize()
                                gmsh.model.mesh.embed(0, [centerTag],
                                                      self.dim, tag)
                                break

                self.geom.synchronize()
                gmsh.model.addPhysicalGroup(0, [centerTag], tag=0)

            # Mark physical domains and boundaries.
            for i, source in enumerate(self.source):
                gmsh.model.addPhysicalGroup(dim=self.dim,
                                            tags=[s[1] for s in source],
                                            tag=300+i)

            for i, boundary in enumerate(self.boundaries):
                if unique_boundary: # same tag for all parts of a boundary
                    gmsh.model.addPhysicalGroup(dim=self.dim-1,
                                                tags=[b[1] for b in boundary],
                                                tag=200+i)
                else: # separate tag for each part of a boundary
                    inc = max([len(x) for x in self.boundaries])
                    for m, dimTag in enumerate(boundary):
                        gmsh.model.addPhysicalGroup(dim=self.dim-1,
                                                    tags=[dimTag[1]],
                                                    tag=200+inc*i+m)

        # gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.model.mesh.generate(dim=self.dim)
        gmsh.model.mesh.removeDuplicateNodes()

        if verbose>=2:
            # Inspect the log:
            log = gmsh.logger.get()
            print("Logger has recorded " + str(len(log)) + " lines")
            gmsh.logger.stop()

        if show_mesh:
            gmsh.fltk.run()

        # save the generated mesh
        if filename is not None:
            fullfilename = get_meshsource(filename, ext='.vtk')
            gmsh.write(fileName=fullfilename)
            # convert the mesh to a .vtk file readable by Sfepy
            if convert:
                self.outfiles = converter4sfepy(
                    fullfilename, ignoreTags=ignoreTags,
                    exterior=exterior, Rcut=self.Rcut)
            else:
                self.outfiles.append(fullfilename)

        # To be called when you are done using the Gmsh Python API
        gmsh.clear()
        gmsh.finalize()

        return self.outfiles

    @staticmethod
    def pointInCurve(point_tag, curve_tag):
        """ @author: Hugo Lévy """
        return gmsh.model.occ.fragment([(0, point_tag)], [(1, curve_tag)])[0]

    def add_shapes(self, shapes_1, shapes_2):
        """
        @author: Chad Briddon
        Fusses together elements of 'shapes_1' and 'shapes_2' to form new group
        of shapes.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples representing the new group of shapes.

        """

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.fuse(shapes_1, shapes_2,
                                           removeObject=False,
                                           removeTool=False)

            # Get rid of unneeded shapes.
            for shape in shapes_1:
                if shape not in new_shapes:
                    self.geom.remove([shape], recursive=True)

            for shape in shapes_2:
                if shape not in new_shapes:
                    self.geom.remove([shape], recursive=True)

        else:
            new_shapes = shapes_1 + shapes_2

        return new_shapes

    def subtract_shapes(self, shapes_1, shapes_2,
                        removeObject=False, removeTool=False):
        """
        @author: Chad Briddon
        Subtracts elements of 'shapes_2' from 'shapes_1' to form new group of
        shapes.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples representing a groups of shapes. Each tuple
            contains the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
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

    def intersect_shapes(self, shapes_1, shapes_2):
        """
        @author: Chad Briddon
        Creates group of shapes consisting of the intersection of elements
        from 'shapes_1' and 'shapes_2'.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples representing the new group of shapes.

        """

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.intersect(shapes_1, shapes_2)
        else:
            self.geom.remove(shapes_1 + shapes_2, recursive=True)
            new_shapes = []

        return new_shapes

    def non_intersect_shapes(self, shapes_1, shapes_2):
        """
        @author: Chad Briddon
        Creates group of shapes consisting of the non-intersection of elements
        from 'shapes_1' and 'shapes_2'.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

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
                        to_remove.append(shape_fragments.pop(i+1))
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

    def rotate_x(self, shapes, rot_fraction):
        """
        @author: Chad Briddon
        Rotates group of shapes around the x-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        rot_fraction : float
            Fraction of a full rotation the group will be rotated by.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.rotate(shapes, x=0, y=0, z=0, ax=1, ay=0, az=0,
                         angle=2*np.pi*rot_fraction)

        return shapes

    def rotate_y(self, shapes, rot_fraction):
        """
        @author: Chad Briddon
        Rotates group of shapes around the y-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        rot_fraction : float
            Fraction of a full rotation the group will be rotated by.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.rotate(shapes, x=0, y=0, z=0, ax=0, ay=1, az=0,
                         angle=2*np.pi*rot_fraction)

        return shapes

    def rotate_z(self, shapes, rot_fraction):
        """
        @author: Chad Briddon
        Rotates group of shapes around the z-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        rot_fraction : float
            Fraction of a full rotation the group will be rotated by.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.rotate(shapes, x=0, y=0, z=0, ax=0, ay=0, az=1,
                         angle=2*np.pi*rot_fraction)

        return shapes

    def translate_x(self, shapes, dx):
        """
        @author: Chad Briddon
        Translates group of shapes in the x-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dx : float
            Amount the group of shapes is to be translated by in the posative
            x-direction. If negative then translation will be in the negative
            x-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.translate(shapes, dx=dx, dy=0, dz=0)

        return shapes

    def translate_y(self, shapes, dy):
        """
        @author: Chad Briddon
        Translates group of shapes in the y-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dy : float
            Amount the group of shapes is to be translated by in the posative
            y-direction. If negative then translation will be in the negative
            y-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.translate(shapes, dx=0, dy=dy, dz=0)

        return shapes

    def translate_z(self, shapes, dz):
        """
        @author: Chad Briddon
        Translates group of shapes in the z-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples representing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dz : float
            Amount the group of shapes is to be translated by in the posative
            z-direction. If negative then translation will be in the negative
            z-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples representing the group of shapes. Is identical to
            input 'shapes'.

        """

        self.geom.translate(shapes, dx=0, dy=0, dz=dz)

        return shapes

    def create_ellipse(self, rx=0.1, ry=0.1, xc=0, yc=0):
        """
        @author: Chad Briddon
        Generates an ellipse in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        rx : float, optional
            Ellipse radial size along x-axis. The default is 0.1.
        ry : float, optional
            Ellipse radial size along y-axis. The default is 0.1.

        Returns
        -------
        ellipse : list tuple
            List containing tuple representing the ellipse.

        """

        Rx = max(self.Min_length, abs(rx))
        Ry = max(self.Min_length, abs(ry))

        if Rx >= Ry:
            ellipse = [(2, self.geom.addDisk(xc=xc, yc=yc, zc=0,
                                             rx=Rx,ry=Ry))]
        else:
            ellipse = [(2, self.geom.addDisk(xc=xc, yc=yc, zc=0,
                                             rx=Ry, ry=Rx))]
            self.geom.rotate(ellipse, x=xc, y=yc, z=0, ax=0, ay=0, az=1,
                             angle=np.pi/2)

        return ellipse

    def create_exterior_mesh(self):
        """
        @author: Hugo Lévy
        Generates a solid sphere whose surface DOFs match the last created
        shape boundary. Because the exterior mesh has to be spatially separated
        from all other geometries, its center of mass is set three radii away
        from the interior domain along the x-axis.

        Returns
        -------
        exterior : list tuple
            List containing tuple representing the exterior domain (sphere).

        """
        # Retrieve Rcut
        from femtoscope.misc.util import numpyit
        bb = numpyit(self.geom.getBoundingBox(3, self.source[-1][0][1]))
        Rcut = abs(bb).min()
        Rcut = np.float64(round((Rcut)))
        self.Rcut = Rcut

        # Retrieve master boundary
        masterTags = [self.boundaries[-1][0][1]]

        # create new shape
        exterior = [(3, self.geom.addSphere(
            xc=3*Rcut, yc=0, zc=0, radius=Rcut))]
        self.geom.synchronize()
        srfTags = [gmsh.model.getBoundary(exterior)[0][1]]
        translation = [1, 0, 0, 3*Rcut, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(2, srfTags, masterTags, translation)

        return exterior

    def create_rectangle(self, dx=0.2, dy=0.2, xll=0, yll=0,
                         centered=False, vperiodic=False, hperiodic=False):
        """
        @author: Chad Briddon & Hugo Lévy
        Generates a rectangle in an open gmsh window with its centre of mass
        at the origin.

        Parameters
        ----------
        dx : float, optional
            Length of rectangle along x-axis. The default is 0.2.
        dy : float, optional
            Length of rectangle along y-axis. The default is 0.2.
        xll : float, optional
            Low left corner x-coordinate. The default is 0.
        yll : float, optional
            Low left corner y-coordinate. The default is 0.
        centered : bool, optional
            Whether the rectangle is origin-centered. The default is False.
        vperiodic : bool, optional
            Whether the mesh is periodic in the vertical direction.
            The default is False.
        hperiodic : bool, optional
            Whether the mesh is periodic in the horizontal direction.
            The default is False.

        Returns
        -------
        rectangle : list tuple
            List containing tuple representing the rectangle.

        """

        Dx = max(self.Min_length, abs(dx))
        Dy = max(self.Min_length, abs(dy))

        if centered:
            rectangle = [(2, self.geom.addRectangle(x=-Dx/2, y=-Dy/2, z=0,
                                                    dx=Dx, dy=Dy))]
        else:
            rectangle = [(2, self.geom.addRectangle(x=xll, y=yll, z=0,
                                                    dx=Dx, dy=Dy))]

        self.geom.synchronize()
        tags = MeshingTools.extractTag(self.geom.getEntities(1)[-4:])

        if vperiodic:
            vtranslation = [1, 0, 0, 0, 0, 1, 0, Dy, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [tags[0]], [tags[2]], vtranslation)

        if hperiodic:
            htranslation = [1, 0, 0, Dx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [tags[3]], [tags[1]], htranslation)

        return rectangle

    def create_ellipsoid(self, rx=0.1, ry=0.1, rz=0.1):
        """
        @author: Chad Briddon
        Generates an ellipsoid in an open gmsh window with its centre of mass
        at the origin.

        Parameters
        ----------
        rx : float, optional
            Ellipsoid radial size along x-axis. The default is 0.1.
        ry : float, optional
            Ellipsoid radial size along y-axis. The default is 0.1.
        rz : float, optional
            Ellipsoid radial size along z-axis. The default is 0.1.

        Returns
        -------
        ellipsoid : list tuple
            List containing tuple representing the ellipsoid.

        """

        Rx = max(self.Min_length, abs(rx))
        Ry = max(self.Min_length, abs(ry))
        Rz = max(self.Min_length, abs(rz))

        ellipsoid = [(3, self.geom.addSphere(xc=0, yc=0, zc=0, radius=1))]
        self.geom.dilate(ellipsoid, x=0, y=0, z=0, a=Rx, b=Ry, c=Rz)

        return ellipsoid

    def create_box(self, dx=0.2, dy=0.2, dz=0.2):
        """
        @author: Chad Briddon
        Generates a box in an open gmsh window with its centre of mass at the
        origin.

        Parameters
        ----------
        dx : float, optional
            Length of box along x-axis. The default is 0.2.
        dy : float, optional
            Length of box along y-axis. The default is 0.2.
        dz : float, optional
            Length of box along z-axis. The default is 0.2.

        Returns
        -------
        box : list tuple
            List containing tuple representing the box.

        """

        Dx = max(self.Min_length, abs(dx))
        Dy = max(self.Min_length, abs(dy))
        Dz = max(self.Min_length, abs(dz))

        box = [(3, self.geom.addBox(x=-Dx/2, y=-Dy/2, z=-Dz/2, dx=Dx,
                                    dy=Dy, dz=Dz))]

        return box

    def create_cylinder(self, Length=0.1, r=0.1):
        """
        @author: Chad Briddon
        Generates a cylinder in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        Length : float, optional
            Length of cylinder. The default is 0.1.
        r : float, optional
            Radial size of cylinder. The default is 0.1.

        Returns
        -------
        cylinder : list tuple
            List containing tuple representing the cylinder.

        """

        L = max(self.Min_length, abs(Length))
        R = max(self.Min_length, abs(r))

        cylinder = [(3, self.geom.addCylinder(x=0, y=0, z=-L/2, dx=0, dy=0,
                                              dz=L, r=R))]

        return cylinder

    def create_cone(self, Length=0.1, r=0.1):
        """
        @author: Chad Briddon
        Generates a cone in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        Length : float, optional
            Length between tip and base of the cone. The default is 0.1.
        r : float, optional
            Radial size at the base of the cone. The default is 0.1.

        Returns
        -------
        cone : list tuple
            List containing tuple representing the cone.

        """

        L = max(self.Min_length, abs(Length))
        R = max(self.Min_length, abs(r))

        cone = [(3, self.geom.addCone(x=0, y=0, z=-L/4, dx=0, dy=0, dz=L,
                                      r1=R, r2=0))]

        return cone

    def create_torus(self, r_hole=0.1, r_tube=0.1):
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
            List containing tuple representing the torus.

        """

        R_hole = max(self.Min_length, abs(r_hole))
        R_tube = max(self.Min_length, abs(r_tube))

        torus = [(3, self.geom.addTorus(x=0, y=0, z=0, r1=R_hole+R_tube,
                                        r2=R_tube))]

        return torus

    def create_disk_from_pts(self, radius, N=100):
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
        list
            List of a single dimTag.

        """
        angle = (np.linspace(0, 2*np.pi, N))[:-1]
        X = (radius * np.cos(angle))[:, np.newaxis]
        Y = (radius * np.sin(angle))[:, np.newaxis]
        Z = np.zeros_like(X)
        coors = np.concatenate((X, Y, Z), axis=1)
        return [self.points_to_surface(coors)]

    def extractTag(dimTags):
        """Fetch tags from a list of (dim, tag) [@author: Hugo Lévy]"""
        out = []
        for pair in dimTags:
            out.append(pair[1])
        return out

def converter4sfepy(fullfilename, ignoreTags=[], exterior=False, Rcut=None):
    """
    @author: Hugo Lévy
    Converts the .vtk mesh written by Gmsh into a new .vtk which facilitates
    topological entity selection with Sfepy. Vertices are given a unique group
    id following the convention:
        [0 - 99]    --> vertex tag (i.e. entity of dimension 0)
        [100 - 199] --> edge tag (i.e. entity of dimension 1)
        [200 - 299] --> facet tag (i.e. entity of dimension D-1)
        [300 - xxx] --> cell tag (i.e. entity of dimension D)
    If one node belong to several topological entities, it will be tagged with
    the lowest dimension group id. For instance, a vertex belonging to a facet
    of tag 200 and a subvolume of tag 300 will be tagged 200. This is
    problematic for subvolume selection. This difficulty is overcome by taking
    advantage of the 'mat_ids' field that is readable by Sfepy.

    Parameters
    ----------
    fullfilename : str
        Absolute pathname of the .vtk mesh.
    ignoreTags : list, optional
        List of tags to be ignored in the new .vtk mesh. The default is [].
    exterior : bool, optional
        If True, assume that the existing meshfile actually contains two
        meshes to be separated into two distinct meshfiles.
        The default is False.

    """

    from sfepy.discrete.fem import Mesh # Sfepy I/O utilities
    mesh = Mesh.from_file(fullfilename)
    cell_dim = mesh.dim

    # All the necessary & sufficient info for defining a mesh
    data = list(mesh._get_io_data(cell_dim_only=cell_dim)) # to be mod

    # Managing vertex groups
    ngroups = np.array([None for _ in range(mesh.n_nod)])
    reader = meshio.read(fullfilename)
    for key in reader.cells_dict.keys():
        conn = reader.cells_dict[key]
        formerTags = list(reader.cell_data_dict.values())[0][key]
        for k, tag in enumerate(formerTags):
            if tag not in ignoreTags:
                for idx in conn[k]:
                    if ngroups[idx] is None:
                        ngroups[idx] = int(tag)
    ngroups[np.where(ngroups==None)] = 400 # default marker for untagged
    data[1] = ngroups.astype(dtype=np.int32)

    # Managing cell groups
    conns = list(reader.cells_dict.values())[-1] # entities of highest dim
    mat_ids = np.max(ngroups[conns], axis=1)
    data[3] = [mat_ids.astype(dtype=np.int32)]

    # Overwrite the former mesh
    if exterior:
        if Rcut == None:
            raise Exception("Rcut needs to be specified!")
        out = split_mesh_spheres(Rcut, data, mesh)
    else:
        mesh = Mesh.from_data(mesh.name, *data)
        mesh.write(fullfilename, None, binary=False)
        out = fullfilename
        
    return out


def mesh_from_geo(geofile, geo_dir=None, param_dic={}, show_mesh=False,
                  ignoreTags=[], convert=True, **kwargs):
    """
    Generate a mesh from an existing .geo file with Gmsh.

    Parameters
    ----------
    geofile : str
        Name of the .geo file.
    geo_dir : str, optional
        Directory where the .geo file is located. The default is None, in which
        case the file will be sought in the MESH_DIR/geo directory.
    param_dic : dict
        Dictionary containing key/value pairs to be modified.
        The default is {}.
    show_mesh : bool, optional
        If True will open a window to allow viewing of the generated mesh.
        The default is False.
    ignoreTags : list, optional
        List of physical groups not to be included in the final .vtk file.
        The default is [].
    convert : bool, optional
        If True, the mesh file exported by Gmsh will be converted so as to
        be readable by Sfepy. The default is True.
        
    Other Parameters
    ----------------
    exterior : bool, optional
        If True, construct a second mesh matching the main one on a
        spherical boundary (for 3D domain-splitting techniques).
        The default is False.
    verbose : bool
        Display user's information. The default is False.

    Returns
    -------
    outfile : str
        Absolute path of output file.

    """

    from pathlib import Path
    from femtoscope.misc.util import write_geo_params
    verbose = kwargs.get('verbose', False)
    exterior = kwargs.get('exterior', False)
    fullFileName = write_geo_params(geofile, param_dic, geo_dir=geo_dir)
    filename = str(Path(geofile).with_suffix('.vtk'))
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.open(fullFileName)
    gmsh.model.geo.synchronize()
    dim = gmsh.model.getDimension()
    
    if verbose:
        print('Model' + gmsh.model.getCurrent() + ' (' + str(dim) + 'D)')
    
    gmsh.model.mesh.generate(dim)
    
    if verbose>=2:
        # Inspect the log:
        log = gmsh.logger.get()
        print("Logger has recorded " + str(len(log)) + " lines")
        gmsh.logger.stop()

    if show_mesh:
        gmsh.fltk.run()

    # save the generated mesh
    fullfilename = get_meshsource(filename, ext='.vtk')
    gmsh.write(fileName=fullfilename)
    gmsh.clear()
    gmsh.finalize()

    # convert the mesh to a .vtk file readable by Sfepy
    if convert:
        outfile = converter4sfepy(fullfilename, ignoreTags=ignoreTags,
                                  exterior=exterior)
    else:
        outfile = fullfilename

    return outfile


def adjust_surface_nodes(file_ref, file_mod, surfTag):
    """
    Re-adjust the DOFs belonging to a given surface from mesh `mesh_mod` to
    match those of the reference mesh `mesh_ref`.

    Parameters
    ----------
    file_ref : str
        Absolute pathname of the .vtk reference mesh.
    file_mod : str
        Absolute pathname of the .vtk mesh to be modified.
    surfTag : int
        Identifier of the entity where the DOFs are to be adjusted.

    Returns
    -------
    None.

    """
    
    import meshio
    mesh_ref = meshio.read(file_ref)
    mesh_mod = meshio.read(file_mod)
    ind_ref = np.where(mesh_ref.point_data['node_groups']==surfTag)[0]
    ind_mod = np.where(mesh_mod.point_data['node_groups']==surfTag)[0]
    assert len(ind_ref)==len(ind_mod), "Number of DOFs does not match!"
    pts_ref = mesh_ref.points[ind_ref]
    pts_mod = mesh_mod.points[ind_mod]
    actually_modified = []
    for kk, ptmod in enumerate(pts_mod):
        if not (pts_ref - ptmod == [0, 0, 0]).prod(axis=1).any():
            ind = abs(pts_ref - ptmod).sum(axis=1).argmin()
            actually_modified.append(ind_mod[kk])
            ptmod = np.copy(pts_ref[ind]) # in place modification
            mesh_mod.points[ind_mod[kk]] = ptmod
    
    if len(actually_modified) != 0:
        # Re-save the modified mesh
        print("Adjusting the position of some boundary nodes...\n")
        from sfepy.discrete.fem import Mesh # Sfepy I/O utilities
        mesh_overwrite = Mesh.from_file(file_mod)
        coors_mod = mesh_mod.points[:, :mesh_overwrite.dim]
        mesh_overwrite.coors[actually_modified] = coors_mod[actually_modified]
        mesh_overwrite.write(file_mod, None, binary=False)
    
    

def mesh_1d_interior(rho_func, Rcut, Nsamples=int(1e5), threshold=1e-3,
                     plot_hist=False):
    """
    @author: Hugo Lévy
    Create 1D mesh with adaptive mesh refinement around local variations of
    the density distribution. Indeed, both the chameleon field and the
    gravitational potential are driven by matter distribution. Consequently,
    they are likely to exhibit important variations at density jumps.

    Parameters
    ----------
    rho_func : function
        Function of a single coordinate returning the density.
    Rcut : float
        Radius that delimits the interior domain.
    Nsamples : int, optional
        Number of samples to generate from the probability distribution
        function. The default is 100 000.
    threshold : float, optional
        Minimum probability in the pdf vector. Values below it will be set to
        the threshold value. The default is 1e-3.
    plot_hist : bool, optional
        Plot the histogram of the draw. The default is False.

    Returns
    -------
    mesh_int : Sfepy Mesh instance
        1D mesh with adaptive mesh refinement.

    """

    from sfepy.discrete.fem import Mesh
    from femtoscope.misc.util import moving_average
    from matplotlib import pyplot as plt

    # Create a Probability Distribution Function (pdf) based on the density
    # gradient.
    rr = np.linspace(0, Rcut, int(1e8))
    rho = rho_func(rr)
    grad = abs(np.gradient(rho, rr))
    grad[grad < 1e-19] = 1e-19
    lograd = np.log10(grad)
    w = 100 # Filter size
    pdf = moving_average(lograd, w, mode='same')
    pdf[-int(w/2)-1:] = pdf[-int(w/2)-1]
    pdf[:int(w/2)+1] = pdf[int(w/2)+1]
    pdf -= pdf.min()
    pdf /= pdf.max()
    pdf[pdf<threshold] = threshold # thresholding
    pdf /= pdf.sum() # normalizing the distribution

    # Generate samples from the pdf
    np.random.seed(42) # repeatability
    coors_in = np.random.choice(rr, Nsamples, p=pdf) # draw samples
    coors_in.sort()
    coors_in[0] = 0.0
    coors_in[-1] = Rcut
    coors_in = np.unique(coors_in) # remove duplicates
    coors_in = coors_in.reshape((-1, 1)) # necessary for Sfepy
    n_tot = coors_in.shape[0]
    conn = np.arange(n_tot, dtype=np.int32).repeat(2)[1:-1].reshape((-1, 2))
    mat_ids = np.zeros(n_tot - 1, dtype=np.int32)
    descs = ['1_2']
    mesh_int = Mesh.from_data('mesh_int', coors_in, None, [conn], [mat_ids],
                              descs)
    # Plot histogram
    if plot_hist:
        figsize = (5, 4)
        plt.figure(figsize=figsize)
        plt.hist(coors_in.squeeze(), bins=50, alpha=0.5, histtype='bar',
                 ec='black')
        plt.xlabel(r'$r$', fontsize=17)
        plt.ylabel('\#DOFs', fontsize=17)
        plt.xlim([0, Rcut])
        plt.tick_params(axis='both', labelsize=15)
        plt.tight_layout()
        plt.show()

    return mesh_int

def mesh_1d_exterior(params, Rcut, Nsamples=int(1e5), plot_hist=False):
    """
    @author: Hugo Lévy
    Create 1D mesh of the exterior domain with mesh refinement around rapid
    variations of the chameleon field in the inverse coordinate eta. An
    analytical study shows that dphi/deta is expected to be maximum at
    $$ \eta_{SB} = \frac{m_{vac} R_{cut}^2}{3} $$

    Parameters
    ----------
    params : dict
        Dictionary of the parameters required for the analytical approximation.
    Rcut : float
        Radius that delimits the interior domain.
    Nsamples : int, optional
        Number of samples to generate from the probability distribution
        function. The default is 100 000.
    plot_hist : bool, optional
        Plot the histogram of the draw. The default is False.

    Returns
    -------
    mesh_ext : Sfepy Mesh instance
        1D mesh with adaptive mesh refinement.

    """

    from sfepy.discrete.fem import Mesh
    from matplotlib import pyplot as plt

    # Read dictionary
    rho_vac = params['rho_vac']
    alpha = params['alpha']
    npot = params['npot']
    m_vac = np.sqrt( (npot+1) / alpha * rho_vac**((npot+2)/(npot+1)) )
    eta_SB = m_vac * Rcut**2 / 3
    eta_min = eta_SB / 2
    eta_max = min(5*eta_SB, Rcut)

    # Create a Probability Distribution Function (pdf) based the analytic
    # approximation of the chameleon field far away from the sources.
    factor = 10
    eta = np.linspace(0, Rcut, int(1e8))
    ind_eta_SB = np.argmin(abs(eta-eta_SB))
    ind_right = np.where((eta >= eta_min) & (eta <= eta_SB))[0]
    ind_left = np.where((eta >= eta_SB) & (eta <= eta_max))[0]
    d_eta_SB = abs(eta-eta_SB)
    pdf = np.ones_like(eta)
    pdf[ind_right] += factor * (1 - d_eta_SB[ind_right]/eta_min)
    pdf[ind_left] += factor * (1 - d_eta_SB[ind_left]/eta_max)
    pdf[ind_eta_SB] = 1.0 + factor
    pdf /= pdf.sum() # normalizing the distribution

    # Generate samples from the pdf
    np.random.seed(42) # repeatability
    coors_out = np.random.choice(eta, Nsamples, p=pdf) # draw samples
    coors_out.sort()
    coors_out[0] = 0.0
    coors_out[-1] = Rcut
    coors_out = np.unique(coors_out) # remove duplicates
    coors_out = coors_out.reshape((-1, 1)) # necessary for Sfepy
    n_tot = coors_out.shape[0]
    conn = np.arange(n_tot, dtype=np.int32).repeat(2)[1:-1].reshape((-1, 2))
    mat_ids = np.zeros(n_tot - 1, dtype=np.int32)
    descs = ['1_2']
    mesh_ext = Mesh.from_data('mesh_ext', coors_out, None, [conn], [mat_ids],
                              descs)
    # Plot histogram
    if plot_hist:
        figsize = (5, 4)
        plt.figure(figsize=figsize)
        plt.hist(coors_out.squeeze(), bins=50, alpha=0.5, histtype='bar',
                 ec='black')
        plt.xlabel(r'$\eta = R_{\mathrm{cut}}^2 / r$', fontsize=17)
        plt.ylabel('\#DOFs', fontsize=17)
        plt.xlim([0, Rcut])
        plt.tick_params(axis='both', labelsize=15)
        plt.tight_layout()
        plt.show()

    return mesh_ext


def split_mesh_spheres(Rcut, data, mesh):
    """
    @author: Hugo Lévy
    Split two meshed sub-shapes that are geometrically separable by a
    hyperplane into two distinct mesh files. The original mesh is then deleted.

    Parameters
    ----------
    Rcut : float
        Radius of the sphere.
    data : list
        Data list of a mesh representation in Sfepy.
    mesh : Mesh instance
        Mesh instance of the original mesh (Sfepy class).

    Returns
    -------
    outs : list
        List of absolute path of output file(s).

    """
    import os # for deleting obsolete mesh file
    from copy import deepcopy # for copying two dimensional arrays
    from sfepy.discrete.fem import Mesh # Sfepy I/O utilities
    data_int = deepcopy(data)
    data_ext = deepcopy(data)

    # classify nodes belonging to sphere_int and sphere_ext
    norms = np.linalg.norm(data[0], axis=1)
    idx_node_int = np.where(norms < Rcut + 1)[0]
    idx_node_ext = np.where(norms > Rcut + 1)[0]

    # Sort out data structures according to node class
    data_int[0] = data_int[0][idx_node_int]
    data_int[1] = data_int[1][idx_node_int]
    data_ext[0] = data_ext[0][idx_node_ext] - np.array([3*Rcut, 0, 0]).reshape(1, 3)
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
    data_int[2][0] -=  min_int
    idx_node_int -= min_int
    min_ext = conn_ext.min()
    conn_ext -= min_ext
    data_ext[2][0] -=  min_ext
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
    name_int = Path(mesh.name).with_name(name+'_int')
    name_ext = Path(mesh.name).with_name(name+'_ext')
    mesh_int = Mesh.from_data(str(name_int), *data_int)
    mesh_int.write(str(name_int.with_suffix('.vtk')), None, binary=False)
    mesh_ext = Mesh.from_data(str(name_ext), *data_ext)
    mesh_ext.write(str(name_ext.with_suffix('.vtk')), None, binary=False)
    os.remove(mesh.name+'.vtk')

    # Return new file names
    outs = []
    outs.append(str(name_int.with_suffix('.vtk')))
    outs.append(str(name_ext.with_suffix('.vtk')))
    return outs


def get_physical_group_ids(meshfile, mesh_dir=None, ext='.vtk'):
    """Returns the physical groups of a given mesh file (absolute pathname)
    [@author: Hugo Lévy]"""
    meshfile = get_meshsource(meshfile, mesh_dir=mesh_dir, ext=ext)
    reader = meshio.read(meshfile)
    return list(np.unique(reader.point_data["node_groups"]))


def get_meshsource(meshfile, mesh_dir=None, ext='.vtk'):
    """Returns the absolute pathname of a given meshfile
    [@author: Hugo Lévy]"""
    if not mesh_dir: # search file in mesh directory (default behaviour)
        meshsource = MESH_DIR / meshfile
    else:
        mesh_dir = Path(mesh_dir)
        meshsource = mesh_dir / meshfile
    if meshsource.suffix=='' or meshsource.suffix!=ext:
        meshsource = meshsource.with_suffix(ext)
    return str(meshsource)


def get_meshdim(meshfile, mesh_dir=None, ext='.vtk'):
    """Return the dimension of a mesh [@author: Hugo Lévy]"""
    meshsource = get_meshsource(meshfile, mesh_dir=mesh_dir, ext=ext)
    reader = meshio.read(meshsource)
    Z = reader.points[:, 2]
    if (Z==0.0).all():
        return 2
    else:
        return 3


def get_rcut(meshfile, mesh_dir=None, ext='.vtk', coorsys='cartesian'):
    """Returns the truncation radius [@author: Hugo Lévy]"""
    meshsource = get_meshsource(meshfile, mesh_dir=mesh_dir, ext=ext)
    reader = meshio.read(meshsource)
    coors = reader.points
    if coorsys == 'cartesian':
        return float(np.max(np.linalg.norm(coors, axis=1)))
    elif coorsys in ['polar', 'polar_mu']:
        return float(np.max(coors[:, 0]))
    else:
        raise Exception("Not implemented coordinates system: %s" %coorsys)
