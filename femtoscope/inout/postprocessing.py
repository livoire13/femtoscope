# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:43:55 2023
Creation of a new class for post-processing .vtk raw data.

@author: hlevy
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, cos, sin, sqrt
from pathlib import Path
import meshio
from collections.abc import Iterable

from femtoscope import RESULT_DIR, MESH_DIR
from femtoscope.inout.meshfactory import get_physical_group_ids
from femtoscope.misc.analytical import PotentialByIntegration
from femtoscope.misc.sph import AxiSymSph
from femtoscope.misc.constants import (C_LIGHT, EV, H_BAR, G_GRAV, MU_EARTH,
                                       R_EARTH, M_EARTH, LAMBDA_DE, M_PL)
from femtoscope.misc.unit_conversion import (compute_Lambda, compute_beta,
                                             compute_phi_0, compute_alpha,
                                             mass_to_nat, nat_to_acc)

class PostProcessResult():
    """
    Class for post-processing FEM ouputs presaved in .vtk files.

    Attributes
    ----------
    meshfiles : str or list of str
        Mesh files (.vtk) are used to reconstruct the problem instance.
    resultfiles : str or list of str
        Result files (.vtk) contain the FEM solution as well as some other
        relevant fields.
    fem_order : int
        The FE approximation order.
    coorsys : str
        The set of coordinates to be used. The default is 'cartesian'.
    conn : str
        Method for linking the interior domain with the exterior domain.
        The default is 'connected'.
    alpha : float
        Physical parameter weighting the laplacian operator of the
        Klein-Gordon equation (dimensionless).
    islinear : bool
        Whether or not the PDE to be solved is linear.
    mesh_dir : str
        Directory where the mesh files are located. The default is `MESH_DIR`.
    result_dir : str
        Directory where the result files are located.
        The default is `RESULT_DIR`.
    int_only : bool
        Whether the solution on the exterior domain should be returned or not.

    """
    def __init__(self, meshfiles, resultfiles, fem_order=2, coorsys=None,
                 conn=None, alpha=None, islinear=False, **kwargs):

        if len(meshfiles) != len(resultfiles):
            raise ValueError("""User should provide as many mesh-files as
                             result files!""")

        self.meshfiles = meshfiles
        self.resultfiles = resultfiles
        self.fem_order = fem_order
        self.coorsys = coorsys if coorsys is not None else 'cartesian'
        if len(meshfiles)==2:
            self.conn = conn if conn is not None else 'connected'
        self.alpha = alpha
        self.islinear = islinear and (alpha is None)
        self.mesh_dir = kwargs.get('mesh_dir', None)
        self.result_dir = kwargs.get('result_dir', RESULT_DIR)
        self.int_only = kwargs.get('int_only', True) or (len(meshfiles)==1)

        self.dic_ext = None
        data = self.retrieve_data()
        if self.int_only:
            self.dic_int = data
        else:
            self.dic_int = data[0]
            self.dic_ext = data[1]


    def retrieve_data(self):

        # Determine the number of physical groups of the interior mesh
        if isinstance(self.meshfiles, list):
            meshfileint = self.meshfiles[0]
        else:
            meshfileint = self.meshfiles
        physical_groups_ids = np.array(get_physical_group_ids(meshfileint))
        nb_pg_vol = int(np.sum(physical_groups_ids>=300))
        nb_pg_fac = int(np.sum((physical_groups_ids>=200)
                             & (physical_groups_ids<300)))
        densities_int = nb_pg_vol * [1]
        densities_ext = 1

        # Problem on Bounded domain
        if len(self.resultfiles)==1:

            # Poisson
            if self.islinear:
                from femtoscope.physics.poisson import PoissonBounded
                pb = PoissonBounded(4*pi, densities_int, [0],
                                    meshfileint, coorsys=self.coorsys,
                                    fem_order=self.fem_order,
                                    mesh_dir=self.mesh_dir)
            # Chameleon
            else:
                from femtoscope.physics.chameleon import ChameleonBounded
                if self.coorsys == 'cartesian':
                    dbc = [0]
                elif self.coorsys in ['polar', 'polar_mu']:
                    dbc = nb_pg_fac * [0]
                else:
                    raise ValueError(
                        "Unknown coordinate system: {}".format(self.coorsys))
                pb = ChameleonBounded(self.alpha, 1, densities_int, dbc,
                                      [0.1, 1], meshfileint,
                                      fem_order=self.fem_order,
                                      coorsys=self.coorsys,
                                      mesh_dir=self.meshdir)
        # Problem on Unbounded domain
        elif len(self.resultfiles)==2:

            # Poisson
            if self.islinear:
                from femtoscope.physics.poisson import PoissonSplit
                pb = PoissonSplit(4*pi, densities_int, densities_ext,
                                  self.meshfiles, coorsys=self.coorsys,
                                  fem_order=self.fem_order, conn=self.conn,
                                  mesh_dir=self.mesh_dir)
            # Chameleon
            else:
                from femtoscope.physics.chameleon import ChameleonSplit
                dirichlet_bc_facet_int = None
                if nb_pg_fac == 2:
                    dirichlet_bc_facet_int = [None, 0]
                pb = ChameleonSplit(self.alpha, 1, densities_int,
                                    densities_ext, [0.1, 1], self.meshfiles,
                                    coorsys=self.coorsys, conn=self.conn,
                                    fem_order=self.fem_order,
                                    dirichlet_bc_facet_int=dirichlet_bc_facet_int)
        else:
            raise ValueError("""1 or 2 result files should be provided:
                             - 1 for FEM on a bounded domain
                             - 2 for domain splitting + Kelvin inversion
                             techniques for dealing withasymtotic BC.""")

        # Quantities of interest
        dic_int = {}
        if isinstance(self.resultfiles, Iterable):
            resultfiles = self.resultfiles
        else:
            resultfiles = [resultfiles]
        fullname_int = str(Path(Path(self.result_dir) / resultfiles[0]))
        datavtk_int = meshio.read(fullname_int)
        dic_int['coors'] = datavtk_int.points[:, :2]
        dic_int['field_int'] = pb.solver.weakforms[0].field
        dic_int['sol_int'] = np.ascontiguousarray(
            datavtk_int.point_data['u1']).byteswap().newbyteorder()
        if not self.islinear:
            dic_int['res'] = np.ascontiguousarray(
                datavtk_int.point_data['residual'].byteswap()).newbyteorder()
            if 'mtx' in datavtk_int.point_data:
                dic_int['res_mtx'] = np.ascontiguousarray(
                    datavtk_int.point_data['mtx'].byteswap()).newbyteorder()
                dic_int['res_rho'] = np.ascontiguousarray(
                    datavtk_int.point_data['rhs_cst'].byteswap()).newbyteorder()
                dic_int['res_pow'] = np.ascontiguousarray(
                    datavtk_int.point_data['rhs_mod'].byteswap()).newbyteorder()

        if not self.int_only:
            dic_ext = {}
            fullname_ext = str(Path(Path(self.result_dir) / resultfiles[1]))
            datavtk_ext = meshio.read(fullname_ext)
            dic_ext['coors'] = datavtk_ext.points[:, :2]
            dic_ext['field_ext'] = pb.solver.weakforms[0].field
            dic_ext['sol_ext'] = np.ascontiguousarray(
                datavtk_ext.point_data['u2']).byteswap().newbyteorder()
            return dic_int, dic_ext

        return dic_int

    def evaluate_fem(self, coor1, coor2, keys, mode='val', domain='int'):
        """
        Evaluate some source DOFs values at (coor1, coor2) using FEM
        interpolation. The user may input DOFs of several types using the
        `keys` argument.

        Parameters
        ----------
        coor1 : 1d-array or float
            First coordinate.
        coor2 : 1d-array or float
            Second coordinate.
        keys : str or list of str
            One or several string corresponding to `dic_int` or `dic_ext` keys.
        mode : {‘val’, ‘grad’, ‘div’, ‘cauchy_strain’}, optional
            The evaluation mode: the field value (default), the field value
            gradient, divergence, or cauchy strain.
        domain : str or list of str, optional
            Are the DOFs values in the interior domain or in the exterior
            domain ? The default is 'int'.

        Returns
        -------
        array or list of arrays
            The interpolated values at (coor1, coor2).

        """
        # Handling of keys, mode, domain
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(mode, str):
            mode = [mode] * len(keys)
        if isinstance(domain, str):
            domain = [domain] * len(keys)

        # Handling of coordinates
        if isinstance(coor1, Iterable):
            coors = np.ascontiguousarray(np.concatenate(
                (coor1[:, np.newaxis], coor2[:, np.newaxis]), axis=1))
        else:
            coors = np.array([float(coor1), float(coor2)]).reshape(-1, 2)

        evs = []
        for kk in range(len(keys)):
            # retrieve the field and source values to evaluate
            if domain[kk] == 'int':
                field = self.dic_int['field_int']
                source_vals = self.dic_int[keys[kk]]
            elif domain[kk] == 'ext':
                field = self.dic_ext['field_ext']
                source_vals = self.dic_ext[keys[kk]]
            else:
                raise ValueError("domain should be 'int' or 'ext'")

            # use Sfepy built-in function to evaluate
            ev = field.evaluate_at(
                coors, source_vals[:, np.newaxis], mode=mode[kk])
            if mode[kk] == 'grad' and self.coorsys == 'polar':
                ev = ev.reshape(-1, 2)
                ev[:, 1] /= coor1
            evs.append(ev.squeeze())

        if len(keys) == 1:
            return evs[0]
        return evs
