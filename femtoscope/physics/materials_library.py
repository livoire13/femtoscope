# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:10:34 2024

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Callable, Union
from functools import wraps

import numpy as np
from numpy import sin, sqrt

valid_coorsys = ('polar', 'cylindrical', 'cartesian')


def check_mode_qp(mat_func):
    """Decorator for not repeating the check on 'mode' parameter in each
    function definition."""
    @wraps(mat_func)
    def wrapper(*args, **kwargs):
        mode = kwargs.get('mode')
        if mode != 'qp':
            return
        return mat_func(*args, **kwargs)

    return wrapper


class AbstractMaterials(ABC):
    """
    Template class for storing material functions of a certain type.
    """

    def get_material(self, dim, coorsys=None, tag='int') -> Callable:
        """
        Retrieve material function according to input parameters.

        Parameters
        ----------
        dim : int
            Dimension of the problem.
        coorsys : str
            Coordinate system. For valid strings, see `valid_coorsys`
        tag : str
            'int' for interior domain and 'ext' for exterior domain.

        Returns
        -------
        func : Callable
            Material function matching input parameters.

        """

        _check_parameters(dim, coorsys, tag)  # perform checks

        if dim == 1:  # Dimension 1
            if coorsys == 'polar':
                if tag == 'int':
                    func = self.mat1dpolint
                else:
                    func = self.mat1dpolext
            elif coorsys == 'cartesian':
                if tag == 'int':
                    func = self.mat3dint
                else:
                    func = self.mat1dcartext
            else:
                raise NotImplementedError

        elif dim == 2:  # Dimension 2
            if coorsys == 'polar':
                if tag == 'int':
                    func = self.mat2dpolint
                else:
                    func = self.mat2dpolext
            elif coorsys == 'cylindrical':
                if tag == 'int':
                    func = self.mat2dcylint
                else:
                    func = self.mat2dcylext
            elif coorsys == 'cartesian':
                if tag == 'int':
                    func = self.mat3dint
                else:
                    func = self.mat2dcartext
            else:
                raise ValueError(f"Invalid coordinate system '{coorsys}'.")

        else:  # Dimension 3
            if coorsys == 'cartesian':
                if tag == 'int':
                    func = self.mat3dint
                else:
                    func = self.mat3dext
            else:
                raise NotImplementedError

        return func

    @staticmethod
    @abstractmethod
    def mat1dpolint(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat1dpolext(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat1dcartext(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat2dpolint(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat2dpolext(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat2dcylint(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat2dcylext(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat2dcartext(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat3dint(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat3dext(*args, **kwargs):
        raise NotImplementedError


class LaplacianMaterials(AbstractMaterials):
    """
    Shelf for storing material functions related to the laplacian term.
    Whatever the dimension and/or the coordinate system, we consider the Laplace
    operator in three dimensions.
    """

    # 1D material functions
    @staticmethod
    @check_mode_qp
    def mat1dpolint(ts, coors, mode=None, **kwargs):
        r = coors.squeeze()
        val = r ** 2
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dpolext(ts, coors, mode=None, Rc=None, **kwargs):
        eta = coors.squeeze()
        val = eta ** 4 / Rc ** 2 * (5 - 4 * eta / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dcartext(ts, coors, mode=None, Rc=None, **kwargs):
        xi = coors.squeeze()
        val = (xi / Rc) ** 4 * (3 - 2 * abs(xi) / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    # 2D polar coordinates material functions
    @staticmethod
    @check_mode_qp
    def mat2dpolint(ts, coors, mode=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 2))
        r, theta = coors[:, 0], coors[:, 1]
        val[:, 0, 0] = sin(theta) * r ** 2
        val[:, 1, 1] = sin(theta)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dpolext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 2))
        eta, theta = coors[:, 0], coors[:, 1]
        val[:, 0, 0] = eta ** 4 / Rc ** 2 * (5 - 4 * eta / Rc) * sin(theta)
        val[:, 1, 1] = eta ** 2 / Rc ** 2 * (5 - 4 * eta / Rc) * sin(theta)
        return {'val': val}

    # 2D cyclindrical coordinates material functions
    @staticmethod
    @check_mode_qp
    def mat2dcylint(ts, coors, mode=None, **kwargs):
        x = coors[:, 0]
        val = abs(x).reshape(-1, 1, 1)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcylext(ts, coors, mode=None, Rc=None, **kwargs):
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val = norm2 ** 2 / Rc ** 4 * abs(xi) * (7 - 6 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dcartext(ts, coors, mode=None, Rc=None, **kwargs):
        norm = np.linalg.norm(coors, axis=1)
        val = (norm / Rc) ** 4 * (5 - 4 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    # 3D cartesian coordinates material functions
    mat3dint = None

    @staticmethod
    @check_mode_qp
    def mat3dext(ts, coors, mode=None, Rc=None, **kwargs):
        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
        norm2 = xi ** 2 + eta ** 2 + zeta ** 2
        norm = sqrt(norm2)
        val = norm2 ** 2 / Rc ** 4 * (7 - 6 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}


class LaplacianMaterialsVacuum(LaplacianMaterials):
    """
    Shelf for storing Laplace material functions in the particular case of the
    Poisson equation where the rhs has compact support.

    Notes
    -----
    Overrides methods mat1dpolext, mat2dpolext, mat2dcylext, mat2dcartext,
    mat3dext from parent class.
    """

    mat1dpolext = None

    @staticmethod
    @check_mode_qp
    def mat1dcartext(ts, coors, mode=None, Rc=None, **kwargs):
        xi = coors.squeeze()
        val = (xi / Rc) ** 2
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dpolext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 2))
        eta = coors[:, 0]
        theta = coors[:, 1]
        val[:, 0, 0] = (3 - 2 * eta / Rc) * sin(theta) * eta ** 2
        val[:, 1, 1] = (3 - 2 * eta / Rc) * sin(theta)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcylext(ts, coors, mode=None, Rc=None, **kwargs):
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val = norm2 / Rc ** 2 * abs(xi) * (5 - 4 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    mat2dcartext = None

    @staticmethod
    @check_mode_qp
    def mat3dext(ts, coors, mode=None, Rc=None, **kwargs):
        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
        norm2 = xi ** 2 + eta ** 2 + zeta ** 2
        norm = sqrt(norm2)
        val = norm2 / Rc ** 2 * (5 - 4 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}


class LapAdvectionMaterials(AbstractMaterials):
    """
    Shelf for storing advection material functions originating from weighting
    the PDE in the exterior domain.
    """

    mat1dpolint = None
    mat2dpolint = None
    mat2dcylint = None
    mat3dint = None

    @staticmethod
    @check_mode_qp
    def mat1dpolext(ts, coors, mode=None, Rc=None, **kwargs):
        eta = coors.squeeze()
        val = 20 * eta ** 3 / Rc ** 2 * (1 - eta / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dcartext(ts, coors, mode=None, Rc=None, **kwargs):
        xi = coors.squeeze()
        val = 6 * xi ** 3 / Rc ** 5 * (Rc - abs(xi))
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dpolext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 1))
        eta, theta = coors[:, 0], coors[:, 1]
        val[:, 0, 0] = 20 * eta ** 3 / Rc ** 2 * (1 - eta / Rc) * sin(theta)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcylext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 1))
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val[:, 0, 0] = abs(xi) * 42 * norm2 / Rc ** 4 * (1 - norm / Rc) * xi
        val[:, 1, 0] = abs(xi) * 42 * norm2 / Rc ** 4 * (1 - norm / Rc) * eta
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcartext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 1))
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val[:, 0, 0] = 20 * norm2 / Rc ** 5 * (Rc - norm) * xi
        val[:, 1, 0] = 20 * norm2 / Rc ** 5 * (Rc - norm) * eta
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat3dext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 3, 1))
        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
        norm2 = xi ** 2 + eta ** 2 + zeta ** 2
        norm = sqrt(norm2)
        val[:, 0, 0] = 42 * norm2 / Rc ** 4 * (1 - norm / Rc) * xi
        val[:, 1, 0] = 42 * norm2 / Rc ** 4 * (1 - norm / Rc) * eta
        val[:, 2, 0] = 42 * norm2 / Rc ** 4 * (1 - norm / Rc) * zeta
        return {'val': val}


class LapAdvectionMaterialsVacuum(LapAdvectionMaterials):
    """
    Shelf for storing advection material functions originating from weighting
    the PDE in the exterior domain in the particular case of the Poisson
    equation where the rhs has compact support.

    Notes
    -----
    Overrides methods 'mat1dpolext', 'mat1dcartext' 'mat2dpolext',
    'mat2dcylext', 'mat2dcartext' 'mat3dext' from parent class.
    """

    mat1dpolext = None
    mat1dcartext = None

    @staticmethod
    @check_mode_qp
    def mat2dpolext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 1))
        eta, theta = coors[:, 0], coors[:, 1]
        val[:, 0, 0] = 6 * eta * (1 - eta / Rc) * sin(theta)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcylext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 1))
        xi, eta = coors[:, 0], coors[:, 1]
        norm2 = xi ** 2 + eta ** 2
        norm = sqrt(norm2)
        val[:, 0, 0] = 20 * abs(xi) / Rc ** 2 * (1 - norm / Rc) * xi
        val[:, 1, 0] = 20 * abs(xi) / Rc ** 2 * (1 - norm / Rc) * eta
        return {'val': val}

    mat2dcartext = None

    @staticmethod
    @check_mode_qp
    def mat3dext(ts, coors, mode=None, Rc=None, **kwargs):
        val = np.zeros((coors.shape[0], 3, 1))
        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
        norm = sqrt(xi ** 2 + eta ** 2 + zeta ** 2)
        val[:, 0, 0] = 20 / Rc ** 2 * (1 - norm / Rc) * xi
        val[:, 1, 0] = 20 / Rc ** 2 * (1 - norm / Rc) * eta
        val[:, 2, 0] = 20 / Rc ** 2 * (1 - norm / Rc) * zeta
        return {'val': val}


class LapSurfaceMaterials(AbstractMaterials):
    """
    Shelf for storing material functions associated with the surface term
    arising from the Laplacian.
    """

    mat1dpolext = None
    mat1dcartext = None
    mat2dpolext = None
    mat2dcylext = None
    mat2dcartext = None
    mat3dext = None
    mat3dint = None

    @staticmethod
    @check_mode_qp
    def mat1dpolint(ts, coors, mode=None, **kwargs):
        r = coors.squeeze()
        val = np.zeros((coors.shape[0], 1, 1))
        val[:, 0, 0] = r ** 2
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dpolint(ts, coors, mode=None, **kwargs):
        val = np.zeros((coors.shape[0], 2, 2))
        r, theta = coors[:, 0], coors[:, 1]
        val[:, 0, 0] = sin(theta) * r ** 2
        val[:, 1, 1] = sin(theta)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcylint(ts, coors, mode=None, **kwargs):
        x = coors[:, 0]
        val = np.zeros((coors.shape[0], 2, 2))
        val[:, 0, 0] = abs(x)
        val[:, 1, 1] = abs(x)
        return {'val': val}


class DensityMaterials(AbstractMaterials):
    """
    Shelf for storing material functions related to a density term.
    """

    @staticmethod
    @check_mode_qp
    def mat1dpolint(ts, coors, mode=None, rho=None, **kwargs):
        if mode != 'qp': return
        r = coors.squeeze()
        if callable(rho): rho = rho(r)
        val = rho * r ** 2
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dpolext(ts, coors, mode=None, rho=None, Rc=None, **kwargs):
        eta = coors.squeeze()
        if callable(rho): rho = rho(eta)
        val = rho * Rc ** 2 * (5 - 4 * eta / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dcartext(ts, coors, mode=None, rho=None, Rc=None, **kwargs):
        xi = coors.squeeze()
        if callable(rho): rho = rho(xi)
        val = rho * (3 - 2 * abs(xi) / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dpolint(ts, coors, mode=None, rho=None, **kwargs):
        if callable(rho): rho = rho(coors)
        r, theta = coors[:, 0], coors[:, 1]
        val = rho * sin(theta) * r ** 2
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dpolext(ts, coors, mode=None, rho=None, Rc=None, **kwargs):
        if callable(rho): rho = rho(coors)
        eta, theta = coors[:, 0], coors[:, 1]
        val = rho * Rc ** 2 * sin(theta) * (5 - 4 * eta / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dcylint(ts, coors, mode=None, rho=None, **kwargs):
        if callable(rho): rho = rho(coors)
        x = coors[:, 0]
        val = (rho * abs(x)).reshape(-1, 1, 1)
        return {'val': val}

    @staticmethod
    @check_mode_qp
    def mat2dcylext(ts, coors, mode=None, rho=None, Rc=None, **kwargs):
        if callable(rho): rho = rho(coors)
        xi, eta = coors[:, 0], coors[:, 1]
        norm = sqrt(xi ** 2 + eta ** 2)
        val = rho * abs(xi) * (7 - 6 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dcartext(ts, coors, mode=None, rho=None, Rc=None, **kwargs):
        if callable(rho): rho = rho(coors)
        norm = np.linalg.norm(coors, axis=1)
        val = (5 - 4 * norm / Rc) * rho
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat3dint(ts, coors, mode=None, rho=None, **kwargs):
        if callable(rho): rho = rho(coors)
        val = rho * np.ones(coors.shape[0], dtype=np.float64)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat3dext(ts, coors, mode=None, rho=None, Rc=None, **kwargs):
        if callable(rho): rho = rho(coors)
        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
        norm = sqrt(xi ** 2 + eta ** 2 + zeta ** 2)
        val = rho * (7 - 6 * norm / Rc)
        return {'val': val.reshape(-1, 1, 1)}


class NonLinearMaterials(AbstractMaterials):
    """
    Shelf for storing material functions associated with 'mod' terms.
    """

    @staticmethod
    @check_mode_qp
    def mat1dpolint(ts, coors, mode=None, vec_qp=None, nl_fun=None, **kwargs):
        r = coors.squeeze()
        val = r ** 2 * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dpolext(
            ts, coors, mode=None, vec_qp=None, nl_fun=None, Rc=None, **kwargs):
        eta = coors.squeeze()
        val = Rc ** 2 * (5 - 4 * eta / Rc) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat1dcartext(
            ts, coors, mode=None, vec_qp=None, nl_fun=None, Rc=None, **kwargs):
        xi = coors.squeeze()
        val = (3 - 2 * abs(xi) / Rc) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dpolint(ts, coors, mode=None, vec_qp=None, nl_fun=None, **kwargs):
        r, theta = coors[:, 0], coors[:, 1]
        val = sin(theta) * r ** 2 * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dpolext(
            ts, coors, mode=None, vec_qp=None, nl_fun=None, Rc=None, **kwargs):
        eta, theta = coors[:, 0], coors[:, 1]
        val = Rc ** 2 * (5 - 4 * eta / Rc) * sin(theta) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dcylint(ts, coors, mode=None, vec_qp=None, nl_fun=None, **kwargs):
        x = coors[:, 0]
        val = abs(x) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dcylext(
            ts, coors, mode=None, vec_qp=None, nl_fun=None, Rc=None, **kwargs):
        xi, eta = coors[:, 0], coors[:, 1]
        norm = sqrt(xi ** 2 + eta ** 2)
        val = abs(xi) * (7 - 6 * norm / Rc) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat2dcartext(
            ts, coors, mode=None, vec_qp=None, nl_fun=None, Rc=None, **kwargs):
        norm = np.linalg.norm(coors, axis=1)
        val = (5 - 4 * norm / Rc) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat3dint(ts, coors, mode=None, vec_qp=None, nl_fun=None, **kwargs):
        val = nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}

    @staticmethod
    @check_mode_qp
    def mat3dext(
            ts, coors, mode=None, vec_qp=None, nl_fun=None, Rc=None, **kwargs):
        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
        norm = sqrt(xi ** 2 + eta ** 2 + zeta ** 2)
        val = (7 - 6 * norm / Rc) * nl_fun(vec_qp)
        return {'val': val.reshape(-1, 1, 1)}


def _check_parameters(dim, coorsys, tag):
    # Check dimension
    if dim not in (1, 2, 3):
        raise ValueError(f"dim = {dim} is not a valid dimension!")
    # Check coordinate system
    if coorsys is not None and coorsys not in valid_coorsys:
        raise ValueError(f"coorsys = {coorsys} is not a valid coordinate "
                         f"system, should be in {valid_coorsys}.")
    # Check tag
    if tag not in ('int', 'ext'):
        raise ValueError("'tag' should be either 'int' or 'ext'.")
