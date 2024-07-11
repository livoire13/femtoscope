# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:48:36 2024

Semi-analytical method for computing the Newtonian potential created by a given
distribution of mass.

Additional documentation
------------------------
Scipy `tpl_quad` method
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tplquad.html

BSpline implementation
https://stackoverflow.com/questions/28279060/splines-with-python-using-control-knots-and-endpoints
https://github.com/kawache/Python-B-spline-examples

"""

from functools import partial

import numpy as np
from numpy import sin, cos, pi, sqrt
from scipy.integrate import tplquad


class PotentialByIntegration:
    """
    Class for computing the gravitational potential through direct numerical
    integration (hence we refer to this method as 'semi-analytical').

    Attributes
    ----------
    coorsys : str
        The set of coordinates to be used ('cartesian' or 'spherical' or
        'cylindrical').
    determinant : function
        Pre-factor of the volume element (coordinate-system-dependent).
    denom : function
        Denominator |r-r'| of the integrand.
    integrand : function
        Function to be integrated with `scipy.integrate.tplquad`, which gives
        the gravitational potential value.
    rho : function
        Density function (arbitrary function with compact support).
    a, b, gfun, hfun, qfun, rfun : parameters (see scipy doc in the link below)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tplquad.html

    Methods
    -------
    eval_mass :
        Computes the total mass based on the mass distribution provided
        by the user.
    eval_potential :
        Computes the gravitational potential at the desired location(s).

    """

    def __init__(self, coorsys):
        self.coorsys = coorsys
        self.determinant = None
        self.denom = None
        self.integrand = None
        self.rho = None
        self.a = None
        self.b = None
        self.gfun = None
        self.hfun = None
        self.qfun = None
        self.rfun = None

    def set_up(self, preset=None, rho=1.0, a=None, b=None, gfun=None, hfun=None,
               qfun=None, rfun=None, **kwargs):
        """
        Set up the `PotentialByIntegration` instance. If 'preset' is set to
        None, all the subsequent keyword arguments must be specified.

        """

        coorsys = self.coorsys

        if coorsys == 'cartesian':
            self.determinant = lambda x, y, z: 1.0

            def denom(x1, x2, y1, y2, z1, z2):
                return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

            def integrand(xp, yp, zp, xeval=None, yeval=None, zeval=None):
                return rho(xp, yp, zp) * self.determinant(xp, yp, zp) \
                    / denom(xeval, xp, yeval, yp, zeval, zp)

        elif coorsys == 'spherical':
            self.determinant = lambda r, theta, phi: r ** 2 * sin(theta)

            def denom(r1, r2, theta1, theta2, phi1, phi2):
                return sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * (
                        sin(theta1) * sin(theta2) * cos(phi1 - phi2)
                        + cos(theta1) * cos(theta2)))

            def integrand(rp, thetap, phip, reval=None, thetaeval=None,
                          phieval=None):
                return rho(rp, thetap, phip) * \
                    self.determinant(rp, thetap, phip) \
                    / denom(reval, rp, thetaeval, thetap, phieval, phip)

        elif coorsys == 'spherical_mu':
            self.determinant = lambda r, mu, phi: r ** 2

            def denom(r1, r2, mu1, mu2, phi1, phi2):
                return sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * (
                        sqrt((1 - mu1 ** 2) * (1 - mu2 ** 2))
                        * cos(phi1 - phi2) + mu1 * mu2))

            def integrand(rp, mup, phip, reval=None, mueval=None, phieval=None):
                return rho(rp, mup, phip) * self.determinant(rp, mup, phip) \
                    / denom(reval, rp, mueval, mup, phieval, phip)

        elif coorsys == 'cylindrical':
            self.determinant = lambda r, phi, z: r

            def denom(r1, r2, phi1, phi2, z1, z2):
                return sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * cos(phi1 - phi2)
                            + (z1 - z2) ** 2)

            def integrand(rp, phip, zp, reval=None, phieval=None, zeval=None):
                return rho(rp, phip, zp) * self.determinant(rp, phip, zp) \
                    / denom(reval, rp, phieval, phip, zeval, zp)

        else:
            raise ValueError(
                "{} is not a known coordinate system!".format(coorsys))

        self.denom = denom
        self.integrand = integrand

        if preset == 'sphere':
            radius = kwargs.get('radius', 1.0)
            rho_val = rho

            if coorsys == 'cartesian':
                def rho(x, y, z):
                    d = sqrt(x ** 2 + y ** 2 + z ** 2)
                    if d <= radius:
                        return rho_val
                    else:
                        return 0.0

                self.a = -1
                self.b = +1
                self.gfun = lambda x: -1
                self.hfun = lambda x: +1
                self.qfun = lambda x, y: -sqrt(
                    max(0, radius ** 2 - x ** 2 - y ** 2))
                self.rfun = lambda x, y: +sqrt(
                    max(0, radius ** 2 - x ** 2 - y ** 2))

            elif coorsys == 'spherical':
                def rho(r, theta, phi):
                    if r <= radius:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = 0.0
                self.hfun = pi
                self.qfun = 0.0
                self.rfun = radius

            elif coorsys == 'spherical_mu':
                def rho(r, mu, phi):
                    if r <= radius:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = -1
                self.hfun = +1
                self.qfun = 0.0
                self.rfun = radius

            elif coorsys == 'cylindrical':
                def rho(r, phi, z):
                    d = sqrt(r ** 2 + z ** 2)
                    if d <= radius:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = lambda x: 0.0
                self.hfun = lambda x: radius
                self.qfun = lambda theta, r: -sqrt(max(0, radius ** 2 - r ** 2))
                self.rfun = lambda theta, r: +sqrt(max(0, radius ** 2 - r ** 2))

            else:
                raise NotImplementedError(
                    "{} is not yet implemented for preset = {}".format(
                        coorsys, preset))
            self.rho = rho

        elif preset == 'mountainC0':
            radius = kwargs.get('radius', 1.0)
            thetam = kwargs.get('thetam', 0.07)
            hm = kwargs.get('hm', 1e-2)
            rho_val = rho

            if coorsys == 'spherical':

                def R_of_theta(theta):
                    if theta > thetam:
                        return radius
                    else:
                        return radius + hm - hm / thetam * theta

                def rho(r, theta, phi):
                    Rmax = R_of_theta(theta)
                    if r <= Rmax:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = lambda x: 0.0
                self.hfun = lambda x: pi
                self.qfun = lambda phi, theta: 0.0
                self.rfun = lambda phi, theta: R_of_theta(theta)

            else:
                raise NotImplementedError("{} not available".format(coorsys))

            self.rho = rho

        elif preset == 'mountainC1':
            radius = kwargs.get('radius', 1.0)
            thetam = kwargs.get('thetam', 0.07)
            hm = kwargs.get('hm', 1e-2)
            rho_val = rho

            if coorsys == 'spherical':

                # Use 3rd polynomial
                def R_of_theta(theta):
                    """R(theta) for a smooth polynomial profile [mountainC1]"""
                    expr = radius + hm + 2 * hm * (theta / thetam) ** 3 \
                           - 3 * hm * (theta / thetam) ** 2
                    if np.isscalar(theta):
                        if theta > thetam:
                            return radius
                        else:
                            return expr
                    elif isinstance(theta, np.ndarray):
                        return np.where(theta > thetam, radius, expr)
                    else:
                        raise ValueError()

                def rho(r, theta, phi):
                    Rmax = R_of_theta(theta)
                    if r <= Rmax:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = lambda x: 0.0
                self.hfun = lambda x: pi
                self.qfun = lambda phi, theta: 0.0
                self.rfun = lambda phi, theta: R_of_theta(theta)

            else:
                raise NotImplementedError("{} not available".format(coorsys))

            self.rho = rho

        elif preset == 'mountainBSpline':
            from scipy import interpolate
            radius = kwargs.get('radius', 1.0)
            hm = kwargs.get('hm', 1e-2)
            rho_val = rho

            if coorsys == 'spherical_mu':
                mum = kwargs.get('mum', 1e-2)
                plist = [(radius, 1 - mum), (radius, 1 - mum),
                         (radius, 1 - 0.7 * mum),
                         (radius + hm, 1 - 0.3 * mum), (radius + hm, 1),
                         (radius + hm, 1)]
                ctr = np.array(plist)
                x = ctr[:, 0]
                y = ctr[:, 1]
                l = len(x)
                t = np.linspace(0, 1, l - 2, endpoint=True)
                t = np.append([0, 0, 0], t)
                t = np.append(t, [1, 1, 1])
                tck = [t, [x, y], 3]
                u3 = np.linspace(0, 1, (max(l * 2, 500)), endpoint=True)
                out = np.array(interpolate.splev(u3, tck))

                def R_of_mu(mu):
                    return np.interp(mu, out[1], out[0])

                def rho(r, mu, phi):
                    Rmax = R_of_mu(mu)
                    if r <= Rmax:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = lambda x: -1
                self.hfun = lambda x: +1
                self.qfun = lambda phi, mu: 0.0
                self.rfun = lambda phi, mu: R_of_mu(mu)

            elif coorsys == 'spherical':
                thetam = kwargs.get('thetam', 1e-2)
                plist = [(radius + hm, 0), (radius + hm, 0),
                         (radius + hm, 0.3 * thetam), (radius, 0.7 * thetam),
                         (radius, thetam), (radius, thetam)]
                ctr = np.array(plist)
                x = ctr[:, 0]
                y = ctr[:, 1]
                l = len(x)
                t = np.linspace(0, 1, l - 2, endpoint=True)
                t = np.append([0, 0, 0], t)
                t = np.append(t, [1, 1, 1])
                tck = [t, [x, y], 3]
                u3 = np.linspace(0, 1, (max(l * 2, 500)), endpoint=True)
                out = np.array(interpolate.splev(u3, tck))

                def R_of_theta(theta):
                    return np.interp(theta, out[1], out[0])

                def rho(r, theta, phi):
                    Rmax = R_of_theta(theta)
                    if r <= Rmax:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2 * pi
                self.gfun = lambda x: 0.0
                self.hfun = lambda x: pi
                self.qfun = lambda phi, theta: 0.0
                self.rfun = lambda phi, theta: R_of_theta(theta)

            else:
                raise NotImplementedError("{} not available".format(coorsys))

            self.rho = rho

        elif preset is None:
            assert (callable(rho) and a is not None and b is not None
                    and callable(gfun) and callable(hfun)
                    and callable(qfun) and callable(rfun)), "missing arguments"
            self.rho = rho
            self.a = a
            self.b = b
            self.gfun = gfun
            self.hfun = hfun
            self.qfun = qfun
            self.rfun = rfun
        else:
            raise ValueError("{} is not a valid preset".format(preset))

    def eval_mass(self, verbose=False):
        """Evaluation of the total mass of the source based on the integration
        of the denisty function."""

        if self.coorsys == 'cartesian':
            def _integrand(x, y, z):
                return self.rho(x, y, z) * self.determinant(x, y, z)

        elif self.coorsys == 'spherical':
            def _integrand(r, theta, phi):
                return self.rho(r, theta, phi) * self.determinant(r, theta, phi)

        elif self.coorsys == 'spherical_mu':
            def _integrand(r, mu, phi):
                return self.rho(r, mu, phi) * self.determinant(r, mu, phi)

        elif self.coorsys == 'cylindrical':
            def _integrand(z, r, phi):
                return self.rho(r, phi, z) * self.determinant(r, phi, z)

        else:
            raise ValueError("Unknown coordinate system!")

        mass, err = tplquad(_integrand, self.a, self.b, self.gfun,
                            self.hfun, self.qfun, self.rfun)
        if verbose:
            print("mass = {:.5e} ; err = {:.2e}".format(mass, err))
        return mass, err

    def eval_potential(self, coor1, coor2, coor3, verbose=False):
        """Evaluation of the gravitational potential at coordinates
        (`coor1`, `coor2`, `coor3`) in a pre-determined coordinate system.
        Acts as a wrapper for `_eval_pot_point`."""

        if isinstance(coor1, (list, np.ndarray, tuple)):
            pots, errs = [], []
            for kk in range(len(coor1)):
                res = self._eval_pot_point(coor1[kk], coor2[kk], coor3[kk])
                pots.append(-res[0])
                errs.append(+res[1])
                if verbose:
                    PotentialByIntegration.print_coors(
                        self.coorsys, coor1[kk], coor2[kk], coor3[kk])
                    print("pot = {:.5e} ; err = {:.2e}\n".format(pots[-1],
                                                                 errs[-1]))
            return np.array(pots), np.array(errs)

        elif np.isscalar(coor1):
            res = self._eval_pot_point(coor1, coor2, coor3)
            pot, err = -res[0], res[1]
            if verbose:
                PotentialByIntegration.print_coors(self.coorsys, coor1, coor2,
                                                   coor3)
                print("pot = {:.5e} ; err = {:.2e}".format(pot, err))
            return pot, err

        else:
            raise ValueError(
                "coordinates must be specified as scalars or lists")

    def _eval_pot_point(self, coor1, coor2, coor3):
        """Evaluation of potential at point (coor1, coor2, coor3)."""

        if self.coorsys == 'cartesian':
            def _integrand(x, y, z):
                func = partial(self.integrand, xeval=coor1,
                               yeval=coor2, zeval=coor3)
                return func(x, y, z)

        elif self.coorsys == 'spherical':
            def _integrand(r, theta, phi):
                func = partial(self.integrand, reval=coor1, thetaeval=coor2,
                               phieval=coor3)
                return func(r, theta, phi)

        elif self.coorsys == 'spherical_mu':
            def _integrand(r, mu, phi):
                func = partial(self.integrand, reval=coor1, mueval=coor2,
                               phieval=coor3)
                return func(r, mu, phi)

        elif self.coorsys == 'cylindrical':
            def _integrand(z, r, phi):
                func = partial(self.integrand, reval=coor1, phieval=coor2,
                               zeval=coor3)
                return func(r, phi, z)

        else:
            raise ValueError(
                "{} is not a known coordinate system!".format(self.coorsys))

        return tplquad(_integrand, self.a, self.b, self.gfun, self.hfun,
                       self.qfun, self.rfun)

    @staticmethod
    def print_coors(coorsys, coor1, coor2, coor3):
        if coorsys == 'cartesian':
            print("x = {:.3f}, y = {:.3f}, z = {:.3f}".format(
                coor1, coor2, coor3))
        elif coorsys == 'spherical':
            print("r = {:.3f}, theta = {:.3f}, phi = {:.3f}".format(
                coor1, coor2, coor3))
        elif coorsys == 'cylindrical':
            print("r = {:.3f}, phi = {:.3f}, z = {:.3f}".format(
                coor1, coor2, coor3))
        else:
            raise ValueError(
                "{} is not a known coordinate system".format(coorsys))
