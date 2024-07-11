# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:06:53 2022

Collection of analytical solutions for various test-cases.

NB: source for the BSpline implementation
https://stackoverflow.com/questions/28279060/splines-with-python-using-control-knots-and-endpoints
https://github.com/kawache/Python-B-spline-examples

@author: hlevy
"""

import numpy as np
from numpy import sqrt, pi, cos, sin, arctan, arcsin
from scipy.integrate import tplquad
from functools import partial

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
    def __init__(self, coorsys, preset=None, rho=1.0, a=None, b=None,
                 gfun=None, hfun=None, qfun=None, rfun=None, **kwargs):
        """
        Construct a `PotentialByIntegration` instance. If preset is set to None,
        all the subsequent keyword arguments must be specified.

        Parameters
        ----------
        See above class docstring.

        """

        self.coorsys = coorsys

        if coorsys == 'cartesian':
            self.determinant = lambda x, y, z : 1.0
            def denom(x1, x2, y1, y2, z1, z2):
                return sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            def integrand(xp, yp, zp, xeval=None, yeval=None, zeval=None):
                return rho(xp, yp, zp)*self.determinant(xp, yp, zp) \
                    / denom(xeval, xp, yeval, yp, zeval, zp)

        elif coorsys == 'spherical':
            self.determinant  = lambda r, theta, phi : r**2 * sin(theta)
            def denom(r1, r2, theta1, theta2, phi1, phi2):
                return sqrt(r1**2 + r2**2 - 2*r1*r2*(
                    sin(theta1)*sin(theta2)*cos(phi1-phi2) \
                        + cos(theta1)*cos(theta2)))
            def integrand(rp, thetap, phip, reval=None, thetaeval=None,
                          phieval=None):
                return rho(rp, thetap, phip)*self.determinant(rp, thetap, phip) \
                    / denom(reval, rp, thetaeval, thetap, phieval, phip)

        elif coorsys == 'spherical_mu':
            self.determinant = lambda r, mu, phi : r**2
            def denom(r1, r2, mu1, mu2, phi1, phi2):
                return sqrt(r1**2 + r2**2 - 2*r1*r2*(
                    sqrt((1-mu1**2)*(1-mu2**2))*cos(phi1-phi2) + mu1*mu2))
            def integrand(rp, mup, phip, reval=None, mueval=None, phieval=None):
                return rho(rp, mup, phip)*self.determinant(rp, mup, phip) \
                    / denom(reval, rp, mueval, mup, phieval, phip)

        elif coorsys == 'cylindrical':
            self.determinant = lambda r, phi, z : r
            def denom(r1, r2, phi1, phi2, z1, z2):
                return sqrt(r1**2 + r2**2 - 2*r1*r2*cos(phi1-phi2) \
                            + (z1-z2)**2)
            def integrand(rp, phip, zp, reval=None, phieval=None, zeval=None):
                return rho(rp, phip, zp)*self.determinant(rp, phip, zp) \
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
                    d = sqrt(x**2 + y**2 + z**2)
                    if d <= radius:
                        return rho_val
                    else:
                        return 0.0
                self.a = -1
                self.b = +1
                self.gfun = lambda x : -1
                self.hfun = lambda x : +1
                self.qfun = lambda x, y : -sqrt(max(0, radius**2-x**2-y**2))
                self.rfun = lambda x, y : +sqrt(max(0, radius**2-x**2-y**2))

            elif coorsys == 'spherical':
                def rho(r, theta, phi):
                    if r <= radius:
                        return rho_val
                    else:
                        return 0.0
                self.a = 0.0
                self.b = 2*pi
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
                self.a  = 0.0
                self.b = 2*pi
                self.gfun = -1
                self.hfun = +1
                self.qfun = 0.0
                self.rfun = radius

            elif coorsys == 'cylindrical':
                def rho(r, phi, z):
                    d = sqrt(r**2 + z**2)
                    if d <= radius:
                        return rho_val
                    else:
                        return 0.0
                self.a = 0.0
                self.b = 2*pi
                self.gfun = lambda x : 0.0
                self.hfun = lambda x : radius
                self.qfun = lambda theta, r : -sqrt(max(0, radius**2-r**2))
                self.rfun = lambda theta, r : +sqrt(max(0, radius**2-r**2))

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
                        return radius + hm - hm/thetam * theta

                def rho(r, theta, phi):
                    Rmax = R_of_theta(theta)
                    if r <= Rmax:
                        return rho_val
                    else:
                        return 0.0

                self.a = 0.0
                self.b = 2*pi
                self.gfun = lambda x : 0.0
                self.hfun = lambda x : pi
                self.qfun = lambda phi, theta : 0.0
                self.rfun = lambda phi, theta : R_of_theta(theta)

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
                    expr = radius + hm + 2*hm*(theta/thetam)**3 \
                        - 3*hm*(theta/thetam)**2
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
                self.b = 2*pi
                self.gfun = lambda x : 0.0
                self.hfun = lambda x : pi
                self.qfun = lambda phi, theta : 0.0
                self.rfun = lambda phi, theta : R_of_theta(theta)

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
                plist = [(radius, 1-mum), (radius, 1-mum), (radius, 1-0.7*mum),
                         (radius+hm, 1-0.3*mum), (radius+hm, 1), (radius+hm, 1)]
                ctr = np.array(plist)
                x = ctr[:, 0]
                y = ctr[:, 1]
                l = len(x)
                t = np.linspace(0, 1, l-2, endpoint=True)
                t = np.append([0,0,0], t)
                t = np.append(t, [1,1,1])
                tck = [t, [x,y], 3]
                u3 = np.linspace(0, 1, (max(l*2, 500)), endpoint=True)
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
                self.b = 2*pi
                self.gfun = lambda x : -1
                self.hfun = lambda x : +1
                self.qfun = lambda phi, mu : 0.0
                self.rfun = lambda phi, mu : R_of_mu(mu)

            elif coorsys == 'spherical':
                thetam = kwargs.get('thetam', 1e-2)
                plist = [(radius+hm, 0), (radius+hm, 0),
                         (radius+hm, 0.3*thetam), (radius, 0.7*thetam),
                         (radius, thetam), (radius, thetam)]
                ctr = np.array(plist)
                x = ctr[:, 0]
                y = ctr[:, 1]
                l = len(x)
                t = np.linspace(0, 1, l-2, endpoint=True)
                t = np.append([0,0,0], t)
                t = np.append(t, [1,1,1])
                tck = [t, [x,y], 3]
                u3 = np.linspace(0, 1, (max(l*2, 500)), endpoint=True)
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
                self.b = 2*pi
                self.gfun = lambda x : 0.0
                self.hfun = lambda x : pi
                self.qfun = lambda phi, theta : 0.0
                self.rfun = lambda phi, theta : R_of_theta(theta)

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
                return self.rho(x, y, z)*self.determinant(x, y, z)

        elif self.coorsys == 'spherical':
            def _integrand(r, theta, phi):
                return self.rho(r, theta, phi)*self.determinant(r, theta, phi)

        elif self.coorsys == 'spherical_mu':
            def _integrand(r, mu, phi):
                return self.rho(r, mu, phi)*self.determinant(r, mu, phi)

        elif self.coorsys == 'cylindrical':
            def _integrand(z, r, phi):
                return self.rho(r, phi, z)*self.determinant(r, phi, z)

        else:
            raise ValueError("Unknown coordinate system!")

        mass, err  =  tplquad(_integrand, self.a, self.b, self.gfun,
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
                    print("pot = {:.5e} ; err = {:.2e}\n".format(pots[-1], errs[-1]))
            return np.array(pots), np.array(errs)

        elif np.isscalar(coor1):
            res = self._eval_pot_point(coor1, coor2, coor3)
            pot, err = -res[0], res[1]
            if verbose:
                PotentialByIntegration.print_coors(self.coorsys, coor1, coor2, coor3)
                print("pot = {:.5e} ; err = {:.2e}".format(pot, err))
            return pot, err

        else:
            raise ValueError("coordinates must be specified as scalars or lists")

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
            raise ValueError("{} is not a known coordinate system".format(coorsys))

def potential_sphere(r, R, G, M=None, rho=None):
    """
    Gravitational potential created by a perfect homogeneous solid
    sphere. Note that `M` and `rho` cannot be simultaneously None.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the mass center.
    R : float
        Radius of the sperical body.
    G : float
        Gravitational constant.
    M : float, optional
        Mass of the body. The default is None.
    rho : float
        Density of the body. The default is None.

    Returns
    -------
    ndarray
        Gravitational potential at radii `r` from the center of the spherical
        body.

    See Also
    --------
    potential_ellipsoid : generalization to an oblate ellipsoid.

    """
    r = np.array(r).reshape(-1)
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho*4*pi*R**3)/3
    with np.errstate(divide='ignore'):
        pot = np.where(r<R, 0.5*G*M*((r/R)**2-3)/R, -G*M/r)
    if len(pot) == 1: pot = pot[0]
    return pot

def grad_potential_sphere(r, R, G, M=None, rho=None):
    """
    Derivative(with respect to r) of the gravitational potential created by a
    perfect homogeneous solid sphere. Note that `M` and `rho` cannot be
    simultaneously None.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the mass center.
    R : float
        Radius of the sperical body.
    G : float
        Gravitational constant.
    M : float, optional
        Mass of the body. The default is None.
    rho : float
        Density of the body. The default is None.

    Returns
    -------
    ndarray
        Gravitational field at radii `r` from the center of the spherical
        body.
    """
    r = np.array(r).reshape(-1)
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho*4*pi*R**3)/3
    return np.where(r<R, G*M*r/R**3, G*M/r**2)

def _potential_sphere_eta(eta, R, G, Rcut, M=None, rho=None):
    r"""
    Gravitational potential created by a perfect homogeneous solid
    sphere in the `eta` coordinate. Note that `M` and `rho` cannot be
    simultaneously None.

    $$ \eta = \dfrac{Rcut}{r} $$

    Parameters
    ----------
    eta : float or ndarray
        Radial distance from the mass center.
    R : float
        Radius of the sperical body.
    G : float
        Gravitational constant.
    Rcut : float
        Truncation radius (appearing in the definition of eta).
    M : float, optional
        Mass of the body. The default is None.
    rho : float
        Density of the body. The default is None.

    Returns
    -------
    ndarray
        Gravitational potential at coordinate `eta` from the center of the
        spherical body.
    """
    eta = np.array(eta).reshape(-1)
    for x in eta:
        if x > Rcut*1.01:
            import warnings
            warnings.warn("""eta is usually set below Rcut.
                          Consider using `potential_sphere` in this case""")
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho*4*pi*R**3)/3
    return -G*M*eta/Rcut**2

def _grad_potential_sphere_eta(eta, R, G, Rcut, M=None, rho=None):
    """
    Derivative(with respect to r) of the gravitational potential created by a
    perfect homogeneous solid sphere. Note that `M` and `rho` cannot be
    simultaneously None.

    Parameters
    ----------
    eta : float or ndarray
        Radial distance from the mass center.
    R : float
        Radius of the sperical body.
    G : float
        Gravitational constant.
    Rcut : float
        Truncation radius (appearing in the definition of eta).
    M : float, optional
        Mass of the body. The default is None.
    rho : float
        Density of the body. The default is None.

    Returns
    -------
    ndarray
        Gravitational field at coordinate `eta` from the center of the
        spherical body.
    """
    eta = np.array(eta).reshape(-1)
    for x in eta:
        if x > Rcut*1.01:
            import warnings
            warnings.warn("""eta is usually set below Rcut.
                          Consider using `potential_sphere` in this case""")
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho*4*pi*R**3)/3
    return G*M*eta**2/Rcut**4

def potential_ellipsoid(coors_cart, sa, G, ecc=None,
                        sc=None, M=None, rho=None):
    r"""
    Gravitational potential inside a homogeneous ellipsoid of revolution
    bounded by the surface $ X^2 + Y^2 + Z^2/(1-e^2) = a^2 $. Note that `ecc`
    & `sc` cannot be simultaneously None and `M` & `rho` cannot be
    simultaneously None either.

    Parameters
    ----------
    coors_cart : ndarray
        Cartesian coordinates (numpy array (:, 3)-shapped).
    sa : float
        Spheroid semi-major-axis.
    G : float
        Gravitational constant.
    ecc : float, optional
        Spheroid eccentricity. The default is None.
    sc : float, optional
        Spheroid semi-minor-axis. The default is None.
    M : float, optional
        Mass of the body. The default is None.
    rho : float, optional
        Density of the body. The default is None.

    Returns
    -------
    ndarray
        Gravitational potential at `coors` locations.

    See Also
    --------
    potential_sphere : Particular case of the perfect homogeneous sphere.

    References
    ----------
    This implementation rely on [1]_ for the potential inside a homogeneous
    ellipsoid of revolution and [2]_ for the potential inside & outside.

    .. [1] DOI:10.1093/oso/9780198786399.003.0015.

    .. [2] HVOŽDARA, M., & KOHÚT, I. (2011). "Gravity field due to a
           homogeneous oblate spheroid: Simple solution form and numerical
           calculations". Contributions to Geophysics and Geodesy, 41(4),
           307-327. https://doi.org/10.2478/v10126-011-0013-0.

    """
    coors_cart = np.array(coors_cart)
    if len(coors_cart.shape)==1:
        coors_cart = coors_cart[:, np.newaxis].T
    X, Y, Z = coors_cart[:,0], coors_cart[:,1], coors_cart[:,2]

    # Assertions
    assert ecc is not None or sc is not None
    assert M is not None or rho is not None

    # Some conversions
    if ecc is None:
        ecc = sqrt(1-(sc/sa)**2)
    elif sc is None:
        sc = sa*sqrt(1-ecc**2)
    if M is None:
        M = 4*rho*pi*sc*sa**2/3
    elif rho is None:
        rho = 3*M/(4*pi*sc*sa**2)
    boolin = (X**2+Y**2)/sa**2 + Z**2/sc**2 < 1.0

    # Some function definitions
    def P2(t):
        return 0.5 * (3*t**2 - 1)
    def P2i(t):
        return -0.5 * (3*t**2 + 1)
    def q2(t):
        return 0.5 * ((3*t**2 + 1)*arctan(1/t) - 3*t)
    def q2prime(t):
        return -(2+3*t**2)/(1+t**2) + 3*t*arctan(1/t)
    def potin(w0, chalpha, cosbeta, E0, E2, shalpha):
        return -w0*(chalpha**2*(P2(cosbeta)-1)+E0+E2*P2i(shalpha)*P2(cosbeta))
    def potout(shalpha, cosbeta, f):
        return -(G*M*(arctan(1/shalpha) + q2(shalpha) * P2(cosbeta))/f)

    # Some coefficients
    f = sqrt(sa**2 - sc**2) # f = sa*ecc
    E0 = (sa/f)**2 * (1 + 2*(sc/f)*arctan(f/sc))
    ch02 = (sa/f)**2
    sh0 = sc/f
    w0 = 2*pi*G*rho*f**2 / 3
    E2 = -ch02 * (ch02*q2prime(sh0) - 2*sh0*q2(sh0))
    r = sqrt(X**2 + Y**2)
    chalpha = 0.5 * (sqrt((r-f)**2 + Z**2) + sqrt((r+f)**2 + Z**2)) / f
    shalpha = sqrt(chalpha**2 - 1)
    cosbeta = Z / (f*shalpha)

    # Formula from Maclaurin
    I = 2*arcsin(ecc)/ecc
    A1 = (arcsin(ecc)-ecc*sqrt(1-ecc**2))/ecc**3
    A3 = 2*(ecc-sqrt(1-ecc**2)*arcsin(ecc))/(sqrt(1-ecc**2)*ecc**3)
    potmac = -pi*G*rho*sqrt(1-ecc**2)*(sa**2*I-(X**2+Y**2)*A1-Z**2*A3)

    # return np.where(boolin, potmac, potout(shalpha, cosbeta, f))
    return np.where(boolin, potmac, potout(shalpha, cosbeta, f))


def chameleon_radial(r, R_A, rho_in, rho_vac, alpha, npot, plot=False,
                     verbose=False):
    """
    Approximate analytical solution of the chameleon field within and around a
    perfect homogeneous solid sphere.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the mass center.
    R_A : float
        Radius of the sperical body.
    rho_in : float
        Density inside the spherical body.
    rho_vac : float
        Vacuum density.
    alpha : float
        Physical parameter weighting the laplacian operator of the Klein-Gordon
        equation (dimensionless).
    npot : int
        Exponent (parameter of the chameleon model).
    plot : bool
        Whether or not to plot the resulting profile. The default is False.
    verbose : bool
        Display user's information. The default is False.

    Returns
    -------
    phi : ndarray
        Chameleon scalar field at distance `r` from the body center.

    Notes
    -----
    The solution returned by this function is only an approximate solution and
    by no means should be used to compute the *error* of a FEM result for
    instance.

    References
    ----------
    See [1]_ or [2]_ for the analytical derivation of the solution of the
    Klein-Gordon equation.

    .. [1] "Chameleon cosmology", Justin Khoury and Amanda Weltman,
           Phys. Rev. D 69, 044026 – Published 27 February 2004.

    .. [2] "Testing gravity in space : towards a realistic treatment of
           chameleon gravity in the MICROSCOPE mission"", Martin Pernot-Borràs,
           PhD thesis manuscript, November 2020.

    """
    r = np.array(r).reshape(-1)
    phi = np.zeros_like(r, dtype=np.float64)
    m_vac = np.sqrt((npot+1)/alpha * rho_vac**((npot+2)/(npot+1)))
    phi_in = rho_in**(-1/(npot+1))
    phi_vac = rho_vac**(-1/(npot+1))
    poly = np.array([m_vac/(3*alpha*(1+m_vac*R_A)), -1/(2*alpha), 0,
                     R_A**2/(6*alpha)*(2/(1+m_vac*R_A)+1)
                     + (phi_in-phi_vac)/rho_in])
    roots = np.roots(poly)
    roots = np.real(roots[np.where(np.isreal(roots))[0]])
    thin_shell = np.where((roots>=0) & (roots<=R_A))[0].size != 0

    # Thin-shell regime
    if thin_shell:
        if verbose: print("thin-shell")
        R_TS = roots[np.where((roots>=0) & (roots<=R_A))[0]]
        # Keep the solution that makes K>0
        R_TS = np.double(R_TS[np.where(R_TS<=R_A)].squeeze())
        K = rho_in * (R_A**3 - R_TS**3) / (3*alpha*(1+R_A*m_vac))
        r2 = r[np.where((r>=R_TS) & (r<=R_A))[0]]
        r3 = r[np.where(r>R_A)[0]]
        phi[np.where(r<R_TS)[0]] = phi_in
        phi[np.where((r>=R_TS) & (r<=R_A))[0]] = phi_in + rho_in/(3*alpha) * \
            (r2**2/2 + R_TS**3/r2 - 3*R_TS**2/2)
        phi[np.where(r>R_A)[0]] = phi_vac - K/r3 * np.exp(-m_vac*(r3-R_A))

    # Thick-shell regime
    else:
        if verbose: print("thick-shell")
        K = rho_in/(3*alpha) * R_A**3 / (1 + m_vac*R_A)
        phi_min = phi_vac - K/R_A - rho_in*R_A**2/(6*alpha)
        r1 = r[np.where(r<R_A)[0]]
        r2 = r[np.where(r>=R_A)[0]]
        phi[np.where(r<R_A)[0]] = phi_min + rho_in/(6*alpha) * r1**2
        phi[np.where(r>=R_A)[0]] = phi_vac - K/r2 * np.exp(-m_vac*(r2-R_A))

    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(r, phi, color='black')
        plt.show()

    return phi

def thin_shell(r, R_A, rho_in, rho_vac, alpha, npot):
    """Compute the Thin-Shell thickness [see `chameleon_radial` function for
    full documentation]."""
    r = np.array(r).reshape(-1)
    phi = np.zeros_like(r, dtype=np.float64)
    m_vac = np.sqrt((npot+1)/alpha * rho_vac**((npot+2)/(npot+1)))
    phi_in = rho_in**(-1/(npot+1))
    phi_vac = rho_vac**(-1/(npot+1))
    poly = np.array([m_vac/(3*alpha*(1+m_vac*R_A)), -1/(2*alpha), 0,
                     R_A**2/(6*alpha)*(2/(1+m_vac*R_A)+1)
                     + (phi_in-phi_vac)/rho_in])
    roots = np.roots(poly)
    roots = np.real(roots[np.where(np.isreal(roots))[0]])
    thin_shell = np.where((roots>=0) & (roots<=R_A))[0].size != 0

    # Thin-shell regime
    if thin_shell:
        R_TS = roots[np.where((roots>=0) & (roots<=R_A))[0]]
        R_TS = np.double(R_TS[np.where(R_TS<=R_A)].squeeze())
        return R_A - R_TS

    # Thick-shell regime
    else:
        return R_A

def param_to_alpha(Lambda, beta, npot, L_0=1.0, rho_0=1.0):
    r"""
    Mapping from the chameleon space parameter to the $\alpha$ parameter used
    in the dimensionless Klein-Gordon equation.

    Parameters
    ----------
    Lambda : float or 1D array
        Energy scale.
    beta : float or 1D array
        Coupling constant.
    npot : int or 1D array
        Exponent appearing in the Ratra-Peebles inverse power-law potential.
    L_0 : float, optional
        Characteristic length scale. The default is 1.0 m.
    rho_0 : float, optional
        Characteristic density. The default is 1.0 km.m^-3.

    Returns
    -------
    alpha : float
        Dimensionless parameter built from chameleon parameters and physical
        constants.

    """

    from femtoscope.misc.unit_conversion import compute_alpha
    alpha = compute_alpha(Lambda, beta, npot, L_0, rho_0)
    return alpha


def plot_alpha_map(Lambda_bounds, beta_bounds, npot, L_0=1.0, rho_0=1.0,
                   iso_alphas=[], savefig=False, **kwargs):
    r"""
    Plot the $\alpha$ parameter map.

    Parameters
    ----------
    Lambda_bounds : list
        Lambda range [Lambda_min, Lambda_max].
    beta_bounds : list
        beta range [beta_min, beta_max].
    npot : int
        Exponent appearing in the Ratra-Peebles inverse power-law potential.
    L_0 : float, optional
        Characteristic length scale. The default is 1.0.
    rho_0 : float, optional
        Characteristic density. The default is 1.0.
    iso_alphas : list
        List of the values of alpha to be emphasised on the map.
        The default is [].
    savefig : bool
       Save the figure to pdf format. The default is False
    ax : matplotlib.axes
        Axes object of an external figure. The default is None.
    fig : matplotlib.figure
        Figure object of an external figure. The default is None.
    colors : list
        List of iso-alpha-line colors.
    M_param : bool
        If True, use the M instead of beta (recalling that beta = Mpl/M).
        The default is False.
    figsize : pair of floats
        Width, height in inches. The default is (17.0, 7.0)
    """

    from matplotlib import pyplot as plt

    M_param = kwargs.get('M_param', False)
    sign = 1 - 2*int(M_param)
    figsize = kwargs.get('figsize', (17.0, 7.0))
    cmap = kwargs.get('cmap', 'viridis')
    ax, fig = kwargs.get('ax', None), kwargs.get('fig', None)
    nblevels = 500
    if iso_alphas:
        colors = kwargs.get('colors', None)
        if colors is None:
            colors = plt.cm.get_cmap('Dark2', len(iso_alphas)).colors

    if isinstance(npot, int):
        npot = [npot]

    Nplot = len(npot)
    vmin, vmax = np.inf, -np.inf
    if ax is None: fig, ax = plt.subplots(1, Nplot, figsize=figsize)
    Lambda = 10**(np.linspace(np.log10(Lambda_bounds[0]),
                              np.log10(Lambda_bounds[1]), nblevels))
    beta = 10**(np.linspace(np.log10(beta_bounds[0]),
                            np.log10(beta_bounds[1]), nblevels))
    beta, Lambda = np.meshgrid(beta, Lambda)
    extent = [sign*np.log10(beta_bounds[0]), sign*np.log10(beta_bounds[1]),
              np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1])]
    params = np.array(list(zip(beta.ravel(), Lambda.ravel())))
    betas, Lambdas = params[:, 0], params[:, 1]
    alpha = []
    for k in range(Nplot):

        alphas = param_to_alpha(Lambdas, betas, npot[k], L_0=L_0, rho_0=rho_0)
        alpha.append(alphas.reshape(Lambda.shape))

        if alphas.min() < vmin:
            vmin = alphas.min()
        if alphas.max() > vmax:
            vmax = alphas.max()

    for k in range(Nplot):
        axk = ax[k] if Nplot>1 else ax
        pc = axk.imshow(np.log10(alpha[k]), cmap=cmap, interpolation='nearest',
                        extent=extent, vmin=np.log10(vmin), vmax=np.log10(vmax),
                        origin='lower', interpolation_stage='data', aspect="auto")
        # pc = axk.imshow(alpha[k], cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
        #                 origin='lower', extent=extent)
        if k==0:
            axk.set_ylabel(r"$\mathrm{log_{10}}(\Lambda / \mathrm{eV})$",
                           fontsize=13)
        axk.set_title(r"$n = {}$".format(npot[k]), fontsize=13)
        axk.tick_params(
            direction='in', axis='both', color='white', length=5.5,
            bottom=True, top=True, left=True, right=True, labelsize=11)
        # axk.set_xlim([np.log10(beta_bounds[0]), np.log10(beta_bounds[1])])
        # axk.set_ylim([np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1])])

        if iso_alphas:
            for iso, c in zip(iso_alphas, colors):
                if iso>vmin and iso<vmax:
                    log_iso = np.log10(iso)
                    xb, yb = _iso_alpha_bounds(log_iso, Lambda_bounds,
                                               beta_bounds, npot[k], L_0, rho_0)
                    # c = 'darkorange' # to be removed
                    xb = [sign*x for x in xb]
                    axk.plot(xb, yb, c=c, linewidth=2.5)

        if M_param:
            axk.set_xlabel(
                r"$\mathrm{log_{10}}(M / M_\mathrm{{Pl}})$", fontsize=13)
            axk.set_xlim(axk.get_xlim()[::-1])
        else:
            axk.set_xlabel(r"$\mathrm{log_{10}}(\beta)$", fontsize=13)

    if Nplot > 1:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=np.log10(vmin), vmax=np.log10(vmax)))
        cbar = fig.colorbar(sm, cax=cbar_ax)
    else:
        cbar = fig.colorbar(pc)
    cbar.set_label(r"$\mathrm{log_{10}}(\alpha)$", rotation=270, fontsize=13,
                    labelpad=25)
    cbar.ax.tick_params(labelsize=11)

    if iso_alphas:
        for iso, c in zip(iso_alphas, colors):
            # c = 'darkorange' # to be removed
            cbar.ax.plot([0, 1], [np.log10(iso)]*2, c=c, linewidth=1.5)

    plt.tight_layout()
    if savefig:
        from femtoscope import RESULT_DIR
        fullFileName = str(RESULT_DIR / 'plot' / 'alpha_map.pdf')
        plt.savefig(fullFileName, format="pdf", bbox_inches="tight")

    plt.show()


def _iso_alpha_bounds(iso, Lambda_bounds, beta_bounds, npot, L_0, rho_0):
    r""" Utility function for plotting curves of iso-alpha on alpha-maps. The
    equation reads
            $$ iso = A + \frac{n+4}{n+1} y - \frac{n+2}{n+1} x $$
    Iso-alpha curves are straight lines in the $(\log \beta , \log \Lambda)$
    plane."""
    from femtoscope.misc import constants
    from femtoscope.misc.constants import H_BAR, C_LIGHT, EV
    M_pl = constants.M_PL
    beta_min = np.log10(beta_bounds[0])
    beta_max = np.log10(beta_bounds[1])
    Lambda_min = np.log10(Lambda_bounds[0])
    Lambda_max = np.log10(Lambda_bounds[1])

    # Compute limits
    A = np.log10(M_pl/(rho_0*L_0**2) * EV/(H_BAR*C_LIGHT))
    A += 1/(npot+1)*np.log10(npot*M_pl/rho_0 * (EV/(H_BAR*C_LIGHT))**3)
    xmin = (npot+1)/(npot+2) * (A - iso + (npot+4)/(npot+1)*Lambda_min)
    xmax = (npot+1)/(npot+2) * (A - iso + (npot+4)/(npot+1)*Lambda_max)
    ymin = (npot+1)/(npot+4) * (iso - A + (npot+2)/(npot+1)*beta_min)
    ymax = (npot+1)/(npot+4) * (iso - A + (npot+2)/(npot+1)*beta_max)

    # crop to get the correct limits
    xmin = beta_min if xmin<beta_min else xmin
    xmax = beta_max if xmax>beta_max else xmax
    ymin = Lambda_min if ymin<Lambda_min else ymin
    ymax = Lambda_max if ymax>Lambda_max else ymax
    return ([xmin, xmax], [ymin, ymax])
