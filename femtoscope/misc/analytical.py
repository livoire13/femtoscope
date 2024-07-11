# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:45:31 2024

Collection of analytical solutions for various test-cases.
"""

from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from numpy import arcsin, arctan, pi, sqrt

from femtoscope.misc.unit_conversion import compute_alpha


def potential_sphere(r: Union[float, np.ndarray], R: float, G: float, M=None,
                     rho=None) -> Union[float, np.ndarray]:
    """
    Gravitational potential created by a perfect homogeneous solid
    sphere. Note that `M` and `rho` cannot be simultaneously None.

    Parameters
    ----------
    r : float or np.ndarray
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
    pot : float or np.ndarray
        Gravitational potential at radii `r` from the center of the spherical
        body.

    See Also
    --------
    potential_ellipsoid : generalization to an oblate ellipsoid.

    """
    r = np.array(r).reshape(-1)
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho * 4 * pi * R ** 3) / 3
    with np.errstate(divide='ignore'):
        pot = np.where(r < R, 0.5 * G * M * ((r / R) ** 2 - 3) / R, -G * M / r)
    if len(pot) == 1:
        pot = pot[0]
    return pot


def grad_potential_sphere(r: Union[float, np.ndarray], R: float, G: float,
                          M=None, rho=None) -> Union[float, np.ndarray]:
    """
    Derivative(with respect to r) of the gravitational potential created by a
    perfect homogeneous solid sphere. Note that `M` and `rho` cannot be
    simultaneously None.

    Parameters
    ----------
    r : float or np.ndarray
        Radial distance from the mass center.
    R : float
        Radius of the sperical body.
    G : float
        Gravitational constant.
    M : float, optional
        Mass of the body. The default is None.
    rho : float or np.ndarray
        Density of the body. The default is None.

    Returns
    -------
    grad_pot : float or np.ndarray
        Gravitational field at radii `r` from the center of the spherical
        body.

    """
    r = np.array(r).reshape(-1)
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho * 4 * pi * R ** 3) / 3
    grad_pot = np.where(r < R, G * M * r / R ** 3, G * M / r ** 2)
    if len(grad_pot) == 1:
        grad_pot = grad_pot[0]
    return grad_pot


def _potential_sphere_eta(eta: Union[float, np.ndarray],
                          R: float, G: float, Rcut: float,
                          M=None, rho=None) -> Union[float, np.ndarray]:
    r"""
    Gravitational potential created by a perfect homogeneous solid
    sphere in the `eta` coordinate. Note that `M` and `rho` cannot be
    simultaneously None.

    $$ \eta = \dfrac{Rc}{r} $$

    Parameters
    ----------
    eta : float or np.ndarray
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
    pot : float or np.ndarray
        Gravitational potential at coordinate `eta` from the center of the
        spherical body.
    """
    eta = np.array(eta).reshape(-1)
    for x in eta:
        if x > Rcut * 1.01:
            import warnings
            warnings.warn("""eta is usually set below Rc.
                          Consider using `potential_sphere` in this case""")
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho * 4 * pi * R ** 3) / 3
    pot = -G * M * eta / Rcut ** 2
    if len(pot) == 1:
        pot = pot[0]
    return pot


def _grad_potential_sphere_eta(eta: Union[float, np.ndarray],
                               R: float, G: float, Rcut: float,
                               M=None, rho=None) -> Union[float, np.ndarray]:
    """
    Derivative(with respect to r) of the gravitational potential created by a
    perfect homogeneous solid sphere. Note that `M` and `rho` cannot be
    simultaneously None.

    Parameters
    ----------
    eta : float or np.ndarray
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
    grad_pot : float or np.ndarray
        Gravitational field at coordinate `eta` from the center of the
        spherical body.
    """
    eta = np.array(eta).reshape(-1)
    for x in eta:
        if x > Rcut * 1.01:
            import warnings
            warnings.warn("""eta is usually set below Rc.
                          Consider using `potential_sphere` in this case""")
    assert M is not None or rho is not None
    if rho is not None:
        M = (rho * 4 * pi * R ** 3) / 3
    grad_pot = G * M * eta ** 2 / Rcut ** 4
    if len(grad_pot) == 1:
        grad_pot = grad_pot[0]
    return grad_pot


def potential_ellipsoid(coors_cart: np.ndarray, sa: float, G: float,
                        ecc=None, sc=None, M=None, rho=None) -> np.ndarray:
    r"""
    Gravitational potential inside a homogeneous ellipsoid of revolution
    bounded by the surface $ X^2 + Y^2 + Z^2/(1-e^2) = a^2 $. Note that `ecc`
    & `sc` cannot be simultaneously None and `M` & `rho` cannot be
    simultaneously None either.

    Parameters
    ----------
    coors_cart : np.ndarray
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
    if len(coors_cart.shape) == 1:
        coors_cart = coors_cart[:, np.newaxis].T
    if coors_cart.shape[1] == 2:
        X, Z = coors_cart[:, 0], coors_cart[:, 1]
        Y = np.zeros_like(X)
    elif coors_cart.shape[1] == 3:
        X, Y, Z = coors_cart[:, 0], coors_cart[:, 1], coors_cart[:, 2]
    else:
        raise ValueError("'coors_cart' has wrong shape!")

    # Assertions
    assert ecc is not None or sc is not None
    assert M is not None or rho is not None

    # Some conversions
    if ecc is None:
        ecc = sqrt(1 - (sc / sa) ** 2)
    elif sc is None:
        sc = sa * sqrt(1 - ecc ** 2)
    if M is None:
        M = 4 * rho * pi * sc * sa ** 2 / 3
    elif rho is None:
        rho = 3 * M / (4 * pi * sc * sa ** 2)
    boolin = (X ** 2 + Y ** 2) / sa ** 2 + Z ** 2 / sc ** 2 < 1.0

    # Some function definitions
    def P2(t):
        return 0.5 * (3 * t ** 2 - 1)

    def P2i(t):
        return -0.5 * (3 * t ** 2 + 1)

    def q2(t):
        return 0.5 * ((3 * t ** 2 + 1) * arctan(1 / t) - 3 * t)

    def q2prime(t):
        return -(2 + 3 * t ** 2) / (1 + t ** 2) + 3 * t * arctan(1 / t)

    def potin(w0, chalpha, cosbeta, E0, E2, shalpha):
        return -w0 * (chalpha ** 2 * (P2(cosbeta) - 1) + E0 + E2 * P2i(
            shalpha) * P2(cosbeta))

    def potout(shalpha, cosbeta, f):
        return -(G * M * (arctan(1 / shalpha) + q2(shalpha) * P2(cosbeta)) / f)

    # Some coefficients
    f = sqrt(sa ** 2 - sc ** 2)  # f = sa*ecc
    E0 = (sa / f) ** 2 * (1 + 2 * (sc / f) * arctan(f / sc))
    ch02 = (sa / f) ** 2
    sh0 = sc / f
    w0 = 2 * pi * G * rho * f ** 2 / 3
    E2 = -ch02 * (ch02 * q2prime(sh0) - 2 * sh0 * q2(sh0))
    r = sqrt(X ** 2 + Y ** 2)
    chalpha = 0.5 * (
            sqrt((r - f) ** 2 + Z ** 2) + sqrt((r + f) ** 2 + Z ** 2)) / f
    shalpha = sqrt(chalpha ** 2 - 1)
    cosbeta = Z / (f * shalpha)

    # Formula from Maclaurin
    I = 2 * arcsin(ecc) / ecc
    A1 = (arcsin(ecc) - ecc * sqrt(1 - ecc ** 2)) / ecc ** 3
    A3 = 2 * (ecc - sqrt(1 - ecc ** 2) * arcsin(ecc)) / (
            sqrt(1 - ecc ** 2) * ecc ** 3)
    potmac = -pi * G * rho * sqrt(1 - ecc ** 2) * (
            sa ** 2 * I - (X ** 2 + Y ** 2) * A1 - Z ** 2 * A3)

    return np.where(boolin, potmac, potout(shalpha, cosbeta, f))


def chameleon_radial(r: Union[float, np.ndarray], R_A: float, rho_in: float,
                     rho_vac: float, alpha: float, npot: int, plot=False,
                     verbose=False) -> Union[float, np.ndarray]:
    """
    Approximate analytical solution of the chameleon field within and around a
    perfect homogeneous solid sphere.

    Parameters
    ----------
    r : float or np.ndarray
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
        Whether to plot the resulting profile. The default is False.
    verbose : bool
        Display user's information. The default is False.

    Returns
    -------
    phi : float or np.ndarray
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
    m_vac = np.sqrt((npot + 1) / alpha * rho_vac ** ((npot + 2) / (npot + 1)))
    phi_in = rho_in ** (-1 / (npot + 1))
    phi_vac = rho_vac ** (-1 / (npot + 1))
    poly = np.array(
        [m_vac / (3 * alpha * (1 + m_vac * R_A)), -1 / (2 * alpha), 0,
         R_A ** 2 / (6 * alpha) * (2 / (1 + m_vac * R_A) + 1)
         + (phi_in - phi_vac) / rho_in])
    roots = np.roots(poly)
    roots = np.real(roots[np.where(np.isreal(roots))[0]])
    thin_shell = np.where((roots >= 0) & (roots <= R_A))[0].size != 0

    # Thin-shell regime
    if thin_shell:
        if verbose: print("thin-shell")
        R_TS = roots[np.where((roots >= 0) & (roots <= R_A))[0]]
        # Keep the solution that makes K>0
        R_TS = np.double(R_TS[np.where(R_TS <= R_A)].squeeze())
        K = rho_in * (R_A ** 3 - R_TS ** 3) / (3 * alpha * (1 + R_A * m_vac))
        r2 = r[np.where((r >= R_TS) & (r <= R_A))[0]]
        r3 = r[np.where(r > R_A)[0]]
        phi[np.where(r < R_TS)[0]] = phi_in
        phi[np.where((r >= R_TS) & (r <= R_A))[0]] = phi_in + rho_in / (
                    3 * alpha) \
                                                     * (
                                                                 r2 ** 2 / 2 + R_TS ** 3 / r2 - 3 * R_TS ** 2 / 2)
        phi[np.where(r > R_A)[0]] = phi_vac - K / r3 * np.exp(
            -m_vac * (r3 - R_A))

    # Thick-shell regime
    else:
        if verbose: print("thick-shell")
        K = rho_in / (3 * alpha) * R_A ** 3 / (1 + m_vac * R_A)
        phi_min = phi_vac - K / R_A - rho_in * R_A ** 2 / (6 * alpha)
        r1 = r[np.where(r < R_A)[0]]
        r2 = r[np.where(r >= R_A)[0]]
        phi[np.where(r < R_A)[0]] = phi_min + rho_in / (6 * alpha) * r1 ** 2
        phi[np.where(r >= R_A)[0]] = phi_vac - K / r2 * np.exp(
            -m_vac * (r2 - R_A))

    if plot:
        plt.figure()
        plt.plot(r, phi, color='black')
        plt.show()

    if len(phi) == 1:
        phi = phi[0]
    return phi


def thin_shell_thickness(R_A: float, rho_in: float, rho_vac: float,
                         alpha: float, npot: int) -> float:
    """Compute the Thin-Shell thickness [see `chameleon_radial` function for
    full documentation]."""
    m_vac = np.sqrt((npot + 1) / alpha * rho_vac ** ((npot + 2) / (npot + 1)))
    phi_in = rho_in ** (-1 / (npot + 1))
    phi_vac = rho_vac ** (-1 / (npot + 1))
    poly = np.array(
        [m_vac / (3 * alpha * (1 + m_vac * R_A)), -1 / (2 * alpha), 0,
         R_A ** 2 / (6 * alpha) * (2 / (1 + m_vac * R_A) + 1)
         + (phi_in - phi_vac) / rho_in])
    roots = np.roots(poly)
    roots = np.real(roots[np.where(np.isreal(roots))[0]])
    thin_shell = np.where((roots >= 0) & (roots <= R_A))[0].size != 0

    # Thin-shell regime
    if thin_shell:
        R_TS = roots[np.where((roots >= 0) & (roots <= R_A))[0]]
        R_TS = np.double(R_TS[np.where(R_TS <= R_A)].squeeze())
        return R_A - R_TS

    # Thick-shell regime
    else:
        return R_A


def param_to_alpha(Lambda: Union[float, np.ndarray],
                   beta: Union[float, np.ndarray],
                   npot: Union[int, np.ndarray],
                   L_0=1.0, rho_0=1.0) -> Union[float, np.ndarray]:
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
    alpha : float or np.ndarray
        Dimensionless parameter built from chameleon parameters and physical
        constants.

    """
    bool_list = [isinstance(Lambda, np.ndarray),
                 isinstance(beta, np.ndarray),
                 isinstance(npot, np.ndarray)]
    if sum(bool_list) > 2:
        raise ValueError("At most two argument can be of type 'ndarray'.")
    alpha = compute_alpha(Lambda, beta, npot, L_0, rho_0)
    return alpha


def plot_alpha_map(Lambda_bounds: list, beta_bounds: list, npot: int, L_0=1.0,
                   rho_0=1.0, iso_alphas=None, savefig=False, **kwargs):
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
        The default is None.
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

    M_param = kwargs.get('M_param', False)
    sign = 1 - 2 * int(M_param)
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
    Lambda = 10 ** (np.linspace(np.log10(Lambda_bounds[0]),
                                np.log10(Lambda_bounds[1]), nblevels))
    beta = 10 ** (np.linspace(np.log10(beta_bounds[0]),
                              np.log10(beta_bounds[1]), nblevels))
    beta, Lambda = np.meshgrid(beta, Lambda)
    extent = [sign * np.log10(beta_bounds[0]), sign * np.log10(beta_bounds[1]),
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
        axk = ax[k] if Nplot > 1 else ax
        pc = axk.imshow(np.log10(alpha[k]), cmap=cmap, interpolation='nearest',
                        extent=extent, vmin=np.log10(vmin), vmax=np.log10(vmax),
                        origin='lower', interpolation_stage='data',
                        aspect="auto")
        # pc = axk.imshow(alpha[k], cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
        #                 origin='lower', extent=extent)
        if k == 0:
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
                if iso > vmin and iso < vmax:
                    log_iso = np.log10(iso)
                    xb, yb = _iso_alpha_bounds(log_iso, Lambda_bounds,
                                               beta_bounds, npot[k], L_0, rho_0)
                    # c = 'darkorange'  # to be removed
                    xb = [sign * x for x in xb]
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
            # c = 'darkorange'  # to be removed
            cbar.ax.plot([0, 1], [np.log10(iso)] * 2, c=c, linewidth=1.5)

    plt.tight_layout()
    if savefig:
        from femtoscope import RESULT_DIR
        fullFileName = str(RESULT_DIR / 'plot' / 'alpha_map.pdf')
        plt.savefig(fullFileName, format="pdf", bbox_inches="tight")

    plt.show()


def _iso_alpha_bounds(iso: float, Lambda_bounds: list, beta_bounds: list,
                      npot: int, L_0: float, rho_0: float) -> tuple:
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
    A = np.log10(M_pl / (rho_0 * L_0 ** 2) * EV / (H_BAR * C_LIGHT))
    A += 1 / (npot + 1) * np.log10(
        npot * M_pl / rho_0 * (EV / (H_BAR * C_LIGHT)) ** 3)
    xmin = (npot + 1) / (npot + 2) * (
            A - iso + (npot + 4) / (npot + 1) * Lambda_min)
    xmax = (npot + 1) / (npot + 2) * (
            A - iso + (npot + 4) / (npot + 1) * Lambda_max)
    ymin = (npot + 1) / (npot + 4) * (
            iso - A + (npot + 2) / (npot + 1) * beta_min)
    ymax = (npot + 1) / (npot + 4) * (
            iso - A + (npot + 2) / (npot + 1) * beta_max)

    # crop to get the correct limits
    xmin = beta_min if xmin < beta_min else xmin
    xmax = beta_max if xmax > beta_max else xmax
    ymin = Lambda_min if ymin < Lambda_min else ymin
    ymax = Lambda_max if ymax > Lambda_max else ymax
    return [xmin, xmax], [ymin, ymax]
