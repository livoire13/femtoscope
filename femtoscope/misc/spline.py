# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:50:42 2023

@author: hlevy
"""

from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline, splev, splrep
import numpy as np
# https://stackoverflow.com/questions/32046582/spline-with-constraints-at-border

def guess(x, y, k, s, w=None):
    """Do an ordinary spline fit to provide knots.
    x, y : array_like
        The data points defining a curve y = f(x).
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as x and y.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the y values have standard-deviation given by the
        vector d, then w should be 1/d. Default is ones(len(x)).
    k : int, optional
        The degree of the spline fit. It is recommended to use cubic splines.
        Even values of k should be avoided especially with small s values.
        1 <= k <= 5
    s : float, optional
        A smoothing condition. The amount of smoothness is determined by
        satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)
        is the smoothed interpolation of (x,y). The user can use s to control
        the tradeoff between closeness and smoothness of fit. Larger s means
        more smoothing while smaller values of s indicate less smoothing.
        Recommended values of s depend on the weights, w. If the weights
        represent the inverse of the standard-deviation of y, then a good s
        value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
        the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
        weights are supplied. s = 0.0 (interpolating) if no weights are
        supplied."""
    return splrep(x, y, w, k=k, s=s)

def err(c, x, y, t, k, w=None):
    """The error function to minimize"""
    diff = y - splev(x, (t, c, k))
    if w is None:
        diff = np.einsum('...i,...i', diff, diff)
    else:
        diff = np.dot(diff*diff, w)
    return np.abs(diff)

def spline_neumann(x, y, xbc=None, k=3, s=0, w=None):
    """
    Spline fitting of 1D data with null derivative at the interval boundaries.

    Parameters
    ----------
    x, y : array_like
        The data points defining a curve y = f(x).
    xbc : list of length 2
        The bounds of the interval where the constraint is applied. The
        constraint on one of the two bounds can be relaxed using None e.g. with
        syntax [x0, None]. The default is [min(x), max(x)].
    k, s, w : See docstring of `guess` function.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if xbc is None:
        xbc = [x.min(), x.max()]
    t, c0, k = guess(x, y, k, s, w=w)
    if (xbc[0] is not None and xbc[1] is not None):
        fun = lambda c : splev(xbc[0], (t, c, k), der=1)**2  \
            + splev(xbc[1], (t, c, k), der=1)**2
    elif xbc[0] is not None:
        fun = lambda c : splev(xbc[0], (t, c, k), der=1)
    elif xbc[1] is not None:
        fun = lambda c : splev(xbc[1], (t, c, k), der=1)
    else:
        return UnivariateSpline._from_tck((t, c0, k))
    con = {'type' : 'eq',
           'fun' : fun}
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))