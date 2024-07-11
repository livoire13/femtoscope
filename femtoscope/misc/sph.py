# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:52:25 2023
Collection of spherical harmonic routines using PyshTools

@author: hlevy
"""

import pyshtools as pysh
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt

class AxiSymSph():
    """
    Class for computing and storing spherical harmonic coefficients of a
    axi-symmetric scalar field (the field does not depend on longitude).
    
    Attributes
    ----------
    c_lm : `pyshtools.shclasses.shcoeffs.SHRealCoeffs` class instance
        Pyshtools Class for spherical harmonic coefficients.
    grid : `pyshtools.shclasses.shgrid.DHRealGrid` class instance
        Pyshtools Class for real Driscoll and Healy (1994) sampled grids.
    f_lm : numpy 2d-array
        Array containing the spherical harmonic coefficients (all degrees /
        all orders).
    f_l0 : numpy 1d-array
        Array containing the spherical harmonic coefficients of order m=0.
    lats : numpy 1d-array
        Array of the latitudes of each row of the gridded data.
    lons : numpy 1d-array
        Array of the longitudes of each column of the gridded data.
    thetas : numpy 1d-array
        Array of the co-latitudes of each row of the gridded data.
    
    """
    def __init__(self, Lmax, normalization='ortho', **kwargs):
        """
        Construct a AxiSymSph instance.

        Parameters
        ----------
        Lmax : int
            Maximum degree of the expansion.
        normalization : str, optional
            The normalization of the coefficients (see Pyshtools docs).
            The default is 'ortho'.
        """
        self.c_lm = pysh.SHCoeffs.from_zeros(Lmax, kind='real',
                                            normalization=normalization)
        self.grid = self.c_lm.expand(grid='DH2')
        self.lats = self.grid.lats(degrees=False)
        self.lons = self.grid.lons(degrees=False)
        self.thetas = pi/2 - self.lats
        
        eval_func = kwargs.get('eval_func')
        if eval_func is not None:
            self.compute_expansion(eval_func)
        else:
            self.f_lm = None
            self.f_l0 = None
            
    def compute_expansion(self, eval_func):
        val = np.array(eval_func(self.thetas))[:, np.newaxis]
        self.grid.data = val.repeat(self.lons.shape[0], axis=1)
        self.c_lm = self.grid.expand(normalization=self.c_lm.normalization)
        self.f_lm = extract_coeffs(self.c_lm.coeffs)
        ind_m0 = self.f_lm.shape[0] - 1
        self.f_l0 = self.f_lm[:, ind_m0]
        
    def plot_spectrum(self, **kwargs):
        """
        Plot the spectrum as a function of spherical harmonic degree. See
        https://github.com/SHTOOLS/SHTOOLS/blob/master/pyshtools/shclasses/shcoeffs.py#L2310
        for kwargs usage.
        """
        kwargs.pop('show', None)
        fig, ax = self.c_lm.plot_spectrum(show=False)
    
    def plot_spectrum2d(self, **kwargs):
        """
        Plot the spectrum as a function of spherical harmonic degree and order.
        See https://github.com/SHTOOLS/SHTOOLS/blob/master/pyshtools/shclasses/shcoeffs.py#L2654
        for kwargs usage.

        """
        kwargs.pop('show', None)
        fig, ax = self.c_lm.plot_spectrum2d(show=False, **kwargs)

    def plot_grid(self, dim=2, **kwargs):
        """
        Plot the 'raw' data on a grid (2d map if dim==2, spheroid if dim==3).
        See https://github.com/SHTOOLS/SHTOOLS/blob/master/pyshtools/shclasses/shgrid.py#L1202
        for kwargs usage.
        """
        kwargs.pop('show', None)
        if dim==2:
            kwargs.pop('colorbar', None)
            fig, ax = self.grid.plot(colorbar='right', show=False, **kwargs)
        elif dim==3:
            fig, ax = self.grid.plot3d(show=False, **kwargs) # can be slow!
        else:
            raise ValueError("dim should be either 2 or 3")
            
    def plot_f_l0(self, logbool=True, monopole=False):
        """Wrapper for `_plot_f_l0`, see below."""
        _plot_f_l0(self.f_l0, logbool=logbool, monopole=monopole)


def extract_coeffs(CSlm):
    r"""
    Convert $(C_{lm}, S_{lm})$ coefficients (pyshtools' output) to $\phi_{lm}$.

    Parameters
    ----------
    CSlm : 3d-array
        Array containing $C_{lm}$ and $S_{lm}$ coefficients.
            1st dimension : C or S
            2nd dimension : degree l
            3rd dimension : order m

    Returns
    -------
    f_lm. 2d-array
        Spherical harmonic coefficients
    """
    C_lm = CSlm[0]
    S_lm = CSlm[1]
    L = max(C_lm.shape) - 1 # Max degree
    f_lm = np.zeros((L+1, 2*L+1), dtype=np.float64)
    f_lm[:, L:] = C_lm
    f_lm[:, :L] = np.flip(S_lm[:, 1:], axis=1)
    return f_lm

def _plot_f_l0(f_l0, logbool=True, monopole=False):
    """
    Plot sph coefficients of order m=0 as a function of degree l.

    Parameters
    ----------
    f_l0 : numpy 1d-array
        Array containing the spherical harmonic coefficients of order m=0.
    logbool : bool, optional
        Use a log-scale for the y-axis. The default is True.
    monopole : bool, optional
        Include the monopole l=0. The default is False.

    """
    L = f_l0.shape[0] - 1
    plt.figure(figsize=(5,4))
    if monopole:
        plt.bar(list(range(0, L+1)), abs(f_l0[0:]))
    else:
        plt.bar(list(range(1, L+1)), abs(f_l0[1:]))
    if logbool:
        plt.yscale('log')
    plt.xlabel(r"$l$ (Degree)", fontsize=15)
    plt.ylabel(r"$f_{l0}$", fontsize=15)
    plt.tight_layout()
    plt.show()