# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:38:17 2024

@author: hlevy
"""

from matplotlib import pyplot as plt
import numpy as np
from numpy import pi

from femtoscope.misc.unit_conversion import (mass_to_nat, length_to_nat,
                                             density_to_nat, compute_phi_0)
from femtoscope.misc.analytical import plot_alpha_map


npot = 1
rho0 = 1  # kg / m^3
L0 = 1  # m
npot = 1
beta_bounds = np.array([1e-1, 1e18])
Lambda_bounds = np.array([1e-7, 1e1])

rho_bg = 1e-20  # kg / m^3


def compute_alpha_screened(mass, size, rho_bg):
    """

    Parameters
    ----------
    mass : kg
        Mass of the satellite
    size : m
        Size of the satellite.
    rho_bg : kg / m^3
        Background density.

    """

    rho_sat = mass / (4/3 * pi * (size/2)**3)
    return 1/3 * (rho_sat/rho0) * (size/L0)**2 * (rho_bg/rho0)**(1/(npot+1))

masses = np.logspace(-1.3, 4.3, 100)
sizes = np.logspace(-1.2, 1.2, 100) # divide by two to get the radius

M, S = np.meshgrid(masses, sizes)
A = compute_alpha_screened(M, S, rho_bg)

mass_cubesat = 1  # kg
size_cubesat = 10e-2  # m
alpha_cubesat = compute_alpha_screened(mass_cubesat, size_cubesat, rho_bg)

mass_microscope = 330  # kg
size_microscope = 120e-2  # m (the satellite is approximately one cubic meter)
alph_microscope = compute_alpha_screened(mass_microscope, size_microscope, rho_bg)

mass_galileo = 675  # kg
size_galileo = 2  # m
alpha_galileo = compute_alpha_screened(mass_galileo, size_galileo, rho_bg)

#%%

cticks = [-12, -10, -8, -6]
xticks = [-15, -10, -5, 0]
yticks = np.arange(np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1])+1)
xtick_labels = [r'$1$', r'$10^{-5}$', r'$10^{-10}$', r'$10^{-15}$'][::-1]
ytick_labels = ['', r'$10^{-6}$', '', r'$10^{-4}$', '', r'$10^{-2}$', '', r'$1$', '']

fig, axs = plt.subplots(figsize=(8, 3.5), nrows=1, ncols=2)
ax1, ax2 = axs
ctf = ax1.contourf(S, M, np.log10(A), levels=100, cmap='viridis')
ct = ax1.contour(S, M, np.log10(A),levels=[np.log10(alpha_cubesat)])
ax1.scatter(size_cubesat, mass_cubesat, marker='s', color='lightgray')
ax1.scatter(size_microscope, mass_microscope, s=70, marker=r'$\heartsuit$',
            color='maroon')
ax1.scatter(size_galileo, mass_galileo, marker='*')
cbar = fig.colorbar(ctf, ax=ax1, ticks=cticks, aspect=12)
cbar.ax.plot([0, 1], [np.log10(alpha_cubesat), np.log10(alpha_cubesat)])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='both', direction='in', length=5.5,
                bottom=True, top=True, left=True, right=True, labelsize=11,
                color='white')
ax1.set_xlabel(r"Size [m]", fontsize=13)
ax1.set_ylabel(r"Mass [kg]", fontsize=13)

ax1.tick_params(axis='both', which='minor',
                bottom=False, top=False, left=False, right=False)
plot_alpha_map(Lambda_bounds, beta_bounds, npot, ax=ax2, fig=fig, M_param=True)
ax2.set_xlabel(r"$M / M_{\mathrm{Pl}}$", fontsize=13)
ax2.set_ylabel(r"$\Lambda \, \mathrm{[eV]}$")
ax2.set_xticks(xticks, labels=xtick_labels)
ax2.set_yticks(yticks, labels=ytick_labels)
