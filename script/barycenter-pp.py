# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:14:36 2024

@author: hlevy
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from femtoscope.inout.postprocess import ResultsPostProcessor
from femtoscope.misc.unit_conversion import compute_alpha, compute_beta, \
    compute_acc_0
from femtoscope.misc.analytical import grad_potential_sphere, potential_sphere
from femtoscope.misc.constants import G_GRAV, LAMBDA_DE

# Chameleon parameters
rho0 = 1  # kg/m^3
L0 = 0.1  # m
npot = 1
Lambda = LAMBDA_DE
beta = 1e3
acc0 = compute_acc_0(rho0, L0, npot, Lambda=Lambda, beta=beta)
# alpha = compute_alpha(Lambda, beta, npot, L0, rho0)

# Mesh parameters
R1 = 0.5  # radius of sphere no 1
R2 = 1.5  # radius of sphere no 2
d = 3.2  # distance between the two spheres
Rc = 6  # truncation radius

# Densities & chameleon parameters
rho_sp1 = 2700
rho_sp2 = 1e2
rho_vac = 1e-1
rho_max = max(rho_sp1, rho_sp2)
phi_max = rho_vac ** (-1 / (npot + 1))

# Fetching the solution array
result_pp = ResultsPostProcessor.from_files("barycenter")
sol_int = result_pp.sol_int
yy = np.linspace(-d/2, d/2, 700)
xx = np.zeros_like(yy)
cline = np.concatenate((xx[:, np.newaxis], yy[:, np.newaxis]), axis=1)
philine = result_pp.evaluate_at(cline)
gradline = result_pp.evaluate_at(cline, mode='grad')[:, 1]
a_cham = - acc0 * gradline

# Newtonian gravity
def newton_acc(x):
    acc_sp1 = grad_potential_sphere(abs(x+d/2)*L0, R1*L0, G_GRAV, rho=rho_sp1)
    acc_sp2 = grad_potential_sphere(abs(x-d/2)*L0, R2*L0, G_GRAV, rho=rho_sp2)
    return -acc_sp1 + acc_sp2


def newton_pot(x):
    pot_sp1 = potential_sphere(abs(x + d / 2)*L0, R1 * L0, G_GRAV, rho=rho_sp1)
    pot_sp2 = potential_sphere(abs(x - d / 2)*L0, R2 * L0, G_GRAV, rho=rho_sp2)
    return pot_sp2


a_newton = newton_acc(yy)
# pot_newton = newton_pot(yy)

fig, axs = plt.subplots(figsize=(10, 3.5), nrows=1, ncols=3)

axs[0].plot(yy*L0/1e-2, a_cham)
inset_ax = inset_axes(axs[0], width="35%", height="35%", loc='lower right')
inset_ax.plot(yy*L0/1e-2, a_cham)
inset_ax.set_xlim(-3, 1)  # Zoom in around (0, 0)
inset_ax.set_ylim(-1e-8, 5e-8)
inset_ax.axvline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
inset_ax.axhline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
inset_ax.tick_params(direction='in', labelsize=10, left=True, right=True,
                     bottom=True, top=True)
axs[0].indicate_inset([-3, -1e-8, 4, 6e-8], inset_ax=inset_ax)

axs[1].plot(yy*L0/1e-2, a_newton)
inset_ax = inset_axes(axs[1], width="35%", height="35%", loc='lower right')
inset_ax.plot(yy*L0/1e-2, a_newton)
inset_ax.set_xlim(-3, 1)  # Zoom in around (0, 0)
inset_ax.set_ylim(-3e-9, 1e-9)
inset_ax.axvline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
inset_ax.axhline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
inset_ax.tick_params(direction='in', labelsize=10, left=True, right=True,
                     bottom=True, top=True)
axs[1].indicate_inset([-3, -3e-9, 4, 4e-9], inset_ax=inset_ax)

axs[2].plot(yy*L0/1e-2, a_newton + a_cham)
inset_ax = inset_axes(axs[2], width="35%", height="35%", loc='lower right')
inset_ax.plot(yy*L0/1e-2, a_newton + a_cham)
inset_ax.set_xlim(-3, 1)  # Zoom in around (0, 0)
inset_ax.set_ylim(-1e-8, 5e-8)
inset_ax.axvline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
inset_ax.axhline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
inset_ax.tick_params(direction='in', labelsize=10, left=True, right=True,
                     bottom=True, top=True)
axs[2].indicate_inset([-3, -1e-8, 4, 6e-8], inset_ax=inset_ax)

for ax in axs:
    ax.axvspan((-d / 2 - R1) * L0 / 1e-2, (-d / 2 + R1) * L0 / 1e-2,
                   facecolor='gray', edgecolor=None, alpha=0.05, zorder=-1)
    ax.axvspan((d / 2 - R2) * L0 / 1e-2, (d / 2 + R2) * L0 / 1e-2, facecolor='gray',
               edgecolor=None, alpha=0.05, zorder=-1)
    ax.set_xlim(-d / 2 * L0 / 1e-2, +d / 2 * L0 / 1e-2)
    ax.axvline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
    ax.axhline(0, linestyle='dashed', color='red', linewidth=1, zorder=-1, alpha=0.4)
    ax.set_xlabel(r"$x \, \mathrm{[cm]}$", fontsize=15)
    ax.tick_params(direction='in', left=True, right=True, bottom=True, top=True)
plt.tight_layout()
plt.show()
