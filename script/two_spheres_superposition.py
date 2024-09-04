# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:14:36 2024

@author: hlevy
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from femtoscope.inout.postprocess import ResultsPostProcessor

# geometric parameters
R1 = 1  # radius of sphere no 1
R2 = 0.8  # radius of sphere no 2
d = 3  # distance between the two spheres
Rc = 6  # truncation radius

# Fetching the solution array
res_two_spheres = ResultsPostProcessor.from_files("two-spheres-case-2")
res_sphere_1 = ResultsPostProcessor.from_files("sphere-1-case-2")
res_sphere_2 = ResultsPostProcessor.from_files("sphere-2-case-2")
sol_two_spheres_int = res_two_spheres.sol_int
sol_sphere_1_int = res_sphere_1.sol_int
sol_sphere_2_int = res_sphere_2.sol_int

coors_pos = res_two_spheres.coors_int
coors_neg = np.copy(coors_pos)
coors_neg[:, 0] *= -1
coors = np.concatenate((coors_pos, coors_neg), axis=0)
sol_two_spheres = np.concatenate((sol_two_spheres_int, sol_two_spheres_int))
sol_sphere_1 = np.concatenate((sol_sphere_1_int, sol_sphere_1_int))
sol_sphere_2 = np.concatenate((sol_sphere_2_int, sol_sphere_2_int))
X = coors[:, 1]
Y = -coors[:, 0]

# Superposition
sol_super = sol_sphere_1 + sol_sphere_2
sol_rel = abs(sol_super - sol_two_spheres) / abs(sol_two_spheres-(1e-1)**(-1/2))
# rel_log = np.log10(sol_rel)

plt.figure()
ax = plt.gca()
trictf = ax.tricontourf(coors[:, 1], -coors[:, 0], sol_super, levels=100)
plt.ylim(-3, 3)
plt.xlim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.08)
plt.colorbar(trictf, cax=cax, ticks=[20, 30, 40, 50])
for c in trictf.collections:
    c.set_rasterized(True)
plt.savefig('sphere_super.pdf', dpi=400)
plt.show()

plt.figure()
ax = plt.gca()
trictf = ax.tricontourf(coors[:, 1], -coors[:, 0], sol_rel, levels=100)
plt.ylim(-3, 3)
plt.xlim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.08)
plt.colorbar(trictf, cax=cax, ticks=[1e-4, 2e-4, 3e-4, 4e-4])
for c in trictf.collections:
    c.set_rasterized(True)
plt.savefig('sphere_rel_log.pdf', dpi=400)
plt.show()
