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
R1 = 0.5  # radius of sphere no 1
R2 = 0.5  # radius of sphere no 2
d = 4  # distance between the two spheres
Rc = 6  # truncation radius

# Fetching the solution array
result_pp = ResultsPostProcessor.from_files("two-spheres-super-2")
sol_int = result_pp.sol_int
residual = result_pp.vars_dict['residual']
coors_pos = result_pp.coors_int
coors_neg = np.copy(coors_pos)
coors_neg[:, 0] *= -1
coors = np.concatenate((coors_pos, coors_neg), axis=0)
sol = np.concatenate((sol_int, sol_int))
res = abs(np.concatenate((residual, residual)))
resort = np.sort(res)
resmin = resort[np.where(resort > 0)[0][0]]
res[np.where(res==0)[0]] = resmin
X = coors[:, 1]
Y = -coors[:, 0]

def symmetric_gradient():
    aux = np.copy(coors)
    aux[:, 0] = abs(coors[:, 0])
    grad = result_pp.evaluate_at(aux, mode='grad')
    return grad

yy = np.linspace(-5, 5, 700)
xx = np.zeros_like(yy)
cline = np.concatenate((xx[:, np.newaxis], yy[:, np.newaxis]), axis=1)
philine = result_pp.evaluate_at(cline)
gradline = result_pp.evaluate_at(cline, mode='grad')[:, 1]

# Fig residual
plt.figure()
ax = plt.gca()
trictf = ax.tricontourf(coors[:, 1], -coors[:, 0], np.log10(res), levels=100,
                        cmap='cividis')
plt.ylim(-3, 3)
plt.xlim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.08)
plt.colorbar(trictf, cax=cax, ticks=None)
for c in trictf.collections:
    c.set_rasterized(True)
plt.savefig('residual_super-0.pdf', dpi=400)
plt.show()

# Fig tricontourf phi
plt.figure()
ax = plt.gca()
trictf = ax.tricontourf(coors[:, 1], -coors[:, 0], sol_super-0.1**(-1/2), levels=100)
plt.ylim(-3, 3)
plt.xlim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.08)
plt.colorbar(trictf, cax=cax, ticks=None)
for c in trictf.collections:
    c.set_rasterized(True)
# plt.savefig('two_sphere.pdf', dpi=400)
plt.show()

# Fig tricontourf grad
grad = symmetric_gradient()
grad_norm = np.linalg.norm(grad, axis=1)
plt.figure()
ax = plt.gca()
trictf = ax.tricontourf(coors[:, 1], -coors[:, 0], grad_norm, levels=100,
                        cmap='magma')
plt.ylim(-3, 3)
plt.xlim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.08)
plt.colorbar(trictf, cax=cax, ticks=[0, 10, 20, 30])
for c in trictf.collections:
    c.set_rasterized(True)
plt.savefig('two_sphere.pdf', dpi=400)
plt.show()

# Fig phi along x axis
plt.figure(figsize=(7, 4))
plt.plot(yy, philine)
plt.axvspan(-d/2-R1, -d/2+R1, facecolor='gray', edgecolor=None, alpha=0.1, zorder=-1)
plt.axvspan(d/2-R2, d/2+R2, facecolor='gray', edgecolor=None, alpha=0.1, zorder=-1)
plt.xlim(-5, 5)
plt.show()

# Fig grad phi along x axis
plt.figure(figsize=(7, 4))
plt.plot(yy, gradline)
plt.axvspan(-d/2-R1, -d/2+R1, facecolor='gray', edgecolor=None, alpha=0.1, zorder=-1)
plt.axvspan(d/2-R2, d/2+R2, facecolor='gray', edgecolor=None, alpha=0.1, zorder=-1)
plt.xlim(-5, 5)
plt.show()
