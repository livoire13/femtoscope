# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:27:50 2023
Illustration des transformations {compactification + kelvin inversion} sur
une image simple générée en Python

@author: hlevy
"""

import numpy as np
from numpy import pi, sqrt, arctan2, cos, sin, arctan
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
import colorcet

if __name__ == '__main__':

    Rc = 3.0
    cmap = matplotlib.colormaps.get_cmap("viridis")
    # cvals = sqrt(np.linspace(0, 1, 100))
    cvals = np.linspace(0, 1, 100)
    colors = cmap(cvals)

    #%% Generate the "flat space" image

    size = 10
    xmin, xmax = -size, size
    ymin, ymax = -size, size
    xlevels = np.arange(xmin, xmax+1)
    ylevels = np.arange(ymin, ymax+1)
    radii = np.arange(1, xmax+5)
    circles = []

    plt.figure(figsize=(10, 8))
    for kk in range(len(xlevels)):
        plt.plot([xmin, xmax], [ylevels[kk], ylevels[kk]], c='lightgray')
    for jj in range(len(ylevels)):
        plt.plot([xlevels[jj], xlevels[jj]], [ymin, ymax], c='lightgray')
    for R in radii:
        # indc = int((R/(1+R)-radii[0]/(1+radii[0]))*len(colors))
        # indc = int(2/pi * (arctan(0.3*(R-radii[0]))) * len(colors))
        indc = int(((R-radii[0])/R)**4 * len(colors))
        c = colors[indc]
        circles.append(plt.Circle((0,0), R, color=c, fill=False, linewidth=3,
                                  alpha=0.5, zorder=2))

    fig = plt.gcf()
    ax = fig.gca()
    for circ in circles:
        ax.add_patch(circ)
    plt.grid(False)
    plt.axis('off')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    plt.show()

    #%% Generate the compactified image

    cmap = matplotlib.colormaps.get_cmap("viridis")
    cvals = sqrt(np.linspace(0, 1, 500))
    # cvals = np.linspace(0, 1, 100)
    colors = cmap(cvals)

    def X_to_U(x, y):
        r = sqrt(x**2 + y**2)
        theta = arctan2(y, x)
        u = Rc*r/(1+r) * cos(theta)
        v = Rc*r/(1+r) * sin(theta)
        return u, v

    def U_to_X(u, v):
        eta = sqrt(u**2 + v**2)
        x = eta/(Rc-eta) * u
        y = eta/(Rc-eta) * v
        return x, y

    # X = np.linspace(-100, 100, 1000)
    # Y = np.ones_like(X)
    # U, V = X_to_U(X, Y)
    # plt.figure()
    # plt.plot(X, Y)
    # plt.plot(U, V)
    # plt.ylim([0, None])
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    N2lines = 18
    Ncircles = 100
    circles = []
    radii = np.arange(1, Ncircles+1)

    plt.figure(figsize=(10, 8))
    for yy in range(-N2lines, N2lines+1):
        X = np.linspace(-100, 100, 1000)
        Y = yy * np.ones_like(X)
        U, V = X_to_U(X, Y)
        plt.plot(U, V, c='lightgray', alpha=(1-abs(yy/N2lines))**2,
                 linewidth=2-2*abs(yy/N2lines))
    for xx in range(-N2lines, N2lines+1):
        Y = np.linspace(-100, 100, 1000)
        X = xx * np.ones_like(Y)
        U, V = X_to_U(X, Y)
        plt.plot(U, V, c='lightgray', alpha=(1-abs(xx/N2lines))**2,
                 linewidth=2-2*abs(xx/N2lines))
    for R in radii:
        Rtilde = Rc*R/(1+R)
        indc = int((Rtilde/Rc - radii[0]/(1+radii[0])) * len(colors))
        c = colors[indc]
        circles.append(plt.Circle((0,0), Rtilde, color=c, fill=False, zorder=2,
                                  linewidth=5*((1-Rtilde/Rc)**1), alpha=0.5))
    great_circle = plt.Circle((0,0), Rc, fill=False, linewidth=4, zorder=2)
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(great_circle)
    for circ in circles:
        ax.add_patch(circ)
    plt.grid(False)
    plt.axis('off')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.axis("equal")
    # plt.savefig("compactified.svg", transparent=True)
    plt.show()

    #%% Generate the Kelvin inversion image
    Rc = 3.0
    cmap = matplotlib.colormaps.get_cmap("viridis_r")
    cvals = sqrt(np.linspace(0, 1, 100))
    colors = cmap(cvals)

    def X_to_U(x, y):
        r2 = x**2 + y**2
        u = Rc**2/r2 * x
        v = Rc**2/r2 * y
        return u, v

    N2lines = 15
    Ncircles = 500
    circles = []
    radii = np.arange(int(Rc), int(Rc)+Ncircles+1)

    plt.figure(figsize=(10, 8))
    for yy in range(-N2lines, N2lines+1):
        if abs(yy) >= Rc:
            X = np.linspace(-10, 10, 1000)
            Xtra = np.logspace(1, 10, 100)[1:]
            X = np.concatenate((np.flip(-Xtra), X, Xtra))
            Y = yy * np.ones_like(X)
            U, V = X_to_U(X, Y)
            plt.plot(U, V, c='lightgray', alpha=0.9,
                     linewidth=1)
    for xx in range(-N2lines, N2lines+1):
        if abs(xx) >= Rc:
            Y = np.linspace(-10, 10, 1000)
            Ytra = np.logspace(1, 10, 100)[1:]
            Y = np.concatenate((np.flip(-Ytra), Y, Ytra))
            X = xx * np.ones_like(Y)
            U, V = X_to_U(X, Y)
            plt.plot(U, V, c='lightgray', alpha=0.9,
                      linewidth=1)
    for R in radii:
        Rtilde = Rc**2/R
        indc = int((Rc/R) * (len(colors)-1))
        circles.append(plt.Circle((0,0), Rtilde, color=colors[indc], fill=False, zorder=2,
                                  linewidth=2*(Rtilde/Rc)**0.5, alpha=1*(Rtilde/Rc)**0.2))
    great_circle = plt.Circle((0,0), Rc, fill=False, linewidth=4, zorder=2)
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(great_circle)
    for circ in circles:
        ax.add_patch(circ)
    plt.grid(False)
    plt.axis('off')
    plt.xlim([-Rc, Rc])
    plt.ylim([-Rc, Rc])
    plt.axis("equal")
    plt.savefig("kelvin-inversed.svg", transparent=True)
    plt.show()