# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:16:52 2022

Definition of all the useful physical constants (S.I. units).

@author: hlevy
"""

from numpy import sqrt, pi

# Speed of light
C_LIGHT = 299792458 # [m/s]

# Gravitational constant
G_GRAV = 6.6743e-11 # [m^3 . kg^-1 . s^-2]

# Reduced Planck constant
H_BAR = 1.054571817e-34 # [J . s]

# Reduced Planck mass
M_PL = sqrt(H_BAR*C_LIGHT/(8*pi*G_GRAV)) # [kg]

# Electron-Volt
EV = 1.602176634e-19 # [J]

# Density conversion
RHO_FACTOR = (H_BAR**3 * C_LIGHT**5) / EV**4

# Earth radius
R_EARTH = 6371e3 # [m]

# Earth mass
M_EARTH = 5.9722e24 # [kg]

# Gravitational parameter of the Earth
MU_EARTH = 398600441.8e6 # [m^3 . s^-2]

# Dark energy scale
LAMBDA_DE = 2.4e-3 # eV

# Sun radius
R_SUN = 696340e3 # m
