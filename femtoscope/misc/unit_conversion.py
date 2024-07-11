# -*- coding: utf-8 -*-
r"""
Created on Wed Jun 15 08:29:43 2022

Convertion between SI units and natural units (in which $c = \hbar = 1$).

"""

from femtoscope.misc.constants import C_LIGHT, H_BAR, EV, M_PL, LAMBDA_DE


# Conversion routines
def mass_to_nat(mass):
    return mass * C_LIGHT ** 2 / EV


def nat_to_mass(nat):
    return nat * EV / C_LIGHT ** 2


def length_to_nat(L):
    return L * EV / (H_BAR * C_LIGHT)


def nat_to_length(nat):
    return nat * H_BAR * C_LIGHT / EV


def force_to_nat(force):
    return force * H_BAR * C_LIGHT / EV ** 2


def nat_to_force(nat):
    return nat * EV ** 2 / (H_BAR * C_LIGHT)


def acc_to_nat(acc):
    return acc * H_BAR / (C_LIGHT * EV)


def nat_to_acc(nat):
    return nat * C_LIGHT * (EV / H_BAR)


def density_to_nat(rho):
    return rho * H_BAR ** 3 * C_LIGHT ** 5 / EV ** 4


def nat_to_density(nat):
    return nat * EV ** 4 / (H_BAR ** 3 * C_LIGHT ** 5)


# Computation of alpha from SI units
def compute_alpha(Lambda, beta, npot, L_0, rho_0):
    """
    Lambda : [eV]
    beta   : []
    npot   : []
    L_0    : [m]
    rho_0  : [kg/m^3]
    alpha  : []
    """
    t1 = M_PL * Lambda / (beta * rho_0 * L_0 ** 2) * EV / (H_BAR * C_LIGHT)
    t2 = npot * M_PL * Lambda ** 3 / (beta * rho_0) * (
            EV / (H_BAR * C_LIGHT)) ** 3
    return t1 * t2 ** (1 / (npot + 1))


# Computation of beta from SI units
def compute_beta(Lambda, alpha, npot, L_0, rho_0):
    """
    Lambda : [eV]
    alpha  : []
    npot   : []
    L_0    : [m]
    rho_0  : [kg/m^3]
    beta   : []
    """
    t1 = M_PL * Lambda / (alpha * rho_0 * L_0 ** 2) * EV / (H_BAR * C_LIGHT)
    t2 = npot * M_PL * Lambda ** 3 / rho_0 * (EV / (H_BAR * C_LIGHT)) ** 3
    return t1 ** ((npot + 1) / (npot + 2)) * t2 ** (1 / (npot + 2))


# Computation of Lambda from SI units
def compute_Lambda(beta, alpha, npot, L_0, rho_0):
    """
    beta   : []
    alpha  : []
    npot   : []
    L_0    : [m]
    rho_0  : [kg/m^3]
    Lambda : [eV]
    """
    t1 = M_PL / (alpha * beta * rho_0 * L_0 ** 2) * EV / (H_BAR * C_LIGHT)
    t2 = npot * M_PL / (rho_0 * beta) * (EV / (H_BAR * C_LIGHT)) ** 3
    return t1 ** (-(npot + 1) / (npot + 4)) * t2 ** (-1 / (npot + 4))


# Computation of phi_0 from SI units
def compute_phi_0(Lambda, beta, npot, rho_0):
    """
    beta   : []
    npot   : []
    rho_0  : [kg/m^3]
    Lambda : [eV]
    """
    t = npot * M_PL * Lambda ** (npot + 4) / (beta * rho_0) * (
            EV / (H_BAR * C_LIGHT)) ** 3
    return t ** (1 / (npot + 1))


# computation of a0 (m/s^2)
def compute_acc_0(rho_0, L_0, npot=1, Lambda=LAMBDA_DE, alpha=None, beta=None):
    """
    Compute the coefficient that needs to multiply the dimensionless chameleon
    field gradient to recover the chameleon acceleration expressed in m/s^2.

    rho_0  : [kg/m^3]
    L_0    : [m]
    npot   : []
    Lamnda : [eV]
    alpha  : []
    beta   : []
    """
    assert [alpha, beta].count(None) != 2, "Need to specify alpha or beta"
    if alpha is not None and beta is not None:
        Lambda = compute_Lambda(beta, alpha, npot, L_0, rho_0)
    elif alpha is not None and beta is None:
        beta = compute_beta(Lambda, alpha, npot, L_0, rho_0)
    else:
        pass
    gamma_n = EV / (M_PL * L_0) * (npot * M_PL / rho_0
                   * (EV / (H_BAR * C_LIGHT)) ** 3) ** (1 / (npot + 1))
    return gamma_n * Lambda ** ((npot+4)/(npot+1)) * beta ** (npot/(npot+1))
