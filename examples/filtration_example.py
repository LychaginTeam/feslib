#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of isoentropy one component filtration in disc
constant boundary condion, permiability and viscosity

Source/sink in the disk center
"""

import numpy as np
import scipy.optimize as opt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from feslib.VdWMix import VdWMix
from feslib.Filtration import Filtration


GasConstant = 8.314472# * Pa m3 / (K mol)


#%%


filtr = Filtration()

filtr.c = np.array([1e+2])# source/sink intensity
filtr.a = np.array([[0.0, 0.0]])# source/sink coordinations


fld = VdWMix()

#water
# fld.tc = np.array([647.29901123])# critical temperature, K
# fld.pc = np.array([22120.0 * 1e+3])# critical pressure, Pa
# fld.nu = np.array([6.0])# degree of freedom

#methane
fld.tc = np.array([190.69900513])# critical temperature, K
fld.pc = np.array([4.64068017578125e+6])# critical pressure, Pa
fld.nu = np.array([6.0])# degree of freedom

fld.perm = 1e-12# permiability, m2
fld.visc = 1e-6# viscosity, kg / (m s)

fld.PrepForCalculation()


pf = 10.0 * 101325.0
tf = 303.15
sf = fld.EntropyPhaseCalc(tf, pf, 0, 0)# constant entropy
filtr.uc = fld.QIsoEntropyCalc(pf, sf, 0)
print("uoc", fld.QIsoEntropyCalc(pf, sf, 0))


# find Leibenson function Q
nn = 100
x0 = 1.0
x1 = 1000.0
xm = np.linspace(x0, x1, nn)
u0m = np.zeros(nn)
for i in range(nn):
    x = np.array([xm[i], 0.0])
    u0m[i] = filtr.U0DCalc(x, r0=x1)

plt.figure(dpi=300)
plt.plot(xm, u0m, ".")
plt.title("u0(x)")
plt.grid()
plt.show()


# Find P from Q
p0 = pf
p1 = np.exp(opt.root_scalar(
    lambda t: fld.QIsoEntropyCalc(np.exp(t), sf, 0) / u0m[0] - 1.0,\
    x0 = np.log(1000 * pf),\
    fprime = lambda t: np.exp(t) * fld.DQIsoEntropyCalc(np.exp(t), sf, 0) / u0m[0],\
    method="newton").root)
pIm = np.linspace(p0, p1, nn)

qIm = fld.QmIsoEntropyCalc(pIm, sf, 0)

if qIm[0] < qIm[1]:
    pqF = CubicSpline(qIm, pIm)
else:
    pqF = CubicSpline(np.flip(qIm), np.flip(pIm))

pm = np.zeros(nn)
tm = np.zeros(nn)
fracVm = np.zeros(nn)
for i in range(nn):
    pm[i] = pqF(u0m[i])
    tm[i], _, _, fracVm[i] = fld.EquilPSCalc(pm[i], sf, 0)


plt.figure(dpi=300)
plt.plot(xm, pm / pf, ".")
plt.title("Pr(x)")
plt.grid()
plt.show()

plt.figure(dpi=300)
plt.plot(xm, tm, ".")
plt.title("T(x)")
plt.grid()
plt.show()

plt.figure(dpi=300)
plt.plot(xm, fracVm)
plt.title("frav V(x)")
plt.grid()
plt.show()
