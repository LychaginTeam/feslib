#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some exapmles of fluid
"""

import numpy as np
import matplotlib.pyplot as plt
from feslib.VdWMix import VdWMix


#%%

#universal gas const
R = 8.314472


#%% Shatlyk gasfield
#methane, ethane, propane, butan, pentane, nitrogen, CO2


fld = VdWMix()
fld.tc = np.array([190.69900513, 305.42800903, 369.89801025, 425.19900513,
                    469.6000061, 126.19400024, 304.1000061])
fld.pc = np.array([4640680.17578125, 4883850.09765625, 4256660.15625,
                    3796620.1171875 , 3375120.1171875 , 3394370.1171875 ,
                    7370000.])
fld.a = fld.ACalc(fld.tc, fld.pc)
fld.b = fld.BCalc(fld.tc, fld.pc)


#singularity curve in Tp
nComp = fld.a.shape[0]
nn0 = 100
nn1 = 300
nn = nn0 + nn1
v0m = 2 * fld.b
v2m = 10 * fld.b
# v1m = 10000 * np.ones((nComp,)) * units.m3 * 1e-6 / units.mol
v1m = 10000 * np.ones((nComp,))  * 1e-6

vm = np.zeros((nComp, nn))
vm0 = np.zeros((nComp, nn0))
vm1 = np.zeros((nComp, nn1))
for i in range(nComp):
    vm0 = np.linspace(v0m[i], v2m[i], nn0)
    vm1 = np.linspace(v2m[i] + fld.b[i], v1m[i], nn1)
    # vm[i] = np.linspace(v0m[i], v1m[i], nn)
    vm[i] = np.concatenate((vm0, vm1))
Tm = np.zeros((nComp, nn))
pm = np.zeros((nComp, nn))
for i in range(nn):
    for i1 in range(nComp):
        Tm[i1,i] = fld.SingCurvTCalc(vm[i1,i], i1)
        pm[i1,i] = fld.SingCurvPCalc(vm[i1,i], i1)
    Tm[i1,-1] = 0.
    pm[i1,-1] = 0.



compName = ["Methane", "Ethane", "Propane", "n-Butane", "n-Pentane",
            "Nitrogen", "CO2"]

#singularity curves and thermodynamicly unstable set
fig = plt.figure(dpi=300)
for i1 in range(nComp):
    plt.plot(Tm[i1], pm[i1] * 1e-6, label=compName[i1])
    plt.fill(Tm[i1], pm[i1] * 1e-6, alpha=0.3)
plt.legend()
plt.xlabel("T, K")
plt.ylabel("P, MPa           ", rotation=0)
plt.grid()
plt.show()



#%% methane and water


# fld = VdWMix()
# fld.tc = np.array([190.69900513, 647.29901123])
# fld.pc = np.array([4.64068017578125, 22.120]) * 1e+6
# fld.nu = np.array([6.0, 6.0])
# fld.PrepForCalculation()

# t0 = 275
# p0 = 0.25
# comp = np.array([0.8, 0.2])
# v, vcm = fld.EquilMixTPCalc(t0, p0, comp)



# tt = 280.
# l = 0.9
# comp = np.array([l, 1.0 - l])
# p0 = 10 * 1e+3
# p1 = 300 * 1e+3
# nn = 300

# pm = np.zeros(comp.shape[0])
# v0m = np.zeros(comp.shape[0])
# v1m = np.zeros(comp.shape[0])
# for i in range(comp.shape[0]):
#     pm[i], v0m[i], v1m[i] = fld.PPhTrCalc(tt, i)

# pM = np.linspace(p0, p1, nn)
# vM = np.zeros((nn,))
# v0M = np.zeros((nn,))
# v1M = np.zeros((nn,))
# for i in range(nn):
#     v, vcm = fld.EquilMixTPCalc(tt, pM[i], comp)
#     vM[i] = v
#     v0M[i] = vcm[0]
#     v1M[i] = vcm[1]


# plt.figure(dpi=300)
# plt.plot(vM, pM)
# plt.plot(v0M, pM, '--')
# plt.plot(v1M, pM, '--')
# plt.xlim([0, 0.012])
# plt.ylim([150 * 1e+3,300 * 1e+3])
# plt.title("isotherm")
# plt.xlabel("vol")
# plt.ylabel("press")
# plt.legend(["mix", "methane", "water"])
# plt.grid()
# plt.show()




# # isotherm for Helmholtz potential
# # calc TP equilibrium
# tt = 280.
# l = 0.9
# comp = np.array([l, 1.0 - l])
# p0 = 10 * 1e+3
# p1 = 300 * 1e+3
# nn = 300

# pm = np.zeros(comp.shape[0])
# v0m = np.zeros(comp.shape[0])
# v1m = np.zeros(comp.shape[0])
# for i in range(comp.shape[0]):
#     pm[i], v0m[i], v1m[i] = fld.PPhTrCalc(tt, i)

# pM = np.linspace(p0, p1, nn)
# vM = np.zeros((nn,))
# alphaM = np.zeros((nn,))
# alpha0M = np.zeros((nn,))
# alpha1M = np.zeros((nn,))
# for i in range(nn):
#     v, vcm = fld.EquilMixTPCalc(tt, pM[i], comp)
#     vM[i] = v
#     v0M[i] = vcm[0]
#     v1M[i] = vcm[1]
#     alphaM[i] = fld.HelmholtzMixTPCalc(tt, pM[i], comp)
#     alpha0M[i] = fld.Helmholtz0Calc(v0M[i], tt, 0)
#     alpha1M[i] = fld.Helmholtz0Calc(v1M[i], tt, 1)


# plt.figure(dpi=300)
# plt.plot(vM, alphaM)
# plt.plot(v0M, alpha0M, '--')
# plt.plot(v1M, alpha1M, '--')
# plt.xlim([0.01, 0.016])
# plt.ylim([4.1e+4, 4.2e+4])
# plt.xlabel("vol")
# plt.ylabel("Helmholtz")
# plt.legend(["mix", "methane", "water"])
# plt.grid()
# plt.show()



#%% wather isotherm

# fld = VdWMix()
# fld.tc = np.array([647.29901123])
# # fld.pc = np.array([22120.]) * 1e+3
# fld.pc = np.array([22.120])
# fld.nu = np.array([6.0])
# fld.PrepForCalculation()


# nn1 = 900
# nn2 = 100
# nn = nn1 + nn2
# t0 = 570
# v0 = 31
# v1 = 400
# vMixM = np.linspace(v0, v1, nn)
# gM = np.zeros(nn)

# for i in range(nn):
#     gM[i] = fld.Gibbs0Calc(vMixM[i], t0, 0)

# pp, vv0, vv1 = fld.PPhTrCalc(t0, 0)
# gg = fld.Gibbs0Calc(vv0, t0, 0)

# plt.figure(dpi=300)
# plt.plot(vMixM, gM)
# plt.plot([v0, v1], [gg, gg])
# plt.legend(["water"])
# # plt.xlim([0, 200])
# plt.ylim([gM.min(), -2.3e+4])
# # plt.yscale("log")
# plt.grid()
# plt.show()


# t0 = 570
# pM = np.zeros(nn)
# for i in range(nn):
#     pM[i] = fld.PCalc(vMixM[i], t0, 0)

# plt.figure(dpi=300)
# plt.plot(vMixM, pM)
# plt.legend(["water"])
# # plt.xlim([0, 200])
# plt.ylim([pM.min(), 17])
# # plt.yscale("log")
# plt.grid()
# plt.show()


# vM = fld.v3Calc(t0, 7, fld.a[0], fld.b[0])
# print(fld.gamma0Calc(vM[1], t0) - fld.gamma0Calc(vM[0], t0))


# vv0, vv1, pp = fld.PhTrCalc(t0)
# print(vv0, vv1, pp)
