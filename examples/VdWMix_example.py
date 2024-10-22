#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import measurementunits as units
import matplotlib.pyplot as plt
from VdWMix import VdWMix, SecantMethod


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
fld.a = fld.aCalc(fld.tc, fld.pc)
fld.b = fld.bCalc(fld.tc, fld.pc)


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
        Tm[i1,i] = fld.phivvTCalc(vm[i1,i], a=fld.a[i1], b=fld.b[i1])
        pm[i1,i] = fld.phivvPCalc(vm[i1,i], a=fld.a[i1], b=fld.b[i1])
    Tm[i1,-1] = 0.
    pm[i1,-1] = 0.


#singularity curv
compName = ["Methane", "Ethane", "Propane", "n-Butane", "n-Pentane",
            "Nitrogen", "CO2"]
plt.figure(dpi=300)
for i1 in range(nComp):
    plt.plot(Tm[i1], pm[i1], label=compName[i1])
plt.legend()
plt.ylim([0, pm.max()])
plt.grid()
plt.show()


#thermodynamicly unstable set
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


fld = VdWMix()
fld.tc = np.array([190.69900513, 647.29901123])
fld.pc = np.array([4640.68017578125, 22120.]) * 1e+3
fld.a = fld.aCalc(fld.tc, fld.pc)
fld.b = fld.bCalc(fld.tc, fld.pc)



tt = 280.
l = 0.9
z = np.array([l, 1.0 - l])
p0 = 10 * 1e+3
p1 = 300 * 1e+3
nn = 300
v0m, v1m, pm = fld.PhTrCalc(tt)
pM = np.linspace(p0, p1, nn)
vM = np.zeros((nn,))
v0M = np.zeros((nn,))
v1M = np.zeros((nn,))
for i in range(nn):
    v, vcm = fld.EquilTPCalc(tt, pM[i], z)
    vM[i] = v
    v0M[i] = vcm[0]
    v1M[i] = vcm[1]


plt.figure(dpi=300)
plt.plot(vM, pM)
plt.plot(v0M, pM, '--')
plt.plot(v1M, pM, '--')
plt.xlim([0, 0.012])
plt.ylim([150 * 1e+3,300 * 1e+3])
plt.title("isotherm")
plt.xlabel("vol")
plt.ylabel("press")
plt.legend(["mix", "methane", "water"])
plt.grid()
plt.show()




# isotherm for Helmholtz potential
# calc TP equilibrium
tt = 280.
l = 0.9
z = np.array([l, 1.0 - l])
p0 = 10 * 1e+3
p1 = 300 * 1e+3
nn = 300
v0m, v1m, pm = fld.PhTrCalc(tt)
pM = np.linspace(p0, p1, nn)
vM = np.zeros((nn,))
alphaM = np.zeros((nn,))
alpha0M = np.zeros((nn,))
alpha1M = np.zeros((nn,))
for i in range(nn):
    v, vcm = fld.EquilTPCalc(tt, pM[i], z)
    vM[i] = v
    v0M[i] = vcm[0]
    v1M[i] = vcm[1]
    alphaM[i] = fld.alpha0MixCalc0(tt, pM[i], z)
    alpha0M[i] = fld.alpha0Calc(v0M[i], tt, fld.a[0], fld.b[0])
    alpha1M[i] = fld.alpha0Calc(v1M[i], tt, fld.a[1], fld.b[1])


plt.figure(dpi=300)
plt.plot(vM, alphaM)
plt.plot(v0M, alpha0M, '--')
plt.plot(v1M, alpha1M, '--')
plt.xlim([0.006, 0.035])
plt.ylim([8e+3,11.5e+3])
plt.title("isotherm, TP")
plt.xlabel("vol")
plt.ylabel("Helmholtz")
plt.legend(["mix", "methane", "water"])
plt.grid()
plt.show()



# isotherm for Helmholtz potential
# calc VT equilibrium
tt = 280.
l = 0.9
z = np.array([l, 1.0 - l])
p0 = 10 * 1e+3
p1 = 300 * 1e+3
nn = 300
v0m, v1m, pm = fld.PhTrCalc(tt)
alphaM = np.zeros((nn,))
alpha0M = np.zeros((nn,))
alpha1M = np.zeros((nn,))
for i in range(nn):
    p, vcm = fld.EquilVTCalc(vM[i], tt, z)
    v0M[i] = vcm[0]
    v1M[i] = vcm[1]
    alphaM[i] = fld.alpha0MixCalc1(vM[i], tt, z)
    alpha0M[i] = fld.alpha0Calc(v0M[i], tt, fld.a[0], fld.b[0])
    alpha1M[i] = fld.alpha0Calc(v1M[i], tt, fld.a[1], fld.b[1])


plt.figure(dpi=300)
plt.plot(vM, alphaM)
plt.plot(v0M, alpha0M, '--')
plt.plot(v1M, alpha1M, '--')
plt.xlim([0.006, 0.035])
plt.ylim([8e+3,11.5e+3])
plt.title("isotherm, VT")
plt.xlabel("vol")
plt.ylabel("Helmholtz")
plt.legend(["mix", "methane", "water"])
plt.grid()
plt.show()