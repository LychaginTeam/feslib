#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example methane and ethane
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
#import h5py
from numba import jit
from scipy.integrate import quad, nquad
from scipy.optimize import fsolve, root, minimize
from time import time

from MSLVMix import MSLVMix


#%%

#universal gas const
R = 8.314472


#%%

fluid = MSLVMix()
fluid.acPure = np.array([267576.04999955, 634217.7545973])
fluid.bPure = np.array([29.48676014, 43.21622085])
fluid.dPure = np.array([35.54066954, 46.1522217])
fluid.cPure = np.array([35.54658734, 46.15235265])
fluid.wPure = np.array([0.11e-1, 0.99e-1])
fluid.mPure = np.array([0.39157219967999995, 0.5246782540799999])
fluid.k = np.array([[0,-0.0026], [-0.0026,0]])

fluid.vcPure = np.array([98.63, 145.5])
fluid.TcPure = np.array([190.56, 305.32])
fluid.pcPure = np.array([4.5992, 4.8724])

# methane pure
fluid.comp = np.array([1., 0.])
fluid.vc = 116.5407548
fluid.Tc = 187.0254283
fluid.pc = 4.10188288

# 20% methane + 80% ethane
# fluid.comp = np.array([0.2, 0.8])
# fluid.vc = 159.9186596
# fluid.Tc = 281.1015168
# fluid.pc = 4.49269506

fluid.b = fluid.bCalc()
fluid.d = fluid.dCalc()
fluid.c = fluid.cCalc()
fluid.br = fluid.b / fluid.vc
fluid.dr = fluid.d / fluid.vc
fluid.cr = fluid.c / fluid.vc


#%% VL and SV phase transition

TM1VL, pM1VL, v1M1VL, v2M1VL = fluid.PhTrVLr(0.25, 0.3)
TM2VL, pM2VL, v1M2VL, v2M2VL = fluid.PhTrVLr(0.3, 0.99, nn=50)

TMVL = np.concatenate((TM1VL, TM2VL))
pMVL = np.concatenate((pM1VL, pM2VL))
v1MVL = np.concatenate((v1M1VL, v1M2VL))
v2MVL = np.concatenate((v2M1VL, v2M2VL))


#%% SL phase transition

TM1SL, pM1SL, v1M1SL, v2M1SL = fluid.PhTrSLr(0.25, 0.3, nn=30)
TM2SL, pM2SL, v1M2SL, v2M2SL = fluid.PhTrSLr(0.3+1e-3, 1-1e-3, nn=50)
TM3SL, pM3SL, v1M3SL, v2M3SL = fluid.PhTrSLr(1., 1.1, nn=10)

TMSL = np.concatenate((TM1SL, TM2SL, TM3SL))
pMSL = np.concatenate((pM1SL, pM2SL, pM3SL))
v1MSL = np.concatenate((v1M1SL, v1M2SL, v1M3SL))
v2MSL = np.concatenate((v2M1SL, v2M2SL, v2M3SL))

# triple point
Tt, pt, it = fluid.TripleFind(TMSL, pMSL, TMVL, pMVL)

nSL = pMSL[pMSL>0.].shape[0]
NSL = pMSL.shape[0]
TMSL = TMSL[NSL-nSL:]
pMSL = pMSL[NSL-nSL:]
v1MSL = v1MSL[NSL-nSL:]
v2MSL = v2MSL[NSL-nSL:]

# pT diagram
plt.figure()
plt.plot(np.concatenate((TMVL[it:], [1.])), np.concatenate((pMVL[it:], [1.])))
plt.plot(np.concatenate(([Tt], TMSL)), np.concatenate(([pt], pMSL)))
plt.plot(np.concatenate((TMVL[:it], [Tt])), np.concatenate((pMVL[:it], [pt])))
plt.plot(Tt, pt, 'o')
plt.plot(1., 1., 'o')
plt.ylim([1e-9,5e+2])
plt.yscale('log')
plt.title('pT-diagram')
plt.legend(['LV phase transition', 'SL phase transition',\
            'SV phase transition', 'Crit point', 'Triple point'])
plt.grid()
plt.show()
