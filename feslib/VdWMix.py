#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Van der Waals
methane + water
1 -- methane, 2 -- water
mixture without chemical reaction
"""


import numpy as np
# import measurementunits as units
import matplotlib.pyplot as plt
import scipy.optimize as opt
# from freezing import freeze
from copy import copy


GasConstant = 8.314472# * units.Pa * units.m3 / (units.K * units.mol)


def Newton(f, x0, df, args=(), dargs=None, eps=1e-5, dxMax = 0.1,\
           xb0=-np.inf, xb1=np.inf, MaxIters=1000, log=False):
    """
    Newton's method
    
    x1 = x0 - f(x0)/f'(x0)
    """
    
    if dargs is None:
        dargs = copy(args)
    
    t0 = x0
    y0 = f(t0, *args)
    flag = True
    if np.abs(y0) < eps:
        flag = False
        t1 = t0
    else:
        flag = True
    
    i = 0
    while flag:
        if i > MaxIters:
            break
        dt = y0/df(t0, *args)
        if np.abs(dt) < 1e-12:
            t1 = t0
            break
        if np.abs(dt) > dxMax:
            dt = np.sign(dt)*dxMax
        t1 = t0 - dt
        if t1 < xb0:
            t1 = t0 - np.sign(dt) * (t0 - xb0) * 0.5
        elif t1 > xb1:
            t1 = t0 - np.sign(dt) * (xb1 - t0) * 0.5
        y1 = f(t1, *args)
        if log:
            print(t1, y1)
        
        if np.abs(y1) < eps and np.abs(dt) < eps:
            flag = False
        else:
            t0 = t1
            y0 = y1
        
        i = i + 1
        #end while
    # print('Newton:', i)
    if log:
        return t1, i
    else:
        return t1



def SecantMethod(f, x0, x1, args=(), f0=0., epsx=1e-6, epsf=1e-6,
                 maxIters=100, maxIters1=5):
    """
    secant method for solving non-linear equation
    """
    
    y0 = f(x0, *args) - f0
    y1 = f(x1, *args) - f0
    
    isConv = False
    i = 0
    while (not isConv):
        if i < maxIters1:
            x = 0.5 * (x0 + x1)
        else:
            x = x1 - y1 * (x1 - x0) / (y1 - y0)
        y = f(x, *args) - f0
        # print(i)
        # print("x", x0, x, x1)
        # print("p", y0, y, y1)
        
        if abs(y) < epsf and min(abs(x0 - x), abs(x1 - x)) < epsx:
            isConv = True
        elif i > maxIters:
            break
        else:
            i = i + 1
            if y0 < 0.:
                if y < 0.0:
                    x0 = x
                    y0 = y
                else:
                    x1 = x
                    y1 = y
            else:
                if y > 0.0:
                    x0 = x
                    y0 = y
                else:
                    x1 = x
                    y1 = y
    # print(i)
    return x


# @freeze
class VdWMix:
    
    def __init__(self):
        self.vc = None
        self.tc = None #component critical temperatures
        self.pc = None #component critical pressures
        self.a = None #component a coefficient of VdW EoS
        self.b = None #component b coefficient of VdW EoS
        self.comp = None
    
    def aCalc(self, tc, pc):
        """
        calculate component a coefficient of VdW EoS
        """
        return 27. * (tc * GasConstant)**2 / (64. * pc)
    
    def bCalc(self, tc, pc):
        """
        calculate component b coefficient of VdW EoS
        """
        return tc * GasConstant / (8. * pc)
    
    def RCubCalc(self, t, p, a, b):
        """
        -discrim\n
        R>0 -> oner real root\n
        R<0 -> three real roots\n
        """
        A = a * p / (GasConstant * t)**2
        B = b * p / (GasConstant * t)
        a0 = -A*B
        a1 = A
        a2 = -B-1
        p1 = (3*a1 - a2**2)/3
        q1 = (2*a2**3 - 9*a2*a1 + 27*a0)/27
        return q1**2 / 4 + p1**3 / 27
    
    def ZOneCalc(self, t, p, a, b):
        """
        root of Z cubic EOS
        if one root
        """
        A = a * p / (GasConstant * t)**2
        B = b * p / (GasConstant * t)
        a0 = -A*B
        a1 = A
        a2 = -B-1
        p1 = (3*a1 - a2**2)/3
        q1 = (2*a2**3 - 9*a2*a1 + 27*a0)/27
        R1 = q1**2 / 4 + p1**3 / 27
        P = np.cbrt(-q1/2 + np.sqrt(R1))
        Q = np.cbrt(-q1/2 - np.sqrt(R1))
        return P + Q - a2/3
    
    def ZCubCalc(self, t, p, a, b, theta0):
        """
        root of Z cubic EOS
        if 3 real roots
        """
        A = a * p / (GasConstant * t)**2
        B = b * p / (GasConstant * t)
        a0 = -A*B
        a1 = A
        a2 = -B-1
        p1 = (3*a1 - a2**2)/3
        q1 = (2*a2**3 - 9*a2*a1 + 27*a0)/27
        m1 = 2*np.sqrt(-p1/3)
        theta = np.arccos(3*q1/(p1*m1))/3
        return m1*np.cos(theta + theta0) - a2/3
    
    def ZCalc(self, t, p, a, b, ph):
        """
        calculate root of Z cubic EOS
        """
        RCub = self.RCubCalc(t, p, a, b)
        if RCub > 0.:
            #one real root
            Z = self.ZOneCalc(t, p, a, b)
        else:
            #three real root
            Z1 = self.ZCubCalc(t, p, a, b, 0.)
            Z2 = self.ZCubCalc(t, p, a, b, 2*np.pi/3)
            Z3 = self.ZCubCalc(t, p, a, b, 4*np.pi/3)
            if ph == 0:
                #vap
                Z = np.max([Z1, Z2, Z3])
            elif ph == 1:
                #liq
                B = b * p / (GasConstant * t)
                ZM = np.sort([Z1, Z2, Z3])
                for z in ZM:
                    if z - B > 0:
                        Z = z
                        break
        return Z
    
    def ZCalc(self, t, p, a, b):
        """
        calculate root of Z cubic EOS
        """
        RCub = self.RCubCalc(t, p, a, b)
        if RCub > 0.:
            #one real root
            Z = self.ZOneCalc(t, p, a, b)
        else:
            #three real root
            Z1 = self.ZCubCalc(t, p, a, b, 0.)
            Z2 = self.ZCubCalc(t, p, a, b, 2*np.pi/3)
            Z3 = self.ZCubCalc(t, p, a, b, 4*np.pi/3)
            
            v1 = Z1 * GasConstant * t / p
            v2 = Z2 * GasConstant * t / p
            v3 = Z3 * GasConstant * t / p
            
            g1 = self.gamma0Calc(v1, t, a, b)
            g2 = self.gamma0Calc(v2, t, a, b)
            g3 = self.gamma0Calc(v3, t, a, b)
            
            i = np.array([g1, g2, g3]).argmin()
            Z = np.array([v1, v2, v3])[i] * p / (GasConstant * t)
        return Z
    
    def vCalc(self, t, p, a, b):
        """
        calculate root of Z cubic EOS
        """
        RCub = self.RCubCalc(t, p, a, b)
        if RCub > 0.:
            #one real root
            v = self.ZOneCalc(t, p, a, b) * GasConstant * t / p
        else:
            #three real root
            Z1 = self.ZCubCalc(t, p, a, b, 0.)
            Z2 = self.ZCubCalc(t, p, a, b, 2*np.pi/3)
            Z3 = self.ZCubCalc(t, p, a, b, 4*np.pi/3)
            
            v1 = Z1 * GasConstant * t / p
            v2 = Z2 * GasConstant * t / p
            v3 = Z3 * GasConstant * t / p
            
            g1 = self.gamma0Calc(v1, t, a, b)
            g2 = self.gamma0Calc(v2, t, a, b)
            g3 = self.gamma0Calc(v3, t, a, b)
            
            i = np.array([g1, g2, g3]).argmin()
            v = np.array([v1, v2, v3])[i]
        return v
    
    def pCalc(self, v, T, a=None, b=None):
        """
        calculate pressure by EOS for all components
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return GasConstant * T / (v - b) - a / v**2
    
    def p1Calc(self, v, T, a=None, b=None, vp=None):
        """
        calculate pressure by EOS for all components
        phase transition is respected
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        if vp is None:
            v0, v1, pm = self.PhTrCalc(T, a, b)
        else:
            v0 = vp[0]
            v1 = vp[1]
            pm = vp[2]
        
        if type(a) == np.ndarray:
            nn = a.shape[0]
            p = np.zeros((nn,))
            for i in range(nn):
                if v0[i] < v < v1[i]:
                    p[i] = pm[i]
                else:
                    p[i] = GasConstant * T / (v - b[i]) - a[i] / v**2
        else:
            if v0 < v < v1:
                p = pm
            else:
                p = GasConstant * T / (v - b) - a / v**2
        
        return p
    
    def vMixCalc(self, t, p, a=None, b=None, pPhTr=None):
        """
        calculate mixture molar volume from temperature and pressure
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        vMix = 0.
        for i1 in range(a.shape[0]):
            if p > pPhTr[i1]:
                ph = 1
            else:
                ph = 0
            # print(i1, t, p)
            z = self.ZCalc(t, p, a[i1], b[i1], ph)
            vComp = GasConstant * t * z / p
            vMix = vMix + self.comp[i1] * vComp
        
        return vMix
    
    def alpha0Calc(self, v, T, a=None, b=None):
        """
        calculate Helmholtz potential without IG term for all components
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return -GasConstant * T * np.log(v - b) - a / v
    
    def phi0Calc(self, v, T, a=None, b=None):
        """
        calculate Massie-Planck potential without IG term for all components
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return GasConstant * np.log(v - b) + a / (v * T)
    
    def gamma0Calc(self, v, T, a=None, b=None):
        """
        calculate Gibbs potential without IG term for all components
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        # return v * self.pCalc(v, T, a, b) - T * self.phi0Calc(v, T, a, b)
        return v * self.pCalc(v, T, a, b) + self.alpha0Calc(v, T, a, b)
    
    def alphavvValc(self, v, T, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return -2 * a / v**3 + GasConstant * T / (v - b)**2
    
    def phivvCalc(self, v, T, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return -GasConstant / (v - b)**2 + 2 * a / (T * v**3)
    
    
    def phivvTCalc(self, v, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return 2 * a * (v - b)**2 / (GasConstant * v**3)
    
    def phivvPCalc(self, v, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return a * (v - 2 * b) / v**3
    
    def PhTrCalc(self, T, a=None, b=None, epsx=1e-1, epsf=1e-1, maxIters=100):
        """
        calculate molar volume for phase transition at temperature T
        for all components
        """
        
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        vc = 3 * b
        
        nn = a.shape[0]
        # print("nn", nn)
        v0m = np.zeros((nn,))
        v1m = np.zeros((nn,))
        pm = np.zeros((nn,))
        
        for i in range(nn):
            if T > self.tc[i]:
                v0m[i] = 0.
                v1m[i] = 0.
                pm[i] = 0.
            else:
                vMin = opt.minimize(self.pCalc, 0.5 * (b[i] + vc[i]),
                    args=(T, a[i], b[i]),
                    bounds=((1.01 * b[i], vc[i]),)).x[0]
                vMax = opt.minimize(lambda v, : -self.pCalc(v, T, a[i], b[i]),
                    2 * vc[i], bounds=((vc[i], 1e+3),)).x[0]
                # print("vMin, vMax", vMin, vMax)
                
                p0 = self.pCalc(vMin, T, a[i], b[i])
                # print("p0", p0)
                if p0 < 0.:
                    p0 = 0.
                    v1 = 1e+6
                else:
                    v1 = SecantMethod(self.pCalc, vMax, 1e+3,
                                      args=(T, a[i], b[i]), f0=p0)
                p1 = self.pCalc(vMax, T, a[i], b[i])
                v0 = SecantMethod(self.pCalc, 1.01 * b[i], vMin,
                                  args=(T, a[i], b[i]), f0=p1)
                
                disc0 = self.gamma0Calc(v1, T, a[i], b[i]) -\
                    self.gamma0Calc(vMin, T, a[i], b[i])
                disc1 = self.gamma0Calc(vMax, T, a[i], b[i]) -\
                    self.gamma0Calc(v0, T, a[i], b[i])
                # print("v0, v1", v0, v1)
                # print("p0, p1", p0, p1)
                # print("disc", disc0, disc1)
                
                isConv = False
                j = 0
                while(not isConv):
                    # p = p1 - disc1 * (p1 - p0) / (disc1 - disc0)
                    # T = TR - discR * (TR - TL) / (HR - HL)
                    p = 0.5 * (p0 + p1)
                    v0 = SecantMethod(self.pCalc, 1.01 * b[i], vMin,
                                      args=(T, a[i], b[i]), f0=p, maxIters1=100)
                    v1 = SecantMethod(self.pCalc, vMax, 1e+3,
                                      args=(T, a[i], b[i]), f0=p, maxIters1=100)
                    disc = self.gamma0Calc(v1, T, a[i], b[i]) -\
                        self.gamma0Calc(v0, T, a[i], b[i])
                    # print(j, v0, v1)
                    
                    # print(j, p, disc)
                    # print(j, "p", p0, p1)
                    # print(j, "disc", disc0, disc1)
                    
                    # if abs(disc) < epsf and min(abs(p-p0), abs(p-p1)) < epsx:
                    if abs(disc) < epsf and abs(p-p0) < epsx:
                        isConv = True
                    elif j > maxIters:
                        break
                    else:
                        j = j + 1
                        if disc < 0.0:
                            p0 = p
                            # disc0 = disc
                        else:
                            p1 = p
                            # disc1 = disc
                v0m[i] = v0
                v1m[i] = v1
                pm[i] = p
        
        return v0m, v1m, pm
    
    
    def alpha0MixCalc0(self, t, p, z, a=None, b=None):
        """
        calculate Helmholtz potential for mixture by
        temperature and pressure
        """
        
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        v, vm = self.EquilTPCalc(t, p, z, a, b)
        
        alpha = 0.0
        nn = a.shape[0]
        for i in range(nn):
            alpha0 = self.alpha0Calc(vm[i], t, a[i], b[i])
            alpha += z[i] * alpha0
        
        return alpha
    
    
    def alpha0MixCalc1(self, v, t, z, a=None, b=None):
        """
        calculate Helmholtz potential for mixture by
        temperature and pressure
        """
        
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        p, vm = self.EquilVTCalc(v, t, z, a, b)
        
        alpha = 0.0
        nn = a.shape[0]
        for i in range(nn):
            alpha0 = self.alpha0Calc(vm[i], t, a[i], b[i])
            alpha += z[i] * alpha0
        
        return alpha
    
    
    def EquilTPCalc(self, t, p, z, a=None, b=None):
        """
        calculate equilibrium at given temperature, pressure
        and composition
        """
        
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        v = 0.0
        nn = a.shape[0]
        vm = np.zeros((nn,))
        for i in range(nn):
            v0 = self.vCalc(t, p, a[i], b[i])
            vm[i] = v0
            v += z[i] * v0
        
        return v, vm
    
    
    def EquilVTCalc(self, v, t, z, a=None, b=None, epsv=1e-6, maxIters=100):
        """
        calculate equilibrium at given mixture molar volume,
        temperature, and composition
        """
        
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        v0m, v1m, pm = self.PhTrCalc(t, a, b)
        
        p0 = np.inf
        p1 = 0.0
        nn = a.shape[0]
        for i in range(nn):
            if (pm[i] != 0.0):
                if (v < v0m[i] or v > v1m[i]):
                    p = self.pCalc(v, t, a[i], b[i])
                else:
                    p = pm[i]
            else:
                p = self.pCalc(v, t, a[i], b[i])
            
            if (p < p0):
                p0 = p
            if (p > p1):
                p1 = p
        
        disc0 = self.EquilTPCalc(t, p0, z, a, b)[0] - v
        disc1 = self.EquilTPCalc(t, p1, z, a, b)[0] - v
        
        i = 0
        while (i < maxIters):
            p = p1 - disc1 * (p1 - p0) / (disc1 - disc0)
            vv, vm = self.EquilTPCalc(t, p, z, a, b)
            disc = vv - v
            
            if (disc < epsv):
                break
            else:
                i += 1
        
        return p, vm