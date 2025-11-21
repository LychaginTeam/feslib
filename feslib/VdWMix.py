#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Van der Waals mixture class
mixture without chemical reaction
"""


import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
from feslib.EqSolver import EqSolver


GasConstant = 8.314472# * units.Pa * units.m3 / (units.K * units.mol)


class VdWMix:

    def __init__(self):
        self.vc = None
        self.tc = None #component critical temperatures
        self.pc = None #component critical pressures
        self.a = None #component a coefficient of VdW EoS
        self.b = None #component b coefficient of VdW EoS
        self.comp = None
        self.nu = None #degree of freedom
        self.c = None #constant for Helmholtz potential
        self.perm = None #Permeability of porous media
        self.visc = None #Dynamic viscosity


    def ACalc(self, tc, pc):
        """
        calculate component a coefficient of VdW EoS
        """
        return 27. * (tc * GasConstant)**2 / (64. * pc)


    def BCalc(self, tc, pc):
        """
        calculate component b coefficient of VdW EoS
        """

        return tc * GasConstant / (8. * pc)


    def PrepForCalculation(self):
        """
        Prepare constants for calculation
        """

        self.a = self.ACalc(self.tc, self.pc)
        self.b = self.BCalc(self.tc, self.pc)
        self.vc = 3.0 * self.b
        t0 = 293.15
        p0 = 101325
        v0 = GasConstant * t0 / p0
        self.c = GasConstant * np.log(v0) + self.nu * GasConstant * np.log(t0)


    def RCubCalc(self, t, p, ic):
        """
        -discrim
        R>0 -> oner real root
        R<0 -> three real roots
        """

        # A = self.a[ic] * p / (GasConstant * t)**2
        # B = self.b[ic] * p / (GasConstant * t)
        # a0 = -A*B
        # a1 = A
        # a2 = -B-1
        # p1 = (3*a1 - a2**2)/3
        # q1 = (2*a2**3 - 9*a2*a1 + 27*a0)/27

        rt = GasConstant * t
        ap = self.a[ic] * p
        bp = self.b[ic] * p
        p1 = -1.0 / 3.0 -\
            2.0 * bp / (3.0 * rt) -\
            (bp / rt)**2 / 3.0 +\
            ap / (rt)**2
        q1 = -2.0 / 27.0 -\
            2.0 * bp / (9.0 * rt) -\
            2.0 * (bp / rt)**2 / 9.0 -\
            2.0 * (bp / rt)**3 / 27.0 -\
            2.0 * ap / (3.0 * rt**2) * (bp / rt) +\
            ap / (3.0 * rt**2)

        return q1**2 / 4 + p1**3 / 27

    def ZOneCalc(self, t, p, ic):
        """
        root of Z cubic EOS if one root
        """
        # A = self.a[ic] * p / (GasConstant * t)**2
        # B = self.b[ic] * p / (GasConstant * t)
        # a0 = -A*B
        # a1 = A
        # a2 = -B-1
        # p1 = (3*a1 - a2**2)/3
        # q1 = (2*a2**3 - 9*a2*a1 + 27*a0)/27

        rt = GasConstant * t
        ap = self.a[ic] * p
        bp = self.b[ic] * p
        p1 = -1.0 / 3.0 -\
            2.0 * bp / (3.0 * rt) -\
            (bp / rt)**2 / 3.0 +\
            ap / (rt)**2
        q1 = -2.0 / 27.0 -\
            2.0 * bp / (9.0 * rt) -\
            2.0 * (bp / rt)**2 / 9.0 -\
            2.0 * (bp / rt)**3 / 27.0 -\
            2.0 * ap / (3.0 * rt**2) * (bp / rt) +\
            ap / (3.0 * rt**2)
        a2 = -bp / rt - 1.0

        R1 = q1**2 / 4 + p1**3 / 27

        P = np.cbrt(-q1/2 + np.sqrt(R1))
        Q = np.cbrt(-q1/2 - np.sqrt(R1))

        return P + Q - a2/3


    def ZCubCalc(self, t, p, theta0, ic):
        """
        root of Z cubic EOS if 3 real roots
        """
        # A = self.a[ic] * p / (GasConstant * t)**2
        # B = self.b[ic] * p / (GasConstant * t)
        # a0 = -A*B
        # a1 = A
        # a2 = -B-1
        # p1 = (3*a1 - a2**2)/3
        # q1 = (2*a2**3 - 9*a2*a1 + 27*a0)/27

        rt = GasConstant * t
        ap = self.a[ic] * p
        bp = self.b[ic] * p
        p1 = -1.0 / 3.0 -\
            2.0 * bp / (3.0 * rt) -\
            (bp / rt)**2 / 3.0 +\
            ap / (rt)**2
        q1 = -2.0 / 27.0 -\
            2.0 * bp / (9.0 * rt) -\
            2.0 * (bp / rt)**2 / 9.0 -\
            2.0 * (bp / rt)**3 / 27.0 -\
            2.0 * ap / (3.0 * rt**2) * (bp / rt) +\
            ap / (3.0 * rt**2)
        a2 = -bp / rt - 1.0

        m1 = 2*np.sqrt(-p1/3)
        theta = np.arccos(3*q1/(p1*m1))/3

        return m1*np.cos(theta + theta0) - a2/3


    def Z0Calc(self, t, p, ph, ic):
        """
        calculate root of Z cubic EOS
        """

        RCub = self.RCubCalc(t, p, ic)
        if RCub > 0.:
            #one real root
            Z = self.ZOneCalc(t, p, ic)
        else:
            #three real root
            Z1 = self.ZCubCalc(t, p, 0., ic)
            Z2 = self.ZCubCalc(t, p, 2*np.pi/3, ic)
            Z3 = self.ZCubCalc(t, p, 4*np.pi/3, ic)
            # print(Z1, Z2, Z3, t, p)
            if ph == 0:
                #vap
                Z = np.max([Z1, Z2, Z3])
            elif ph == 1:
                #liq
                B = self.b[ic] * p / (GasConstant * t)
                ZM = np.sort([Z1, Z2, Z3])
                for z in ZM:
                    if z - B > 0:
                        Z = z
                        break

        return Z


    def V0Calc(self, t, p, ph, ic):
        return self.Z0Calc(t, p, ph, ic) * GasConstant * t / p


    def VCalc(self, t, p, ic):
        """
        calculate v
        """

        RCub = self.RCubCalc(t, p, ic)
        if RCub > 0.:
            #one real root
            v = self.ZOneCalc(t, p, ic) * GasConstant * t / p
        else:
            #three real root
            Z1 = self.ZCubCalc(t, p, 0., ic)
            Z2 = self.ZCubCalc(t, p, 2*np.pi/3, ic)
            Z3 = self.ZCubCalc(t, p, 4*np.pi/3, ic)

            v1 = Z1 * GasConstant * t / p
            v2 = Z2 * GasConstant * t / p
            v3 = Z3 * GasConstant * t / p

            g1 = self.Gibbs0Calc(v1, t, ic)
            g2 = self.Gibbs0Calc(v2, t, ic)
            g3 = self.Gibbs0Calc(v3, t, ic)

            i = np.array([g1, g2, g3]).argmin()
            v = np.array([v1, v2, v3])[i]

        return v


    def V3Calc(self, t, p, ic):
        """
        return all stability roots of v cubic EOS
        """

        RCub = self.RCubCalc(t, p, ic)
        if RCub > 0.:
            #one real root
            v = self.ZOneCalc(t, p, ic) * GasConstant * t / p

            return np.array([v])
        else:
            #three real root
            Z1 = self.ZCubCalc(t, p, 0., ic)
            Z2 = self.ZCubCalc(t, p, 2*np.pi/3, ic)
            Z3 = self.ZCubCalc(t, p, 4*np.pi/3, ic)

            v1 = Z1 * GasConstant * t / p
            v2 = Z2 * GasConstant * t / p
            v3 = Z3 * GasConstant * t / p

            vM = np.array([v1, v2, v3])

            return np.array([vM.min(), vM.max()])


    def PCalc(self, v, t, ic):
        """
        calculate pressure by EOS for ic component
        """

        return GasConstant * t / (v - self.b[ic]) - self.a[ic] / v**2


    # def P1Calc(self, v, t, ic, pv=None):
    #     """
    #     calculate pressure by EOS for all components
    #     phase transition is respected
    #     """

    #     if pv is None:
    #         pm, v0, v1 = self.PPhTrCalc(t, ic)
    #     else:
    #         pm = pv[0]
    #         v0 = pv[1]
    #         v1 = pv[2]

    #     if v0 < v < v1:
    #         return pm
    #     else:
    #         return self.PCalc(v, t, ic)


    def VMixCalc(self, t, p, pPhTr=None):
        """
        calculate mixture specific volume from temperature and pressure
        """

        if pPhTr is None:
            pPhTr = np.zeros(self.a.shape[0])
            for i in range(self.a.shape[0]):
                pPhTr[i], _, _ = self.PPhTrCalc(t, i)

        vMix = 0.
        for i in range(self.a.shape[0]):
            if p > pPhTr[i]:
                ph = 1
            else:
                ph = 0
            # vComp = self.VCalc(t, p, i)
            vComp = self.V0Calc(t, p, ph, i)
            vMix = vMix + self.comp[i] * vComp

        return vMix


    def Helmholtz0Calc(self, v, t, ic):
        """
        calculate specific Helmholtz potential for ic component
        """

        return -self.a[ic] / v - GasConstant * t * np.log(v - self.b[ic]) -\
            0.5 * self.nu[ic] * GasConstant * t * np.log(t) + t * self.c[ic]


    def IntEnergy0Calc(self, v, t, ic):
        """
        calculate specific internal energy for ic component
        """

        return -self.a[ic] / v + 0.5 * self.nu * GasConstant * t


    def Gibbs0Calc(self, v, t, ic):
        """
        calculate specific Gibbs potential for ic component
        """

        return self.Helmholtz0Calc(v, t, ic) + v * self.PCalc(v, t, ic)


    def Entropy0Calc(self, v, t, ic):
        """
        calculate specific entropy for ic component
        """

        return GasConstant * np.log(v - self.b[ic]) +\
            0.5 * self.nu[ic] * GasConstant * np.log(t) + 0.5 * self.nu[ic] * GasConstant - self.c[ic]


    # def EntropyCalc(self, v, t, ic):
    #     """

    #     """

    #     pm, v0, v1 = self.PPhTrCalc(t, ic)
    #     if v <= v0 or v >= v1:
    #         return self.Entropy0Calc(v, t, ic)
    #     else:
    #         s0 = self.Entropy0Calc(v0, t, ic)
    #         s1 = self.Entropy0Calc(v1, t, ic)
    #         frac = (v1 - v) / (v0 - v1)

    #         return frac * s0 + (1.0 - frac) * s1


    def EntropyPhaseCalc(self, t, p, ph, ic):
        """
        calculate specific entropy for ic component
        """

        v = self.V0Calc(t, p, ph, ic)

        return self.Entropy0Calc(v, t, ic)


    def SingCurvTCalc(self, v, ic):
        """
        Calculate temperature for singular curve for ic component in TP plane
        """

        return 2 * self.a[ic] * (v - self.b[ic])**2 / (GasConstant * v**3)


    def SingCurvPCalc(self, v, ic):
        """
        Calculate pressure for singular curve for ic component in TP plane
        """

        return self.a[ic] * (v - 2 * self.b[ic]) / v**3


    def PPhTrCalc(self, t, ic, epsx=1e-1, epsf=1e-1, maxIters=100):
        """
        calculate pressure and specific volume for phase transition
        at temperature t for ic component
        """

        if t > self.tc[ic]:
            v0m = 0.0
            v1m = 0.0
            pm = 0.0
        else:
            vMin = opt.minimize(lambda x: self.PCalc(x, t, ic),\
                0.5 * (self.b[ic] + self.vc[ic]),
                bounds=((1.01 * self.b[ic], self.vc[ic]),)).x[0]
            vMax = opt.minimize(lambda x, : -self.PCalc(x, t, ic),
                2 * self.vc[ic], bounds=((self.vc[ic], 1e+3),)).x[0]
            # print("vMin, vMax", vMin, vMax)

            p0 = self.PCalc(vMin, t, ic)
            # print("p0", p0)
            if p0 < 0.:
                p0 = 0.
                v1 = 1e+6
            else:
                v1 = EqSolver.SecantMethod(
                    lambda x: self.PCalc(x, t, ic) - p0, vMax, 1e+3)
            p1 = self.PCalc(vMax, t, ic)
            v0 = EqSolver.SecantMethod(
                lambda x: self.PCalc(x, t, ic) - p1, 1.01 * self.b[ic], vMin)

            # disc0 = self.Gibbs0Calc(v1, t, a[i], b[i]) -\
            #     self.Gibbs0Calc(vMin, t, a[i], b[i])
            # disc1 = self.Gibbs0Calc(vMax, t, a[i], b[i]) -\
            #     self.Gibbs0Calc(v0, t, a[i], b[i])
            # print("v0, v1", v0, v1)
            # print("p0, p1", p0, p1)
            # print("disc", disc0, disc1)

            isConv = False
            j = 0
            while(not isConv):
                # p = p1 - disc1 * (p1 - p0) / (disc1 - disc0)
                # t = TR - discR * (TR - TL) / (HR - HL)
                p = 0.5 * (p0 + p1)
                v0 = EqSolver.SecantMethod(
                    lambda x: self.PCalc(x, t, ic) - p,\
                    1.01 * self.b[ic], vMin, maxIters1=100)
                v1 = EqSolver.SecantMethod(
                    lambda x: self.PCalc(x, t, ic) - p,\
                    vMax, 1e+3, maxIters1=100)
                disc = self.Gibbs0Calc(v1, t, ic) -\
                    self.Gibbs0Calc(v0, t, ic)
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
            v0m = v0
            v1m = v1
            pm = p

        return pm, v0m, v1m


    def TPhTrCalc(self, p, ic, epsx=1e-1, epsf=1e-1, maxIters=100):
        """
        calculate temperature and specific volume for phase transition
        at pressure p for ic component
        """

        if p > self.pc[ic]:
            tm = 0.0
            v0m = 0.0
            v1m = 0.0
        else:
            t0 = 0.0
            t1 = self.tc[ic]
            disc0 = -1.0#>0
            disc1 = 1.0#<0

            i = 0
            while True:
                if disc0 > 0.0 and disc1 < 0.0:
                    tm = (t1 - t0) * (0.0 - disc0) / (disc1 - disc0) + t0
                else:
                    tm = 0.5 * (t0 + t1)

                if self.RCubCalc(tm, p, ic) > 0.0:
                    # t0 = tm
                    vm = self.ZOneCalc(tm, p, ic) * GasConstant * tm / p
                    if vm < self.vc[ic]:
                        t0 = tm
                    else:
                        t1 = tm
                else:
                    v0m = self.V0Calc(tm, p, 0, ic)
                    v1m = self.V0Calc(tm, p, 1, ic)
                    disc = self.Gibbs0Calc(v0m, tm, ic) - self.Gibbs0Calc(v1m, tm, ic)

                    if abs(disc) < epsf and min(abs(tm - t0), abs(t1 - t0)) < epsx:
                        break
                    elif disc > 0.0:
                        t0 = tm
                        disc0 = disc
                    elif disc < 0.0:
                        t1 = tm
                        disc1 = disc

                    if i >= maxIters:
                        break
                    else:
                        i += 1

        return tm, v1m, v0m


    def HelmholtzMixTPCalc(self, t, p, comp):
        """
        calculate Helmholtz potential for mixture by
        temperature and pressure
        """

        v, vm = self.EquilMixTPCalc(t, p, comp)

        alpha = 0.0
        nn = self.a.shape[0]
        for i in range(nn):
            alpha0 = self.Helmholtz0Calc(vm[i], t, i)
            alpha += comp[i] * alpha0

        return alpha


    def HelmholtzMixVTCalc(self, v, t, comp):
        """
        calculate Helmholtz potential for mixture by
        temperature and pressure
        """

        p, vm = self.EquilMixVTCalc(v, t, comp)

        alpha = 0.0
        nn = self.a.shape[0]
        for i in range(nn):
            alpha0 = self.Helmholtz0Calc(vm[i], t, i)
            alpha += comp[i] * alpha0

        return alpha


    def EquilMixTPCalc(self, t, p, comp):
        """
        calculate equilibrium of mixture at given
        temperature, pressure and composition
        """

        v = 0.0
        nn = self.a.shape[0]
        vm = np.zeros((nn,))
        for i in range(nn):
            v0 = self.VCalc(t, p, i)
            vm[i] = v0
            v += comp[i] * v0

        return v, vm


    def EquilMixVTCalc(self, v, t, comp, epsv=1e-6, maxIters=100):
        """
        calculate equilibrium of mixture at given
        specific volume, temperature, and composition
        """

        p0 = np.inf
        p1 = 0.0
        nn = self.a.shape[0]
        for i in range(nn):
            pm, v0m, v1m = self.PPhTrCalc(t, i)
            if (pm != 0.0):
                if (v < v0m or v > v1m):
                    p = self.PCalc(v, t, i)
                else:
                    p = pm
            else:
                p = self.PCalc(v, t, i)

            if (p < p0):
                p0 = p
            if (p > p1):
                p1 = p

        disc0 = self.EquilMixTPCalc(t, p0, comp)[0] - v
        disc1 = self.EquilMixTPCalc(t, p1, comp)[0] - v

        i = 0
        while (i < maxIters):
            p = p1 - disc1 * (p1 - p0) / (disc1 - disc0)
            vv, vm = self.EquilMixTPCalc(t, p, comp)
            disc = vv - v

            if (disc < epsv):
                break
            else:
                i += 1

        return p, vm


    def EquilTPCalc(self, t, p, ic):
        """
        Calculate equilibrium of ic component at given
        temperature and pressure
        """

        vm = self.V3Calc(t, p, ic)
        if vm.shape[0] == 2:
            g0 = self.Gibbs0Calc(vm[0], t, ic)
            g1 = self.Gibbs0Calc(vm[1], t, ic)
            if g0 < g1:
                v = vm[0]
                fracV = 0.0
            else:
                v = vm[1]
                fracV = 1.0
        else:#if vm.shape[0] == 1:
            v = vm[0]
            if v > self.vc[ic]:
                fracV = 1.0
            else:
                fracV = 0.0

        return v, fracV


    def EquilPVFCalc(self, p, fracV, ic):
        """
        Calculate equilibrium of ic component at given
        pressure and vapour fraction
        """

        tt, vt0, vt1 = self.TPhTrCalc(p, ic)
        if fracV == 0.0:
            return tt, vt0
        elif fracV == 1.0:
            return tt, vt1
        else:
            return tt, fracV * vt1 + (1.0 - fracV) * vt0


    def EquilPSCalc(self, p, s, ic, epsv=1e-6, maxIters=100):
        """
        Calculate equilibrium of ic component at given
        pressure and entropy
        """

        if p < self.pc[ic]:
            tt, vt0, vt1 = self.TPhTrCalc(p, ic)
            s0 = self.Entropy0Calc(vt0, tt, ic)
            s1 = self.Entropy0Calc(vt1, tt, ic)

            if s < s0:
                fracV = 0.0

                t0 = 0.0
                t1 = tt
                disc0 = None
                disc1 = None
                while (True):
                    if disc0 is not None and disc1 is not None:
                        t = EqSolver.SecantMethod(
                            lambda x: self.EntropyPhaseCalc(x, p, 1, ic) / s - 1.0, t0, t1)
                        break
                    else:
                        t = 0.5 * (t0 + t1)
                        disc = self.EntropyPhaseCalc(t, p, 1, ic) - s
                        if disc < 0.0:
                            t0 = t
                            disc0 = disc
                        elif disc > 0.0:
                            t1 = t
                            disc1 = disc

                # t = opt.root(lambda x: self.EntropyPhaseCalc(x, p, 1, ic) / s - 1.0, 0.5 * (0.0 + tt)).x[0]
                vV = None
                vL = self.V0Calc(t, p, 1, ic)
            elif s > s1:
                fracV = 1.0
                # t0 = tt
                # t1 = self.tc[ic]
                # disc0 = s1 - s
                # disc1 = None
                # while (True):
                #     if disc0 is not None and disc1 is not None:
                #         t = EqSolver.SecantMethod(
                #             lambda x: self.EntropyPhaseCalc(x, p, 0, ic) / s - 1.0, t0, t1)
                #         break
                #     else:
                #         t = 0.5 * (t0 + t1)
                #         disc = self.EntropyPhaseCalc(t, p, 0, ic) - s
                #         if disc < 0.0:
                #             t0 = t
                #             disc0 = disc
                #         elif disc > 0.0:
                #             t1 = t
                #             disc1 = disc

                t = opt.root(lambda x: self.EntropyPhaseCalc(x, p, 0, ic) / s - 1.0, 0.5 * (tt + self.tc[ic])).x[0]
                vV = self.V0Calc(t, p, 0, ic)
                vL = None
            else:
                fracL = (s1 - s) / (s1 - s0)
                fracV = 1.0 - fracL
                t = tt
                vV = vt1
                vL = vt0
        else:
            fracV = 0.0
            t = opt.root(lambda x: self.EntropyPhaseCalc(x, p, 0, ic) / s - 1.0, 1.1 * self.tc[ic]).x[0]
            vV = self.V0Calc(t, p, 0, ic)
            vL = None

        return t, vV, vL, fracV


    # def PhTrSolve(self, t, p01, p02, eps1=1e-3, ic=0):
    #     """
    #     решаю систему для фазового перехода перебором
    #     при заданных t и p нахожу значения v в интервалах (b,c) и (c,vcr)
    #     далее проверяю выполнение 2го уравнения
    #     Если не выполнено, то меняю p
    #     """

    #     scs = False

    #     # v11, v12 = V2Calc(t, p01, ph2)
    #     vM = self.V3Calc(t, p01, self.a[ic], self.b[ic])
    #     if vM.shape[0] == 1:
    #         disc1 = -np.inf
    #     else:
    #         disc1 = self.Gibbs0Calc(vM[1], t, self.a[ic], self.b[ic]) -\
    #             self.Gibbs0Calc(vM[0], t, self.a[ic], self.b[ic])
    #         if np.abs(disc1) < eps1:
    #             return vM[0], vM[1], True

    #     # v21, v22 = V2Calc(t, p02, ph2)
    #     vM = self.V3Calc(t, p02, self.a[ic], self.b[ic])
    #     if vM.shape[0] == 1:
    #         disc2 = np.inf
    #     else:
    #         disc2 = self.Gibbs0Calc(vM[1], t, self.a[ic], self.b[ic]) -\
    #             self.Gibbs0Calc(vM[0], t, self.a[ic], self.b[ic])
    #         if np.abs(disc2) < eps1:
    #             return vM[0], vM[1], True

    #     n1 = 1e+2
    #     i = 0
    #     while True:
    #         if i >= n1:
    #             break

    #         p0 = 0.5 * (p01 + p02)
    #         # v1, v2 = V2Calc(t, p0, ph2)
    #         vM = self.V3Calc(t, p0, self.a[ic], self.b[ic])
    #         if vM.shape[0] == 1:
    #             if vM[0] > self.vc:
    #                 disc = -np.inf
    #             else:
    #                 disc = np.inf
    #         if vM.shape[0] > 1:
    #             disc = self.Gibbs0Calc(vM[1], t, self.a[ic], self.b[ic]) -\
    #                 self.Gibbs0Calc(vM[0], t, self.a[ic], self.b[ic])

    #         if np.abs(disc) < eps1:
    #             scs = True
    #             break
    #         else:
    #             i = i + 1
    #             if np.sign(disc) == np.sign(disc1):
    #                 p01 = p0
    #                 disc1 = disc
    #             else:
    #                 p02 = p0

    #     return vM[0], vM[1], scs



    # def QiCalc(self, rho, t, ic):
    #     """
    #     The integrand of the Leibenson function Q
    #     for a given density rho
    #     """

    #     return -2.0 * self.a[ic] * rho**3 / 3.0 +\
    #         GasConstant * t * (np.log(1.0 - self.b[ic] * rho) -\
    #             1.0 / (self.b[ic] * rho - 1.0)) / (self.b[ic]**2)


    # def QCalc(self, rho, t, ic):
    #     """
    #     The Leibenson function Q for a given density rho
    #     """

    #     return integrate.quad(lambda x: self.QiCalc(x, t, ic), 0, rho)[0]


    # def Qi1Calc(self, rho, t, rho0, rho1, ic):
    #     """
    #     Считает функцию Лейбензона Q при заданной плотности rho
    #     Учитывает фазовый переход
    #     """

    #     if rho1 < rho < rho0:
    #         return 0.0
    #     else:
    #         return self.QiCalc(rho, t, ic)


    # def Q1Calc(self, rho, t, ic):
    #     """
    #     Функция Лейбензона Q при заданной плотности rho
    #     """

    #     pm, v0, v1 = self.PPhTrCalc(t, ic)
    #     rho0 = 1.0 / v0
    #     rho1 = 1.0 / v1

    #     if rho <= rho1:
    #         return integrate.quad(lambda x: self.QiCalc(x, t, ic), 0, rho)[0]
    #     elif rho1 < rho <= rho0:
    #         return integrate.quad(lambda x: self.QiCalc(x, t, ic), 0, rho1)[0]
    #     else:
    #         return integrate.quad(lambda x: self.QiCalc(x, t, ic), 0, rho1)[0] +\
    #             integrate.quad(lambda x: self.QiCalc(x, t, ic), rho0, rho)[0]

    #     # return integrate.quad(lambda x: self.Qi1Calc(x, t, a, b, 1.0 / v0, 1.0 / v1), 0, rho)[0]


    def DQIsoEntropyCalc(self, p, s, ic):
        """
        The integrand of the Leibenson function Q of isoentropy filtration
        for a given pressure and entropy
        """

        if p != 0:
            _, vV, vL, fracV = self.EquilPSCalc(p, s, ic)

            if vV is not None and vL is not None:
                v = fracV * vV + (1.0 - fracV) * vL
            elif vV is not None:
                v = vV
            else:
                v = vL

            return self.perm / self.visc / v
        else:
            return 0.0


    def QIsoEntropyCalc(self, p, s, ic, p0=0.0):
        """
        The Leibenson function Q of isoentropy filtration
        for a given pressure and entropy
        """

        # return integrate.quad(lambda x: self.DQIsoEntropyCalc(x, s, ic), p0, p)[0]

        nn = 10
        dqm = np.zeros(nn)
        pm = np.linspace(p0, p, nn)
        for i in range(1, nn):
            dqm[i] = self.DQIsoEntropyCalc(pm[i], s, ic)

        return integrate.simpson(dqm, x=pm)


    def QmIsoEntropyCalc(self, pm, s, ic):
        """
        The array of Leibenson function Q of isoentropy filtration
        for a given pressures and entropy
        """

        qm = np.zeros(pm.shape[0])
        qm[0] = self.QIsoEntropyCalc(pm[0], s, ic)
        for i in range(1, pm.shape[0]):
            # print(pm[i])
            qm[i] = qm[i-1] + self.QIsoEntropyCalc(pm[i], s, ic, p0=pm[i-1])

        return qm

    #ver2 QIsoEntropyCalc

    # def PsPhTrCalc(self, s, p0, p1, ic, epsS=1e-3, maxIters=100):
    #     """
    #     Считает давление для заданной энтропии
    #     при котором начинается фазовый переход
    #     """

    #     sc = self.Entropy0Calc(self.vc[ic], self.tc[ic], ic)

    #     # s < sc then disc0 < 0.0, disc1 > 0.0 else disc0 > 0.0, disc1 < 0.0
    #     disc0 = None
    #     disc1 = None

    #     if s > sc:
    #         t, v0, v1 = self.TPhTrCalc(p1, ic)
    #         s1 = self.Entropy0Calc(v1, t, ic)
    #         if s < s1:
    #             return p1
    #         else:
    #             disc1 = s - s1

    #         i = 0
    #         while True:
    #             if disc0 is not None:
    #                 p = p0 + (p1 - p0) * (0.0 - disc0) / (disc1 - disc0)
    #             else:
    #                 p = 0.5 * (p0 + p1)
    #             t, v0, v1 = self.TPhTrCalc(p, ic)
    #             s1 = self.Entropy0Calc(v1, t, ic)
    #             disc = s - s1

    #             if abs(disc) < epsS or i >= maxIters:
    #                 return p

    #             if disc < 0.0:
    #                 p0 = p
    #                 disc0 = disc
    #             else:
    #                 p1 = p
    #                 disc1 = disc

    #             i += 1
    #     else:
    #         t, v0, v1 = self.TPhTrCalc(p1, ic)
    #         s0 = self.Entropy0Calc(v0, t, ic)
    #         if s > s0:
    #             return p1
    #         else:
    #             disc1 = s - s0

    #         i = 0
    #         while True:
    #             if disc0 is not None:
    #                 p = p0 + (p1 - p0) * (0.0 - disc0) / (disc1 - disc0)
    #             else:
    #                 p = 0.5 * (p0 + p1)
    #             t, v0, v1 = self.TPhTrCalc(p, ic)
    #             s0 = self.Entropy0Calc(v0, t, ic)
    #             disc = s - s0

    #             if abs(disc) < epsS or i >= maxIters:
    #                 return p

    #             if disc > 0.0:
    #                 p0 = p
    #                 disc0 = disc
    #             else:
    #                 p1 = p
    #                 disc1 = disc

    #             i += 1

    # def DQIsoEntropyCalc(self, p, s, ic, ps=0):
    #     """
    #     Подинтегральная функция функции Лейбензона Q
    #     """

    #     if p < ps:
    #         tt, vt0, vt1 = self.TPhTrCalc(p, ic)
    #         s0 = self.Entropy0Calc(vt0, tt, ic)
    #         s1 = self.Entropy0Calc(vt1, tt, ic)
    #         frac = (s1 - s) / (s1 - s0)
    #         v = frac * vt0 + (1.0 - frac) * vt1
    #     else:


    #     if p != 0:
    #         _, v = self.EquilPSCalc(p, s, ic)

    #         return 1e-12 * 1.0 / v
    #     else:
    #         return 0.0
