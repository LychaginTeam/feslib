#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from time import time


#%%

#universal gas const
R = 8.314472


#%%

class MSLVMix:
    """
    comp -- composition
    vcPure, TcPure, pcPure -- critical values of components
    vc, Tc, pc -- critical values of mixture
    aPure, acPure, bPure, cPure -- EOS parameters of components
    a, b, c -- EOS parameters of mixture
    ar, br, cr -- dimentionless EOS parameters of mixture
    wPure -- acentric factor of components
    kPure -- binary iteraction parameters
    """

    def __init__(self):
        self.comp = None

        self.vcPure = None
        self.TcPure = None
        self.pcPure = None

        self.vc = None
        self.Tc = None
        self.pc = None

        # self.aPure = None
        self.acPure = None
        self.bPure = None
        self.dPure = None
        self.cPure = None
        self.wPure = None
        self.mPure = None
        self.k = None

        # self.a = None
        self.b = None
        self.d = None
        self.c = None

        self.br = None
        self.dr = None
        self.cr = None
        self.Zr = None

    def bCalc(self):
        return np.dot(self.comp, self.bPure)

    def dCalc(self):
        return np.dot(self.comp, self.dPure)

    def cCalc(self):
        return np.dot(self.comp, self.cPure)

    def aPure(self, T):
        return self.acPure*(1 + self.mPure -\
                            self.mPure*np.sqrt(T/self.TcPure))**2

    def a(self, T):
        M = np.sqrt(np.outer(self.aPure(T), self.aPure(T)))*(1-self.k)
        return np.dot(np.dot(self.comp, M), self.comp)

    def ar(self, Tr):
        T = Tr*self.Tc
        return self.q(T)

    def f(self, v):
        """
        1st term of EoS for mixture
        """
        return R * (v-self.d) / ((v-self.b) * (v-self.c))

    def fr(self, vr):
        """
        1st term of reduced EoS for mixture
        """
        v = vr * self.vc
        return self.f(v) * self.Tc / self.pc

    def g(self, v):
        """
        2nd term of EoS for mixture
        """
        return -1. / (v**2 + 2*self.b*v - self.b**2)

    def gr(self, vr):
        """
        2nd term of reduced EoS for mixture
        """
        v = vr * self.vc
        return self.g(v) / self.pc

    def df(self, v):
        """
        derivative of 1st term of EoS for mixture
        """
        term1 = R*(self.b*self.c - self.b*self.d -\
            self.c*self.d - v**2 + 2*self.d*v)
        term2 = (v-self.b)**2 * (v-self.c)**2
        return term1/term2

    def dfr(self, vr):
        """
        dimensionless derivative of 1st term of EoS for mixture
        """
        v = vr * self.vc
        return self.df(v) * self.Tc * self.vc / self.pc

    def dg(self, v):
        """
        derivative of 2nd term of EoS for mixture
        """
        return 2*(v + self.b) / ((v**2 + 2*self.b*v - self.b**2)**2)

    def dgr(self, vr):
        """
        dimensionless derivative of 2nd term of EoS for mixture
        """
        v = vr * self.vc
        return self.dg(v) * self.vc / self.pc

    def d2f(self, v):
        """
        2 derivative of 1st term of EoS for mixture
        """
        term1 = 2*R*(self.b**2 * self.c - self.b**2 * self.d +\
            self.b * self.c**2 - self.b*self.c*self.d -\
            3*self.b*self.c*v + 3*self.b*self.d*v -\
            self.c**2 * self.d + 3*self.c*self.d*v -\
            3*self.d * v**2 + v**3)
        term2 = (v-self.b)**3 * (v-self.c)**3
        return term1/term2

    def d2g(self, v):
        """
        2 derivative of 2nd term of EoS for mixture
        """
        term1 = 10 * self.b**2 + 12*self.b*v + 6 * v**2
        term2 = (self.b**2 - 2*self.b*v - v**2)**3
        return term1/term2

    def F(self, v):
        """
        integral of 1st term of EoS for mixture
        """
        term1 = np.log(np.abs(v-self.b))
        term2 = np.log(np.abs(v-self.c))
        term3 = (self.b-self.c)
        return R*((self.b-self.d) * term1 + (self.d-self.c) * term2) / term3

    def Fr(self, vr):
        """
        integral of 1st term of EoS for mixture
        """
        v = vr * self.vc
        return self.F(v) * self.Tc / (self.pc * self.vc)

    def G(self, v):
        """
        integral of 2nd term of EoS for mixture
        """
        term1 = np.log(np.abs(np.sqrt(2)*self.b + self.b + v)) -\
            np.log(np.abs(np.sqrt(2)*self.b - self.b - v))
        term2 = (2*np.sqrt(2)*self.b)
        return term1 / term2

    def Gr(self, vr):
        """
        integral of 2nd term of EoS for mixture
        """
        v = vr * self.vc
        return self.G(v) / (self.pc * self.vc)

    def EOS1(self, v, T):
        """
        EoS for Mix
        """
        return T*self.f(v) + self.a(T)*self.g(v)

    def EOS1r(self, vr, Tr):
        """
        EoS for Mix
        """
        v = vr * self.vc
        T = Tr * self.Tc
        return self.EOS1(v,T) / self.pc

    def phi(self, v, T):
        """
        потенциал Масье-Планка без функции K(T)
        """
        return (self.F(v) + self.a(T)*self.G(v)/T) / R

    def phir(self, vr, Tr):
        """
        потенциал Масье-Планка без функции K(T)
        """
        v = vr*self.vc
        T = Tr*self.Tc
        return self.phi(v,T)*self.Tc / (self.pc*self.vc)

    def phi_v(self, v, T):
        """
        частная производная потенциала Масье-Планка по v
        """
        return (self.f(v) + self.a(T)*self.g(v)/T) / R

    def phir_v(self, vr, Tr):
        """
        частная производная потенциала Масье-Планка по v
        """
        return self.EOS1r(vr,Tr) / (R*Tr)

    def phi0_vv(self, v, T):
        """
        вторая частная производная потенциала Масье-Планка по v
        """
        return (self.df(v) + self.a(T)*self.dg(v)/T) / R

    def phir_vv(self, vr, Tr):
        """
        вторая частная производная потенциала Масье-Планка по v
        """
        v = vr * self.vc
        T = Tr * self.Tc
        return self.phi0_vv(v,T) * self.vc / self.Tc

    def PhTrEq1(self, v1, v2, T):
        """
        Первое уравнение фазового перехода
        сохраняется давление
        """
        return self.phi_v(v2,T) - self.phi_v(v1,T)

    def PhTrEq1r(self, v1r, v2r, Tr):
        """
        Первое уравнение фазового перехода
        сохраняется давление
        """
        v1 = v1r * self.vc
        v2 = v2r * self.vc
        T = Tr * self.Tc
        return self.PhTrEq1(v1,v2,T) / self.pc

    def PhTrEq2(self, v1, v2, T):
        """
        второе уравнение фазового перехода
        сохраняется химического потенциала Гиббса
        """
        return self.phi(v2,T) - self.phi(v1,T) -\
            v2*self.phi_v(v2,T) + v1*self.phi_v(v1,T)

    def PhTrEq2r(self, v1r, v2r, Tr):
        """
        второе уравнение фазового перехода
        сохраняется химического потенциала Гиббса
        """
        v1 = v1r * self.vc
        v2 = v2r * self.vc
        T = Tr * self.Tc
        return self.PhTrEq2(v1,v2,T) * self.Tc / (self.pc * self.vc)

    def PhTrV2(self, vr):
        """
        Апроксимация графика фазового прехода в плоскости (v1,v2)
        аппроксимация гиперболой
        """
        return (1.-self.br)*(1.-self.br)/(vr-self.br) + self.br

    def PhTrVLr(self, T0, T1, nn=30):
        TMVL = np.linspace(T0, T1, nn)
        v1MVL = np.zeros(nn)
        v2MVL = np.zeros(nn)
        scsMVL = np.zeros(nn, dtype=bool)

        tt = time()
        v11 = self.br+1e-3
        for i in range(nn):
            print(i)
            # v12 = FindMin(self, 1, TMVL[i])
            # v21 = FindMax(self, 1, TMVL[i])
            v12 = minimize(self.EOS1r, 0.5, args=(TMVL[i],), bounds=((self.cr+1e-8,1.),)).x[0]
            v21 = minimize(lambda v: -self.EOS1r(v, TMVL[i]), 1.1, bounds=((1.,np.inf),)).x[0]
            p01 = self.EOS1r(v12,TMVL[i])
            if p01 < 0:
                p01 = 1e-10
            v22 = R*TMVL[i]/p01 + self.b
            p02 = self.EOS1r(v21,TMVL[i])
            v1, v2, p, scs = Bisection2(self.EOS1r, self.PhTrEq2r,\
                v11, v12, v21, v22, p01, p02,\
                args=(TMVL[i],), eps1=1e-9, eps2=1e-6)
            v1MVL[i] = v1
            v2MVL[i] = v2
            scsMVL[i] = 1*scs
        print(time()-tt)

        plt.figure()
        plt.plot(scsMVL)
        plt.grid()
        plt.show()

        pMVL = np.vectorize(self.EOS1r)(v1MVL, TMVL)

        return TMVL, pMVL, v1MVL, v2MVL

    def PhTrSLr(self, T0, T1, nn=30):
        TMSL = np.linspace(T0, T1, nn)
        pMSL = np.zeros(nn)
        v1MSL = np.zeros(nn)
        v2MSL = np.zeros(nn)
        scsMSL = np.zeros(nn, dtype=bool)

        tt = time()
        p02 = 200.
        v11 = self.br+1e-8
        v12 = self.dr-1e-8
        v21 = self.cr+1e-8
        for i in range(nn):
            print(i)
            # v22 = FindMin(self, 1, TMSL[i])
            v22 = minimize(self.EOS1r, 0.5, args=(TMSL[i],), bounds=((v21,1.),)).x[0]
            p01 = self.EOS1r(v22, TMSL[i])
            v1, v2, p, scs = Bisection2(self.EOS1r, self.PhTrEq2r,\
                v11, v12, v21, v22, p01, p02,\
                args=(TMSL[i],), eps1=1e-9, eps2=1e-5)
            v1MSL[i] = v1
            v2MSL[i] = v2
            scsMSL[i] = scs
            pMSL[i] = self.EOS1r(v1,TMSL[i])
        print(time()-tt)

        plt.figure()
        plt.plot(scsMSL)
        plt.grid()
        plt.show()

        return TMSL, pMSL, v1MSL, v2MSL

    def TripleFind(self, T1, p1, T2, p2):
        """
        поиск тройной точки

        для обеих крывых одной температуре должны соответствовать
        точки с одинаковым номером
        """
        nn = T1.shape[0]-1
        for i in range(nn):
            x11 = T1[i]
            x12 = T1[i+1]
            y11 = p1[i]
            y12 = p1[i+1]
            x21 = T2[i]
            x22 = T2[i+1]
            y21 = p2[i]
            y22 = p2[i+1]
            y = -((x12*y11*y21 - x22*y11*y21 - x11*y12*y21 + x22*y12*y21 -\
                   x12*y11*y22 + x21*y11*y22 + x11*y12*y22 - x21*y12*y22)/\
                  (-x21*y11 + x22*y11 + x21*y12 - x22*y12 +\
                   x11*y21 - x12*y21 - x11*y22 + x12*y22))

            if y21 <= y <= y22:
                x = -((-x12*x21*y11 + x12*x22*y11 + x11*x21*y12 - x11*x22*y12 +\
                       x11*x22*y21 - x12*x22*y21 - x11*x21*y22 + x12*x21*y22)/\
                      (x21*y11 - x22*y11 - x21*y12 + x22*y12 -\
                       x11*y21 + x12*y21 + x11*y22 - x12*y22))
                break
        return x, y, i

#end class


def Bisection(f, x1, x2, args=(), eps=1e-8, y0=0.):
    """
    метод бисекции
    """
    MaxIters = 100
    scs = True
    t1 = x1
    t2 = x2

    y1 = f(t1, *args) - y0
    y2 = f(t2, *args) - y0

    if np.abs(y1) < eps:
        flag = False
        t3 = t1
    elif np.abs(y2) < eps:
        flag = False
        t3 = t2
    elif np.sign(y1) * np.sign(y2) > 0.:
        print('Bisection: no root in range')
        flag = False
        scs = False
        t3 = (t2 + t1) * 0.5
    else:
        flag = True

    i = 0
    while flag:
        if i > MaxIters:
            print('Bisection: break')
            scs = False
            break
        i = i + 1

        t3 = (t2 + t1) * 0.5
        y3 = f(t3, *args) - y0
        if np.abs(y3) < eps:
            flag = False
        if np.sign(y1) * np.sign(y3) < 0:
            t2 = t3
            y2 = y3
        else:
            t1 = t3
            y1 = y3
    return t3, scs

def Bisection2(f1, f2, x11, x12, x21, x22, y1, y2, args=(), eps1=1e-8, eps2=1e-5):
    """
    метод бисекции
    """
    MaxIters = 150
    scs = True
    u1 = y1
    u2 = y2

    t11, scs1 = Bisection(f1, x11, x12, args=args, eps=eps1, y0=u1)
    t12, scs1 = Bisection(f1, x21, x22, args=args, eps=eps1, y0=u1)

    g1 = f2(t11, t12, *args)

    t21, scs1 = Bisection(f1, x11, x12, args=args, eps=eps1, y0=u2)
    t22, scs1 = Bisection(f1, x21, x22, args=args, eps=eps1, y0=u2)

    g2 = f2(t21, t22, *args)

    if np.abs(g1) < eps2:
        flag = False
        t1 = t11
        t2 = t12
        u3 = u1
    elif np.abs(g2) < eps2:
        flag = False
        t1 = t21
        t2 = t22
        u3 = u2
    elif np.sign(g1) * np.sign(g2) > 0.:
        print('Bisection2: no root in range')
        flag = False
        scs = False
        u3 = (u2 + u1) * 0.5
        t1, scs1 = Bisection(f1, x11, x12, args=args, eps=eps1, y0=u3)
        t2, scs1 = Bisection(f1, x21, x22, args=args, eps=eps1, y0=u3)
    else:
        flag = True

    i = 0
    while flag:
        if i > MaxIters:
            print('Bisection2: break')
            # scs = False
            break
        i = i + 1

        u3 = (u2 + u1) * 0.5
        t1, scs1 = Bisection(f1, x11, x12, args=args, eps=eps1, y0=u3)
        t2, scs1 = Bisection(f1, x21, x22, args=args, eps=eps1, y0=u3)
        g3 = f2(t1, t2, *args)

        if np.abs(g3) < eps2:
            flag = False
        if np.sign(g1) * np.sign(g3) < 0:
            u2 = u3
            g2 = g3
        else:
            u1 = u3
            g1 = g3

    # print(i)
    return t1, t2, u3, scs
