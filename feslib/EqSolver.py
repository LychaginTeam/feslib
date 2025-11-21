#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equation solver class
"""

import numpy as np
from copy import copy



class EqSolver:
    def __init__(self):
        None


    def CubSolveTrig(a):
        P = (3 * a[1] - a[2]**2) / 9.0
        Q = (2 * a[2]**3 - 9 * a[2] * a[1] + 27 * a[0]) / 54.0
        if P == 0:
            if Q == 0:
                np.zeros(1)
            else:
                np.array([np.sqrt(2) * np.cbrt(-Q)])
        else:
            R = Q**2 + P**3
            if R == 0:
                x0 = 2 * Q / P - a[2] / 3.0
                x1 = -Q / P - a[2] / 3.0
                return np.array([x0, x1])
            elif R > 0:
                pp = np.cbrt(-Q + np.sqrt(R))
                qq = np.cbrt(-Q - np.sqrt(R))
                return np.array([pp + qq - a[2] / 3.0])
            else:
                theta = np.arccos(Q / (P * np.sqrt(-P))) / 3.0
                x0 = 2 * np.sqrt(-P) * np.cos(theta) - a[2] / 3.0
                x1 = 2 * np.sqrt(-P) * np.cos(theta + 2 * np.pi / 3.0) - a[2] / 3.0
                x2 = 2 * np.sqrt(-P) * np.cos(theta + 4 * np.pi / 3.0) - a[2] / 3.0
                return np.array([x0, x1, x2])


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
