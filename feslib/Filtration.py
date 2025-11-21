#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isoentropy filtration in disk
constant boundary condition
"""

import numpy as np


GasConstant = 8.314472# * units.Pa * units.m3 / (units.K * units.mol)


#%%


class Filtration:
    
    def __init__(self):
        self.uc = None# boundary condition for Dirichlet problem
        self.c = None# array of source/sink intensity
        self.a = None# array of source/sink coortinates
    
    def ParamCircle(self, r, phi):
        return np.array([r * np.cos(phi), r * np.sin(phi)])

    def G1(self, x, eta, r0=1.0):
        """
        Green function in disk of Dirichlet problem
        """
        
        for j in range(self.c.shape[0]):
            if(np.linalg.norm(eta) > 1e-10):
                eta1 = r0**2 / np.dot(eta, eta) * eta
                g1 = 0.5 * np.log(np.dot(eta, eta) * np.dot(x - eta1, x - eta1) / np.dot(x - eta, x - eta)) - np.log(r0)
            else:
                g1 = np.log(r0) - 0.5 * np.log(np.dot(x, x))
        g1 /= 2.0 * np.pi
        
        return g1

    def DG1(self, x, eta, r0=1.0):
        """
        Green function derivative in circle of Dirichlet problem at circle
        """
        
        return (np.dot(x, x) - r0**2) / np.dot(x - eta, x - eta) / (2.0 * np.pi * r0)

    def U0DfCalc(self, x, eta, r0=1.0):
        """
        Integrand for the Dirichlet problem
        """
        
        return self.uc * self.DG1(x, eta, r0=r0)


    def U0DCalc(self, x, r0=1.0):
        """
        Solve of Dirichlet problem in disk
        """
        
        if (np.linalg.norm(x) / r0 < 1.0 - 1e-4):
            t1 = 0
            for j in range(self.c.shape[0]):
                t1 += -self.c[j] * self.G1(x, self.a[j], r0=r0)
        
            # t2 = -integrate.quad(lambda t: U0DfCalc(x, ParamCircle(r0, t), r0=r0), 0.0, 2.0 * np.pi)[0] * r0
            t2 = self.uc# if const
        
            return t1 + t2
        else:
            return self.uc