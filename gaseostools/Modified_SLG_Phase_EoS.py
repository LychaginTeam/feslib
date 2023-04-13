# -*- coding: utf-8 -*-
"""
считаю границы фаз для метана
уравнение Modified SLG Phase EoS
"""

#%%пакеты

import numpy as np

# %matplotlib qt
import matplotlib.pyplot as plt
import scipy.optimize
from numba import jit
from scipy.integrate import quad, nquad
from scipy.optimize import fsolve, root
from time import time

#%%константы

arc = 0.4902264
brc = 0.2989634
crc = 0.3604034
drc = 0.3603434
omega = 0.11e-1
m = 0.37464 + 1.54226*omega - 0.26992*omega*omega
br = brc
cr = crc
dr = drc
R = 8.314472
Zc = 0.286

# vcr = 1.359597691
# Tcr = 0.9609196532
# pcr = 0.801678632091705

#критические значения исходных величин
pc = 4.599
Tc = 190.56
vc = 98.6

# Zc = pc*vc/(R*Tc)

#критические значения для безразмерного уравнения
pcr = 0.892400952
Tcr = 0.9816970074
vcr = 1.181595414

#тройная точка (грубо)
pt = 0.006100277449929686
Tt = 0.50918597921202

#%%функции

@jit("float64(float64)")
def ar(T):
  return arc*(1+m-m*np.sqrt(T))**2

@jit("float64(float64)")
def dar(T):
  return -arc*m*(1+m-m*np.sqrt(T))/np.sqrt(T)

@jit("float64(float64)")
def f(v):
  return (v-dr)/(Zc*(v-br)*(v-cr))

@jit("float64(float64)")
def g(v):
  return -1./(Zc**2 * (v**2 + 2*br*v - br**2))

@jit("float64(float64)")
def df(v):
  a = br*cr - br*dr - cr*dr - v**2 + 2*dr*v
  b = Zc*(v-br)**2 * (v-cr)**2
  return a/b

@jit("float64(float64)")
def dg(v):
  return 2*(v+br)/(Zc**2 * (v**2 + 2*br*v - br**2)**2)

@jit("float64(float64)")
def F(v):
  b = Zc*(br-cr)
  a1 = np.log(np.abs(v-br))
  a2 = np.log(np.abs(v-cr))
  return ((br-dr)*a1 + (dr-cr)*a2)/b

@jit("float64(float64)")
def G(v):
  a = np.log(np.abs(np.sqrt(2)*br+br+v)) - np.log(np.abs(np.sqrt(2)*br-br-v))
  b = (2*np.sqrt(2)*br*Zc**2)
  return a/b

@jit("float64(float64,float64)")
def p0(v,T):
  """
  термическое уравнение состояния
  """
  # a1 = T/(Zc*(v-br))
  # a2 = ar(T)/(Zc**2 * (v**2+2*br*v-br**2))
  return T*f(v) + ar(T)*g(v)

@jit("float64(float64,float64)")
def phi0(v,T):
  """
  потенциал Масье-Планка без функции K(T)
  """
  # a1 = np.log(v-br)/Zc
  # b2 = complex(ar(T)/(T*Zc*Zc*br*np.sqrt(2)))
  # a2 = np.arctanh(complex((v+br)/(np.sqrt(2)*br),0.))*b2
  # return a1 + a2.real
  return F(v) + ar(T)*G(v)/T

@jit("float64(float64,float64)")
def phi0_v(v,T):
  """
  частная производная потенциала Масье-Планка по v
  """
  return f(v) + ar(T)*g(v)/T

@jit("float64(float64,float64)")
def phi0_T(v,T):
  """
  частная производная потенциала Масье-Планка по T
  """
  return G(v)*(dar(T)/T-ar(T)/(T*T))

@jit("float64(float64,float64)")
def phi0_vv(v,T):
  """
  вторая частная производная потенциала Масье-Планка по v
  """
  # a1 = -1./(Zc*(v-br)*(v-br))
  # a2 = 2*ar(T)*(v+br)/(Zc*Zc*(v*v+2*br*v-br*br)*(v*v+2*br*v-br*br))
  # return a1 + a2
  return df(v) + ar(T)*dg(v)/T

@jit("float64(float64,float64)")
def phi0_vT(v,T):
  """
  частная производная потенциала Масье-Планка по v и T
  """
  return g(v)*(dar(T)/T-ar(T)/(T*T))

@jit("float64(float64)")
def Tphi2(v):
  """
  T(v) из phi_vv = 0
  """
  a = arc*m**2*dg(v)**2 - df(v)*dg(v) - 2*m*dg(v)*np.sqrt(-arc*df(v)*dg(v))
  b = (df(v) + arc*m**2*dg(v))**2
  return arc*(m+1)**2*a/b

@jit("float64(float64)")
def Tp0(v):
  """
  T(v) из p0 = 0
  """
  a = arc*m**2*g(v)**2 - f(v)*g(v) + 2*m*g(v)*np.sqrt(-arc*f(v)*g(v))
  b = (f(v) + arc*m**2*g(v))**2
  return arc*(m+1)**2*a/b

@jit("float64(float64,float64,float64)")
def PhTrEq1(v1,v2,T):
  """
  Первое уравнение фазового перехода
  сохраняется давление
  """
  return phi0_v(v2,T) - phi0_v(v1,T)

@jit("float64(float64,float64,float64)")
def PhTrEq2(v1,v2,T):
  """
  второе уравнение фазового перехода
  сохраняется химического потенциала Гиббса
  """
  return phi0(v2,T) - phi0(v1,T) - v2*phi0_v(v2,T) + v1*phi0_v(v1,T)
  # return phi0(v2,T) - phi0(v1,T) - phi0_v(v2,T)*(v2-v1)

@jit("float64[::1](float64[::1],float64)")
def PhTrEq12(vT,v2):
  """
  оба уравнения фазового прехода
  эта функция используется в fsolve
  vT -- массив, 0я координата -- v, 1я -- T
  """
  # print(vT)
  return np.array([PhTrEq1(vT[0],v2,vT[1]), PhTrEq2(vT[0],v2,vT[1])])

@jit("float64[::1](float64,float64,float64)")
def dPhTrEq1(v1,v2,T):
  """
  градиент первого уравнения фазового перехода
  """
  return np.array([-phi0_vv(v1,T),
                   phi0_vT(v2,T) - phi0_vT(v1,T)])

@jit("float64[::1](float64,float64,float64)")
def dPhTrEq2(v1,v2,T):
  """
  градиент второго уравнения фазового перехода
  """
  return np.array([-phi0_v(v1,T) + phi0_v(v1,T) + v1*phi0_vv(v1,T),
                   phi0_T(v2,T) - phi0_T(v1,T) - v2*phi0_vT(v2,T) + v1*phi0_vT(v1,T)])

# @jit("float64[::2](float64[::1],float64)")
def dPhTrEq12(vT,v2):
  """
  якобиан уравнений фазового перехода
  эта функция используется в fsolve
  vT -- массив, 0я координата -- v, 1я -- T
  """
  # print(vT)
  return np.array([dPhTrEq1(vT[0],v2,vT[1]),
                   dPhTrEq2(vT[0],v2,vT[1])])

@jit("float64(float64,float64)")
def fg12(v1,v2):
  # if np.isclose(v1, v2):
  if np.abs(v1-v2)<1e-5:
    # print(v1)
    # print(v2)
    c1 = Zc*(v1**2+2*v1*br-br**2)**2
    c2 = (v1**2-2*dr*v1-br*cr+br*dr+cr*dr)
    b1 = 2 * (v1-br)**2 * (v1-cr)**2 * (br+v1)
    return c1*c2/b1
  else:
    return (f(v2)-f(v1))/(g(v1)-g(v2))

@jit("float64(float64,float64)")
def PhTrT1(v1,v2):
  """
  значение температуры при фазовом переходе
  """
  fg = fg12(v1,v2)
  a1 = (1+m)*(arc*m + np.sqrt(arc*fg))
  b1 = arc*m**2 - fg
  return (a1/b1)**2

@jit("float64(float64,float64)")
def PhTrT(v1,v2):
  """
  значение температуры при фазовом переходе
  """
  fg = fg12(v1,v2)
  a1 = (1+m)*(arc*m - np.sqrt(arc*fg))
  b1 = arc*m**2 - fg
  return (a1/b1)**2

@jit("float64(float64)")
def PhTrV2(v):
  """
  Апроксимация графика фазового прехода в плоскости (v1,v2)
  аппроксимация гиперболой
  """
  return (vcr-br)*(vcr-br)/(v-br) + br

# def PhTr_p(v1):
#   """
#   значение p при фазовом переходе при v
#   функция принимает v1, далее из PhTr_v12 находим v2 и из PhTr_p12 находим p
#   """
#   if v1 > vcr:
#     v0 = 0.1
#   elif v1 > 2.9:
#     v0 = 50.
#   else:
#     v0 = 100.
#   v2 = fsolve(np.vectorize(PhTr_v12), v0, args=v1)[0]
#   return PhTr_p12(v1,v2)

# def PhTr_T(v1):
#   """
#   значение T при фазовом переходе при v
#   функция принимает v1, далее из PhTr_v12 находим v2 и из PhTr_T12 находим T
#   """
#   if v1 > vcr:
#     v0 = 0.1
#   elif v1 > 2.9:
#     v0 = 50.
#   else:
#     v0 = 100.
#   v2 = fsolve(np.vectorize(PhTr_v12), v0, args=v1)[0]
#   return PhTr_T12(v1,v2)

# def fsolver(v1, T, v0=cr+0.0001, eps=1e-3, d=1e-2):
#   """
#   решаю систему для фазового перехода перебором
#   перебор начинаю с v=v0, и иду вперёд по v, при этом T не изменяется
#   когда нахожу две точки, в которых невязка отличается знаком, то уменьшаю шаг
#   иду до vcr
#   """
#   ni = 10**6
#   v2 = v0
#   dis1 = PhTrEq2(v1,v2,T)
#   if np.abs(dis1) < eps:
#     flag = False
#   else:
#     flag = True
#   i = 0
#   while flag and i<ni:
#     i += 1
#     v2 += d
#     dis2 = PhTrEq2(v1,v2,T)
#     if np.abs(dis2) < eps:
#       flag = False
#     if dis1*dis2 < 0:
#       v2 = v2-d
#       d = d*0.5
#     else:
#       dis1 = dis2

#   return v2

def fsolver(f, x0, x1, T, y0=0., drct=1, eps=1e-3, dx=1e-3):
  """
  решаю методом перебора в диапозоне [x0,x1]
  """

  if drct == 1:
    x = x0
  else:
    x = x1
  n1 = 10**4
  dis1 = f(x,T) - y0
  if np.abs(dis1) < eps:
    flag1 = False
  else:
    flag1 = True
  i = 0
  dx1 = dx
  while flag1:
    i = i+1
    x = x + drct*dx1
    if (not x0 < x < x1) or (i > n1):
      flag1 = False
      x = np.nan
    else:
      dis2 = f(x,T) - y0
      if np.abs(dis2) < eps:
        flag1 = False
      else:
        if dis1*dis2 < 0:
          x = x - drct*dx1
          dx1 = 0.5*dx1
        else:
          dis1 = dis2

  return x

def PhTrSolve(T, p01, p02, eps1=1e-3, eps2=1e-3, dp=1e-2, dv=1e-3):
  """
  решаю систему для фазового перехода перебором
  при заданных T и p нахожу значения v в интервалах (br,cr) и (cr,vcr)
  далее проверяю выполнение 2го уравнения
  Если не выполнено, то меняю p
  """

  # disM = ()
  # dpM = ()
  # v1M = ()
  # v2M = ()

  p00 = p01

  v1 = fsolver(p0, br+1e-4, cr-1e-6, T, p00, drct=-1, eps=eps2)
  v2 = fsolver(p0, cr+1e-6, vcr-1e-4, T, p00, drct=1, eps=eps2, dx=1e-1)
  # v1M = np.append(v1M,v1)
  # v2M = np.append(v2M,v2)

  if not (v1 == np.nan or v2 == np.nan):
    dis1 = PhTrEq2(v1,v2,T)
    # disM = np.append(disM,dis1)
    if np.abs(dis1) < eps1:
      flag1 = False
    else:
      flag1 = True
  else:
    flag1 = True

  scs = 1

  dp1 = dp
  # dpM = np.append(dpM,dp1)
  n1 = 10**5
  i = 0
  while flag1:
    i = i + 1
    p00 = p00 + dp1
    # dpM = np.append(dpM,dp1)
    if (not p01 <= p00 <= p02) or i > n1:
      flag1 = False
      v1 = np.nan
      v2 = np.nan
      scs = 0
    else:
      v1 = fsolver(p0, br+1e-4, cr-1e-6, T, p00, drct=-1, eps=eps2)
      v2 = fsolver(p0, cr+1e-6, vcr-1e-4, T, p00, drct=1, eps=eps2, dx=1e-1)
      # v1M = np.append(v1M,v1)
      # v2M = np.append(v2M,v2)
      if not (v1 == np.nan or v2 == np.nan):
        dis2 = PhTrEq2(v1,v2,T)
        # disM = np.append(disM,dis2)
        if np.abs(dis1) < eps1:
          flag1 = False
        else:
          flag1 = True
      else:
        flag1 = True
      if np.abs(dis2) < eps1:
        flag1 = False
      else:
        if dis1*dis2 < 0:
          p00 = p00 - dp1
          dp1 = 0.5*dp1
        else:
          dis1 = dis2

  # return v1, v2, scs, disM, dpM, v1M, v2M
  return v1, v2, scs

def PhTrSolve1(T, p01, p02, eps1=1e-3, eps2=1e-3, dp=1e-2, dv=1e-3):
  """
  решаю систему для фазового перехода перебором
  при заданных T и p нахожу значения v в интервалах (br,cr) и (cr,vcr)
  далее проверяю выполнение 2го уравнения
  Если не выполнено, то меняю p
  """

  disM = ()
  dpM = ()
  v1M = ()
  v2M = ()

  p00 = p01

  v1 = fsolver(p0, br+1e-4, cr-1e-8, T, p00, drct=-1, eps=eps2)
  v2 = fsolver(p0, cr+1e-8, vcr-1e-4, T, p00, drct=1, eps=eps2, dx=1e-1)
  v1M = np.append(v1M,v1)
  v2M = np.append(v2M,v2)

  if not (v1 == np.nan or v2 == np.nan):
    dis1 = PhTrEq2(v1,v2,T)
    disM = np.append(disM,dis1)
    if np.abs(dis1) < eps1:
      flag1 = False
    else:
      flag1 = True
  else:
    flag1 = True

  scs = 1

  dp1 = dp
  dpM = np.append(dpM,dp1)
  n1 = 10**5
  i = 0
  while flag1:
    i = i + 1
    p00 = p00 + dp1
    dpM = np.append(dpM,dp1)
    if (not p01 <= p00 <= p02) or i > n1:
      flag1 = False
      v1 = np.nan
      v2 = np.nan
      scs = 0
    else:
      v1 = fsolver(p0, br+1e-4, cr-1e-6, T, p00, drct=-1, eps=eps2)
      v2 = fsolver(p0, cr+1e-8, vcr-1e-4, T, p00, drct=1, eps=eps2, dx=1e-1)
      v1M = np.append(v1M,v1)
      v2M = np.append(v2M,v2)
      if not (v1 == np.nan or v2 == np.nan):
        dis2 = PhTrEq2(v1,v2,T)
        disM = np.append(disM,dis2)
        if np.abs(dis1) < eps1:
          flag1 = False
        else:
          flag1 = True
      else:
        flag1 = True
      if np.abs(dis2) < eps1:
        flag1 = False
      else:
        if dis1*dis2 < 0:
          p00 = p00 - dp1
          dp1 = 0.5*dp1
        else:
          dis1 = dis2

  return v1, v2, scs, disM, dpM, v1M, v2M
  # return v1, v2, scs

#%%программа

#%%Фазовый переход V-L (считал напрямую)

nn1 = 200
nn2 = 200
nn = nn1 + nn2
# v1M = np.linspace(0.3,dr,nn)
v1M = np.concatenate((np.linspace(0.3,dr-1e-3,nn1),np.linspace(cr+1e-2,vcr-0.01,nn2)))
v2M = np.zeros(nn)
TM = np.zeros(nn)
scs = np.zeros(nn)

for i in range(nn):
  vv0 = PhTrV2(v1M[i])
  TT0 = PhTrT(v1M[i],vv0)
  rt0 = root(PhTrEq12, np.array([vv0,TT0]), args=v1M[i])
  rt = rt0.x
  v2M[i] = rt[0]
  TM[i] = rt[1]
  scs[i] = 1*rt0.success

# plt.figure()
# plt.plot(v1M,scs)
# plt.grid()
# plt.show()

pM = np.vectorize(p0)(v1M,TM)

# plt.figure()
# plt.plot(TM,pM,".")
# plt.plot(Tcr,pcr,".g")
# # plt.xlim([0,1])
# plt.grid()
# plt.show()

# pM01 = np.vectorize(p0)(v1M,TM)
# pM02 = np.vectorize(p0)(v2M,TM)
# pM11 = np.vectorize(p0)(v1M,TM1)
# pM12 = np.vectorize(p0)(v2M,TM1)


#%%Фазовый переход S-L (считал напрямую)

nn = 100
TMSL = np.linspace(Tt,1.5,nn)
v1MSL = np.zeros(nn)
v2MSL = np.zeros(nn)
scsMSL = np.zeros(nn)

tt = time()
for i in range(nn):
  # print(i)
  v1,v2,scs = PhTrSolve(TMSL[i],0.,100.,eps1=1e-5, eps2=1e-5, dp=1e-1)
  v1MSL[i] = v1
  v2MSL[i] = v2
  scsMSL[i] = scs
print(time()-tt)

plt.figure()
plt.plot(TMSL,scsMSL)
plt.grid()
plt.show()

plt.figure()
plt.plot(v1MSL,v2MSL)
plt.grid()
plt.show()

pM1SL = np.vectorize(p0)(v1MSL,TMSL)
# pM2SL = np.vectorize(p0)(v2MSL[1:],TMSL[1:])

plt.figure()
plt.plot(TMSL,pM1SL)
plt.grid()
plt.show()

plt.figure(figsize=(9,7))
plt.plot(TM,pM)
plt.plot(TMSL,pM1SL)
plt.yscale("log")
plt.xlim([0.2,1.5])
plt.grid()
plt.show()

# plt.figure(figsize=(10,10))
# plt.plot(TM,pM)
# plt.xlim([0.2,1.])
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(disM)
# plt.grid()
# plt.show()


#%%Графики на лагранжевом многообразии

# nn = nn1 + nn2
# vM = v1M
# TM = np.linspace(0,Tcr,nn)
# # pM = np.vectorize(p0)(vM,TM)

# #лагранжево многообразие
# vM1,TM1 = np.meshgrid(vM,TM)
# pM = np.zeros((nn,nn))
# for i in range(nn):
#   for j in range(nn):
#     pp = p0(vM1[i,j],TM1[i,j])
#     if pp > 1 or pp < 0:
#       pM[i,j] = np.nan
#     else:
#       pM[i,j] = pp

# #phi_vv = 0
# vphi2M = np.linspace(0.8,vcr,nn)
# Tphi2M = np.vectorize(Tphi2)(vphi2M)
# pphi2M = np.vectorize(p0)(vphi2M,Tphi2M)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(vM1,TM1,pM, alpha=0.5)
# ax.plot3D(vphi2M,Tphi2M,pphi2M)
# ax.set_xlabel('v')
# ax.set_ylabel('T')
# ax.set_zlabel('p')
# plt.show()


#%%черновик

# vM = np.linspace(0.355, 0.365, 1000)
# pM = np.vectorize(p0)(vM,0.5*np.ones(1000))

# plt.figure()
# plt.plot(vM,pM)
# # plt.yscale('log')
# plt.grid()
# plt.show()


# def func(vT, v2):
#   eq1 = vT[0] * np.cos(vT[1]) - v2
#   eq2 = vT[1] * vT[0] - vT[1] - v2-1
#   return [eq1, eq2]

# rt = fsolve(func, np.array([1.1,1.5]), args=4.)
# print(rt)


# rt = fsolve(PhTrEq12, np.array([2.,0.8]), args=0.4)
# print(rt)
# print(PhTrEq12(rt,0.4))
# print(np.isclose(PhTrEq12(rt,0.4),[0.,0.]))


#%%

# @jit("float64(float64)")
# def fPhTrT(fg):
#   """
#   значение температуры при фазовом переходе
#   """
#   # a1 = arc*(m+1.)*(m+1.)*(arc*m*m+fg+2*np.sqrt(arc*fg))
#   # b1 = (fg-arc*m*m)*(fg-arc*m*m)
#   a1 = (1+m)*(arc*m + np.sqrt(arc*fg))
#   b1 = arc*m**2 - fg
#   return (a1/b1)**2

# @jit("float64(float64)")
# def fPhTrT1(fg):
#   """
#   значение температуры при фазовом переходе
#   """
#   # a1 = arc*(m+1.)*(m+1.)*(arc*m*m+fg-2*np.sqrt(arc*fg))
#   # b1 = (fg-arc*m*m)*(fg-arc*m*m)
#   # return a1/b1
#   a1 = (1+m)*(arc*m - np.sqrt(arc*fg))
#   b1 = arc*m**2 - fg
#   return (a1/b1)**2

# nn = 200
# fgM = np.linspace(0, 0.2, nn)
# M0 = np.vectorize(fPhTrT)(fgM)
# M1 = np.vectorize(fPhTrT1)(fgM)

# plt.figure()
# # plt.plot(fgM,M0)
# plt.plot(fgM,M1,'r')
# # plt.ylim([0,40])
# # plt.yscale("log")
# plt.grid()
# plt.show()


# TM = np.linspace(0,1, 100)
# arM = np.vectorize(ar)(TM)/TM

# plt.figure()
# plt.plot(TM,arM)
# plt.grid()
# plt.show()


# @jit("float64(float64)")
# def f2PhTrT(fg):
#   """
#   значение температуры при фазовом переходе
#   """
#   # a1 = arc*(m+1.)*(m+1.)*(arc*m*m+fg+2*np.sqrt(arc*fg))
#   # b1 = (fg-arc*m*m)*(fg-arc*m*m)
#   a1 = (1+m)*(arc*m + np.sqrt(arc*fg))
#   b1 = arc*m**2 - fg
#   return -(a1/b1)**2

# opt = scipy.optimize.minimize(np.vectorize(f2PhTrT), 0.5)


# TM = np.linspace(0,0.1, 100)
# arM = np.vectorize(f2PhTrT)(TM)/TM

# plt.figure()
# plt.plot(TM,arM)
# plt.grid()
# plt.show()





# @jit("float64(float64,float64)")
# def fg12(v1,v2):
#   #здесь неправильно написано выражение при близких v1 и v2
#   if np.isclose(v1, v2):
#     c1 = Zc*(v1**2+2*v1*br-br**2)**2
#     c2 = (v1**2-2*dr*v1-br*cr+br*dr+cr*dr)
#     b1 = 2 * (v1-br)**2 * (v1-cr)**2 * (br+v1)
#     return c1*c2/b1
#   else:
#     return (f(v2)-f(v1))/(g(v1)-g(v2))

# nn1 = 10
# nn = 100
# v1M = np.linspace(0.3,dr,nn1)
# v2M = np.linspace(0.3,dr,nn)
# fgM = np.zeros((nn1,nn))
# for i in range(nn1):
#   fgM[i] = np.vectorize(fg12)(v1M[i]*np.ones(nn),v2M)

# plt.figure()
# for i in range(nn1):
#   plt.plot(v2M,fgM[i])
# plt.grid()
# plt.ylim([0,500])
# plt.legend(v1M)
# plt.show()

# plt.figure()
# plt.plot(v2M,fgM[-1])
# plt.grid()
# plt.show()

#%%

# def test(v1,v2):
#   fg = (f(v2)-f(v1))/(g(v1)-g(v2))
#   T0 = PhTrT(v1,v2)
#   return fg - ar(T0)/T0

# def test1(v1,v2):
#   fg = (f(v2)-f(v1))/(g(v1)-g(v2))
#   T0 = PhTrT1(v1,v2)
#   return fg - ar(T0)/T0

# vv1M = v1M[pM>0]
# vv2M = v2M[pM>0]

# nn1 = vv1M.shape[0]
# disM = np.zeros(nn1)
# for i in range(nn1):
#   disM[i] = test(vv1M[i],vv2M[i])

# plt.figure()
# plt.plot(disM)
# plt.grid()
# plt.show()

# nn1 = vv1M.shape[0]
# disM1 = np.zeros(nn1)
# for i in range(nn1):
#   disM1[i] = test1(vv1M[i],vv2M[i])

# plt.figure()
# plt.plot(disM1)
# plt.grid()
# plt.show()

#%%

# nn = 10000

# vv0 = 0.35
# TT0 = 0.55
# v0M = vv0*np.ones(nn)
# T0M = TT0*np.ones(nn)
# # vv1 = 0.360419355517

# vM2 = np.linspace(dr,0.4,nn)
# eq1M = np.vectorize(PhTrEq1)(v0M,vM2,T0M)
# eq2M = np.vectorize(PhTrEq2)(v0M,vM2,T0M)

# plt.figure()
# plt.plot(eq2M[150:])
# plt.grid()
# plt.show()


# # plt.figure()
# # plt.plot(eq1M)
# # plt.plot(eq2M)
# # plt.grid()
# # plt.show()

# plt.figure()
# plt.plot(eq1M[150:])
# plt.grid()
# plt.show()

# pM = np.vectorize(p0)(np.linspace(br+0.001,0.4,nn),T0M)

# plt.figure()
# plt.plot(np.linspace(br+0.001,0.4,nn),pM)
# plt.ylim([-10,10])
# plt.grid()
# plt.show()

# T1M = np.vectorize(PhTrT)(v0M,vM2)

# plt.figure()
# plt.plot(T1M)
# plt.grid()
# plt.show()



# nn = 1000
# vM = np.linspace(0.355, 0.37, nn)
# pM = np.vectorize(p0)(vM, 0.8*np.ones(nn))

# plt.figure()
# plt.plot(pM)
# plt.ylim([0.,1.5])
# plt.grid()
# plt.show()




# vv1,vv2,disM = PhTrSolve(0.55,1e-7,21.)

# plt.figure()
# plt.plot(disM)
# plt.grid()
# plt.show()


#%%

# rt0 = root(PhTrEq12, np.array([cr+1e-5,0.507]), args=cr-1e-3)

# nn = 100
# vM = np.linspace(br,1.,nn)
# gM = np.vectorize(g)(vM)

# plt.figure()
# plt.plot(vM,gM)
# plt.grid()
# plt.show()

# TM = np.linspace(0.,1.,nn)
# aM = np.vectorize(ar)(vM)

# plt.figure()
# plt.plot(TM,aM)
# plt.grid()
# plt.show()

# nn = 100
# v10 = 0.3
# T0 = 0.51
# vM = np.linspace(cr+1e-6,0.36041,nn)
# fM1 = np.vectorize(PhTrEq2)(v10*np.ones(nn),vM,0.51*np.ones(nn))
# fM2 = np.vectorize(PhTrEq2)(v10*np.ones(nn),vM,1.*np.ones(nn))

# plt.figure()
# plt.plot(vM,fM1)
# plt.plot(vM,fM2)
# plt.grid()
# plt.show()


# nnT = 30
# TMSL = np.linspace(0.507,1.,nnT)

# v1,v2,scs,disM,dpM,v1M,v2M = PhTrSolve1(0.507,-1.,100.,eps1=1e-4, eps2=1e-5, dp=1e-1)

# plt.figure()
# plt.plot(disM[:100])
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(dpM)
# # plt.yscale('log')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(v1M)
# plt.plot(v2M)
# plt.legend(['v1','v2'])
# plt.grid()
# plt.show()

# nn = 100
# v10 = 0.3
# T0 = TMSL[7]
# vM = np.linspace(cr+1e-6,0.36041,nn)
# fM1 = np.vectorize(PhTrEq2)(v10*np.ones(nn),vM,T0*np.ones(nn))

# plt.figure()
# plt.plot(vM,fM1)
# plt.grid()
# plt.show()


plt.figure(figsize=(9,7))
plt.plot(TM,pM)
plt.plot(TMSL,pM1SL)
# plt.plot(TM,pM,'.r')
plt.xlim([0.4,0.8])
plt.ylim([0.,0.1])
plt.grid()
plt.show()

plt.figure(figsize=(9,7))
plt.plot(TM)
plt.grid()
plt.show()

plt.figure(figsize=(9,7))
plt.plot(pM)
# plt.ylim([0,0.4])
plt.yscale('log')
plt.xlim([190,210])
plt.grid()
plt.show()

# for i in range(100):
#   if TM[i] > TMSL[1]:
#     print(i)
#     break

# plt.figure(figsize=(9,7))
# plt.plot(TM[66:68],pM[66:68])
# plt.plot(TMSL[0:2],pM1SL[0:2])
# plt.plot(xx,yy,'.r')
# plt.xlim([0.4,0.6])
# plt.ylim([0.,0.02])
# plt.grid()
# plt.show()


# for i in range(nn-1):
#   if np.abs(pM[i]-pM[i+1]) > 1e-2:
#     print(i)
#     break

MM1 = np.concatenate((v1M,np.linspace(vcr,10.,100)))
MM2 = np.vectorize(Tphi2)(MM1)
MM21 = np.vectorize(Tp0)(MM1)

plt.figure(figsize=(9,7))
plt.plot(MM1,MM2)
plt.plot(MM1,MM21)
plt.grid()
plt.show()

MM3 = np.vectorize(p0)(MM1,MM2)

plt.figure(figsize=(9,7))
plt.plot(MM2,MM3)
plt.ylim([0,1])
plt.grid()
plt.show()
