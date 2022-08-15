#!/usr/bin/env python
#
# The adjoint codes are rewritten by line-by-line conversion
# By Yosuke Fujii
#
#
import numpy as np
import numpy.random as ran
from pendulum import *

class Pendulum_adjoint:
  def __init__(self):
    self.a = (2.0*np.pi / Tperiod)**2
    self.omf = 2.0*np.pi / Tforce


  def modifiedEuler_adjoint(self, xa, t, xf):
    ## Forward code (same as the original)
    g0 = np.zeros((Ndim))
    
    g0[0] = xf.x[1]
    g0[1] = - self.a*np.sin(xf.x[0]) - gamma*xf.x[1] + forcing*np.sin(self.omf*t)*np.cos(xf.x[0])
    
    xt = xf.x[:] + dt * g0 / 2
    tt = t + dt / 2
    
    ## Initialization
    g0a = np.zeros((Ndim))
    g1a = np.zeros((Ndim))
    xta = np.zeros((Ndim))

    g1a[:] = dt * xa.x[:]

    xta[0] = ( - self.a*np.cos(xt[0])
                - forcing*np.sin(self.omf*tt)*np.sin(xt[0]) ) * g1a[1]
    xta[1] = - gamma * g1a[1]
    xta[1] += g1a[0]

    xa.x[:] += xta[:]

    g0a[:] += (dt * 0.5) * xta[:]
    xa.x[0] += ( - self.a*np.cos(xf.x[0])
                - forcing*np.sin(self.omf*t)*np.sin(xf.x[0]) ) * g0a[1]
    xa.x[1] += - gamma * g0a[1]
    xa.x[1] += g0a[0]


  def Euler_adjoint(self, xa, t, xf):
    ## Forward code (same as the original)
    g0 = np.zeros((Ndim))
    
    g0[0] = xf.x[1]
    g0[1] = - self.a*np.sin(xf.x[0]) - gamma*xf.x[1] + forcing*np.sin(self.omf*t)*np.cos(xf.x[0])
    
    xt = xf.x[:] + dt * g0
    tt = t + dt / 2
    
    # Initialization
    g0a = np.zeros((Ndim))
    #xa = np.zeros((Ndim))

    #xa.x[0] += xa.x[0]
    
    xa.x[1] += dt * xa.x[0]    
    
    xa.x[0] += ( - self.a*np.cos(xf.x[1])
                - forcing*np.sin(self.omf*t)*np.sin(xf.x[1]) )* dt* xa.x[1]
    
    xa.x[1] += (- gamma*dt) * xa.x[1]
    