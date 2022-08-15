## Ensemble Kalman filter
##  
##  by S. Nakano (Jul. 2020)

import numpy as np
import copy
from pendulum import *


class Filter():
  def __init__( self, xens, sysmodel, obsmodel ):
    self.nptcls = len(xens)
    self.sym = sysmodel
    self.obm = obsmodel
    self.weight = (1.0 / self.nptcls) * np.ones(( self.nptcls ))


  def prediction(self, xens, t):
    for i in range(self.nptcls):
      #self.sym.modifiedEuler( xens[i], t )
      self.sym.Euler( xens[i], t )    #課題①


  def enkf(self, xens, ydata):
    Ymat = np.zeros((Nobs, self.nptcls))

    for i in range(self.nptcls):
      Ymat[:,i] = self.obm.hobs(xens[i])

    ymean = np.reshape(np.mean(Ymat, 1), (Nobs,1))
    DY = Ymat - ymean
    YYmat = DY.dot(DY.T)/(self.nptcls - 1.0) + self.obm.Rmat

    Yinov = np.reshape(ydata, (Nobs,1)) - Ymat
    for i in range(self.nptcls):
      Yinov[:,i] += self.obm.obs_perturbation()

###
    Wmat = (DY.T).dot(np.linalg.inv(YYmat)) / (self.nptcls - 1.0)

    Kgain = np.zeros(( Ndim, Nobs ))
    for i in range(Ndim):
      for j in range(Nobs):
        for k in range(self.nptcls):
          Kgain[i,j] += xens[k].x[i] * Wmat[k,j]

    for i in range(self.nptcls):
      xens[i].x += Kgain.dot(Yinov[:,i])

### Alternative
    # Wmat = DY.T / (self.nptcls - 1.0)
    # Wmat = Wmat.dot((np.linalg.inv(YYmat)).dot(Yinov))

    # xorg = copy.deepcopy( xens )
    # for i in range(self.nptcls):
    #   for j in range(self.nptcls):
    #     xens[i].x += Wmat[j,i] * xorg[j].x
###



  def estimation(self, xens ):
    xensmean = Pendulum_state( )
    xensvar = Pendulum_state( )

    for i in range(self.nptcls):
      xensmean.x += self.weight[i] * xens[i].x
      xensvar.x += self.weight[i] * xens[i].x**2

    xensvar.x = xensvar.x - xensmean.x**2

    return xensmean,xensvar
