#!/usr/bin/env python

## Model for a pendulum oscillator
##  
##  by S. Nakano (Jul. 2020)

import numpy as np
import numpy.random as ran

Ndim = 2
Nobs = 1
dt = 0.2
Tmax = 240
Kmax = int(Tmax / dt)
Tobs = 6
Kobs_intvl = int(Tobs / dt)

Tperiod = 30.0
Tforce = 70.0
gamma = 0.01
forcing = 0.03


class Pendulum_state:
  def __init__(self):
    self.x = np.zeros(( Ndim ))


class Pendulum_model:
  def __init__(self):
    self.a = (2.0*np.pi / Tperiod)**2
    self.omf = 2.0*np.pi / Tforce


  def Euler(self, xst, t):
    g0 = np.zeros((Ndim))

    g0[0] = xst.x[1]
    g0[1] = - self.a*np.sin(xst.x[0]) - gamma*xst.x[1] + forcing*np.sin(self.omf*t)*np.cos(xst.x[0])

    xst.x[:] += dt * g0


  def modifiedEuler(self, xst, t):
    g0 = np.zeros((Ndim))
    g1 = np.zeros((Ndim))

    g0[0] = xst.x[1]
    g0[1] = - self.a*np.sin(xst.x[0]) - gamma*xst.x[1] + forcing*np.sin(self.omf*t)*np.cos(xst.x[0])

    xt = xst.x[:] + dt * g0 / 2
    tt = t + dt / 2

    g1[0] = xt[1]
    g1[1] = - self.a*np.sin(xt[0]) - gamma*xt[1] + forcing*np.sin(self.omf*tt)*np.cos(xt[0])

    xst.x[:] += dt * g1



class Pendulum_observation:
  def __init__(self, rsig):
    self.rsigma = rsig
    self.Rmat = rsig*rsig*np.eye( Nobs )
    self.Hmat = np.zeros((Nobs, Ndim))
    for i in range(Nobs):
      self.Hmat[i,i] = 1.0

  def hobs(self, xst):
    yob = self.Hmat.dot(xst.x[:])
    return yob

  def obs_perturbation(self):
    return self.rsigma*ran.randn(Nobs)



if __name__ == '__main__':
  import matplotlib.pylab as plt
  import csv

## State vector
  xstate = Pendulum_state( )


## Pendulum model
  xsys = Pendulum_model( )

## For synthetic observation
  robs = 0.1
  obsm = Pendulum_observation( robs )

  xstate.x[0] = 0.0
  xstate.x[1] = 0.2

  truefile = open( "true.dat", 'w' )
  obsfile = open( "obs.dat", 'w' )
  truefile.write("{0:6.1f}{1:12.6f}{2:12.6f}\n".format(0, xstate.x[0], xstate.x[1]))

## Main loop
  for k in range(1,Kmax+1):
    xsys.modifiedEuler( xstate, k*dt )
    #xsys.Euler( xstate, k*dt )

    if k % Kobs_intvl == 0:
      yobs = obsm.hobs( xstate ) + obsm.obs_perturbation()
      obsfile.write( "{0:6.1f} ".format(k*dt) )
      for i in range(Nobs-1):
        obsfile.write( "{0:15.6f} ".format(yobs[i]) )
      obsfile.write( "{0:15.6f}\n".format(yobs[Nobs-1]) )

    truefile.write("{0:6.1f}{1:12.6f}{2:12.6f}\n".format(k*dt, xstate.x[0], xstate.x[1]))

  truefile.close()
  obsfile.close()



## Plot
  xarr = np.loadtxt("true.dat")
  yarr = np.loadtxt("obs.dat")
  plt.plot( xarr[:,0], xarr[:,1], color='b' )
  plt.plot( xarr[:,0], xarr[:,2], color='g' )
  plt.scatter( yarr[:,0], yarr[:,1], color='r' )
  plt.grid()
  plt.savefig("pendulum.png")
  plt.close()
