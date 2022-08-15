#!/usr/bin/env python

#
# Assimilation with adjoint method
#

import numpy as np
import numpy.random as ran
from pendulum import *
from pendulum_adj import *
import matplotlib.pylab as plt
import copy

Kwin = 120 ## Assimilation window

## DA parameters
Niter = 30
epsilon = 0.0001
ecri = 1.0e-6
robs = 0.1 # Standard deviation of observation noise

xsysm = Pendulum_model( )
xadjm = Pendulum_adjoint( )

obmodel = Pendulum_observation( robs )
RHt = obmodel.Hmat.T / obmodel.rsigma**2

Binv = np.diag([1.0, 25.0])

yobs = np.loadtxt("obs.dat")
xest = np.zeros((Kmax+1, Ndim))

delx = np.zeros((Ndim))
xinit = Pendulum_state( )

for kt in range(0, Kmax, Kwin):
# Minimizing the cost function with gradient descent method
  for m in range(Niter):
    j0 = int(kt / Kobs_intvl)
    j1 = int((kt + Kwin) / Kobs_intvl)
    yf = yobs[j0:j1,1:].copy()

    xf = [None] * (Kwin+1)
    xf[0] = copy.deepcopy(xinit)

    Jcost = 0.0
## Forward
    for k in range(1,Kwin+1):
      xf[k] = copy.deepcopy(xf[k-1]) ## Save the history
      xsysm.modifiedEuler( xf[k], (kt+k)*dt )

      if k % Kobs_intvl == 0:
        j = int(k / Kobs_intvl) - 1
        yf[j,:] -= obmodel.hobs( xf[k] )
        Jcost += yf[j,:].dot(yf[j,:])/obmodel.rsigma**2

    Jcost = Jcost / 2
#    print( "Iter {0}: J={1}".format(m, Jcost) )

## Adjoint
    xad = Pendulum_state( )
    for k in range(Kwin,0,-1):
      if k % Kobs_intvl == 0:
        j = int(k / Kobs_intvl) - 1
        xad.x[:] += RHt.dot( yf[j,:] )
      #xadjm.modifiedEuler_adjoint( xad, (kt+k-1)*dt, xf[k-1] )
      xadjm.Euler_adjoint( xad, (kt+k-1)*dt, xf[k-1] )
                    
    delx = epsilon * (xad.x[:] - Binv.dot(xinit.x[:]))
    xinit.x += delx

    if np.sum( delx * delx ) < ecri:
      break
## Iteration loop end

  print( "Iter {0}: J={1}".format(m, Jcost) )
  print(xinit.x[:])

  xf[0] = copy.deepcopy(xinit)
  xest[kt,:] = xf[0].x[:]

  for k in range(1,Kwin+1):
    xsysm.modifiedEuler( xf[0], (kt+k)*dt )
    xest[kt+k,:] = xf[0].x[:]

  xinit = copy.deepcopy(xf[0])
  print(xinit.x[:])

## Plot
xtruth = np.loadtxt( "true.dat",)
plt.plot( xtruth[:,0], xtruth[:,1], color='LightSkyBlue' )
plt.plot( xtruth[:,0], xtruth[:,2], color='LightGreen' )

plt.plot( xtruth[:,0], xest[:,0], color='b' )
plt.plot( xtruth[:,0], xest[:,1], color='g' )

plt.scatter( yobs[:,0], yobs[:,1], color='r' )
plt.grid()
plt.savefig("pendulum_adj.png")
plt.close()
