#!/usr/bin/env python

#
# Estimation with ensemble Kalman filter
#

import numpy as np
import numpy.random as ran
from pendulum import *
from EnKF import *
import matplotlib.pylab as plt


Npt = 20 # Ensemble size
robs = 0.1 # Standard deviation of observation noise

#!!* Seed
iseed = 100
ran.seed(iseed)

xp = [None] * Npt
for i in range(Npt):
  xp[i] = Pendulum_state()
  xp[i].x[0] = ran.normal( 0.0, 1.0 )
  xp[i].x[1] = ran.normal( 0.0, 0.2 )


xsysm = Pendulum_model( )
obmodel = Pendulum_observation( robs )
filt = Filter( xp, xsysm, obmodel )

yobs = np.loadtxt("obs.dat")
xest = np.zeros((Kmax+1, Ndim))

qmean = np.zeros((Ndim))
Qmat = 1.0e-6*np.eye(Ndim)

## EnKF loop
xmean,xvar = filt.estimation( xp )
xest[0,:] = xmean.x[:]

for k in range(1,Kmax+1):
  filt.prediction( xp, k*dt )
  for i in range(Npt):
    xp[i].x[:] += ran.multivariate_normal( qmean, Qmat )

  if k % Kobs_intvl == 0:
    j = int(k / Kobs_intvl) - 1
    filt.enkf( xp, yobs[j,1:] )
    print("t = {0}".format(k))

  xmean,xvar = filt.estimation( xp )

  xest[k,:] = xmean.x[:]
  if np.isnan(np.sum(xest[k,:])):
    print(k)
    break
## EnKF loop end


## Plot
xtruth = np.loadtxt( "true.dat",)
plt.plot( xtruth[:,0], xtruth[:,1], color='LightSkyBlue' )
plt.plot( xtruth[:,0], xtruth[:,2], color='LightGreen' )

plt.plot( xtruth[:,0], xest[:,0], color='b' )
plt.plot( xtruth[:,0], xest[:,1], color='g' )

plt.scatter( yobs[:,0], yobs[:,1], color='r' )
plt.grid()
plt.savefig("pendulum_enkf.png")
plt.close()
