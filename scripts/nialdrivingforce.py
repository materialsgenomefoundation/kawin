from kawin.thermo import MulticomponentThermodynamics, BinaryThermodynamics
import numpy as np
import matplotlib.pyplot as plt

elements = ['NI', 'AL']
phases = ['FCC_A1', 'FCC_L12']
therm = BinaryThermodynamics('examples//NiCrAl.tdb', elements, phases)
therm.setDrivingForceMethod('tangent')

xs = np.ones(100)*0.1
ts = np.linspace(700, 1300, 100)

dg, xb = therm.getDrivingForce(0.1, 1300, returnComp=True)

print(dg, xb)

gm = therm._pointsPrec['FCC_L12'].GM.values.ravel()
ocm = therm._orderingPoints['FCC_L12'].OCM.values.ravel()
xcomp = therm._orderingPoints['FCC_L12'].X.sel(component='AL').values.ravel()

fig, ax = plt.subplots(1,2)
ax[0].scatter(xcomp, ocm, s=2)
ax[1].scatter(xcomp, gm, s=2)
plt.show()