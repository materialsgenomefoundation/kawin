from kawin.ShapeFactors import ShapeFactor
from kawin.ElasticFactors import StrainEnergy
import numpy as np
import matplotlib.pyplot as plt
import time

#Strain energy parameters
se = StrainEnergy()
se.setEigenstrain([6.67e-3, 6.67e-3, 2.86e-2])
se.setModuli(G=57.1e9, nu=0.33)
se.setEllipsoidal()
se.setup()

#Shape factor parameters (only the shape needs to be defined)
sf = ShapeFactor()
sf.setPlateShape()

#Calculate equilibrium aspect ratio
gamma = 0.02375
Rsph = np.linspace(1e-10, 10e-9, 25)

t0 = time.time()
eqAR = se.eqAR_bySearch(Rsph, gamma, sf)
tf = time.time()
print(tf - t0)

#Convert spherical radius to diameter of the plate
R = 2*Rsph / np.cbrt(eqAR**2)*eqAR

#Plot diameter vs. aspect ratio
plt.plot(R, eqAR)
#plt.scatter(R, eqAR, s=8, color='k')
plt.xlim([0, 40e-9])
plt.ylim([1, 9])
plt.xlabel('Diameter (m)')
plt.ylabel('Aspect Ratio')
plt.show()