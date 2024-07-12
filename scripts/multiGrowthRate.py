from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics
from pycalphad import equilibrium, variables as v
from kawin.thermo.LocalEquilibrium import local_equilibrium
import numpy as np
import matplotlib.pyplot as plt
import time

therm = MulticomponentThermodynamics('examples//NiCrAl.tdb', ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')

#Gibbs-Thomson contribution from radius
gamma = 0.023        #Interfacial energy between FCC-Ni and Ni3Al
Vm = 1e-5           #Molar volume
R = np.linspace(1e-10, 1e-6, 10000)
R = np.logspace(-10, -6, 500)
G = 2 * gamma * Vm / R

fig, ax = plt.subplots(1,1)

#Calculate growth rate for different sets of compositions
xset = {'Ni-3Cr-1Al': [0.03, 0.01], 'Ni-3Cr-3Al': [0.03, 0.03], 'Ni-3Cr-5Al': [0.03, 0.05], 'Ni-3Cr-10Al': [0.03, 0.10], 'Ni-3Cr-15Al': [0.03, 0.15], 'Ni-3Cr-17.5Al': [0.03, 0.175], 'Ni-3Cr-20Al': [0.03, 0.2]}
xset = {'Ni-1Cr-1Al': [0.01, 0.01], 'Ni-1Cr-0.01Al': [1e-2, 1e-4], 'Ni-1Cr-0.0001Al': [1e-2, 1e-6], 'Ni-1Cr-0.000001': [1e-2, 1e-8]}
#xset = {'Ni-1Cr-1Al': [0.01, 0.01], 'Ni-0.01Cr-1Al': [1e-4, 1e-2], 'Ni-0.0001Cr-1Al': [1e-6, 1e-2], 'Ni-0.000001Cr-1Al': [1e-8, 1e-2]}
xset = {'Ni-1Cr-1Al': [0.01, 0.01], 'Ni-0.01Cr-0.01Al': [1e-4, 1e-4], 'Ni-0.0001Cr-0.0001Al': [1e-6, 1e-6], 'Ni-0.000001Cr-0.000001Al': [1e-8, 1e-8]}
T = 573
for x in xset:
    #Clear cache since the compositions are quite different in values
    therm.clearCache()

    #Calculate driving force and growth rate
    dg, xb = therm.getDrivingForce(xset[x], T, returnComp=True)
    gr, ca, cb, caeq, cbeq = therm.getGrowthAndInterfacialComposition(xset[x], T, dg, R, G, searchDir=xb)
    ax.plot(R, gr, label=x)
    #ax[0,1].plot(R, ca[:,0])
    #ax[0,2].plot(R, ca[:,1])
    #ax[1,0].plot(R, cb[:,0])
    #ax[1,1].plot(R, cb[:,1])
    print('interface', caeq, cbeq)

ax.set_xlim([0, 5e-8])
#ax.set_ylim([-4e-13, 4e-13])
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Growth Rate (m/s)')
ax.plot([0, 1e-6], [0,0], color='k', linestyle='--')
ax.legend(xset.keys())

plt.show()