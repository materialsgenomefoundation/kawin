from kawin.thermo import MulticomponentThermodynamics
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.solver import SolverType
import numpy as np
import matplotlib.pyplot as plt

elements = ['NI', 'AL', 'CR', 'VA']
phases = ['FCC_A1', 'FCC_L12']

therm = MulticomponentThermodynamics('examples//NiCrAl.tdb', elements, phases, drivingForceMethod='tangent')

model = PrecipitateModel(elements=['Al', 'Cr'])

model.setInitialComposition([0.098, 0.083])
model.setInterfacialEnergy(0.023)

T = 1073
model.setTemperature(T)

a = 0.352e-9        #Lattice parameter
Va = a**3           #Atomic volume of FCC-Ni
Vb = Va             #Assume Ni3Al has same unit volume as FCC-Ni
atomsPerCell = 4    #Atoms in an FCC unit cell
model.setVolumeAlpha(Va, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
model.setVolumeBeta(Vb, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
model.setInfinitePrecipitateDiffusivity(False)

#Set nucleation sites to dislocations and use defualt value of 5e12 m/m3
#model.setNucleationSite('dislocations')
#model.setNucleationDensity(dislocationDensity=5e12)
model.setNucleationSite('bulk')
model.setNucleationDensity(bulkN0=1e30)

model.setThermodynamics(therm)

model.solve(2e6, solverType=SolverType.EXPLICITEULER, verbose=True, vIt = 500)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

bounds = [1e-1, 1e6]
model.plot(axes[0,0], 'Precipitate Density', bounds)
model.plot(axes[0,1], 'Volume Fraction', bounds)
model.plot(axes[1,0], 'Average Radius', bounds, color='C0', label='Avg. R')
model.plot(axes[1,0], 'Critical Radius', bounds, color='C1', label='R*')
axes[1,0].legend(loc='upper left')
model.plot(axes[1,1], 'Composition', bounds)
model.plot(axes[1,1], 'Eq Composition Alpha', bounds, color='k', linestyle='--')

#model.save('scripts/NiCrAl_zero')

fig.tight_layout()
plt.show()
