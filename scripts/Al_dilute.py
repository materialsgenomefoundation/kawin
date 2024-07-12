import numpy as np
from pycalphad import Database, equilibrium, variables as v
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.thermo import BinaryThermodynamics
import matplotlib.pyplot as plt

therm = BinaryThermodynamics('examples//AlScZr.tdb', ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'])
therm.setGuessComposition(0.24)

# T = np.linspace(50, 250, 250) + 273.15
# g = np.zeros(250)
# xa, xb = therm.getInterfacialComposition(T, g)
# plt.plot(T, xa)
# plt.ylim([1e-10, 1e-3])
# plt.yscale('log')
# plt.xlabel('Temperature')
# plt.ylabel('X_AL in FCC_A1')
# plt.show()

# T = 150+273.15
# x = np.logspace(-10, -3, 100)
# dg, _ = therm.getDrivingForce(x, np.ones(x.shape)*T)
# plt.plot(x, dg)
# plt.xlim([1e-10, 1e-3])
# plt.xscale('log')
# plt.xlabel('X_AL')
# plt.ylabel('Driving force (J/mol)')
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

init_comps = [4e-3, 4e-4, 4e-5, 4e-6, 4e-7]
for x in init_comps:
    print(x)
    model = PrecipitateModel()

    model.setInitialComposition(x)
    model.setTemperature(150 + 273.15)

    gamma = 0.1         #Interfacial energy (J/m2)
    model.setInterfacialEnergy(0.01)
    Diff = lambda x, T: 100000 * 0.0768 * np.exp(-242000 / (8.314 * T))
    model.setDiffusivity(Diff)

    a = 0.405e-9        #Lattice parameter
    atomsPerCell = 4    #Atoms in an FCC unit cell
    model.setVolumeAlpha(a**3, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    model.setVolumeBeta(a**3, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)

    #Average grain size (um) and dislocation density (1e15)
    model.setNucleationDensity(grainSize = 1, dislocationDensity = 1e15)
    model.setNucleationSite('dislocations')

    #Set thermodynamic functions
    model.setThermodynamics(therm, addDiffusivity=False)

    model.solve(500*3600, verbose=True, vIt=1000)

    label = r'$X_0$ = {:.2e}'.format(x)
    model.plot(axes[0,0], 'Precipitate Density', label=label)
    model.plot(axes[0,1], 'Volume Fraction', label=label)
    model.plot(axes[1,0], 'Average Radius', label=label)
    model.plot(axes[1,1], 'Composition', label=label)

axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
axes[1,1].legend()

fig.tight_layout()
plt.show()

