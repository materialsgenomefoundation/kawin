from kawin.thermo.Thermodynamics import BinaryThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.ElasticFactors import StrainEnergy
import matplotlib.pyplot as plt
import numpy as np

model = PrecipitateModel(1e-3, 1e5, 5000, phases=['CU4TI'], linearTimeSpacing=True, elements=['TI'])

therm = BinaryThermodynamics('database//CuTi.tdb', ['CU', 'TI'], ['FCC_A1', 'CU4TI'], interfacialCompMethod='equilibrium')
therm.setMobilityCorrection('all', 100)
therm.setGuessComposition(0.15)

model.setInitialComposition(0.019)
model.setTemperatureArray(np.array([0, 1e5])/3600, np.array([400, 200])+273.15)
#model.setTemperature(350 + 273.15)
model.setInterfacialEnergy(0.035)
model.setThermodynamics(therm)

VmAlpha = 7.11e-6
model.setVmAlpha(VmAlpha, 4)

VaBeta = 0.25334e-27
model.setVaBeta(VaBeta, 20)

model.setNucleationSite('bulk')
model.setNucleationDensity(bulkN0=1e30)

se = StrainEnergy()
se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)
se.setEigenstrain([0.022, 0.022, 0.003])

model.setStrainEnergy(se, calculateAspectRatio=True)

#Set precipitate shape
#Since we're calculating the aspect ratio, it does not have to be defined
#Otherwise, a constant value or function can be inputted
model.setAspectRatioNeedle()

model.solve(verbose=True, vIt=2000)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

model.plot(axes[0,0], 'Precipitate Density', bounds=[1e-2, 1e4], timeUnits='min')
axes[0,0].set_ylim([1e10, 1e28])
axes[0,0].set_yscale('log')

model.plot(axes[0,1], 'Composition', bounds=[1e-2, 1e4], timeUnits='min', label='Composition')
model.plot(axes[0,1], 'Eq Composition Alpha', bounds=[1e-2, 1e4], timeUnits='min', label='Equilibrium')
axes[0,1].legend()

model.plot(axes[1,0], 'Average Radius', bounds=[1e-2, 1e4], timeUnits='min', label='Radius')
axes[1,0].set_ylim([0, 7e-9])

ax1 = axes[1,0].twinx()
model.plot(ax1, 'Aspect Ratio', bounds=[1e-2, 1e4], timeUnits='min', label='Aspect Ratio', linestyle=':')
ax1.set_ylim([1,4])

model.plot(axes[1,1], 'Size Distribution Density', label='PSD')

ax2 = axes[1,1].twinx()
model.plot(ax2, 'Aspect Ratio Distribution', label='Aspect Ratio', linestyle=':')
axes[1,1].set_xlim([0, 1.5e-8])
ax2.set_ylim([1,7])

fig.tight_layout()