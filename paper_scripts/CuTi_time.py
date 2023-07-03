from kawin.KWNEuler import PrecipitateModel
from kawin.thermo.Thermodynamics import BinaryThermodynamics
from kawin.ElasticFactors import StrainEnergy
from kawin.thermo.Surrogate import BinarySurrogate
import numpy as np
import matplotlib.pyplot as plt
import time

tstart = time.time()

model = PrecipitateModel(1e-3, 1e5, 5000, phases=['CU4TI'], linearTimeSpacing=False, elements=['TI'])
model.setConstraints(minRadius=0.3e-10)
model.setBetaBinary(2)

therm = BinaryThermodynamics('paper_scripts//CuTi.tdb', ['CU', 'TI'], ['FCC_A1', 'CU4TI'], interfacialCompMethod='equilibrium')
therm.setMobilityCorrection('all', 100)
therm.setGuessComposition(0.17)

t1 = time.time()

model.setInitialComposition(0.019)
model.setTemperature(350 + 273.15)
model.setInterfacialEnergy(0.035)

model.setThermodynamics(therm)

VmAlpha = 7.11e-6
model.setVmAlpha(VmAlpha, 4)

VmBeta = 7.628e-6
model.setVmBeta(VmBeta, 20)
#print(model.VmBeta)

model.setNucleationSite('bulk')
#model.setNucleationDensity(bulkN0=1e30)
model.setNucleationDensity(bulkN0=8.4699e28)
#model.setNucleationDensity(dislocationDensity=1e30)

se = StrainEnergy()
se.setInterfacialEnergyMethod('eqradius')
se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)
se.setEigenstrain([0.022, 0.022, 0.003])

t2 = time.time()

model.solve(verbose=True, vIt=1000)
model.save('paper_scripts//outputs//CuTi_TC_sphere2')

t3 = time.time()

model.reset()
therm.clearCache()
model.setStrainEnergy(se, calculateAspectRatio=True)

#Set precipitate shape
#Since we're calculating the aspect ratio, it does not have to be defined
#Otherwise, a constant value or function can be inputted
model.setAspectRatioNeedle()

model.solve(verbose=True, vIt=2000)
model.save('paper_scripts//outputs//CuTi_TC_needle2')

t4 = time.time()

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(2, 2, figsize=(10,8))

modelS = PrecipitateModel.load('paper_scripts//outputs//CuTi_TC_sphere2.npz')
modelN = PrecipitateModel.load('paper_scripts//outputs//CuTi_TC_needle2.npz')

modelS.plot(axes[0,0], 'Precipitate Density', bounds=[1e-2, 1e4], timeUnits='min')
modelN.plot(axes[0,0], 'Precipitate Density', bounds=[1e-2, 1e4], timeUnits='min')
axes[0,0].set_ylim([1e10, 1e28])
axes[0,0].set_yscale('log')

modelS.plot(axes[0,1], 'Composition', bounds=[1e-2, 1e4], timeUnits='min', label='Composition')
modelN.plot(axes[0,1], 'Composition', bounds=[1e-2, 1e4], timeUnits='min', label='Composition')
axes[0,1].legend()

modelS.plot(axes[1,0], 'Average Radius', bounds=[1e-2, 1e4], timeUnits='min', label='Radius')
modelN.plot(axes[1,0], 'Average Radius', bounds=[1e-2, 1e4], timeUnits='min', label='Radius')
axes[1,0].set_ylim([0, 7e-9])

ax1 = axes[1,0].twinx()
modelS.plot(ax1, 'Aspect Ratio', bounds=[1e-2, 1e4], timeUnits='min', label='Aspect Ratio', linestyle=':')
modelN.plot(ax1, 'Aspect Ratio', bounds=[1e-2, 1e4], timeUnits='min', label='Aspect Ratio', linestyle=':')
ax1.set_ylim([1,4])

modelS.plot(axes[1,1], 'Size Distribution Density', label='PSD')
modelN.plot(axes[1,1], 'Size Distribution Density', label='PSD')

ax2 = axes[1,1].twinx()
modelS.plot(ax2, 'Aspect Ratio Distribution', label='Aspect Ratio', linestyle=':')
modelN.plot(ax2, 'Aspect Ratio Distribution', label='Aspect Ratio', linestyle=':')
axes[1,1].set_xlim([0, 1.5e-8])
ax2.set_ylim([1,7])

fig.tight_layout()

tfinish = time.time()
print('{:.5f} seconds - total'.format(tfinish-tstart))
print('{:.5f} seconds - load thermodynamic database'.format(t1-tstart))
print('{:.5f} seconds - input model parameters'.format(t2-t1))
print('{:.5f} seconds - solve spherical model'.format(t3-t2))
print('{:.5f} seconds - solve needle model'.format(t4-t3))
print('{:.5f} seconds - plot results'.format(tfinish-t4))

plt.show()