from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.thermo.Surrogate import MulticomponentSurrogate, generateTrainingPoints
import numpy as np
import matplotlib.pyplot as plt

elements = ['NI', 'AL', 'CR']
phases = ['FCC_A1', 'FCC_L12']
therm = MulticomponentThermodynamics('paper_scripts//NiCrAl.tdb', elements, phases, drivingForceMethod='approximate')

t0, tf, steps = 1e-2, 1e6, 1000
model = PrecipitateModel(t0, tf, steps, elements=['AL', 'CR'])
print(model.minNucleateDensity)

model.setInitialComposition([0.098, 0.083])
model.setInterfacialEnergy(0.012)

T = 1073
model.setTemperature(T)

a = 0.352e-9
Va = a**3
Vb = Va
atomsPerCell = 4
model.setVaAlpha(Va, atomsPerCell)
model.setVaBeta(Vb, atomsPerCell)
print(model.VmAlpha)

#model.setNucleationSite('dislocations')
#model.setNucleationDensity(dislocationDensity=1e30)
model.setNucleationSite('bulk')
#model.setNucleationDensity(bulkN0 = 1e30)
model.setNucleationDensity(bulkN0 = 9.1717e28)

model.setThermodynamics(therm, removeCache=True)

model.solve(verbose=True, vIt=100)
model.save('paper_scripts//outputs//NiCrAl_TC_nocache')

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(2, 2, figsize=(8,8))

modelLoad = PrecipitateModel.load('paper_scripts//outputs//NiCrAl_TC_nocache.npz')

modelLoad.plot(axes[0,0], 'Precipitate Density', linewidth=2)
axes[0,0].set_ylim([1e10, 1e27])
axes[0,0].set_yscale('log')

#modelLoad.plot(axes[0,1], 'Composition', linewidth=2)
modelLoad.plot(axes[0,1], 'Volume Fraction', linewidth=2)
axes[0,1].set_ylim([6e-3, 2e0])
axes[0,1].set_yscale('log')
modelLoad.plot(axes[1,0], 'Average Radius', color='C0', linewidth=2, label='Avg. R')
modelLoad.plot(axes[1,0], 'Critical Radius', color='C1', linewidth=2, linestyle='--', label='R*')
axes[1,0].legend(loc='upper left')
modelLoad.plot(axes[1,1], 'Size Distribution Density', linewidth=2, color='C0')
axes[1,0].set_ylim([8e-11, 2e-6])
axes[1,0].set_yscale('log')

axes[0,0].set_xlim([4e-3, 2e6])
axes[1,0].set_xlim([4e-3, 2e6])
axes[0,1].set_xlim([4e-3, 2e6])

fig.tight_layout()
plt.show()