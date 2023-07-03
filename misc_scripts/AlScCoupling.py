from kawin.KWNEuler import PrecipitateModel
from kawin.thermo.Thermodynamics import BinaryThermodynamics
import matplotlib.pyplot as plt
import numpy as np

therm = BinaryThermodynamics('database//AlScZr.tdb', ['AL', 'SC'], ['FCC_A1', 'AL3SC'])
therm.setGuessComposition(0.24)
model = PrecipitateModel(0, 250*3600, 1e4, linearTimeSpacing=False)

model.setInitialComposition(0.002)
model.setTemperature(400+273.15)
model.setInterfacialEnergy(0.1)
model.setNucleationDensity(bulkN0 = 1e30)

model.bulkN0array = np.zeros(len(model.time))

def func(model, dt, i):
    model.gamma[0] = 0.105 - model.xComp[i] * 10

model.addCouplingFunction('func', func)

Va = (0.405e-9)**3
Vb = (0.4196e-9)**3
model.setVaAlpha(Va, 4)
model.setVaBeta(Vb, 4)

model.setThermodynamics(therm)

diff = lambda x, T: 1.9e-4 * np.exp(-164000 / (8.314*T)) 
model.setDiffusivity(diff)

model.solve(verbose=True, vIt=5000)
    
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

model.plot(axes[0,0], 'Precipitate Density')
model.plot(axes[0,1], 'Volume Fraction')
model.plot(axes[1,0], 'Average Radius', label='Average Radius')
model.plot(axes[1,0], 'Critical Radius', label='Critical Radius')
#axes[1,1].plot(model.time, model.bulkN0array)
axes[1,0].legend()
#sm.plotStrength(axes[1,1], model, plotContributions=True)

fig.tight_layout()

plt.show()