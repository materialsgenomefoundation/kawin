from kawin.thermo import BinaryThermodynamics
from kawin.precipitation import PrecipitateModel, VolumeParameter
import numpy as np
from kawin.solver.Iterators import _startIteration
import matplotlib.pyplot as plt

def MidPointIterator(f, t, X_old, dtfunc, dtmin, dtmax, correctdXdt, flattenX, unflattenX):
    dt, dXdt = _startIteration(t, X_old, f, dtfunc, dtmin, dtmax)

    X_flat = flattenX(X_old)
    correctdXdt(dt/2, X_old, dXdt)
    xmid = unflattenX(X_flat + dt/2 * flattenX(dXdt), X_old)

    dxdt_mid = f(t+dt/2, xmid)
    correctdXdt(dt, X_old, dxdt_mid)

    return unflattenX(X_flat + dt * flattenX(dxdt_mid), X_old), dt
    

therm = BinaryThermodynamics('examples//AlScZr.tdb', ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='tangent')
therm.setGuessComposition(0.24)
model = PrecipitateModel()

model.setInitialComposition(4e-3)
model.setTemperature(450 + 273.15)
model.setInterfacialEnergy(0.1)
Diff = lambda x, T: 0.0768 * np.exp(-242000 / (8.314 * T))
model.setDiffusivity(Diff)
a = 0.405e-9        #Lattice parameter
model.setVolumeAlpha(a**3, VolumeParameter.ATOMIC_VOLUME, 4)
model.setVolumeBeta(a**3, VolumeParameter.ATOMIC_VOLUME, 4)

model.setNucleationDensity(grainSize = 1, dislocationDensity = 1e15)
model.setNucleationSite('dislocations')

model.setThermodynamics(therm, addDiffusivity=False)

model.solve(500*3600, solverType=MidPointIterator, verbose=True, vIt=5000)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

model.plot(axes[0,0], 'Precipitate Density')
model.plot(axes[0,1], 'Volume Fraction')
model.plot(axes[1,0], 'Average Radius', label='Average Radius')
model.plot(axes[1,0], 'Critical Radius', label='Critical Radius')
axes[1,0].legend()
model.plot(axes[1,1], 'Size Distribution Density')

fig.tight_layout()
plt.show()

