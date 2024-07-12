import numpy as np
import matplotlib.pyplot as plt
from kawin.precipitation.coupling import GrainGrowthModel
from kawin.solver import SolverType
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.thermo import BinaryThermodynamics

np.random.seed(5)

therm = BinaryThermodynamics('examples//AlScZr.tdb', ['AL', 'SC'], ['FCC_A1', 'AL3SC'])
therm.setGuessComposition(0.24)

model = PrecipitateModel()

model.setInitialComposition(0.002)
model.setTemperature(400+273.15)
model.setInterfacialEnergy(0.1)

model.PBM[0].enableRecording()

Va = (0.405e-9)**3
Vb = (0.4196e-9)**3
model.setVolumeAlpha(Va, VolumeParameter.ATOMIC_VOLUME, 4)
model.setVolumeBeta(Vb, VolumeParameter.ATOMIC_VOLUME, 4)

diff = lambda x, T: 1.9e-4 * np.exp(-164000 / (8.314*T)) 
model.setDiffusivity(diff)

model.setThermodynamics(therm, addDiffusivity=False)

gg = GrainGrowthModel(1e-10, 1e-5, solverType=SolverType.RK4)
data = np.random.lognormal(size=10000) * 1e-6
gg.setGrainBoundaryMobility(1e-16)
gg.LoadDistribution(data)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
gg.plotDistribution(axes[1,1])
ylim = axes[1,1].get_ylim()


times = np.concatenate(([0], np.logspace(np.log10(9e1), np.log10(9e5), 10)))
for i in range(len(times)-1):
    model.solve(times[i+1] - times[i], verbose=True, vIt=1000)
    gg.computeZenerRadius(model)
    gg.solve(times[i+1] - times[i], verbose=True, vIt=1000)



model.plot(axes[0,0], 'Precipitate Density')
model.plot(axes[0,1], 'Volume Fraction')
axL = axes[0,1].twinx()
model.plot(axL, 'Average Radius')
gg.plotRadiusvsTime(axes[1,0])
gg.plotDistribution(axes[1,1])
axes[1,1].set_ylim(ylim)

plt.show()