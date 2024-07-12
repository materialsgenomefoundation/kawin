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

gg = GrainGrowthModel(1e-10, 2.5e-5, solverType=SolverType.RK4)
#data = np.random.lognormal(mean=np.log(1e-5), sigma=0.2, size=10000)
func = lambda x: 1/(x*0.2*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-np.log(5e-6))**2 / (2*0.2**2))
gg.setGrainBoundaryMobility(1e-16)
#gg.LoadDistribution(data)
gg.LoadDistributionFunction(func)
gg.pbm.enableRecording()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
gg.plotDistribution(axes[1,1])
ylim = axes[1,1].get_ylim()

#gg.solve(9e5, solverType=gg.solverType, verbose=True, vIt=100)
#gg.plotDistribution(axes[1,1])
#gg.plotRadiusvsTime(axes[1,0])
#gg.reset()

model.addCouplingModel(gg)
model.solve(9e5, solverType=SolverType.RK4, verbose=True, vIt=1000)



model.plot(axes[0,0], 'Precipitate Density')
model.plot(axes[0,1], 'Volume Fraction')
axL = axes[0,1].twinx()
model.plot(axL, 'Average Radius')
gg.plotRadiusvsTime(axes[1,0])
gg.plotDistribution(axes[1,1])
axes[1,1].set_ylim(ylim)

fig, ax = plt.subplots(1,1)
print(gg.pbm._recordedBins.shape)
x = np.linspace(0, 1, gg.pbm.maxBins)
y = np.linspace(0, 1, len(gg.pbm._recordedBins))
X, Y = np.meshgrid(x, y)
ax.pcolormesh(X, Y, gg.pbm._recordedPSD)

plt.show()