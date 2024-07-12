import numpy as np
import matplotlib.pyplot as plt
from kawin.precipitation.coupling import GrainGrowthModel
from kawin.solver import SolverType
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.thermo import BinaryThermodynamics
from kawin.GenericModel import Coupler

np.random.seed(5)

class CustomCoupledModel(Coupler):
    def getdXdt(self, t, x):
        self.models[1].computeZenerRadiusByN(self.models[0], x[0])
        return super().getdXdt(t, x)

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

gg = GrainGrowthModel(1e-10, 1e-5)
gg.setGrainBoundaryMobility(1e-15)
data = np.random.lognormal(mean=np.log(1e-6), sigma=0.5, size=10000)
gg.LoadDistribution(data)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
gg.plotDistribution(axes[1,1])
ylim = axes[1,1].get_ylim()

gc = CustomCoupledModel([model, gg])
gc.solve(9e5, solverType=SolverType.RK4, verbose = True, vIt = 500)

model.plot(axes[0,0], 'Precipitate Density')
model.plot(axes[0,1], 'Volume Fraction')
axL = axes[0,1].twinx()
model.plot(axL, 'Average Radius')
gg.plotRadiusvsTime(axes[1,0])
gg.plotDistribution(axes[1,1])
axes[1,1].set_ylim(ylim)

plt.show()

x = np.linspace(0, 1, model.PBM[0].maxBins)
y = np.linspace(0, 1, len(model.PBM[0]._recordedBins))
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X, Y, model.PBM[0]._recordedPSD)
plt.show()