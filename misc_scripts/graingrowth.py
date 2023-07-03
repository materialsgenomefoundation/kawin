import numpy as np
import matplotlib.pyplot as plt
from kawin.PopulationBalance import PopulationBalanceModel

class GrainGrowth:
    def __init__(self, cMin = 1e-10, cMax = 1e-8, bins = 150, minBins = 100, maxBins = 200):
        self.pbm = PopulationBalanceModel(cMin, cMax, bins, minBins, maxBins)
        self.gbe = 1
        self.M = 1
        self.alpha = 1

    def setGrainBoundaryEnergy(self, gbe):
        self.gbe = gbe

    def setGrainBoundaryMobility(self, M):
        self.M = M

    def setAlpha(self, alpha):
        self.alpha = alpha

    @property
    def Rcr(self):
        return self.pbm.SecondMoment() / self.pbm.FirstMoment()

    @property
    def Rm(self):
        return np.cbrt(self.pbm.ThirdMoment() / self.pbm.ZeroMoment())

    def grainGrowth(self):
        return self.alpha * self.M * self.gbe * (1 / self.Rcr - 1 / self.pbm.PSDbounds)

    def Normalize(self):
        self.pbm.PSD *= 1 / self.pbm.ThirdMoment()

    def Update(self, dt):
        totalT = 0
        while totalT < dt:
            growthRate = self.grainGrowth()
            subDT = self.pbm.getDTEuler(dt, growthRate, 1e-5, 0)
            if totalT + subDT > dt:
                subDT = dt - totalT
            self.pbm.UpdateEuler(subDT, growthRate)
            totalT += subDT
            self.Normalize()

    def plotDistribution(self, ax, *args, **kwargs):
        self.pbm.plotCurve(ax, *args, **kwargs)

    def plotDistributionDensity(self, ax, *args, **kwargs):
        self.pbm.PlotDistributionDensity(ax, *args, **kwargs)

ggm = GrainGrowth(1e-10, 1e-8)
ggm.setGrainBoundaryEnergy(0.5)
ggm.setGrainBoundaryMobility(1e-14)
ggm.setAlpha(1)
ggm.pbm.PSD = 1e3 * np.exp(-(ggm.pbm.PSDsize - 0.5e-8)**2 / (0.5e-9)**2)
ggm.Normalize()

t0 = 0
tf = 1e4
i = 0

fig, ax = plt.subplots(1, 1)
ggm.plotDistributionDensity(ax)
ax.legend()
plt.show()
t = np.linspace(0, 1e4, 1000)
dt = t[1] - t[0]
r = np.zeros(1000)
r[0] = ggm.Rm
for i in range(1, len(t)):
    ggm.Update(dt)
    r[i] = ggm.Rm
    if i % 100 == 0:
        print(t[i])
        fig, ax = plt.subplots(1, 2)
        ggm.plotDistributionDensity(ax[0])
        ax[1].plot(t[:i], r[:i]**2)
        plt.show()
