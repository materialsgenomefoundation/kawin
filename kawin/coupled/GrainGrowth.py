import numpy as np
import matplotlib.pyplot as plt
from kawin.PopulationBalance import PopulationBalanceModel

class GrainGrowthModel:
    '''
    Model for grain growth that can be coupled with the KWN model to account for Zener pinning

    Parameters
    ----------
    cMin : float (optional)
        Minimum grain size (default is 1e-10)
    cMax : float (optional)
        Maximum grain size (default is 1e-8)
    bins : int (optional)
        Initial bins (default is 150)
    minBins : int (optional)
        Minimum number of bins (default is 100)
    maxBins : int (optional)
        Maximum number of bins (default is 200)
    '''
    def __init__(self, cMin = 1e-10, cMax = 1e-8, bins = 150, minBins = 100, maxBins = 200):
        self.pbm = PopulationBalanceModel(cMin, cMax, bins, minBins, maxBins)
        self.gbe = 1
        self.M = 1
        self.alpha = 1

        self.m, self.K = {'all': 1}, {'all': 4/3}

        self.time = np.array([0])
        self.avgR = np.array([0])

    def setGrainBoundaryEnergy(self, gbe):
        '''
        Parameters
        ----------
        gbe : float
            Grain boundary energy
        '''
        self.gbe = gbe

    def setGrainBoundaryMobility(self, M):
        '''
        Parameters
        ----------
        M : float
            Grain boundary mobility
        '''
        self.M = M

    def setAlpha(self, alpha):
        '''
        Correction factor

        Parameters
        ----------
        alpha : float
        '''
        self.alpha = alpha

    def setZenerParameters(self, m, K, phase='all'):
        '''
        Parameters for defining zener radius

        Zener radius is defined as
        Rz = K * r / f^m

        Parameters
        ----------
        m : float
            Exponential factor for volume fraction
        K : float
            Scaling factor
        phase : str (optional)
            Precipitate phase to apply parameters to
            Default is 'all'
        '''
        self.m[phase] = m
        self.K[phase] = K

    def LoadDistribution(self, data):
        '''
        Creates a particle size distribution from a set of data
        
        Parameters
        ----------
        data : array of floats
            Array of data to be inserted into PSD
        '''
        self.pbm.PSD, self.pbm.PSDbounds = np.histogram(data, self.pbm.PSDbounds)
        self.pbm.PSD = self.pbm.PSD.astype('float')
        self.Normalize()

    def LoadDistributionFunction(self, function):
        '''
        Creates a particle size distribution from a function

        Parameters
        ----------
        function : function
            Takes in R and returns density
        '''
        self.pbm.PSD = function(self.pbm.PSDsize)
        self.Normalize()

    @property
    def Rcr(self):
        '''
        Critical radius, grains larger than Rcr will growth while smaller grains will shrink

        Critical radius is defined so that the volume will be constant when applying the growth rate
        '''
        return self.pbm.SecondMoment() / self.pbm.FirstMoment()

    @property
    def Rm(self):
        '''
        Mean radius
        '''
        return np.cbrt(self.pbm.ThirdMoment() / self.pbm.ZeroMoment())

    def grainGrowth(self):
        '''
        Grain growth model
        dRi/dt = alpha * M * gbe * (1/Rcr - 1/Ri)
        '''
        return self.alpha * self.M * self.gbe * (1 / self.Rcr - 1 / self.pbm.PSDbounds)

    def Normalize(self):
        '''
        Normalize PSD to have a third moment of 1

        Ideally, this isn't needed since the grain growth model accounts for constant volume
            But numerical errors will lead to small changes in volume over time
        '''
        self.pbm.PSD *= 1 / self.pbm.ThirdMoment()

    def zenerRadius(self, model, i):
        '''
        Calculates Zener radius from model at iteration i

        Parameters
        ----------
        model : KWNmodel object
        i : iteration
        '''
        z = np.zeros(len(model.phases))
        for p in range(len(model.phases)):
            phaseName = model.phases[p] if model.phases[p] in self.m else 'all'
            if model.avgR[p,i] > 0:
                z[p] += np.power(model.betaFrac[p,i], self.m[phaseName]) / (self.K[phaseName] * model.avgR[p,i])
        return np.sum(z)

    def constrainedGrowth(self, growthRate, z = 0):
        '''
        Constrain growth rate due to zener pinning

        Parameters
        ----------
        growthRate : array
            Growth rate for grain sizes
        z : float (optional)
            Zener radius, default is 0, which will not change the growth rate
        '''
        upper = growthRate + self.alpha * self.M * self.gbe * z
        lower = growthRate - self.alpha * self.M * self.gbe * z
        growIndices = lower > 0
        dissolveIndices = upper < 0
        cG = np.zeros(len(growthRate))
        cG[growIndices] = lower[growIndices]
        cG[dissolveIndices] = upper[dissolveIndices]
        return cG

    def Update(self, dt, z = 0):
        '''
        Updates particle size distribution over a fixed increment

        Parameters
        ----------
        dt : float
            Time increment to update by
        '''
        totalT = 0
        while totalT < dt:
            growthRate = self.grainGrowth()
            growthRate = self.constrainedGrowth(growthRate, z)
            subDT = self.pbm.getDTEuler(dt - totalT, growthRate, 1e-5, 0)
            self.pbm.UpdateEuler(subDT, growthRate)
            totalT += subDT
            self.Normalize()

    def couplingFunction(self, model, dt, i):
        '''
        Coupling function to add to the KWN model
        '''
        z = self.zenerRadius(model, i)
        self.Update(dt, z)
        self.time = np.append(self.time, [model.time[i]])
        self.avgR = np.append(self.avgR, [self.Rm])

    def plotDistribution(self, ax, *args, **kwargs):
        '''
        Plots particle size distribution

        Parameters
        ----------
        ax : matplotlib axes
        '''
        self.pbm.plotCurve(ax, *args, **kwargs)

    def plotDistributionDensity(self, ax, *args, **kwargs):
        '''
        Plots particle size distribution density

        Parameters
        ----------
        ax : matplotlib axes
        '''
        self.pbm.PlotDistributionDensity(ax, *args, **kwargs)

    def plotRadiusvsTime(self, ax, *args, **kwargs):
        ax.plot(self.time, self.avgR, *args, **kwargs)