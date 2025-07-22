import numpy as np
import matplotlib.pyplot as plt
from kawin.GenericModel import GenericModel
from kawin.solver import rk4Iterator
from kawin.precipitation import PrecipitateModel
from kawin.precipitation import PopulationBalanceModel
from kawin.PlotUtils import _get_axis
from kawin.precipitation.PopulationBalance import plotPSD, plotPDF, plotCDF
from kawin.precipitation.Plot import _get_time_axis

class GrainGrowthModel(GenericModel):
    '''
    Model for grain growth that can be coupled with the KWN model to account for Zener pinning

    Following implentation described in
    K. W, J. Jeppsson and P. Mason, J. Phase Equilib. Diffus. 43 (2022) 866-875

    Parameters
    ----------
    gbe: float (optional)
        Grain boundary energy
        Default = 0.5
    M: float (optional)
        Grain boundary mobility
        Default = 1e-14
    alpha: float (optional)
        Correction factor (for when fitting data to the model)
        Default = 1
    m: dict[str, float] (optional)
        Factor related to spatial distribution of precipitates
        Maps a phase name to the factor
        Default = 1 for all phases
    K: dict[str, float] (optional)
        Factor related to spatial distribution of precipitates
        Maps a phase name to the factor
        Default = 4/3 for all phases
    iterator: iterator function (optional)
        Iterator function for model
        Default is rk4Iterator
    pbmKwargs: dict[str, int | float] (optional)
        Arguments to define population balance model
        Default is {cMin = 1e-10, cMax = 1e-8}
    '''
    def __init__(self, gbe=0.5, M=1e-14, alpha=1, m={}, K={}, iterator=rk4Iterator, pbmKwargs={'cMin': 1e-10, 'cMax': 1e-8}):
        super().__init__()
        self.pbm = PopulationBalanceModel(**pbmKwargs)
        self._oldPSD, self._oldPSDbounds = np.array(self.pbm.PSD), np.array(self.pbm.PSDbounds)

        #Model parameters - these are values taken from the paper as general default values
        self.gbe = gbe      #Grain boundary energy (J/m2)
        self.M = M      #Grain boundary mobility (m4/J-s)
        self.alpha = alpha      #Correction factor (for when fitting data to the model)

        #Factors related to spatial distribution of precipitates
        self.m = m
        self.K = K   
        self._mdefault = 1
        self._Kdefault = 4/3

        self.iterator = iterator

        self.maxDissolution = 1e-6

        self.reset()

    def loadDistribution(self, data):
        '''
        Creates a particle size distribution from a set of data
        
        Parameters
        ----------
        data : array of floats
            Array of data to be inserted into PSD
        '''
        self.pbm.reset()
        self.pbm.PSD, self.pbm.PSDbounds = np.histogram(data, self.pbm.PSDbounds)
        self.pbm.PSD = self.pbm.PSD.astype('float')
        self.normalize()
        self.avgR[0] = self.Rm(self.pbm.PSD)
        self._oldPSD, self._oldPSDbounds = np.array(self.pbm.PSD), np.array(self.pbm.PSDbounds)
        self.dissolutionIndex = self.pbm.getDissolutionIndex(self.maxDissolution, 0)

    def loadDistributionFunction(self, function):
        '''
        Creates a particle size distribution from a function

        Parameters
        ----------
        function : function
            Takes in R and returns density
        '''
        self.pbm.reset()
        self.pbm.PSD = function(self.pbm.PSDsize)
        self.normalize()
        self.avgR[0] = self.Rm(self.pbm.PSD)
        self._oldPSD, self._oldPSDbounds = np.array(self.pbm.PSD), np.array(self.pbm.PSDbounds)
        self.dissolutionIndex = self.pbm.getDissolutionIndex(self.maxDissolution, 0)

    def reset(self):
        '''
        Resets model with initially loaded grain size distribution
        '''
        super().reset()
        self.time = np.zeros(1)
        self.avgR = np.zeros(1)
        self._z = 0
        self._growthRate = np.zeros(len(self.pbm.PSDbounds))
        self.pbm.reset()
        self.pbm.PSD, self.pbm.PSDbounds = np.array(self._oldPSD), np.array(self._oldPSDbounds)
        self.dissolutionIndex = 0

    def Rcr(self, x):
        '''
        Critical radius, grains larger than Rcr will growth while smaller grains will shrink

        Critical radius is defined so that the volume will be constant when applying the growth rate

        Parameters
        ----------
        x : np.array
            Grain size distribution corresponding to GrainGrowthModel.pbm.PSDbounds
        '''
        return self.pbm.secondMoment(N=x) / self.pbm.firstMoment(N=x)
    
    def Rm(self, x):
        '''
        Mean radius

        Parameters
        ----------
        x : np.array
            Grain size distribution corresponding to GrainGrowthModel.pbm.PSDbounds
        '''
        return np.cbrt(self.pbm.thirdMoment(N=x) / self.pbm.zeroMoment(N=x))
    
    def grainGrowth(self, x):
        '''
        Grain growth model
        dRi/dt = alpha * M * gbe * (1/Rcr - 1/Ri)

        Parameters
        ----------
        x : np.array
            Grain size distribution corresponding to GrainGrowthModel.pbm.PSDbounds
        '''
        return self.alpha * self.M * self.gbe * (1 / self.Rcr(x) - 1 / self.pbm.PSDbounds)
    
    def normalize(self):
        '''
        Normalize PSD to have a third moment of 1

        Ideally, this isn't needed since the grain growth model accounts for constant volume
            But numerical errors will lead to small changes in volume over time
        '''
        self.pbm.PSD *= 1 / self.pbm.thirdMoment()

    def constrainedGrowth(self, growthRate, z = 0):
        '''
        Constrain growth rate due to zener pinning

        The growth rate given the zener radius is defined by:
            dR/dt = alpha * M * gbe * ((1/Rcr - 1/Ri) +/- 1/Rz)
            Where 1/Rz is added if (1/Rcr - 1/Ri) + 1/Rz < 0 (inhibits grain dissolution)
            And   1/Rz is subtracted in (1/Rcr - 1/Ri) - 1/Rz) > 0 (inhibits grain growth)
            And   dR/dt is 0 for Ri between these two limits
        
        Note: Rather than Rz (zener radius), we use z here which represents the drag force
            But these are related by z = 1/Rz

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
    
    def getCurrentX(self):
        '''
        Returns current time and grain size distribution
        '''
        return [self.pbm.PSD]
    
    def getdXdt(self, t, x):
        '''
        Returns dn_i/dt for the grain size distribution

        Steps:
            1. Get grain growth rate and corrected it with zener drag force
            2. Get dn_i/dt from the PBM given the Eulerian implementation
        '''
        self._growthRate = self.grainGrowth(x[0])
        self._growthRate = self.constrainedGrowth(self._growthRate, self._z)
        return [self.pbm.getdXdtEuler(self._growthRate, 0, 0, x[0])]

    def correctdXdt(self, dt, x, dXdt):
        '''
        Corrects dn_i/dt with the new time step
        '''
        dXdt[0] = self.pbm.correctdXdtEuler(dt, self._growthRate, 0, 0, x[0])

    def getDt(self, dXdt):
        '''
        Calculated a suitable dt with the growth rate and new time step
        We'll limit the max time step to the remaining time for solving
        '''
        return self.pbm.getDTEuler(self.finalTime - self.time[-1], self._growthRate, self.dissolutionIndex)

    def postProcess(self, time, x):
        '''
        Sets grain size distribution to x and record time and average grain size

        Steps:
            1. Set grain size distribution
            2. Adjust PSD size classes
            3. Remove grains below the dissolution threshold
            4. Normalize grain size distribution to 1 (should be a tiny correction factor due to step 3)
            5. Record time and average grain size
        '''
        super().postProcess(time, x)
        self.pbm.updatePBMEuler(time, x[0])
        self.pbm.adjustSizeClassesEuler(True)
        self.dissolutionIndex = self.pbm.getDissolutionIndex(self.maxDissolution, 0)
        self.normalize()
        self.time = np.append(self.time, time)
        self.avgR = np.append(self.avgR, self.Rm(self.pbm.PSD))
        self.updateCoupledModels()
        return [self.pbm.PSD], False
    
    def printHeader(self):
        '''
        Header string before solving
        '''
        print('Iteration\tTime(s)\t\tSim Time(s)\tGrain Size (um)')
    
    def printStatus(self, iteration, modelTime, simTimeElapsed):
        '''
        Status string that prints every n iteration
        '''
        print('{}\t\t{:.1e}\t\t{:.1f}\t\t{:.3e}'.format(iteration, modelTime, simTimeElapsed, self.avgR[-1]*1e6))

    def computeZenerRadius(self, model: PrecipitateModel, x=None):
        '''
        Gets zener radius/drag force from PrecipitateModel and PSD defined by x

        Drag force is defined as z_j = f_j^m_j / (K_j * avgR_j)
            Where f_j is volume fraction for phase j
            And   avgR_j is average radius for phase j
        The total drag force is the sum of z_j over all the phases

        Parameters
        ----------
        model : PrecpitateModel
        x : list[np.array]
            List of particle size distributions in model
        '''
        if x is None:
            x = model.getCurrentX()

        z = np.zeros(len(model.phases))
        for p in range(len(model.phases)):
            volRatio = model.matrix.volume.Vm / model.precipitates[p].volume.Vm
            m = self.m.get(model.phases[p], self._mdefault)
            K = self.K.get(model.phases[p], self._Kdefault)
            Ntot = model.PBM[p].zeroMoment(N=x[p])
            RadSum = model.PBM[p].moment(order=1, N=x[p])
            fBeta = np.amin([volRatio * model.precipitates[p].nucleation.volumeFactor * model.PBM[p].thirdMoment(N=x[p]), 1])
            avgR = 0 if Ntot == 0 else RadSum / Ntot

            if avgR > 0:
                z[p] += np.power(fBeta, m) / (K * avgR)
            
        self._z = np.sum(z)

    def updateCoupledModel(self, model):
        '''
        Computes zener radius/drag force from the PrecipitateModel,
        Then solves the grain growth model with the time step of the PrecipitateModel

        Parameters
        ----------
        model : PrecpitateModel
        '''
        self.computeZenerRadius(model)
        self.solve(model.data.time[model.data.n] - model.data.time[model.data.n-1], iterator=self.iterator)

def _plot_grain_growth_generic(model: GrainGrowthModel, func, ax=None, *args, **kwargs):
    ax = _get_axis(ax)
    func(model.pbm, ax=ax)
    ax.set_xlabel('Grain Radius (m)')
    return ax

def plotGrainPSD(model: GrainGrowthModel, ax=None, *args, **kwargs):
    '''
    Plots grain size distribution

    Parameters
    ----------
    model: GrainGrowthModel
    ax : matplotlib axes
    '''
    return _plot_grain_growth_generic(model, plotPSD, ax=ax, *args, **kwargs)

def plotGrainPDF(model: GrainGrowthModel, ax=None, *args, **kwargs):
    '''
    Plots grain size distribution density

    Parameters
    ----------
    model: GrainGrowthModel
    ax : matplotlib axes
    '''
    return _plot_grain_growth_generic(model, plotPDF, ax=ax, *args, **kwargs)

def plotGrainCDF(model: GrainGrowthModel, ax=None, *args, **kwargs):
    '''
    Plots cumulative grain size distribution

    Parameters
    ----------
    model: GrainGrowthModel
    ax : matplotlib axes
    '''
    return _plot_grain_growth_generic(model, plotCDF, ax=ax, *args, **kwargs)

def plotRadiusvsTime(model: GrainGrowthModel, timeUnits='s', ax=None, *args, **kwargs):
    '''
    Plots average grain size vs time

    Parameters
    ----------
    model: GrainGrowthModel
    timeUnits: str
        's', 'min' or 'hr'
    ax : matplotlib axes
    '''
    ax = _get_axis(ax)
    time = model.time
    timeScale, timeLabel, bounds = _get_time_axis(time, timeUnits)
    ax.plot(timeScale*time, model.avgR, *args, **kwargs)
    ax.set_xlabel(timeLabel)
    ax.set_xlim(bounds)
    ax.set_ylabel('Grain Radius (m)')
    ax.set_ylim(bottom=0)
    return ax