import numpy as np
import matplotlib.pyplot as plt
from kawin.precipitation import PopulationBalanceModel
from kawin.solver import SolverType
from kawin.GenericModel import GenericModel
from kawin.precipitation.Plot import getTimeAxis

class GrainGrowthModel(GenericModel):
    '''
    Model for grain growth that can be coupled with the KWN model to account for Zener pinning

    Following implentation described in
    K. W, J. Jeppsson and P. Mason, J. Phase Equilib. Diffus. 43 (2022) 866-875

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
    def __init__(self, cMin = 1e-10, cMax = 1e-8, bins = 150, minBins = 100, maxBins = 200, solverType = SolverType.RK4):
        super().__init__()
        self.pbm = PopulationBalanceModel(cMin, cMax, bins, minBins, maxBins)
        self._oldPSD, self._oldPSDbounds = np.array(self.pbm.PSD), np.array(self.pbm.PSDbounds)

        #Model parameters - these are values taken from the paper as general default values
        self.gbe = 0.5      #Grain boundary energy (J/m2)
        self.M = 1e-14      #Grain boundary mobility (m4/J-s)
        self.alpha = 1      #Correction factor (for when fitting data to the model)
        self.m, self.K = {'all': 1}, {'all': 4/3}   #Factors related to spatial distribution of precipitates

        self.solverType = solverType

        self.maxDissolution = 1e-6

        self.reset()

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
        self.pbm.reset()
        self.pbm.PSD, self.pbm.PSDbounds = np.histogram(data, self.pbm.PSDbounds)
        self.pbm.PSD = self.pbm.PSD.astype('float')
        self.Normalize()
        self.avgR[0] = self.Rm(self.pbm.PSD)
        self._oldPSD, self._oldPSDbounds = np.array(self.pbm.PSD), np.array(self.pbm.PSDbounds)
        self.dissolutionIndex = self.pbm.getDissolutionIndex(self.maxDissolution, 0)

    def LoadDistributionFunction(self, function):
        '''
        Creates a particle size distribution from a function

        Parameters
        ----------
        function : function
            Takes in R and returns density
        '''
        self.pbm.reset()
        self.pbm.PSD = function(self.pbm.PSDsize)
        self.Normalize()
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
        return self.pbm.SecondMomentFromN(x) / self.pbm.FirstMomentFromN(x)
    
    def Rm(self, x):
        '''
        Mean radius

        Parameters
        ----------
        x : np.array
            Grain size distribution corresponding to GrainGrowthModel.pbm.PSDbounds
        '''
        return np.cbrt(self.pbm.ThirdMomentFromN(x) / self.pbm.ZeroMomentFromN(x))
    
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
    
    def Normalize(self):
        '''
        Normalize PSD to have a third moment of 1

        Ideally, this isn't needed since the grain growth model accounts for constant volume
            But numerical errors will lead to small changes in volume over time
        '''
        self.pbm.PSD *= 1 / self.pbm.ThirdMoment()

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
        self.pbm.UpdatePBMEuler(time, x[0])
        self.pbm.adjustSizeClassesEuler(True)
        self.dissolutionIndex = self.pbm.getDissolutionIndex(self.maxDissolution, 0)
        #self.pbm.PSD[:self.dissolutionIndex] = 0
        self.Normalize()
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

    def computeZenerRadius(self, model):
        '''
        Gets zener radius/drag force from PrecipitateModel

        Drag force is defined as z_j = f_j^m_j / (K_j * avgR_j)
            Where f_j is volume fraction for phase j
            And   avgR_j is average radius for phase j
        The total drag force is the sum of z_j over all the phases

        Parameters
        ----------
        model : PrecpitateModel
        '''
        z = np.zeros(len(model.phases))
        for p in range(len(model.phases)):
            phaseName = model.phases[p] if model.phases[p] in self.m else 'all'
            #if model.avgR[model.n, p] > 0:
            #    z[p] += np.power(model.betaFrac[model.n, p], self.m[phaseName]) / (self.K[phaseName] * model.avgR[model.n, p])
            
            if model.pData.Ravg[model.pData.n,p] > 0:
                z[p] += np.power(model.pData.volFrac[model.pData.n,p], self.m[phaseName]) / (self.K[phaseName] * model.pData.Ravg[model.pData.n,p])
        self._z = np.sum(z)

    def computeZenerRadiusByN(self, model, x):
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
        z = np.zeros(len(model.phases))
        for p in range(len(model.phases)):
            #volRatio = model.VmAlpha / model.VmBeta[p]
            volRatio = model.matrixParameters.volume.Vm / model.precipitateParameters[p].volume.Vm
            phaseName = model.phases[p] if model.phases[p] in self.m else 'all'
            Ntot = model.PBM[p].ZeroMomentFromN(x[p])
            RadSum = model.PBM[p].MomentFromN(x[p], 1)
            #fBeta = np.amin([volRatio * model.GB[p].volumeFactor * model.PBM[p].ThirdMomentFromN(x[p]), 1])
            fBeta = np.amin([volRatio * model.precipitateParameters[p].nucleation.volumeFactor * model.PBM[p].ThirdMomentFromN(x[p]), 1])
            avgR = 0 if Ntot == 0 else RadSum / Ntot

            if avgR > 0:
                z[p] += np.power(fBeta, self.m[phaseName]) / (self.K[phaseName] * avgR)
            
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
        self.solve(model.pData.time[model.pData.n] - model.pData.time[model.pData.n-1], solverType=self.solverType)

    def plotDistribution(self, ax, *args, **kwargs):
        '''
        Plots particle size distribution

        Parameters
        ----------
        ax : matplotlib axes
        '''
        self.pbm.PlotCurve(ax, *args, **kwargs)
        ax.set_xlabel('Grain Radius (m)')

    def plotDistributionDensity(self, ax, *args, **kwargs):
        '''
        Plots particle size distribution density

        Parameters
        ----------
        ax : matplotlib axes
        '''
        self.pbm.PlotDistributionDensity(ax, *args, **kwargs)
        ax.set_xlabel('Grain Radius (m)')

    def plotRadiusvsTime(self, ax, bounds = None, timeUnits = 's', *args, **kwargs):
        '''
        Plots average grain radius over time

        Parameters
        ----------
        ax : matplotlib axes
        '''
        timeScale, timeLabel, bounds = getTimeAxis(self.time, timeUnits, bounds)
        ax.plot(self.time*timeScale, self.avgR, *args, **kwargs)
        ax.set_xlabel(timeLabel)
        ax.set_ylabel('Grain Radius (m)')
        ax.set_ylim([0, 1.1*np.amax(self.avgR)])
        ax.set_xlim(bounds)