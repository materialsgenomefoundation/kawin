import numpy as np
from kawin.precipitation.KWNBase_new import PrecipitateBase
from kawin.precipitation.PopulationBalance_new import PopulationBalanceModel
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
import copy
import csv
from itertools import zip_longest
import time

class PrecipitateModel (PrecipitateBase):
    '''
    Euler implementation of KWN model

    Parameters
    ----------
    phases : list (optional)
        Precipitate phases (array of str)
        If only one phase is considered, the default is ['beta']
    elements : list (optional)
        Solute elements in system
        Note: order of elements must correspond to order of elements set in Thermodynamics module
        If binary system, then defualt is ['solute']
    '''
    def __init__(self, phases = ['beta'], elements = ['solute']):
        super().__init__(phases, elements)

        if self.numberOfElements == 1:
            self._growthRate = self._growthRateBinary
            self._Beta = self._BetaBinary1
        else:
            self._growthRate = self._growthRateMulti
            self._Beta = self._BetaMulti

    def _resetArrays(self):
        '''
        Resets and initializes arrays for all variables

        In addition to PrecipitateBase, the equilibrium aspect ratio area and population balance models are created here
        '''
        super()._resetArrays()
        self.PBM = [PopulationBalanceModel() for p in self.phases]

        self.RdrivingForceIndex = np.zeros(len(self.phases), dtype=np.int32)
        self.dissolutionIndex = np.zeros(len(self.phases), dtype=np.int32)
        
    def reset(self):
        '''
        Resets model results
        '''
        super().reset()

        for i in range(len(self.phases)):
            self.PBM[i].reset()

    def setPBMParameters(self, cMin = 1e-10, cMax = 1e-9, bins = 150, minBins = 100, maxBins = 200, adaptive = True, phase = None):
        '''
        Sets population balance model parameters for each phase

        Parameters
        ----------
        cMin : float
            Minimum bin size
        cMax : float
            Maximum bin size
        bins : int
            Initial number of bins
        minBins : int
            Minimum number of bins - will not be used if adaptive = False
        maxBins : int
            Maximum number of bins - will not be used if adaptive = False
        adaptive : bool
            Sets adaptive bin sizes - bins may still change upon nucleation
        phase : str
            Phase to consider (will set all phases if phase = None or 'all')
        '''
        if phase is None or phase == 'all':
            for p in range(len(self.phases)):
                self.PBM[p] = PopulationBalanceModel(cMin, cMax, bins, minBins, maxBins)
                self.PBM[p].setAdaptiveBinSize(adaptive)
        else:
            index = self.phaseIndex(phase)
            self.PBM[index] = PopulationBalanceModel(cMin, cMax, bins, minBins, maxBins)
            self.PBM[index].setAdaptiveBinSize(adaptive)

    def setPSDrecording(self, record = True, phase = 'all'):
        '''
        Sets recording parameters for PSD of specified phase

        Parameters
        ----------
        record : bool (optional)
            Whether to record PSD, defaults to True
        phase : str (optional)
            Precipitate phase to record for
            Defaults to 'all', which will apply to all precipitate phases
        '''
        if phase is None or phase == 'all':
            for p in self.phases:
                index = self.phaseIndex(p)
                self.PBM[index].setRecording(record)
        else:
            index = self.phaseIndex(phase)
            self.PBM[index].setRecording(record)

    def saveRecordedPSD(self, filename, compressed = True, phase = 'all'):
        '''
        Saves recorded PSD in npz format

        Parameters
        ----------
        filename : str
            File name to save to
            Note: the phase name will be added to the filename if all phases are being saved
        compressed : bool (optional)
            Whether to save in compressed npz format
            Defualts to True
        phase : str (optional)
            Phase to save PSD for
            Defaults to 'all', which will save a file for each phase
        '''
        if phase is None or phase == 'all':
            for p in self.phases:
                index = self.phaseIndex(p)
                self.PBM[index].saveRecordedPSD(filename + '_' + p, compressed)
        else:
            index = self.phaseIndex(phase)
            self.PBM[index].saveRecordedPSD(filename, compressed)

    def loadParticleSizeDistribution(self, data, phase = None):
        '''
        Loads particle size distribution for specified phase

        Parameters
        ----------
        data : array
            Array of data containing precipitate sizes
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.PBM[index].LoadDistribution(data)

    def particleRadius(self, phase = None):
        '''
        Returns PSD bounds of given phase

        Parameters
        ----------
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        return self.PBM[index].PSDbounds
        
    def particleGibbs(self, radius = None, phase = None):
        '''
        Returns Gibbs Thomson contribution of a particle given its radius
        
        Parameters
        ----------
        radius : array (optional)
            Precipitate radaii (defaults to None, which will use boundaries
                of the size classes of the precipitate PSD)
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        if radius is None:
            index = self.phaseIndex(phase)
            radius = self.PBM[index].PSDbounds
        return super().particleGibbs(radius, phase)

    def PSD(self, phase = None):
        '''
        Returns frequency of particle size distribution of given phase

        Parameters
        ----------
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        return self.PBM[index].PSD
    
    def createLookup(self):
        '''
        This creates a lookup table mapping the particle size classes to the interfacial composition
        '''
        #RdrivingForceIndex will find the index of the largest particle size class where the precipitate is unstable
        #This is determined by the interfacial composition function, where it should return -1 or None
        #All compositions from the PSD bounds will be set to the compositions just above RdrivingForceLimit
        #This is just to allow for particles to dissolve instead of pile up in the smallest bin
        self.RdrivingForceIndex = np.zeros(len(self.phases), dtype=np.int32)

        #Keep as separate arrays so that number of PSD classes can change within precipitate phases
        self.PSDXalpha = []
        self.PSDXbeta = []
        
        xEqAlpha = np.zeros((1, len(self.phases), self.numberOfElements))
        xEqBeta = np.zeros((1, len(self.phases), self.numberOfElements))
        T = self._currY[self.TEMPERATURE][0]
        for p in range(len(self.phases)):
            #Interfacial compositions at equilibrium (planar interface)
            xAResult, xBResult = self.interfacialComposition[p](T, 0)
            if xAResult == -1 or xAResult is None:
                xEqAlpha[0,p,0] = 0
                xEqBeta[0,p,0] = 0
            else:
                xEqAlpha[0,p,0] = xAResult
                xEqBeta[0,p,0] = xBResult

            #Interfacial compositions at each size class in PSD
            self.PSDXalpha.append(np.zeros(self.PBM[p].bins + 1))
            self.PSDXbeta.append(np.zeros(self.PBM[p].bins + 1))

            self.PSDXalpha[p], self.PSDXbeta[p] = self.interfacialComposition[p](T, self.particleGibbs(self.PBM[p].PSDbounds, self.phases[p]))
            self.RdrivingForceIndex[p] = np.argmax(self.PSDXalpha[p] != -1)-1
            self.RdrivingForceIndex[p] = 0 if self.RdrivingForceIndex[p] < 0 else self.RdrivingForceIndex[p]
            self.RdrivingForceLimit[p] = self.PBM[p].PSDbounds[self.RdrivingForceIndex[p]]

            #Sets particle radii smaller than driving force limit to driving force limit composition
            #If RdrivingForceIndex is at the end of the PSDX arrays, then no precipitate in the size classes of the PSD is stable
            #This can occur in non-isothermal situations where the temperature gets too high
            if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p]):
                self.PSDXalpha[p][:self.RdrivingForceIndex[p]+1] = self.PSDXalpha[p][self.RdrivingForceIndex[p]+1]
                self.PSDXbeta[p][:self.RdrivingForceIndex[p]+1] = self.PSDXbeta[p][self.RdrivingForceIndex[p]+1]
            else:
                self.PSDXalpha[p] = np.zeros(self.PBM[p].bins + 1)
                self.PSDXbeta[p] = np.zeros(self.PBM[p].bins + 1)

        return xEqAlpha, xEqBeta
            
    def setup(self):
        '''
        Sets up additional variables in addition to PrecipitateBase

        Sets up additional outputs, population balance models, equilibrium aspect ratio and equilibrium compositions
        '''
        if self._isSetup:
            return

        super().setup()

        #Equilibrium aspect ratio and PBM setup
        #If calculateAspectRatio is True, then use strain energy to calculate aspect ratio for each size class in PSD
        #Else, then use aspect ratio defined in shape factors
        self.eqAspectRatio = [None for p in range(len(self.phases))]
        for p in range(len(self.phases)):
            self.PBM[p].reset()

            if self.calculateAspectRatio[p]:
                self.eqAspectRatio[p] = self.strainEnergy[p].eqAR_bySearch(self.PBM[p].PSDbounds, self.gamma[p], self.shapeFactors[p])
                arFunc = lambda R, p1=p : self._interpolateAspectRatio(R, p1)
                self.shapeFactors[p].setAspectRatio(arFunc)
            else:
                self.eqAspectRatio[p] = self.shapeFactors[p].aspectRatio(self.PBM[p].PSDbounds)

        self._currY = [np.array([self.varList[i][self.n]]) for i in range(self.NUM_TERMS)]
        self._currY[self.TIME] = np.array([self.time[self.n]])
        self._currY[self.TEMPERATURE] = np.array([self.getTemperature(self.time[self.n])])
        
        #Setup interfacial composition
        if self.numberOfElements == 1:
            self.xEqAlpha[self.n], self.xEqBeta[self.n] = self.createLookup()
        else:
            self.PSDXalpha = [None for p in range(len(self.phases))]
            self.PSDXbeta = [None for p in range(len(self.phases))]

            #Set first index of eq composition
            for p in range(len(self.phases)):
                #Use arbitrary dg, R and gE since only the eq compositions are needed here
                _, _, _, xEqAlpha, xEqBeta = self.interfacialComposition[p](self.xComp[self.n], self.temperature[self.n], 0, 1, 0)
                if xEqAlpha is not None:
                    self.xEqAlpha[self.n,p] = xEqAlpha
                    self.xEqBeta[self.n,p] = xEqBeta
        
        x = [self.PBM[p].PSD for p in range(len(self.phases))]
        self._calcDrivingForce(self.time[self.n], x)
        self._growthRate()
        self._calcNucleationRate(self.time[self.n], x)
        for i in range(self.NUM_TERMS):
            self.varList[i][self.n] = self._currY[i][0]
    
    def _interpolateAspectRatio(self, R, p):
        '''
        Linear interpolation between self.eqAspectRatio and self.PBM[p].PSDbounds

        Parameters
        ----------
        R : float
            Equivalent spherical radius
        p : int
            Phase index
        '''
        return np.interp(R, self.PBM[p].PSDbounds, self.eqAspectRatio[p])

    def getDt(self, dXdt):
        #Start test dt at 0.01 or previous dt
        i = self.n
        dtPrev = 0.01 if self.n == 0 else self.time[i] - self.time[i-1]
        #dtMax = 100
        dtMax = self.deltaTime - self.time[i]
        
        dtAll = [dtMax]
        if self.checkPSD:
            dtPBM = []
            if i > 1 and self.temperature[i] == self.temperature[i-1]:
                dtPBM = [self.PBM[p].getDTEuler(dtMax, self.growth[p], self.dissolutionIndex[p]) for p in range(len(self.phases))]
            else:
                dtPBM.append(dtPrev)
            dtPBM = np.amin(dtPBM)
            dtAll.append(dtPBM)
        
        if self.checkNucleation and i > 1:
            dtNuc = dtMax * np.ones(len(self.phases))
            nRateCurr = self.nucRate[i]
            nRatePrev = self.nucRate[i-1]
            for p in range(len(self.phases)):
                if nRateCurr[p] > self.minNucleationRate and nRatePrev[p] > self.minNucleationRate and nRatePrev[p] != nRateCurr[p]:
                    dtNuc[p] = self.maxNucleationRateChange * dtPrev / np.abs(np.log10(nRatePrev[p] / nRateCurr[p]))
            dtNuc = np.amin(dtNuc)
            dtAll.append(dtNuc)

        #Temperature change constraint
        if self.checkTemperature and i > 1:
            Tchange = self.temperature[i] - self.temperature[i-1]
            dtTemp = dtMax
            if Tchange > self.maxNonIsothermalDT:
                dtTemp = self.maxNonIsothermalDT * dtPrev / Tchange
                dtTemp = np.amin([dt, dtTemp])
            dtAll.append(dtTemp)

        if self.checkRcrit and i > 1:
            dtRad = dtMax * np.ones(len(self.phases))
            if not all((self.Rcrit[i-1,:] == 0) & (self.Rcrit[i,:] - self.Rcrit[i-1,:] == 0) & (self.dGs[i,:] <= 0)):
                indices = (self.Rcrit[i-1,:] > 0) & (self.Rcrit[i,:] - self.Rcrit[i-1,:] != 0) & (self.dGs[i,:] > 0)
                dtRad[indices] = self.maxRcritChange * dtPrev / np.abs((self.Rcrit[i,:][indices] - self.Rcrit[i-1,:][indices]) / self.Rcrit[i-1,:][indices])
            dtRad = np.amin(dtRad)
            dtAll.append(dtRad)

        if self.checkVolumePre:
            dV = np.zeros(len(self.phases))
            for p in range(len(self.phases)):
                #Calculate estimate volume change based off growth rate and nucleated particles
                #TODO: account for non-spherical precipitates
                dVi = self.PBM[p].PSD * self.PBM[p].PSDsize**2 * 0.5 * (self.growth[p][1:] + self.growth[p][:-1])
                dVi[dVi < 0] = 0
                dV = self.VmAlpha / self.VmBeta[p] * (self.GB[p].areaFactor * np.sum(dVi) + self.GB[p].volumeFactor * self.nucRate[i,p] * self.Rad[i,p]**3)

            dtVol = dtMax * np.ones(len(self.phases))
            for p in range(len(self.phases)):
                if dV != 0:
                    dtVol[p] = self.maxVolumeChange / (2 * np.abs(dV))
            dtVol = np.amin(dtVol)
            dtAll.append(dtVol)

        dt = np.amin(dtAll)
        if dt == dtMax:
            dt = dtPrev
        return dt
    
    def _processX(self, x):
        for p in range(len(self.phases)):
            x[p][:self.dissolutionIndex[p]] = 0
    
    def _calcNucleationRate(self, t, x):
        super()._calcNucleationRate(t, x)
        for p in range(len(self.phases)):
            #If parent phases exists, then calculate the number of potential nucleation sites on the parent phase
            #This is the number of lattice sites on the total surface area of the parent precipitate
            nucleationSites = np.sum([4 * np.pi * self.PBM[p2].SecondMomentFromN(x[p2]) * (self.avo / self.VmBeta[p2])**(2/3) for p2 in self.parentPhases[p]])

            if self.GB[p].nucleationSiteType == GBFactors.BULK:
                #bulkPrec = np.sum([self.GB[p2].volumeFactor * self.PBM[p2].ThirdMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.BULK])
                #nucleationSites += self.bulkN0 - bulkPrec * (self.avo / self.VmAlpha)
                bulkPrec = np.sum([self.PBM[p2].ZeroMomentFromN(x[p2]) for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.BULK])
                nucleationSites += self.bulkN0 - bulkPrec
            elif self.GB[p].nucleationSiteType == GBFactors.DISLOCATION:
                bulkPrec = np.sum([self.PBM[p2].FirstMomentFromN(x[p2]) for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.DISLOCATION])
                nucleationSites += self.dislocationN0 - bulkPrec * (self.avo / self.VmAlpha)**(1/3)
            elif self.GB[p].nucleationSiteType == GBFactors.GRAIN_BOUNDARIES:
                boundPrec = np.sum([self.GB[p2].gbRemoval * self.PBM[p2].SecondMomentFromN(x[p2]) for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.GRAIN_BOUNDARIES])
                nucleationSites += self.GBareaN0 - boundPrec * (self.avo / self.VmAlpha)**(2/3)
            elif self.GB[p].nucleationSiteType == GBFactors.GRAIN_EDGES:
                edgePrec = np.sum([np.sqrt(1 - self.GB[p2].GBk**2) * self.PBM[p2].FirstMomentFromN(x[p2]) for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.GRAIN_EDGES])
                nucleationSites += self.GBedgeN0 - edgePrec * (self.avo / self.VmAlpha)**(1/3)
            elif self.GB[p].nucleationSiteType == GBFactors.GRAIN_CORNERS:
                cornerPrec = np.sum([self.PBM[p2].ZeroMomentFromN(x[p2]) for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.GRAIN_CORNERS])
                nucleationSites += self.GBcornerN0 - cornerPrec
               
            if nucleationSites < 0:
                nucleationSites = 0
            self._currY[self.NUC_RATE][0,p] *= nucleationSites
    
    def _calcMassBalance(self, t, x):
        fBeta = np.zeros((1,len(self.phases)))
        fConc = np.zeros((1, len(self.phases),self.numberOfElements))
        precDens = np.zeros((1,len(self.phases)))
        avgR = np.zeros((1,len(self.phases)))
        avgAR = np.zeros((1,len(self.phases)))
        xComp = np.zeros((1,self.numberOfElements))
        
        for p in range(len(self.phases)):
            Ntot = self.PBM[p].ZeroMomentFromN(x[p])
            RadSum = self.PBM[p].MomentFromN(x[p], 1)
            ARsum = self.PBM[p].WeightedMomentFromN(x[p], 0, self.shapeFactors[p].aspectRatio(self.PBM[p].PSDsize))
            fBeta[0,p] = np.amin([self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].ThirdMomentFromN(x[p]), 1])

            volFactor = self.VmAlpha / self.VmBeta[p]
            if self.infinitePrecipitateDiffusion[p]:
                compAvg = 0.5 * (self.PSDXbeta[p][:-1] + self.PSDXbeta[p][1:])
                if self.numberOfElements == 1:
                    fConc[0,p] = volFactor * self.GB[p].volumeFactor * self.PBM[p].WeightedMomentFromN(x[p], 3, compAvg)
                else:
                    for e in range(self.numberOfElements):
                        fConc[0,p,e] = volFactor * self.GB[p].volumeFactor * self.PBM[p].WeightedMomentFromN(x[p], 3, compAvg[:,e])
            else:
                dt = t - self.t[self.n]
                dR = self.PSD[p].PSDbounds[1:] - self.PSD[p].PSDbounds[:-1]
                y = volFactor * self.GB[p].areaFactor * np.sum(self.PBM[p].PSDbounds[1:]**2 * self.PBM[p]._fv[1:] * dt * self.PSDXbeta[p][1:] * dR)
                fConc[0,p] = self.fConc[self.n,p] + y
                

            if Ntot > 0:
                avgR[0,p] = RadSum / Ntot
                precDens[0,p] = Ntot
                avgAR[0,p] = ARsum / Ntot
            else:
                avgR[0,p] = 0
                precDens[0,p] = 0
                avgAR[0,p] = 0

            #Not sure if needed
            if self.betaFrac[self.n,p] == 1:
                fBeta[0,p] = 1

        if np.sum(fBeta[0]) < 1:
            #print(fConc, fBeta)
            xComp[0] = (self.xComp[0] - np.sum(fConc[0], axis=1)) / (1 - np.sum(fBeta[0]))
            xComp[0,xComp[0] < 0] = self.minComposition

        self._currY[self.VOL_FRAC] = fBeta
        self._currY[self.FCONC] = fConc
        self._currY[self.PREC_DENS] = precDens
        self._currY[self.R_AVG] = avgR
        self._currY[self.AR_AVG] = avgAR
        self._currY[self.COMPOSITION] = xComp

    def _getdXdt(self, t, x):
        return [self.PBM[p].getdXdtEuler(self.growth[p], self._currY[self.NUC_RATE][0,p], self._currY[self.R_NUC][0,p], x[p], self._firstIt) for p in range(len(self.phases))]

    def _singleGrowthBinary(self, p):
        '''
        Calculates growth rate for a single phase
        This is separated from _growthRateBinary since it's used in _calculatePSD

        Matrix/precipitate composition are not calculated here since it's
        already calculated in createLookup
        '''
        xComp = self._currY[self.COMPOSITION][0]
        T = self._currY[self.TEMPERATURE][0]
        growthRate = np.zeros(self.PBM[p].bins + 1)
        #If no precipitates are stable, don't calculate growth rate and set PSD to 0
        #This should represent dissolution of the precipitates
        if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p]):
            superSaturation = (xComp[0] - self.PSDXalpha[p]) / (self.VmAlpha * self.PSDXbeta[p] / self.VmBeta[p] - self.PSDXalpha[p])
            growthRate = self.shapeFactors[p].kineticFactor(self.PBM[p].PSDbounds) * self.Diffusivity(xComp[0], T) * superSaturation / (self.effDiffDistance(superSaturation) * self.PBM[p].PSDbounds)

        return growthRate
    
    def _growthRateBinary(self):
        '''
        Determines current growth rate of all particle size classes in a binary system
        '''
        #Update equilibrium interfacial compositions
        #This will be override if createLookup is called
        T = self._currY[self.TEMPERATURE]
        self.dTemp += T - self.temperature[self.n]
        if np.abs(self.dTemp) > self.maxTempChange:
            xEqAlpha, xEqBeta = self.createLookup()
        else:
            xEqAlpha, xEqBeta = np.array([self.xEqAlpha[self.n]]), np.array([self.xEqBeta[self.n]])
            self.dTemp = 0
        self._currY[self.EQ_COMP_ALPHA] = xEqAlpha
        self._currY[self.EQ_COMP_BETA] = xEqBeta
        
        #growthRate = np.zeros((len(self.phases), self.bins + 1))
        growthRate = []
        for p in range(len(self.phases)):
            growthRate.append(self._singleGrowthBinary(p))
            
        self.growth = growthRate

    def _singleGrowthMulti(self, p):
        '''
        Calculates growth rate for a single phase
        This is separated from _growthRateMulti since it's used in _calculatePSD

        This will also calculate the matrix/precipitate composition 
        for the radius in the PSD as well as equilibrium (infinite radius)
        '''
        xComp = self._currY[self.COMPOSITION][0]
        dGs = self._currY[self.DRIVING_FORCE][0]
        T = self._currY[self.TEMPERATURE][0]
        growth, xAlpha, xBeta, xEqAlpha, xEqBeta = self.interfacialComposition[p](xComp, T, dGs[p] * self.VmBeta[p], self.PBM[p].PSDbounds, self.particleGibbs(phase=self.phases[p]))

        #If two-phase equilibrium not found, two possibilities - precipitates are unstable or equilibrium calculations didn't converge
        if growth is None:
            #If driving force is negative, then precipitates are unstable
            if dGs[p] < 0:
                #Completely reset the PBM, including bounds and number of bins
                #In case nucleation occurs again, the PBM will be at a good length scale
                self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                xEqAlpha = np.zeros(self.numberOfElements)
                xEqBeta = np.zeros(self.numberOfElements)
                growthRate = np.zeros(self.PBM[p].bins + 1)
            #Else, equilibrium did not converge and just use previous values
            #Only the growth rate needs to be updated, since all other terms are previous
            #Also revert the PSD in case this function was called to adjust for the new PSD bins
            else:
                growthRate = self.growth[p]
        else:
            #Update interfacial composition for each precipitate size
            self.PSDXalpha[p] = xAlpha
            self.PSDXbeta[p] = xBeta

            #Add shape factor to growth rate - will need to add effective diffusion distance as well
            growthRate = self.shapeFactors[p].kineticFactor(self.PBM[p].PSDbounds) * growth

        return growthRate, xEqAlpha, xEqBeta
    
    def _growthRateMulti(self):
        '''
        Determines current growth rate of all particle size classes in a multicomponent system
        '''
        xEqAlpha = np.zeros((1,len(self.phases), self.numberOfElements))
        xEqBeta = np.zeros((1,len(self.phases), self.numberOfElements))
        growthRate = []
        for p in range(len(self.phases)):
            growthRate_p, xEqAlpha_p, xEqBeta_p = self._singleGrowthMulti(p)
            growthRate.append(growthRate_p)
            xEqAlpha[0,p] = xEqAlpha_p
            xEqBeta[0,p] = xEqBeta_p
        self._currY[self.EQ_COMP_ALPHA] = xEqAlpha
        self._currY[self.EQ_COMP_BETA] = xEqBeta
        self.growth = growthRate

    def _updateParticleSizeDistribution(self, t, x):
        for p in range(len(self.phases)):
            self.PBM[p].UpdatePBMEuler(t, x[p])
            change, addedIndices = self.PBM[p].adjustSizeClassesEuler(all(self.growth[p] < 0))
            if change:
                if self.calculateAspectRatio[p]:
                    self.eqAspectRatio[p] = self.strainEnergy[p].eqAR_bySearch(self.PBM[p].PSDbounds, self.gamma[p], self.shapeFactors[p])
                else:
                    self.eqAspectRatio[p] = self.shapeFactors[p].aspectRatio(self.PBM[p].PSDbounds)

                self.growth[p] = np.zeros(len(self.PBM[p].PSDbounds))
                if self.numberOfElements == 1:
                    if addedIndices is None:
                        #This is very slow to do
                        self.createLookup()
                    else:
                        self.PSDXalpha[p] = np.concatenate((self.PSDXalpha[p], np.zeros(self.PBM[p].bins+1 - len(self.PSDXalpha[p]))))
                        self.PSDXbeta[p] = np.concatenate((self.PSDXbeta[p], np.zeros(self.PBM[p].bins+1 - len(self.PSDXbeta[p]))))
                        self.PSDXalpha[p][addedIndices:], self.PSDXbeta[p][addedIndices:] = self.interfacialComposition[p](self.temperature[self.n], self.particleGibbs(self.PBM[p].PSDbounds[addedIndices:], self.phases[p]))
                self._growthRate()
            self.PBM[p].PSD[:self.RdrivingForceIndex[p]+1] = 0
            self.PBM[p].PSD[self.PBM[p].PSDsize < self.minRadius] = 0
            self.dissolutionIndex[p] = self.PBM[p].getDissolutionIndex(self.maxDissolution, self.RdrivingForceIndex[p])
            self.PBM[p].PSD[:self.dissolutionIndex[p]] = 0

                