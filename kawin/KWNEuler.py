import numpy as np
import matplotlib.pyplot as plt
from kawin.KWNBase import PrecipitateBase
from kawin.PopulationBalance import PopulationBalanceModel
from kawin.EffectiveDiffusion import effectiveDiffusionDistance, noDiffusionDistance
from kawin.GrainBoundaries import GBFactors
import copy

class PrecipitateModel (PrecipitateBase):
    '''
    Euler implementation of the KWN model designed for binary systems

    Parameters
    ----------
    t0 : float
        Initial time in seconds
    tf : float
        Final time in seconds
    steps : int
        Number of time steps
    rMin : float
        Lower bound of particle size distribution
    rMax : float
        Upper bound of particle size distribution
    bins : int
        Number of size classes in particle size distribution
    phases : list (optional)
        Precipitate phases (array of str)
        If only one phase is considered, the default is ['beta']
    linearTimeSpacing : bool (optional)
        Whether to have time increment spaced linearly or logarithimically
        Defaults to False
    elements : list (optional)
        Solute elements in system
        Note: order of elements must correspond to order of elements set in Thermodynamics module
        If binary system, then defualt is ['solute']
    '''
    def __init__(self, t0, tf, steps, rMin = 1e-10, rMax = 1e-7, bins = 100, phases = ['beta'], linearTimeSpacing = False, elements = ['solute']):
        #Initialize base class
        super().__init__(t0, tf, steps, phases, linearTimeSpacing, elements)

        if self.numberOfElements == 1:
            self._growthRate = self._growthRateBinary
            self._Beta = self._BetaBinary
        else:
            self._growthRate = self._growthRateMulti
            self._Beta = self._BetaMulti

        #Bounds of the bins in PSD
        self.bins = int(bins)
        self.PBM = []
        self.PBM = [PopulationBalanceModel(rMin, rMax, self.bins, True) for p in self.phases]

        #Index of particle size classes which below, precipitates are unstable
        self.RdrivingForceIndex = np.zeros(len(self.phases), dtype=np.int32)

        #Adaptive time stepping
        self._postTimeIncrementCheck = self._noPostCheckDT

    def reset(self):
        '''
        Resets model results
        '''
        super().reset()
        #Bounds of the bins in PSD
        for i in range(len(self.phases)):
            self.PBM[i].reset()

    def save(self, filename, compressed = False):
        '''
        Save results into a numpy .npz format

        TODO: add CSV support

        Parameters
        ----------
        filename : str
        compressed : bool
            If true, will save compressed .npz format
        '''
        variables = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements', \
            'time', 'xComp', 'Rcrit', 'Gcrit', 'Rad', 'avgR', 'avgAR', 'betaFrac', 'nucRate', 'precipitateDensity', 'dGs', 'xEqAlpha', 'xEqBeta']
        vDict = {v: getattr(self, v) for v in variables}
        for i in range(len(self.phases)):
            vDict['PSDdata'+str(i)] = [self.PBM[i].min, self.PBM[i].max, self.PBM[i].bins]
            vDict['PSD' + str(i)] = self.PBM[i].PSD
            vDict['PSDsize' + str(i)] = self.PBM[i].PSDsize
            vDict['PSDbounds' + str(i)] = self.PBM[i].PSDbounds
        if compressed:
            np.savez_compressed(filename, **vDict)
        else:
            np.savez(filename, **vDict)

    def load(filename):
        '''
        Loads data

        Parameters
        ----------
        filename : str

        Returns
        -------
        PrecipitateModel object
            Note: this will only contain model outputs which can be used for plotting
        '''
        data = np.load(filename)
        setupVars = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements']

        #Input arbitrary values for PSD parameters (rMin, rMax, bins) since this will be changed shortly after
        model = PrecipitateModel(data['t0'], data['tf'], data['steps'], 0, 1, 1, data['phases'], data['linearTimeSpacing'], data['elements'])
        for i in range(len(model.phases)):
            PSDvars = ['PSDdata' + str(i), 'PSD' + str(i), 'PSDsize' + str(i), 'PSDbounds' + str(i)]
            setupVars = np.concatenate((setupVars, PSDvars))
            model.PBM[i] = PopulationBalanceModel(data[PSDvars[0]][0], data[PSDvars[0]][1], int(data[PSDvars[0]][2]), True)
            model.PBM[i].PSD = data[PSDvars[1]]
            model.PBM[i].PSDsize = data[PSDvars[2]]
            model.PBM[i].PSDbounds = data[PSDvars[3]]
        for d in data:
            if d not in setupVars:
                setattr(model, d, data[d])
        return model

    def adaptiveTimeStepping(self, adaptive = True):
        '''
        Sets if adaptive time stepping is used

        Parameters
        ----------
        adaptive : bool (optional)
            Defaults to True
        '''
        super().adaptiveTimeStepping(adaptive)
        if adaptive:
            self._postTimeIncrementCheck = self._postCheckDT
        else:
            self._postTimeIncrementCheck = self._noPostCheckDT

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
 
    def createLookup(self, i = 0):
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
        
        for p in range(len(self.phases)):
            #Interfacial compositions at equilibrium (planar interface)
            self.xEqAlpha[p,i], self.xEqBeta[p,i] = self.interfacialComposition[p](self.T[i], 0)
            if self.xEqAlpha[p,i] == -1 or self.xEqAlpha[p,i] is None:
                self.xEqAlpha[p,i] = 0
                self.xEqBeta[p,i] = 0

            #Interfacial compositions at each size class in PSD
            self.PSDXalpha.append(np.zeros(self.PBM[p].bins + 1))
            self.PSDXbeta.append(np.zeros(self.PBM[p].bins + 1))

            for n in range(self.PBM[p].bins + 1):
                self.PSDXalpha[p][n], self.PSDXbeta[p][n] = self.interfacialComposition[p](self.T[i], self.particleGibbs(self.PBM[p].PSDbounds[n], self.phases[p]))

                if self.PSDXalpha[p][n] == -1 or self.PSDXalpha[p][n] is None:
                    self.RdrivingForceLimit[p] = self.PBM[p].PSDbounds[n]
                    self.RdrivingForceIndex[p] = n
                    
            #Sets particle radii smaller than driving force limit to driving force limit composition
            #If RdrivingForceIndex is at the end of the PSDX arrays, then no precipitate in the size classes of the PSD is stable
            #This can occur in non-isothermal situations where the temperature gets too high
            if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p]):
                self.PSDXalpha[p][:self.RdrivingForceIndex[p]+1] = self.PSDXalpha[p][self.RdrivingForceIndex[p]+1]
                self.PSDXbeta[p][:self.RdrivingForceIndex[p]+1] = self.PSDXbeta[p][self.RdrivingForceIndex[p]+1]
            else:
                self.PSDXalpha[p] = np.zeros(self.PBM[p].bins + 1)
                self.PSDXbeta[p] = np.zeros(self.PBM[p].bins + 1)
            
    def setup(self):
        super().setup()
        #Only create lookup table for binary system
        if self.numberOfElements == 1:
            self.createLookup(0)
        else:
            self.PSDXalpha = [None for p in range(len(self.phases))]
            self.PSDXbeta = [None for p in range(len(self.phases))]

            #Set first index of eq composition
            for p in range(len(self.phases)):
                #Use arbitrary dg, R and gE since only the eq compositions are needed here
                _, _, _, xEqAlpha, xEqBeta = self.interfacialComposition[p](self.xComp[0], self.T[0], 0, 1, 0)
                if xEqAlpha is not None:
                    self.xEqAlpha[p,0] = xEqAlpha
                    self.xEqBeta[p,0] = xEqBeta
            

    def _iterate(self, i):
        '''
        Iteration function
        '''
        #Nucleation and growth rate are independent of time increment
        #They can be calculated first and used to determine the time increment for numerical stability
        self._nucleate(i)
        self._growthRate(i)
        self._timeIncrementCheck(i)
        
        postDTCheck = False
        while not postDTCheck:
            dt = self.time[i] - self.time[i-1]
            self._setNucleateRadius(i, dt)
            self._calculatePSD(i, dt)
            self._massBalance(i)

            if i < self.steps - 1:
                postDTCheck = self._postTimeIncrementCheck(i)
            else:
                postDTCheck = True

    def _noCheckDT(self, i):
        '''
        Function if adaptive time stepping is not used
        Will calculated growth rate since it is done in the _checkDT function (not a good way of doing this, but works for now)
        '''
        return

    def _checkDT(self, i):
        '''
        Checks max growth rate and updates dt correspondingly
        '''
        #PBM growth rate constraint - flux leaving a bin cannot be higher than half the frequency in the bin
        #Ignore small radaii (growth rate can be really high, so this prevents dt from being too small)
        #The radaii that are ignore are determined such that the volume CDF up to that radaii is a certain fraction of the total volume
        #We also want to ignore radaii where the PSD is less than a certain threshold (this is useful in multi-phase systems where a phase has completely dissolved)
        dissFrac = [self.maxDissolution * self.PBM[p].ThirdMoment() for p in range(len(self.phases))]
        dissIndices = [np.amax([np.argmax(self.PBM[p].CumulativeMoment(3) > dissFrac[p]), self.RdrivingForceIndex[p]]) for p in range(len(self.phases))]
        growthFilter = [self.growth[p][dissIndices[p]:-1][self.PBM[p].PSD[dissIndices[p]:] > 1] for p in range(len(self.phases))]
        
        gFilter = []
        for g in growthFilter:
            #Possibly temporary!!!
            #If temperature increases too fast and reaches a temp where precipitates are unstable,
            #then the growth rate may by highly negative rather than being 0 with the PSD being reset
            #In this case, only consider the phase for time constraint if isothermal or at least one growth rate is positive
            if len(g) > 0 and (np.amax(g) > 0 or self.T[i] == self.T[i-1]):
                gFilter = np.concatenate([gFilter, g])
        #growthFilter = np.concatenate(([g for g in growthFilter if (len(g) > 0) and (np.amax(g) > 0)]))
        if len(gFilter) == 0:
            dt = self.time[i] - self.time[i-1]
        else:
            maxGrowth = np.amax(np.abs(gFilter))
            dt = (self.PBM[0].PSDbounds[1] - self.PBM[0].PSDbounds[0]) / (2 * maxGrowth)

        if i > 1:
            dtPrev = self.time[i-1] - self.time[i-2]

            #Nucleation rate constraint
            dtNuc = dt * np.ones(len(self.phases))
            for p in range(len(self.phases)):
                if self.nucRate[p,i] > 0 and self.nucRate[p,i-1] > 0 and self.nucRate[p,i-1] != self.nucRate[p,i]:
                    dtNuc[p] = self.maxNucleationRateChange * dtPrev / np.abs(np.log10(self.nucRate[p,i-1] / self.nucRate[p,i]))
            dt = np.amin(dtNuc)

            #Temperature change constraint
            Tchange = self.T[i] - self.T[i-1]
            dtTemp = dt
            if Tchange > self.maxNonIsothermalDT:
                dtTemp = self.maxNonIsothermalDT * (self.time[i] - self.time[i-1]) / Tchange
                dt = np.amin([dt, dtTemp])

        if dt < self.time[i] - self.time[i-1]:
            self._divideTimestep(i, dt)

    def _noPostCheckDT(self, i):
        '''
        Function if no adaptive time stepping is used, no need to do anything in this function
        '''
        return True

    def _postCheckDT(self, i):
        '''
        If adaptive time step is used, this checks new values at iteration i
        and compares with simulation contraints

        If contraints are not met, then remove current values and divide time step
        '''
        #Only perform checks in non-isothermal situations
        if np.abs(self.T[i] - self.T[i-1]) > 1:
            return True

        #Composition and volume change are checks in absolute changes
        #This prevents any unneccessary reduction in time increments for dilute solutions, or
        #if there is a long incubation time until nucleations starts occuring
        volChange = np.abs(self.betaFrac[:,i] - self.betaFrac[:,i-1])
        #If current volume fraction is 0, then ignore (either precipitation has not occured or precipitates has dissolved)
        volChange[self.betaFrac[:,i] == 0] = 0
        volCheck = np.amax(volChange) < self.maxVolumeChange

        if self.numberOfElements == 1:
            compCheck = np.abs(self.xComp[i] - self.xComp[i-1]) < self.maxCompositionChange
        else:
            compCheck = np.amax(np.abs(self.xComp[i,:] - self.xComp[i-1,:])) < self.maxCompositionChange

        checks = [volCheck, compCheck]

        #If any test fails, then reset iteration and divide time increment
        if not any(checks):
            if self.numberOfElements == 1:
                self.xComp[i] = 0
            else:
                self.xComp[i,:] = 0

            self.Rcrit[:,i] = 0
            self.Gcrit[:,i] = 0
            self.Rad[:,i] = 0
            self.avgR[:,i] = 0
            self.avgAR[:,i] = 0
            self.nucRate[:,i] = 0
            self.precipitateDensity[:,i] = 0
            self.betaFrac[:,i] = 0
            self.dGs[:,i] = 0

            self.prevFConc[0] = copy.copy(self.prevFConc[1])

            self._divideTimestep(i, np.amax([(self.time[i] - self.time[i-1]) / 2, self.minDTFraction * (self.tf - self.t0)]))

            return False
        else:
            return True
    
    def _nucleate(self, i):
        '''
        Calculates the nucleation rate at current timestep
        This can be done before the initial time increment checks are performed
        '''
        for p in range(len(self.phases)):
            #If parent phases exists, then calculate the number of potential nucleation sites on the parent phase
            #This is the number of lattice sites on the total surface area of the parent precipitate
            nucleationSites = np.sum([4 * np.pi * self.PBM[p2].SecondMoment() * (self.avo / self.VmBeta[p2])**(2/3) for p2 in self.parentPhases[p]])

            if self.GB[p].nucleationSiteType == GBFactors.BULK:
                #bulkPrec = np.sum([self.GB[p2].volumeFactor * self.PBM[p2].ThirdMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.BULK])
                #nucleationSites += self.bulkN0 - bulkPrec * (self.avo / self.VmAlpha)
                bulkPrec = np.sum([self.PBM[p2].ZeroMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.BULK])
                nucleationSites += self.bulkN0 - bulkPrec
            elif self.GB[p].nucleationSiteType == GBFactors.DISLOCATION:
                bulkPrec = np.sum([self.PBM[p2].FirstMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.DISLOCATION])
                nucleationSites += self.dislocationN0 - bulkPrec * (self.avo / self.VmAlpha)**(1/3)
            elif self.GB[p].nucleationSiteType == GBFactors.GRAIN_BOUNDARIES:
                boundPrec = np.sum([self.GB[p2].gbRemoval * self.PBM[p2].SecondMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.GRAIN_BOUNDARIES])
                nucleationSites += self.GBareaN0 - boundPrec * (self.avo / self.VmAlpha)**(2/3)
            elif self.GB[p].nucleationSiteType == GBFactors.GRAIN_EDGES:
                edgePrec = np.sum([np.sqrt(1 - self.GB[p2].GBk**2) * self.PBM[p2].FirstMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.GRAIN_EDGES])
                nucleationSites += self.GBedgeN0 - edgePrec * (self.avo / self.VmAlpha)**(1/3)
            elif self.GB[p].nucleationSiteType == GBFactors.GRAIN_CORNERS:
                cornerPrec = np.sum([self.PBM[p2].ZeroMoment() for p2 in range(len(self.phases)) if self.GB[p2].nucleationSiteType == GBFactors.GRAIN_CORNERS])
                nucleationSites += self.GBcornerN0 - cornerPrec
               
            if nucleationSites < 0:
                nucleationSites = 0
            self.nucRate[p, i] = nucleationSites * self._nucleationRate(p, i)

    def _setNucleateRadius(self, i, dt):
        for p in range(len(self.phases)):
            #If nucleates form, then calculate radius of precipitate
            #Radius is set slightly larger so preciptate 
            if self.nucRate[p, i] * dt >= 1 and self.Rcrit[p, i] >= self.Rmin[p]:
                self.Rad[p, i] = self.Rcrit[p, i] + 0.5 * np.sqrt(self.kB * self.T[i] / (np.pi * self.gamma[p]))
            else:
                self.Rad[p, i] = 0

    def _calculatePSD(self, i, dt):
        '''
        Updates the PSD using the population balance model from coarsening and nucleation rate
        This also updates the fraction of precipitates, matrix composition and average radius
        '''
        for p in range(len(self.phases)):
            #Check largest class of PSD, if greater than 1, add additional size class
            #Will have to update PSDXalpha and PSDXbeta as well
            if self.PBM[p].PSD[-1] > 1:
                self.PBM[p].addSizeClass()
                self.growth[p] = np.append(self.growth[p], 0)
                if self.numberOfElements == 1:
                    newXalpha, newXbeta = self.interfacialComposition[p](self.T[i], self.particleGibbs(self.PBM[p].PSDbounds[-1], self.phases[p]))
                    self.PSDXalpha[p] = np.append(self.PSDXalpha[p], newXalpha)
                    self.PSDXbeta[p] = np.append(self.PSDXbeta[p], newXbeta)
                else:
                    g, newXalpha, newXbeta, _, _ = self.interfacialComposition[p](self.xComp[i-1], self.T[i], self.dGs[p,i-1] * self.VmBeta[p], self.PBM[p].PSDbounds, self.particleGibbs(phase=self.phases[p]))
                    self.PSDXalpha[p] = newXalpha
                    self.PSDXbeta[p] = newXbeta
                    self.growth[p] = g

            #Add nucleates to PSD after growth rate since new precipitates will have a growth rate of 0
            nRad = 0
            #Find size class for nucleated particles
            for n in range(1, self.PBM[p].bins):
                if self.PBM[p].PSDbounds[n] > self.Rad[p, i]:
                    nRad = n-1
                    break
            self.PBM[p].PSD[nRad] += self.nucRate[p, i] * dt

            self.PBM[p].Update(dt, self.growth[p])
            
            #Set negative frequencies in PSD to 0
            #Also set any less than the minimum possible radius to be 0
            self.PBM[p].PSD[:self.RdrivingForceIndex[p]] = 0

    def _massBalance(self, i):
        '''
        Updates matrix composition and volume fraction of precipitates
        '''
        Ntot = np.zeros(len(self.phases))
        RadSum = np.zeros(len(self.phases))
        ARsum = np.zeros(len(self.phases))
        fBeta = np.zeros(len(self.phases))
        if self.numberOfElements == 1:
            fConc = np.zeros(len(self.phases))
        else:
            fConc = np.zeros((len(self.phases), self.numberOfElements))

        for p in range(len(self.phases)):
            #Sum up particles and average for particles
            Ntot[p] = self.PBM[p].ZeroMoment()
            RadSum[p] = self.PBM[p].Moment(order=1)
            ARsum[p] = self.PBM[p].WeightedMoment(0, self.shapeFactors[p].aspectRatio(self.PBM[p].PSDsize))
            fBeta[p] = self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].ThirdMoment()

            if self.numberOfElements == 1:
                if self.infinitePrecipitateDiffusion[p]:
                    fConc[p] = self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].WeightedMoment(3, 0.5 * (self.PSDXbeta[p][:-1] + self.PSDXbeta[p][1:]))
                else:
                    y = self.VmAlpha / self.VmBeta[p] * self.GB[p].areaFactor * np.sum(self.PBM[p].PSDbounds[1:]**2 * self.PBM[p]._fv[1:] * self.PSDXbeta[p][1:] * (self.PBM[p].PSDbounds[1:] - self.PBM[p].PSDbounds[:-1]))
                    fConc[p] = self.prevFConc[0,p,0] + y
                self.prevFConc[1,p,0] = copy.copy(self.prevFConc[0,p,0])
                self.prevFConc[0,p,0] = fConc[p]
            else:
                if self.infinitePrecipitateDiffusion[p]:
                    for a in range(self.numberOfElements):
                        fConc[p,a] = self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].WeightedMoment(3, 0.5 * (self.PSDXbeta[p][:-1,a] + self.PSDXbeta[p][1:,a]))
                else:
                    for a in range(self.numberOfElements):
                        y = self.VmAlpha / self.VmBeta[p] * self.GB[p].areaFactor * np.sum(self.PBM[p].PSDbounds[1:]**2 * self.PBM[p]._fv[1:] * self.PSDXbeta[p][1:,a] * (self.PBM[p].PSDbounds[1:] - self.PBM[p].PSDbounds[:-1]))
                        fConc[p,a] = self.prevFConc[0,p,a] + y
                self.prevFConc[1,p] = copy.copy(self.prevFConc[0,p])
                self.prevFConc[0,p] = fConc[p]

            #Average radius and precipitate density
            if Ntot[p] > 0:
                self.avgR[p, i] = RadSum[p] * self._GBareaRemoval(p) / Ntot[p]
                self.precipitateDensity[p, i] = Ntot[p]
                self.avgAR[p, i] = ARsum[p] / Ntot[p]
            else:
                self.avgR[p, i] = 0
                self.precipitateDensity[p, i] = 0
                self.avgAR[p, i] = 0
            
            #Volume fraction (max at 1)
            if fBeta[p] > 1:
                fBeta[p] = 1
            if self.betaFrac[p, i-1] == 1:
                fBeta[p] = 1
            
            self.betaFrac[p, i] = fBeta[p]
        
        #Composition (min at 0)
        if self.numberOfElements == 1:
            if np.sum(fBeta) < 1:
                self.xComp[i] = (self.xComp[0] - np.sum(fConc)) / (1 - np.sum(fBeta))
            else:
                self.xComp[i] = 0
        else:
            if np.sum(fBeta) < 1:
                self.xComp[i] = (self.xComp[0] - np.sum(fConc, axis=0)) / (1 - np.sum(fBeta))
            else:
                self.xComp[i] = np.zeros(self.numberOfElements)
    
    def _growthRateBinary(self, i):
        '''
        Determines current growth rate of all particle size classes in a binary system
        '''
        #Update equilibrium interfacial compositions
        #This will be override if createLookup is called
        self.xEqAlpha[:,i] = self.xEqAlpha[:,i-1]
        self.xEqBeta[:,i] = self.xEqBeta[:,i-1]

        #Update lookup table if temperature changes too much
        self.dTemp += self.T[i] - self.T[i-1]
        if np.abs(self.dTemp) > self.maxTempChange:
            self.createLookup(i)
            self.dTemp = 0
        
        #growthRate = np.zeros((len(self.phases), self.bins + 1))
        growthRate = []
        for p in range(len(self.phases)):
            growthRate.append(np.zeros(self.PBM[p].bins + 1))
            #If no precipitates are stable, don't calculate growth rate and set PSD to 0
            #This should represent dissolution of the precipitates
            if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p]):
                superSaturation = (self.xComp[i-1] - self.PSDXalpha[p]) / (self.VmAlpha * self.PSDXbeta[p] / self.VmBeta[p] - self.PSDXalpha[p])
                growthRate[p] = self.shapeFactors[p].kineticFactor(self.PBM[p].PSDbounds) * self.Diffusivity(self.xComp[i-1], self.T[i]) * superSaturation / (self.effDiffDistance(superSaturation) * self.PBM[p].PSDbounds)
            else:
                self.PBM[p].PSD = np.zeros(self.PBM[p].bins)
            
        self.growth = growthRate
    
    def _growthRateMulti(self, i):
        '''
        Determines current growth rate of all particle size classes in a multicomponent system
        '''
        growthRate = []
        for p in range(len(self.phases)):
            growth, xAlpha, xBeta, xEqAlpha, xEqBeta = self.interfacialComposition[p](self.xComp[i-1], self.T[i], self.dGs[p,i-1] * self.VmBeta[p], self.PBM[p].PSDbounds, self.particleGibbs(phase=self.phases[p]))
            
            #If two-phase equilibrium not found, the precipitates are unstable and dissolved
            if growth is None:
                growthRate.append(np.zeros(self.PBM[p].bins + 1))
                self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.PBM[p].PSD = np.zeros(self.PBM[p].bins)
                self.xEqAlpha[p,i] = np.zeros(self.numberOfElements)
                self.xEqBeta[p,i] = np.zeros(self.numberOfElements)
            else:
                #Update interfacial composition for each precipitate size
                self.PSDXalpha[p] = xAlpha
                self.PSDXbeta[p] = xBeta
                self.xEqAlpha[p,i] = xEqAlpha
                self.xEqBeta[p,i] = xEqBeta

                #Add shape factor to growth rate - will need to add effective diffusion distance as well
                growthRate.append(self.shapeFactors[p].kineticFactor(self.PBM[p].PSDbounds) * growth)

        self.growth = growthRate

    def plot(self, axes, variable, bounds = None, *args, **kwargs):
        '''
        Plots model outputs
        
        Parameters
        ----------
        axes : Axis
        variable : str
            Specified variable to plot
            Options are 'Volume Fraction', 'Total Volume Fraction', 'Critical Radius',
                'Average Radius', 'Volume Average Radius', 'Total Average Radius', 
                'Total Volume Average Radius', 'Aspect Ratio', 'Total Aspect Ratio'
                'Driving Force', 'Nucleation Rate', 'Total Nucleation Rate',
                'Precipitate Density', 'Total Precipitate Density', 
                'Temperature', 'Composition',
                'Size Distribution', 'Size Distribution Curve',
                'Size Distribution KDE', 'Size Distribution Density
                'Interfacial Composition Alpha', 'Interfacial Composition Beta'

                Note: for multi-phase simulations, adding the word 'Total' will
                    sum the variable for all phases. Without the word 'Total', the variable
                    for each phase will be plotted separately

                    Interfacial composition terms are more relavent for binary systems than
                    for multicomponent systems
                    
        bounds : tuple (optional)
            Limits on the x-axis (float, float) or None (default, this will set bounds to (initial time, final time))
        *args, **kwargs - extra arguments for plotting
        '''
        sizeDistributionVariables = ['Size Distribution', 'Size Distribution Curve', 'Size Distribution KDE', 'Size Distribution Density']
        compositionVariables = ['Interfacial Composition Alpha', 'Interfacial Composition Beta']

        if variable in compositionVariables:
            if variable == 'Interfacial Composition Alpha':
                yVar = self.PSDXalpha
                ylabel = 'Composition in Alpha phase'
            else:
                yVar = self.PSDXbeta
                ylabel = 'Composition in Beta Phase'

            if (len(self.phases)) == 1:
                axes.semilogx(self.PBM[0].PSDbounds, yVar[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    axes.plot(self.PBM[p].PSDbounds, yVar[p], label=self.phases[p], *args, **kwargs)
                axes.legend()
            axes.set_xlim([self.PBM[0].PSDbounds[0], self.PBM[0].PSDbounds[-1]])
            axes.set_xlabel('Radius (m)')
            axes.set_ylabel(ylabel)

        elif variable in sizeDistributionVariables:
            ylabel = 'Frequency (#/$m^3$)'
            if variable == 'Size Distribution':
                functionName = 'PlotHistogram'
            elif variable == 'Size Distribution KDE':
                functionName = 'PlotKDE'
            elif variable == 'Size Distribution Density':
                functionName = 'PlotDistributionDensity'
                ylabel = 'Distribution Density (#/$m^4$)'
            else:
                functionName = 'PlotCurve'
            

            if len(self.phases) == 1:
                getattr(self.PBM[0], functionName)(axes, scale=self._GBareaRemoval(0), *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    getattr(self.PBM[p], functionName)(axes, label=self.phases[p], scale=self._GBareaRemoval(p), *args, **kwargs)
                axes.legend()
            axes.set_xlabel('Radius (m)')
            axes.set_ylabel(ylabel)

        elif variable == 'Cumulative Size Distribution':
            ylabel = 'CDF'
            if len(self.phases) == 1:
                self.PBM[0].PlotCDF(axes, scale=self._GBareaRemoval(0), *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    self.PBM[p].PlotCDF(axes, label=self.phases[p], scale=self._GBareaRemoval(p), *args, **kwargs)
                axes.legend()
            axes.set_xlabel('Radius (m)')
            axes.set_ylabel(ylabel)
            
        else:
            super().plot(axes, variable, bounds, *args, **kwargs)