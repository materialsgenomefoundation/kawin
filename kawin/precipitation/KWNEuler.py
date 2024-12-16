import numpy as np
from kawin.precipitation.KWNBase import PrecipitateBase
from kawin.precipitation.PopulationBalance import PopulationBalanceModel
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
from kawin.precipitation.Plot import plotEuler

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

        self.eqAspectRatio = [None for p in range(len(self.phases))]

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
            self.PBM[i].resetRecordedData()

    def _addExtraSaveVariables(self, saveDict):
        for p in range(len(self.phases)):
            saveDict['PBM_data_' + self.phases[p]] = [self.PBM[p].min, self.PBM[p].max, self.PBM[p].bins]
            saveDict['PBM_PSD_' + self.phases[p]] = self.PBM[p].PSD
            saveDict['PBM_bounds_' + self.phases[p]] = self.PBM[p].PSDbounds
            saveDict['PBM_size_' + self.phases[p]] = self.PBM[p].PSDsize
            saveDict['eqAspectRatio_' + self.phases[p]] = self.eqAspectRatio[p]

    def load(filename):
        data = np.load(filename)
        model = PrecipitateModel(data['phases'], data['elements'])
        model._loadData(data)
        return model

    def _loadExtraVariables(self, data):
        for p in range(len(self.phases)):
            PBMdata = data['PBM_data_' + self.phases[p]]
            psd = data['PBM_PSD_' + self.phases[p]]
            bounds = data['PBM_bounds_' + self.phases[p]]
            size = data['PBM_size_' + self.phases[p]]
            eqAR = data['eqAspectRatio_' + self.phases[p]]
            self.PBM[p] = PopulationBalanceModel(PBMdata[0], PBMdata[1], int(PBMdata[2]))
            self.PBM[p].PSD = psd
            self.PBM[p].PSDsize = size
            self.PBM[p].PSDbounds = bounds
            self.eqAspectRatio[p] = eqAR

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
            self.PSDXalpha.append(np.zeros((self.PBM[p].bins + 1, 1)))
            self.PSDXbeta.append(np.zeros((self.PBM[p].bins + 1, 1)))

            self.PSDXalpha[p][:,0], self.PSDXbeta[p][:,0] = self.interfacialComposition[p](T, self.particleGibbs(self.PBM[p].PSDbounds, self.phases[p]))
            self.RdrivingForceIndex[p] = np.argmax(self.PSDXalpha[p][:,0] != -1)-1
            self.RdrivingForceIndex[p] = 0 if self.RdrivingForceIndex[p] < 0 else self.RdrivingForceIndex[p]
            self.RdrivingForceLimit[p] = self.PBM[p].PSDbounds[self.RdrivingForceIndex[p]]

            #Sets particle radii smaller than driving force limit to driving force limit composition
            #If RdrivingForceIndex is at the end of the PSDX arrays, then no precipitate in the size classes of the PSD is stable
            #This can occur in non-isothermal situations where the temperature gets too high
            if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p][:,0]):
                self.PSDXalpha[p][:self.RdrivingForceIndex[p]+1,0] = self.PSDXalpha[p][self.RdrivingForceIndex[p]+1,0]
                self.PSDXbeta[p][:self.RdrivingForceIndex[p]+1,0] = self.PSDXbeta[p][self.RdrivingForceIndex[p]+1,0]
            else:
                self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1,1))
                self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1,1))

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
        '''
        The following checks are made
            1) change in number of particles moving between bins
                This is controlled by the implementation in PopulationBalanceModel,
                but essentially limits the number of particles moving between bins
            2) change in nucleation rate
                Time will be proportional to the 1/log(previous nuc rate / new nuc rate)
            3) change in temperature
                Limits how fast temperature can change
            4) change in critical radius
                Proportional to a percent change in critical radius
            5) estimated change in volume fraction
                Estimates the change in volume fraction from the nucleation rate and nucleation radius
        '''
        #Start test dt at 0.01 or previous dt
        i = self.n
        dtPrev = 0.01 if self.n == 0 else self.time[i] - self.time[i-1]
        #Try to slowly increase the time step
        #  Precipitation kinetics is more on a log scale than linear (unless temperature changes are involve)
        #  Thus, we can get away with increasing the time step over time assuming that kinetics are slowing down
        #  Plus, unlike the single phase diffusion module, there's no form way to define a good time step apart from the checks here
        dtPropose = (1 + self.dtScale) * dtPrev
        dtMax = self.finalTime - self.time[i]
        
        dtAll = [dtMax]
        if self.checkPSD:
            dtPBM = [dtMax]
            if i > 0 and self.temperature[i] == self.temperature[i-1]:
                dtPBM += [self.PBM[p].getDTEuler(dtMax, self.growth[p], self.dissolutionIndex[p]) for p in range(len(self.phases))]
            dtPBM = np.amin(dtPBM)
            dtAll.append(dtPBM)
        
        if self.checkNucleation:
            dtNuc = dtMax * np.ones(len(self.phases))
            if i > 0:
                nRateCurr = self.nucRate[i]
                nRatePrev = self.nucRate[i-1]
                for p in range(len(self.phases)):
                    if nRateCurr[p] > self.minNucleationRate and nRatePrev[p] > self.minNucleationRate and nRatePrev[p] != nRateCurr[p]:
                        dtNuc[p] = self.maxNucleationRateChange * dtPrev / np.abs(np.log10(nRatePrev[p] / nRateCurr[p]))
            else:
                for p in range(len(self.phases)):
                    if self.nucRate[i,p] * dtPrev > 1e5:
                        dtNuc[p] = 1e5 / self.nucRate[i,p]
            dtNuc = np.amin(dtNuc)
            dtAll.append(dtNuc)

        #Temperature change constraint
        if self.checkTemperature and i > 0:
            Tchange = self.temperature[i] - self.temperature[i-1]
            dtTemp = dtMax
            if Tchange > self.maxNonIsothermalDT:
                dtTemp = self.maxNonIsothermalDT * dtPrev / Tchange
            dtAll.append(dtTemp)

        if self.checkRcrit and i > 0:
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
        #If all time checks pass, then go back to previous time step and increase it slowly
        #   This is so we don't step at the maximum possible time
        if dt == dtMax:
            dt = dtPropose

        return dt
    
    def _processX(self, x):
        '''
        Quick check to make sure particles below the thresholds are 0
            RdrivingForceIndex - only for binary, where energy from the Gibbs-Thompson effect is high enough
                that the free energy of the precipitate is above the free energy surface of the matrix phase
                and equilibrium cannot be calculated
            minRadius - minimum radius to be considered a precipitate
        '''
        for p in range(len(self.phases)):
            x[p][:self.RdrivingForceIndex[p]+1] = 0
            x[p][self.PBM[p].PSDsize < self.minRadius] = 0
        return
    
    def _calcNucleationRate(self, t, x):
        '''
        The _calcNucleationRate function in KWNBase calculates the nucleation rate as the
            probability that a site can form a nucleate that will continue to grow

        To convert this probability to an actual nucleation rate, we multiply by the amount
            of available nucleation sites

        The number of available sites is determined by:
            Available sites = All sites - used up sites + sites on parent precipitates
            The used up sites depends on the type of nucleation
                Bulk and grain corners - used sites = number of current precipitates
                Dislocation and grain edges - number of sites filled along the edges (assumes average radius of precipitates)
                Grain boundaries - number of sites filled along the faces (assumes average cross sectional area of precipitates)
        '''
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
        '''
        Mass balance to find matrix composition with new particle size distribution

        This also includes: volume fraction, precipitate density, average radius, average aspect ratio and sum of precipitate composition
        '''
        fBeta = np.zeros((1,len(self.phases)))
        fConc = np.zeros((1, len(self.phases),self.numberOfElements))
        precDens = np.zeros((1,len(self.phases)))
        avgR = np.zeros((1,len(self.phases)))
        avgAR = np.zeros((1,len(self.phases)))
        xComp = np.zeros((1,self.numberOfElements))
        
        for p in range(len(self.phases)):
            volRatio = self.VmAlpha / self.VmBeta[p]
            Ntot = self.PBM[p].ZeroMomentFromN(x[p])
            #If no precipitates, then avgR, avgAR, precDens, fConc and fBeta for phase p is all 0
            if Ntot == 0:
                continue
            RadSum = self.PBM[p].MomentFromN(x[p], 1)
            ARsum = self.PBM[p].WeightedMomentFromN(x[p], 0, self.shapeFactors[p].aspectRatio(self.PBM[p].PSDsize))
            fBeta[0,p] = np.amin([volRatio * self.GB[p].volumeFactor * self.PBM[p].ThirdMomentFromN(x[p]), 1])

            '''
            Concentration of the precipitates - needed to get matrix composition

            For a line compound with composition x^beta, this boils down to:
            x_0 = (1-f_v) * x^inf + f_v * x^beta
                Where x_0 is initial composition, f_v is volume fraction and x^inf is matrix composition

            For non-stoichiometric compounds, we want to integrate the precipitate composition as a function of radius
                We'll call this term f_conc (fraction + concentration of precipitates), so:
                x_0 = (1-f_v) * x^inf + f_conc
            
            For infinite precipitate diffusion, the concentration of a single precipitate is assumed to be homogenous
            f_conc = r_vol * vol_factor * sum(n_i * R_i^3 * x_i^beta)
                Where r_vol is V^alpha / V^beta and vol_factor is a factor for converting R^3 to volume (for sphere, this is 4*pi/3)

            For no diffusion in precipitate, the concentration depends on the history of the precipitate compositions and growth rate
            We just have to convert the summation to an integral of the time derivative of the terms inside
            f_conc = r_vol * vol_factor * sum(int(d(n_i * R_i^3 * x_i^beta)/dt, dt))
            We'll assume x_i^beta is constant with time (for 3 or more components, this is not true, but assume it doesn't change significantly per iteration - it'll also be a lot harder to account for)
            d(f_conc)/dt = r_vol * vol_factor * sum(d(n_i)/dt * R_i^3 * x_i^beta + 3 * R_i^3 * d(R_i)/dt * n_i * x_i^beta)
                d(n_i)/dt is the change in precipitates, since we don't record this, this is just (x[p] - self.PBM[p].PSD) / dt - with x[p] being the new number density for phase p
                d(R_i)/dt is the growth rate, however, since we use a eulerian solver, this corresponds to the growth rate of the bins themselves, which is 0
                    If we were to use a langrangian solver, then d(n_i)/dt would be 0 (since the density in each bin would be constant) and d(R_i)/dt would be the growth rate at R_i
            Then we can calculate f_conc per iteration as a time integration like we do with some of the other variables
            '''
            if self.infinitePrecipitateDiffusion[p]:
                compAvg = 0.5 * (self.PSDXbeta[p][:-1] + self.PSDXbeta[p][1:])
                for e in range(self.numberOfElements):
                    fConc[0,p,e] = volRatio * self.GB[p].volumeFactor * self.PBM[p].WeightedMomentFromN(x[p], 3, compAvg[:,e])
            else:
                midX = (self.PSDXbeta[p][1:] + self.PSDXbeta[p][:-1]) / 2
                for e in range(self.numberOfElements):
                    #y = volRatio * self.GB[p].volumeFactor * np.sum((3*midG*self.PBM[p].PSDsize**2*self.PBM[p].PSD*dt + self.PBM[p].PSDsize**3*(x[p]-self.PBM[p].PSD))*midX[:,e])
                    y = volRatio * self.GB[p].volumeFactor * np.sum((self.PBM[p].PSDsize**3*(x[p]-self.PBM[p].PSD))*midX[:,e])
                    fConc[0,p,e] = self.fConc[self.n,p,e] + y

            #Only record these terms if there are non-zero number of precipitates
            #Otherwise we will be dividing by 0 for avgR and avgAR
            #   Argueably, RadSum and ARsum would be 0 if Ntot is 0, so it should be fine to do this
            if Ntot > self.minNucleateDensity:
                avgR[0,p] = RadSum / Ntot
                precDens[0,p] = Ntot
                avgAR[0,p] = ARsum / Ntot
            else:
                avgR[0,p] = 0
                precDens[0,p] = 0
                avgAR[0,p] = 0

            #Not sure if needed, but just in case
            if self.betaFrac[self.n,p] == 1:
                fBeta[0,p] = 1

        if np.sum(fBeta[0]) < 1:
            xComp[0] = (self.xComp[0] - np.sum(fConc[0], axis=0)) / (1 - np.sum(fBeta[0]))
            xComp[0,xComp[0] < 0] = self.minComposition

        self._currY[self.VOL_FRAC] = fBeta
        self._currY[self.FCONC] = fConc
        self._currY[self.PREC_DENS] = precDens
        self._currY[self.R_AVG] = avgR
        self._currY[self.AR_AVG] = avgAR
        self._currY[self.COMPOSITION] = xComp

    def getCurrentX(self):
        '''
        Returns current value of time and X
        In this case, X is the particle size distribution for each phase
        '''
        return [self.PBM[p].PSD for p in range(len(self.phases))]

    def _getdXdt(self, t, x):
        '''
        Returns dn_i/dt for each PBM of each phase
        '''
        return [self.PBM[p].getdXdtEuler(self.growth[p], self._currY[self.NUC_RATE][0,p], self._currY[self.R_NUC][0,p], x[p]) for p in range(len(self.phases))]

    def correctdXdt(self, dt, x, dXdt):
        '''
        Corrects dXdt with the newly found dt, this adjusts the fluxes at the ends of the PBM so that we don't get negative bins
        '''
        for p in range(len(self.phases)):
            dXdt[p] = self.PBM[p].correctdXdtEuler(dt, self.growth[p], self._currY[self.NUC_RATE][0,p], self._currY[self.R_NUC][0,p], x[p])

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
        if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p][:,0]):
            superSaturation = (xComp[0] - self.PSDXalpha[p][:,0]) / (self.VmAlpha * self.PSDXbeta[p][:,0] / self.VmBeta[p] - self.PSDXalpha[p][:,0])
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
        precDens = self._currY[self.PREC_DENS][0]
        if dGs[p] < 0 and precDens[p] <= 0:
            xEqAlpha = np.zeros(self.numberOfElements)
            xEqBeta = np.zeros(self.numberOfElements)
            growthRate = np.zeros(self.PBM[p].bins + 1)
            return growthRate, xEqAlpha, xEqBeta


        growth, xAlpha, xBeta, xEqAlpha, xEqBeta = self.interfacialComposition[p](xComp, T, dGs[p] * self.VmBeta[p], self.PBM[p].PSDbounds, self.particleGibbs(phase=self.phases[p]), searchDir = self._precBetaTemp[p])

        #If two-phase equilibrium not found, two possibilities - precipitates are unstable or equilibrium calculations didn't converge
        #We try to avoid this as much as possible to where if precipitates are unstable, then attempt to get a growth rate from the nearest composition on the phase boundary
        #And if equilibrium calculations didn't converge, try to use the previous calculations assuming the new composition is close to the previous
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
        '''
        Updates particle size distribution with new x

        Steps:
            1. Check if growth rate calculation failed with negative driving force
                We'll reset the PBM since we can't do much from here, but the chances of this happening should be pretty low
            2. Update the PBM with new x
            3. Check if the PBM needs to adjust the size class
                If so, then update the cached aspect ratio and precipitate composition with the new size classes
            4. Remove precipitates below a certain threshold (RdrivingForceIndex and minRadius)
            5. Calculate the dissolution index (index at which below are not considered when calculating dt)
                This is to prevent very small dt as the growth rate increases rapidly when R->0
        '''
        for p in range(len(self.phases)):
            if self.dGs[self.n,p] < 0 and np.all(self.xEqAlpha[self.n,p,:] == 0):
                self.PBM[p].reset()
                self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.growth[p] = np.zeros(self.PBM[p].bins+1)
                continue
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
                        self.PSDXalpha[p] = np.concatenate((self.PSDXalpha[p], np.zeros((self.PBM[p].bins+1 - len(self.PSDXalpha[p]),1))))
                        self.PSDXbeta[p] = np.concatenate((self.PSDXbeta[p], np.zeros((self.PBM[p].bins+1 - len(self.PSDXbeta[p]),1))))
                        self.PSDXalpha[p][addedIndices:,0], self.PSDXbeta[p][addedIndices:,0] = self.interfacialComposition[p](self.temperature[self.n], self.particleGibbs(self.PBM[p].PSDbounds[addedIndices:], self.phases[p]))
                else:
                    self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                    self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self._growthRate()
            self.PBM[p].PSD[:self.RdrivingForceIndex[p]+1] = 0
            self.PBM[p].PSD[self.PBM[p].PSDsize < self.minRadius] = 0
            self.dissolutionIndex[p] = self.PBM[p].getDissolutionIndex(self.maxDissolution, self.RdrivingForceIndex[p])
            #self.PBM[p].PSD[:self.dissolutionIndex[p]] = 0

    def plot(self, axes, variable, bounds = None, timeUnits = 's', radius='spherical', *args, **kwargs):
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
        radius : str (optional)
            For non-spherical precipitates, plot the Average Radius by the -
                Equivalent spherical radius ('spherical')
                Short axis ('short')
                Long axis ('long')
            Note: Total Average Radius and Volume Average Radius will still use the equivalent spherical radius
        *args, **kwargs - extra arguments for plotting
        '''
        plotEuler(self, axes, variable, bounds, timeUnits, radius, *args, **kwargs)


                