import numpy as np

from kawin.Constants import AVOGADROS_NUMBER
from kawin.precipitation.PrecipitationParameters import PrecipitationData
from kawin.precipitation.KWNBase import PrecipitateBase
from kawin.precipitation.PopulationBalance import PopulationBalanceModel
from kawin.precipitation.parameters.Nucleation import NucleationBarrierParameters, DislocationDescription, BulkDescription, GrainBoundaryDescription, GrainCornerDescription, GrainEdgeDescription

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
    def _resetArrays(self):
        '''
        Resets and initializes arrays for all variables

        In addition to PrecipitateBase, the equilibrium aspect ratio area and population balance models are created here
        '''
        super()._resetArrays()
        self.PBM = [PopulationBalanceModel() for p in self.phases]
        self.eqAspectRatio = [None for p in range(len(self.phases))]

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

    def toDict(self):
        data = super().toDict()
        for p in range(len(self.phases)):
            data['PBM_data_' + self.phases[p]] = [self.PBM[p].min, self.PBM[p].max, self.PBM[p].bins]
            data['PBM_PSD_' + self.phases[p]] = self.PBM[p].PSD
            data['PBM_bounds_' + self.phases[p]] = self.PBM[p].PSDbounds
            data['PBM_size_' + self.phases[p]] = self.PBM[p].PSDsize
            data['eqAspectRatio_' + self.phases[p]] = self.eqAspectRatio[p]
        return data
    
    def fromDict(self, data):
        super().fromDict(data)
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

    def saveRecordedPSD(self, filename, phase = 'all'):
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
                self.PBM[index].saveRecordedPSD(filename + '_' + p)
        else:
            index = self.phaseIndex(phase)
            self.PBM[index].saveRecordedPSD(filename)

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
        self.PBM[index].loadDistribution(data)

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
    
    def getPBM(self, phase=None):
        '''
        Returns population balance model of given phase

        Parameters
        ----------
        phase: str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        return self.PBM[self.phaseIndex(phase)]
    
    def _createLookupBinary(self, T):
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
        for p in range(len(self.phases)):
            #Interfacial compositions at equilibrium (planar interface)
            xAResult, xBResult = self.therm.getInterfacialComposition(T, 0, precPhase=self.precipitates[p].phase)
            if xAResult is not None and xAResult != -1:
                xEqAlpha[0,p,0] = xAResult
                xEqBeta[0,p,0] = xBResult

            #Interfacial compositions at each size class in PSD
            self.PSDXalpha.append(np.zeros((self.PBM[p].bins + 1, 1)))
            self.PSDXbeta.append(np.zeros((self.PBM[p].bins + 1, 1)))

            self.PSDXalpha[p][:,0], self.PSDXbeta[p][:,0] = self.therm.getInterfacialComposition(T, self.particleGibbs(self.PBM[p].PSDbounds, self.precipitates[p].phase), precPhase=self.precipitates[p].phase)
            self.RdrivingForceIndex[p] = np.amax([np.argmax(self.PSDXalpha[p][:,0] != -1) - 1, 0])
            self.precipitates[p].RdrivingForceLimit = self.PBM[p].PSDbounds[self.RdrivingForceIndex[p]]

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
    
    def _setupAspectRatio(self):
        #If calculateAspectRatio is True, then use strain energy to calculate aspect ratio for each size class in PSD
        #Else, then use aspect ratio defined in shape factors
        self.eqAspectRatio = [None for p in range(len(self.phases))]
        for p in range(len(self.phases)):
            self.PBM[p].reset()

            if self.precipitates[p].calculateAspectRatio:
                self.eqAspectRatio[p] = self.precipitates[p].strainEnergy.eqAR_bySearch(self.PBM[p].PSDbounds, self.precipitates[p].gamma, self.precipitates[p].shapeFactor)
                arFunc = lambda R, p1=p : self._interpolateAspectRatio(R, p1)
                self.precipitates[p].shapeFactor.setAspectRatio(arFunc)
            else:
                self.eqAspectRatio[p] = self.precipitates[p].shapeFactor.aspectRatio(self.PBM[p].PSDbounds)
            
    def setup(self):
        '''
        Sets up additional variables in addition to PrecipitateBase

        Sets up additional outputs, population balance models, equilibrium aspect ratio and equilibrium compositions
        '''
        if self._isSetup:
            return

        super().setup()

        #Equilibrium aspect ratio and PBM setup
        self._setupAspectRatio()

        #Setup precipitation data for n = 0
        Y = self.data.copySlice(self.data.n)
        Y.time = np.array([self.data.time[self.data.n]])
        Y.temperature = np.array([self.temperatureParameters(Y.time[0])])

        #Setup interfacial composition
        if self.numberOfElements == 1:
            self.data.xEqAlpha[self.data.n], self.data.xEqBeta[self.data.n] = self._createLookupBinary(self.data.temperature[self.data.n])
        else:
            self.PSDXalpha = [None for p in range(len(self.phases))]
            self.PSDXbeta = [None for p in range(len(self.phases))]

            #Set first index of eq composition
            for p in range(len(self.phases)):
                #Use arbitrary dg, R and gE since only the eq compositions are needed here
                growth_result = self.therm.getGrowthAndInterfacialComposition(self.data.composition[self.data.n], self.data.temperature[self.data.n], 0, 1, 0, precPhase=self.precipitates[p].phase, removeCache=self.removeCache)
                if growth_result is not None:
                    _, _, _, c_eq_alpha, c_eq_beta = growth_result
                    self.data.xEqAlpha[self.data.n,p] = c_eq_alpha
                    self.data.xEqBeta[self.data.n,p] = c_eq_beta

        x = [self.PBM[p].PSD for p in range(len(self.phases))]
        Y = self._calcNucleationRate(self.data.time[self.data.n], x, Y)
        self.growth, Y = self._growthRate(Y)
        self.data.setSlice(Y, self.data.n)
    
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
        i = self.data.n
        dtPrev = 0.01 if self.data.n == 0 else self.data.time[i] - self.data.time[i-1]
        #Try to slowly increase the time step
        #  Precipitation kinetics is more on a log scale than linear (unless temperature changes are involve)
        #  Thus, we can get away with increasing the time step over time assuming that kinetics are slowing down
        #  Plus, unlike the single phase diffusion module, there's no form way to define a good time step apart from the checks here
        dtPropose = (1 + self.constraints.dtScale) * dtPrev
        dtMax = self.finalTime - self.data.time[i]
        
        dtAll = [dtMax]
        dtAll.append(self.constraints.computeDTfromPSD(self.data.n, self.data.temperature, self.PBM, self.growth, self.dissolutionIndex, self.phases, dtMax))
        dtAll.append(self.constraints.computeDTfromNucleationRate(self.data.n, self.data.nucRate, self.phases, dtPrev, dtMax))
        dtAll.append(self.constraints.computeDTfromTemperature(self.data.n, self.data.temperature, dtPrev, dtMax))
        dtAll.append(self.constraints.computeDTfromRcrit(self.data.n, self.data.Rcrit, self.data.drivingForce, self.phases, dtPrev, dtMax))
        
        VmAlpha = self.matrix.volume.Vm
        VmBetas = [self.precipitates[p].volume.Vm for p in range(len(self.phases))]
        nucParams = [self.precipitates[p].nucleation for p in range(len(self.phases))]
        dtAll.append(self.constraints.computeDTfromVolume(self.data.n, self.data.nucRate, self.data.Rnuc, self.PBM, self.growth, VmAlpha, VmBetas, nucParams, self.phases, dtMax))

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
            x[p][self.PBM[p].PSDsize < self.constraints.minRadius] = 0
        return
    
    def _calcNucleationSites(self, t, x, p):
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
        VmBetas = [self.precipitates[p2].volume.Vm for p2 in range(len(self.precipitates))]
        nucParams = [self.precipitates[p2].nucleation for p2 in range(len(self.precipitates))]
        parentPhases = [self.precipitates[p2].parentPhases for p2 in range(len(self.precipitates))]

        #If parent phases exists, then calculate the number of potential nucleation sites on the parent phase
        # #This is the number of lattice sites on the total surface area of the parent precipitate
        nucleationSites = np.sum([4*np.pi*self.PBM[p2].SecondMomentFromN(x[p2]) * (AVOGADROS_NUMBER/VmBetas[p2])**(2/3) for p2 in parentPhases[p]])
        # DislocationDescription is a subclass of BulkDescription, so we test this first
        if isinstance(nucParams[p].description, DislocationDescription):
            bulkPrec = np.sum([self.PBM[p2].firstMoment(N=x[p2]) for p2 in range(len(self.phases)) if isinstance(nucParams[p2].description, DislocationDescription)])
            nucleationSites += self.matrix.nucleationSites.dislocationN0 - bulkPrec * (AVOGADROS_NUMBER / self.matrix.volume.Vm)**(1/3)

        elif isinstance(nucParams[p].description, BulkDescription):
            bulkPrec = np.sum([self.PBM[p2].zeroMoment(N=x[p2]) for p2 in range(len(self.phases)) if isinstance(nucParams[p2].description, BulkDescription)])
            nucleationSites += self.matrix.nucleationSites.bulkN0 - bulkPrec

        elif isinstance(nucParams[p].description, GrainBoundaryDescription):
            boundPrec = np.sum([nucParams[p2].gbRemoval * self.PBM[p2].secondMoment(N=x[p2]) for p2 in range(len(self.phases)) if isinstance(nucParams[p2].description, GrainBoundaryDescription)])
            nucleationSites += self.matrix.nucleationSites.GBareaN0 - boundPrec * (AVOGADROS_NUMBER / self.matrix.volume.Vm)**(2/3)

        elif isinstance(nucParams[p].description, GrainEdgeDescription):
            # TODO: where did sqrt(1-GBk^2) come from?
            edgePrec = np.sum([np.sqrt(1 - nucParams[p2].GBk**2) * self.PBM[p2].firstMoment(N=x[p2]) for p2 in range(len(self.phases)) if isinstance(nucParams[p2].description, GrainEdgeDescription)])
            nucleationSites += self.matrix.nucleationSites.GBedgeN0 - edgePrec * (AVOGADROS_NUMBER / self.matrix.volume.Vm)**(1/3)

        elif isinstance(nucParams[p].description, GrainCornerDescription):
            cornerPrec = np.sum([self.PBM[p2].zeroMoment(N=x[p2]) for p2 in range(len(self.phases)) if isinstance(nucParams[p2].description, GrainCornerDescription)])
            nucleationSites += self.matrix.nucleationSites.GBcornerN0 - cornerPrec

        return np.amax([nucleationSites, 0])
    
    def _calcMassBalance(self, t, x, Y : PrecipitationData):
        '''
        Mass balance to find matrix composition with new particle size distribution
        This also includes: volume fraction, precipitate density, average radius, average aspect ratio and sum of precipitate composition

        Notes on computing composition from mass balance
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
            d(f_conc)/dt = r_vol * vol_factor * sum(d(n_i)/dt * R_i^3 * x_i^beta + 3 * R_i^2 * d(R_i)/dt * n_i * x_i^beta)
                d(n_i)/dt is the change in precipitates, since we don't record this, this is just (x[p] - self.PBM[p].PSD) / dt - with x[p] being the new number density for phase p
                d(R_i)/dt is the growth rate, however, since we use a eulerian solver, this corresponds to the growth rate of the bins themselves, which is 0
                    If we were to use a langrangian solver, then d(n_i)/dt would be 0 (since the density in each bin would be constant) and d(R_i)/dt would be the growth rate at R_i
            Then we can calculate f_conc per iteration as a time integration like we do with some of the other variables
        '''
        for p in range(len(self.phases)):
            precParams = self.precipitates[p]
            volRatio = self.matrix.volume.Vm / precParams.volume.Vm
            Y.precipitateDensity[0,p] = self.PBM[p].zeroMoment(N=x[p])
            #If no precipitates, then avgR, avgAR, precDens, fConc and fBeta for phase p is all 0
            if Y.precipitateDensity[0,p] < self.constraints.minNucleateDensity:
                Y.Ravg[0,p] = 0
                Y.ARavg[0,p] = 0
                Y.fconc[0,p] = np.zeros(Y.fconc[0,p].shape)
                Y.volFrac[0,p] = 0
                continue

            Y.Ravg[0,p] = self.PBM[p].firstMoment(N=x[p]) / Y.precipitateDensity[0,p]
            Y.ARavg[0,p] = self.PBM[p].zeroMoment(N=x[p], weights=precParams.shapeFactor.aspectRatio(self.PBM[p].PSDsize)) / Y.precipitateDensity[0,p]
            Y.volFrac[0,p] = np.amin([volRatio * precParams.nucleation.volumeFactor * self.PBM[p].thirdMoment(N=x[p]), 1])
            #Not sure if needed, but just in case
            if self.data.volFrac[self.data.n,p] == 1:
                Y.volFrac[0,p] = 1

            # Compute fconc as described above
            if precParams.infinitePrecipitateDiffusion:
                # f_conc = r_vol * vol_factor * sum(n_i * R_i^3 * x_i^beta)
                compAvg = 0.5 * (self.PSDXbeta[p][:-1] + self.PSDXbeta[p][1:])
                for e in range(self.numberOfElements):
                    Y.fconc[0,p,e] = volRatio * precParams.nucleation.volumeFactor * self.PBM[p].thirdMoment(N=x[p], weights=compAvg[:,e])
            else:
                # d(f_conc)/dt = r_vol * vol_factor * sum(d(n_i)/dt * R_i^3 * x_i^beta)
                # where dt cancels out, so d(f_conc) = r_vol * vol_factor * sum((new_psd - curr_psd) * R_i^3 * x_i^beta)
                midX = (self.PSDXbeta[p][1:] + self.PSDXbeta[p][:-1]) / 2
                for e in range(self.numberOfElements):
                    y = volRatio * precParams.nucleation.volumeFactor * np.sum((self.PBM[p].PSDsize**3*(x[p] - self.PBM[p].PSD))*midX[:,e])
                    Y.fconc[0,p,e] = self.data.fconc[self.data.n,p,e] + y

        if np.sum(Y.volFrac[0]) < 1:
            Y.composition[0] = (self.data.composition[0] - np.sum(Y.fconc[0], axis=0)) / (1 - np.sum(Y.volFrac[0]))
            Y.composition[0,Y.composition[0] < 0] = self.constraints.minComposition

        return Y

    def getCurrentX(self):
        '''
        Returns current value of time and X
        In this case, X is the particle size distribution for each phase
        '''
        return [self.PBM[p].PSD for p in range(len(self.phases))]

    def _getdXdt(self, t, x, Y : PrecipitationData, growth):
        '''
        Returns dn_i/dt for each PBM of each phase
        '''
        return [self.PBM[p].getdXdtEuler(growth[p], Y.nucRate[0,p], Y.Rnuc[0,p], x[p]) for p in range(len(self.phases))]

    def _correctdXdt(self, dt, x, dXdt, Y : PrecipitationData, growth):
        '''
        Corrects dXdt with the newly found dt, this adjusts the fluxes at the ends of the PBM so that we don't get negative bins
        '''
        for p in range(len(self.phases)):
            dXdt[p] = self.PBM[p].correctdXdtEuler(dt, growth[p], Y.nucRate[0,p], Y.Rnuc[0,p], x[p])

    def _growthRate(self, Y : PrecipitationData):
        if self.numberOfElements == 1:
            return self._growthRateBinary(Y)
        else:
            return self._growthRateMulti(Y)

    def _singleGrowthBinary(self, p, Y : PrecipitationData):
        '''
        Calculates growth rate for a single phase
        This is separated from _growthRateBinary since it's used in _calculatePSD

        Matrix/precipitate composition are not calculated here since it's
        already calculated in _createLookupBinary
        '''
        xComp = Y.composition[0]
        T = Y.temperature[0]
        growthRate = np.zeros(self.PBM[p].bins + 1)
        #If no precipitates are stable, don't calculate growth rate and set PSD to 0
        #This should represent dissolution of the precipitates
        if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p][:,0]):
            superSaturation = (xComp[0] - self.PSDXalpha[p][:,0]) / (self.matrix.volume.Vm * self.PSDXbeta[p][:,0] / self.precipitates[p].volume.Vm - self.PSDXalpha[p][:,0])
            D = self.therm.getInterdiffusivity(xComp[0], T, removeCache=self.removeCache)
            growthRate = self.precipitates[p].shapeFactor.kineticFactor(self.PBM[p].PSDbounds) * D * superSaturation / (self.matrix.effectiveDiffusion(superSaturation) * self.PBM[p].PSDbounds)

        return growthRate
    
    def _growthRateBinary(self, Y : PrecipitationData):
        '''
        Determines current growth rate of all particle size classes in a binary system
        '''
        #Update equilibrium interfacial compositions
        #This will be override if _createLookupBinary is called
        T = Y.temperature[0]
        self.dTemp += T - self.data.temperature[self.data.n]
        if np.abs(self.dTemp) > self.constraints.maxTempChange:
            xEqAlpha, xEqBeta = self._createLookupBinary(T)
        else:
            xEqAlpha, xEqBeta = np.array([self.data.xEqAlpha[self.data.n]]), np.array([self.data.xEqBeta[self.data.n]])
            self.dTemp = 0
        Y.xEqAlpha = xEqAlpha
        Y.xEqBeta = xEqBeta
        
        return [self._singleGrowthBinary(p, Y) for p in range(len(self.phases))], Y

    def _singleGrowthMulti(self, p, Y : PrecipitationData):
        '''
        Calculates growth rate for a single phase
        This is separated from _growthRateMulti since it's used in _calculatePSD

        This will also calculate the matrix/precipitate composition 
        for the radius in the PSD as well as equilibrium (infinite radius)
        '''
        xComp = Y.composition[0]
        dGs = Y.drivingForce[0]
        T = Y.temperature[0]
        precDens = Y.precipitateDensity[0]
        if dGs[p] < 0 and precDens[p] <= 0:
            xEqAlpha = np.zeros(self.numberOfElements)
            xEqBeta = np.zeros(self.numberOfElements)
            growthRate = np.zeros(self.PBM[p].bins + 1)
            return growthRate, xEqAlpha, xEqBeta

        growth_result = self.therm.getGrowthAndInterfacialComposition(xComp, T, dGs[p] * self.precipitates[p].volume.Vm, self.PBM[p].PSDbounds, self.particleGibbs(phase=self.precipitates[p].phase), precPhase=self.precipitates[p].phase, removeCache=self.removeCache, searchDir = self._precBetaTemp[p])

        #If two-phase equilibrium not found, two possibilities - precipitates are unstable or equilibrium calculations didn't converge
        #We try to avoid this as much as possible to where if precipitates are unstable, then attempt to get a growth rate from the nearest composition on the phase boundary
        #And if equilibrium calculations didn't converge, try to use the previous calculations assuming the new composition is close to the previous
        if growth_result is None:
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
            growth, xAlpha, xBeta, xEqAlpha, xEqBeta = growth_result
            #Update interfacial composition for each precipitate size
            self.PSDXalpha[p] = xAlpha
            self.PSDXbeta[p] = xBeta

            #Add shape factor to growth rate - will need to add effective diffusion distance as well
            growthRate = self.precipitates[p].shapeFactor.kineticFactor(self.PBM[p].PSDbounds)*growth

        return growthRate, xEqAlpha, xEqBeta
    
    def _growthRateMulti(self, Y : PrecipitationData):
        '''
        Determines current growth rate of all particle size classes in a multicomponent system
        '''
        xEqAlpha = np.zeros((1,len(self.phases), self.numberOfElements))
        xEqBeta = np.zeros((1,len(self.phases), self.numberOfElements))
        growthRate = []
        for p in range(len(self.phases)):
            growthRate_p, xEqAlpha_p, xEqBeta_p = self._singleGrowthMulti(p, Y)
            growthRate.append(growthRate_p)
            xEqAlpha[0,p] = xEqAlpha_p
            xEqBeta[0,p] = xEqBeta_p
        Y.xEqAlpha = xEqAlpha
        Y.xEqBeta = xEqBeta
        return growthRate, Y

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
            if self.data.drivingForce[self.data.n,p] < 0 and np.all(self.data.xEqAlpha[self.data.n,p,:] == 0):
                self.PBM[p].reset()
                self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.growth[p] = np.zeros(self.PBM[p].bins+1)
                continue
            self.PBM[p].updatePBMEuler(t, x[p])
            change, addedIndices = self.PBM[p].adjustSizeClassesEuler(all(self.growth[p] < 0))
            if change:
                if self.precipitates[p].calculateAspectRatio:
                    self.eqAspectRatio[p] = self.precipitates[p].strainEnergy.eqAR_bySearch(self.PBM[p].PSDbounds, self.precipitates[p].gamma, self.precipitates[p].shapeFactor)
                else:
                    self.eqAspectRatio[p] = self.precipitates[p].shapeFactor.aspectRatio(self.PBM[p].PSDbounds)

                self.growth[p] = np.zeros(len(self.PBM[p].PSDbounds))
                if self.numberOfElements == 1:
                    if addedIndices is None:
                        #This is very slow to do
                        self._createLookupBinary(self.data.temperature[self.data.n])
                    else:
                        self.PSDXalpha[p] = np.concatenate((self.PSDXalpha[p], np.zeros((self.PBM[p].bins+1 - len(self.PSDXalpha[p]),1))))
                        self.PSDXbeta[p] = np.concatenate((self.PSDXbeta[p], np.zeros((self.PBM[p].bins+1 - len(self.PSDXbeta[p]),1))))
                        self.PSDXalpha[p][addedIndices:,0], self.PSDXbeta[p][addedIndices:,0] = self.therm.getInterfacialComposition(self.data.temperature[self.data.n], self.particleGibbs(self.PBM[p].PSDbounds[addedIndices:], self.precipitates[p].phase), precPhase=self.precipitates[p].phase)
                else:
                    self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                    self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.growth, _ = self._growthRate(self.data.copySlice(self.data.n))
            self.PBM[p].PSD[:self.RdrivingForceIndex[p]+1] = 0
            self.PBM[p].PSD[self.PBM[p].PSDsize < self.constraints.minRadius] = 0
            self.dissolutionIndex[p] = self.PBM[p].getDissolutionIndex(self.constraints.maxDissolution, self.RdrivingForceIndex[p])


                