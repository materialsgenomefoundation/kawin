import numpy as np

from kawin.GenericModel import GenericModel
from kawin.precipitation.PrecipitationParameters import Constraints, PrecipitationData, MatrixParameters, PrecipitateParameters, TemperatureParameters
import kawin.precipitation.Nucleation as nucfuncs

class PrecipitateBase(GenericModel):
    '''
    Base class for precipitation models

    The iteration method here may seem a bit odd, but it's mainly to reduce redundant calculations
    For a model, we take that dX/dt = f(t, x), so only the time and current state is needed to compute dX/dt
    While this is true for the KWN model, we need to perform quite a bit of calculations from x before being able to compute dX/dt
        This includes mass balance, nucleation rate and growth rate
    For Euler:
        dX/dt is computed using the last state of the iteration, so in _calcDependentTerms, we just copy the last state in pData
    For RK4:
        The first dX/dt is computed the same as for Euler
        Then for subsequent steps, we compute the new state using the current values of x (computed for each step of the iteration)
    In both cases, after updating x, we compute the current state and append it to pData, which is then used for the next iteration

    Parameters
    ----------
    phases : list (optional)
        Precipitate phases (array of str)
        If only one phase is considered, the default is ['beta']
    elements : list (optional)
        Solute elements in system
        Note: order of elements must correspond to order of elements set in Thermodynamics module
                Also, the list here should just be the solutes while the Thermodynamics module needs also the parent element
        If binary system, then default is ['solute']
    '''
    def __init__(self, phases = ['beta'], elements = ['solute']):
        super().__init__()
        self.elements = elements
        self.numberOfElements = len(elements)
        self.phases = np.array(phases)

        self._resetArrays()
        self.resetConstraints()
        self._isSetup = False
        self._currY = None

        #Constants
        self.Rg = 8.314     #Gas constant - J/mol-K
        self.avo = 6.022e23 #Avogadro's number (/mol)
        self.kB = self.Rg / self.avo    #Boltzmann constant (J/K)
        
        self.dTemp = 0
        self.iterationSinceTempChange = 0

        self.matrixParameters = MatrixParameters()
        self.temperatureParameters = TemperatureParameters()
        self.precipitateParameters = [PrecipitateParameters(phases[p]) for p in range(len(phases))]

        #Free energy parameters
        self.dG = [None for i in self.phases]
        self.interfacialComposition = [None for i in self.phases]

        # #Beta function for nucleation rate
        # if self.numberOfElements == 1:
        #     self._Beta = self._BetaBinary1
        # else:
        #     self._Beta = self._BetaMulti
        #     self._betaFuncs = [None for p in phases]
        #     self._defaultBeta = 20
        self.setBetaBinary()

        #Stopping conditions
        self.clearStoppingConditions()

        #Coupling models
        self.clearCouplingModels()

    def phaseIndex(self, phase = None):
        '''
        Returns index of phase in list

        Parameters
        ----------
        phase : str (optional)
            Precipitate phase (defaults to None, which will return 0)
        '''
        return 0 if phase is None else np.where(self.phases == phase)[0][0]
            
    def reset(self):
        '''
        Resets simulation results
        This does not reset the model parameters, however, it will clear any stopping conditions
        '''
        self._resetArrays()
        self.dTemp = 0
        self.iterationSinceTempChange = 0

        self._isSetup = False
        self._currY = None

        #Reset stopping conditions
        for sc in self._stoppingConditions:
            sc.reset()

    def _resetArrays(self):
        '''
        Resets and initializes arrays for all variables
            time, temperature
            matrix composition, equilibrium composition (alpha and beta)
            driving force, impingement factor, nucleation barrier, critical radius, nucleation radius
            nucleation rate, precipitate density
            average radius, average aspect ratio, volume fraction

        Extra variables include incubation offset and incubation sum

        Time dependent variables will be set up as either
            (iterations)                     time, temperature
            (iterations, elements)           composition
            (iterations, phases, elements)   eq composition, total precipitate composition
            (iterations, phases)             Everything else
            This is intended for appending arrays to always be on the first axis
        '''
        self.pData = PrecipitationData(self.phases, self.elements)

        #Temporary storage variables
        self._precBetaTemp = [None for _ in range(len(self.phases))]    #Composition of nucleate (found from driving force)

    def _getVarDict(self):
        '''
        Returns mapping of { variable name : attribute name } for saving
        The variable name will be the name in the .npz file
        '''
        saveDict = {name: name for name in self.pData.ATTRIBUTES}
        return saveDict
    
    def load(filename):
        '''
        Loads data from filename and returns a PrecipitateModel
        '''
        data = np.load(filename)
        model = PrecipitateBase(data['phases'], data['elements'])
        model._loadData(data)
        return model
    
    def _appendArrays(self, newVals):
        '''
        Appends new values to the variable list
        NOTE: newVals must correspond to the same order as _packArrays with first axis as 1
            Ex rCrit is (n, phases) so corresponding new value should be (1, phases)
        Since np append creates a new variable in memory, we have to reassign each term, then pack them into varList again
            TODO: it would be nice to reduce the number of times it copies, perhaps by preallocating some amount (say 1000)
                    for each array and if we have not reached the end of the array, just stick the values at the latest index
                    but once we reach the end of the array, we would append another 1000
                    The after solving, we could clean up the arrays, or just use self.n to state where the end of the simulation is
        I suppose we could make a list of str for each variable and call setattr
        '''
        self.pData.appendToArrays(newVals)

    def resetConstraints(self):
        '''
        Default values for contraints
        '''
        self.constraints = Constraints()

    def setConstraints(self, **kwargs):
        '''
        Sets constraints

        Possible constraints:
        ---------------------
        minRadius - minimum radius to be considered a precipitate (1e-10 m)
        maxTempChange - maximum temperature change before lookup table is updated (only for Euler in binary case) (1 K)

        maxDTFraction - maximum time increment allowed as a fraction of total simulation time (0.1)
        minDTFraction - minimum time increment allowed as a fraction of total simulation time (1e-5)

        checkTemperature - checks max temperature change (True)
        maxNonIsothermalDT - maximum time step when temperature is changing (1 second)

        checkPSD - checks maximum growth rate for particle size distribution (True)
        maxDissolution - maximum relative volume fraction of precipitates allowed to dissolve in a single time step (0.01)

        checkRcrit - checks maximum change in critical radius (False)
        maxRcritChange - maximum change in critical radius (as a fraction) per single time step (0.01)

        checkNucleation - checks maximum change in nucleation rate (True)
        maxNucleationRateChange - maximum change in nucleation rate (on log scale) per single time step (0.5)
        minNucleationRate - minimum nucleation rate to be considered for checking time intervals (1e-5)

        checkVolumePre - estimates maximum volume change (True)
        checkVolumePost - checks maximum calculated volume change (True)
        maxVolumeChange - maximum absolute value that volume fraction can change per single time step (0.001)

        minNucleateDensity - minimum nucleate density to consider nucleation to have occurred (1e-5)
        dtScale - scaling factor to attempt to progressively increase dt over time
        '''
        for key, value in kwargs.items():
            setattr(self.constraints, key, value)

    def setBetaBinary(self, functionType = 1):
        '''
        Sets function for beta calculation in binary systems
            1 for implementation seen in Perez et al, 2008 (default)
            2 for implementation similar to multicomponent systems

        If using a multicomponent system, the beta function defaults to the 2nd
            So this function will not do anything

        Parameters
        ----------
        functionType : int
            ID for function
                1 for implementation seen in Perez et al, 2008 (default)
                2 for implementation similar to multicomponent systems
        '''
        self.betaFuncType = functionType

    def setInitialComposition(self, xInit):
        '''
        Parameters
        
        xInit : float or array
            Initial composition of parent matrix phase in atomic fraction
            Use float for binary system and array of solutes for multicomponent systems
        '''
        self.matrixParameters.initComposition = xInit
        
    def setInterfacialEnergy(self, gamma, phase = None):
        '''
        Parameters
        ----------
        gamma : float
            Interfacial energy between precipitate and matrix in J/m2
        phase : str (optional)
            Phase to input interfacial energy (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.precipitateParameters[index].gamma = gamma
        
    def resetAspectRatio(self, phase = None):
        '''
        Resets aspect ratio variables of defined phase to default

        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.precipitateParameters[index].shapeFactor.setSpherical()

    def setPrecipitateShape(self, precipitateShape, phase = None, ratio = 1):
        '''
        Sets precipitate shape to user-defined shape

        Parameters
        ----------
        precipitateShape : int
            Precipitate shape (ShapeFactor.SPHERE, NEEDLE, PLATE or CUBIC)
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        ratio : float (optional)
            Aspect ratio of precipitate (long axis / short axis)
            If float, must be greater than 1
            If function, must take in radius as input and output float greater than 1
        '''
        index = self.phaseIndex(phase)
        self.precipitateParameters[index].shapeFactor.setPrecipitateShape(precipitateShape, ratio)
    
    def setVolumeAlpha(self, value, valueType, atomsPerCell):
        '''
        Sets volume parameters for parent phase

        Parameters
        ----------
        value : float
            Value for volume parameters (lattice parameter, atomic (unit cell) volume or molar volume)
        valueType : VolumeParameter
            States what volume term that value is
        atomsPerCell : int
            Number of atoms in the unit cell
        '''
        self.matrixParameters.volume.setVolume(value, valueType, atomsPerCell)

    def setVolumeBeta(self, value, valueType, atomsPerCell, phase = None):
        '''
        Sets volume parameters for precipitate phase

        Parameters
        ----------
        value : float
            Value for volume parameters (lattice parameter, atomic (unit cell) volume or molar volume)
        valueType : VolumeParameter
            States what volume term that value is
        atomsPerCell : int
            Number of atoms in the unit cell
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.precipitateParameters[index].volume.setVolume(value, valueType, atomsPerCell)

    def setNucleationDensity(self, grainSize = 100, aspectRatio = 1, dislocationDensity = 5e12, bulkN0 = None):
        '''
        Sets grain size and dislocation density which determines the available nucleation sites
        
        Parameters
        ----------
        grainSize : float (optional)
            Average grain size in microns (default at 100um if this function is not called)
        aspectRatio : float (optional)
            Aspect ratio of grains (default at 1)
        dislocationDensity : float (optional)
            Dislocation density (m/m3) (default at 5e12)
        bulkN0 : float (optional)
            This allows for the use to override the nucleation site density for bulk precipitation
            By default (None), this is calculated by the number of lattice sites containing a solute atom
            However, for calibration purposes, it may be better to set the nucleation site density manually
        '''
        self.matrixParameters.nucleation.setNucleationDensity(grainSize, aspectRatio, dislocationDensity, bulkN0)
        self.matrixParameters.nucleation._parametersSet = True
        
    def setNucleationSite(self, site, phase = None):
        '''
        Sets nucleation site type for specified phase
        If site type is grain boundaries, edges or corners, the phase morphology will be set to spherical and precipitate shape will depend on wetting angle
        
        Parameters
        ----------
        site : str
            Type of nucleation site
            Options are 'bulk', 'dislocations', 'grain_boundaries', 'grain_edges' and 'grain_corners'
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.precipitateParameters[index].GBfactor.setNucleationType(site)
            
    def setParentPhases(self, phase, parentPhases):
        '''
        Sets parent precipitates at which a precipitate can nucleate on the surface of
        
        Parameters
        ----------
        phase : str
            Precipitate phase of interest that will nucleate
        parentPhases : list
            Phases that the precipitate of interest can nucleate on the surface of
        '''
        index = self.phaseIndex(phase)
        for p in parentPhases:
            self.precipitateParameters[index].parentPhases.append(self.phaseIndex(p))
           
    def setGrainBoundaryEnergy(self, energy):
        '''
        Grain boundary energy - this will decrease the critical radius as some grain boundaries will be removed upon nucleation

        Parameters
        ----------
        energy : float
            GB energy in J/m2

        Default upon initialization is 0.3
        Note: GBenergy of 0 is equivalent to bulk precipitation
        '''
        self.matrixParameters.GBenergy = energy
        
    def setTheta(self, theta):
        '''
        This is a scaling factor for the incubation time calculation, default is 2

        Incubation time is defined as 1 / \theta * \beta * Z^2
        \theta differs by derivation. By default, this is set to 2 following the
        Feder derivation. In the Wakeshima derivation, \theta is 4pi

        Parameters
        ----------
        theta : float
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        self.matrixParameters.theta = theta

    def setTemperature(self, *args):
        '''
        Sets temperature parameter

        Options:
            temperature : float
                Isothermal temperature
            temperature : function
                Function takes in time in seconds and returns temperature
            temperature : [times, temps]
                Temperature will be interpolated between the times and temps list
                Each index in the lists will correspond to the time that temperature is reached
                Ex. [0, 15, 20], [100, 500, 400]
                    Temperature starts at 100 and ramps to 500, reaching it at 15 hours
                    Then temperature will drop to 400, reaching it at 20 hours
        '''
        self.temperatureParameters.setTemperatureParameters(*args)
        #self._incubation = self._incubationIsothermal if self.temperatureParameters._isIsothermal else self._incubationNonIsothermal
        
    def setStrainEnergy(self, strainEnergy, phase = None, calculateAspectRatio = False):
        '''
        Sets strain energy class to precipitate

        Parameters
        ----------
        strainEnergy : StrainEnergy object
        phase : str
            Precipitate phase of interest that will nucleate
        calculateAspectRatio : bool
            Will use strain energy to get aspect ratio if True
        '''
        index = self.phaseIndex(phase)
        self.precipitateParameters[index].strainEnergy = strainEnergy
        self.precipitateParameters[index].calculateAspectRatio = calculateAspectRatio

    def setInfinitePrecipitateDiffusivity(self, infinite, phase = None):
        '''
        Sets whether to assuming infinitely fast or no diffusion in phase

        Parameters
        ----------
        infinite : bool
            True will assume infinitely fast diffusion
            False will assume no diffusion
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
            Use 'all' to apply to all phases
        '''
        if phase == 'all':
            for p in range(len(self.precipitateParameters)):
                self.precipitateParameters[p].infinitePrecipitateDiffusion = infinite
        else:
            index = self.phaseIndex(phase)
            self.precipitateParameters[index].infinitePrecipitateDiffusion = infinite

    def setThermodynamics(self, thermodynamics, removeCache = False):
        self.therm = thermodynamics
        self.removeCache = removeCache

    def particleGibbs(self, radius, phase = None):
        '''
        Returns Gibbs Thomson contribution of a particle given its radius

        Parameters
        ----------
        radius : float or array
            Precipitate radius
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        return self.precipitateParameters[index].computeGibbsThomsonContribution(radius)

    def neglectEffectiveDiffusionDistance(self, neglect = True):
        '''
        Whether or not to account for effective diffusion distance dependency on the supersaturation
        By default, effective diffusion distance is considered
        
        Parameters
        ----------
        neglect : bool (optional)
            If True (default), will assume effective diffusion distance is particle radius
            If False, will calculate correction factor from Chen, Jeppson and Agren (2008)
        '''
        self.matrixParameters.effDiffDistance = self.matrixParameters.effDiffFuncs.noDiffusionDistance if neglect else self.matrixParameters.effDiffFuncs.effectiveDiffusionDistance

    def addStoppingCondition(self, condition, mode = 'or'):
        '''
        Adds condition to stop simulation when condition is met

        Parameters
        ----------
        condition: PrecipitateStoppingCondition
        mode: str
            'or' or 'and
            Conditions with 'or' will stop the simulation when at least one condition is met
            Conditions with 'and' will stop the simulation when all conditions are met
        '''
        self._stoppingConditions.append(condition)
        if mode == 'or':
            self._stopConditionMode.append(True)
        else:
            self._stopConditionMode.append(False)
        
    def clearStoppingConditions(self):
        '''
        Clears all stopping conditions
        '''
        self._stoppingConditions = []
        self._stopConditionMode = []

    def setup(self):
        '''
        Sets up hidden parameters before solving
            Nucleation site density
            Grain boundary factors
            Strain energy
        '''
        if self._isSetup:
            return
        
        if not self.matrixParameters.nucleation._parametersSet:
            #Set nucleation density assuming grain size of 100 um and dislocation density of 5e12 m/m3 (Thermocalc default)
            print('Nucleation density not set.\nSetting nucleation density assuming grain size of {:.0f} um and dislocation density of {:.0e} #/m2'.format(100, 5e12))
            self.matrixParameters.nucleation.setNucleationDensity(100, 1, 5e12)
            self.matrixParameters.nucleation._parametersSet = True
        self.matrixParameters.nucleation.setupNucleationDensity(self.matrixParameters.initComposition, self.matrixParameters.volume.Vm)
        for p in range(len(self.phases)):
            self.precipitateParameters[p].setup()

        self.pData.composition[0] = self.matrixParameters.initComposition
        self.pData.temperature[0] = self.temperatureParameters(self.pData.time[0])
        self._isSetup = True

    def printHeader(self):
        '''
        Overloads printHeader from GenericModel to do nothing
        since status displays the necessary outputs
        '''
        return

    def printStatus(self, iteration, modelTime, simTimeElapsed):
        '''
        Prints various terms at latest step

        Will print:
            Model time, simulation time, temperature, matrix composition
            For each phase
                Phase name, precipitate density, volume fraction, avg radius and driving force
        '''
        i = self.pData.n
        #For single element, we just print the composition as matrix comp in terms of the solute
        if self.numberOfElements == 1:
            print('N\tTime (s)\tSim Time (s)\tTemperature (K)\tMatrix Comp')
            print('{:.0f}\t{:.1e}\t\t{:.1f}\t\t{:.0f}\t\t{:.4f}\n'.format(i, modelTime, simTimeElapsed, self.pData.temperature[i], 100*self.pData.composition[i,0]))
        #For multicomponent systems, print each element
        else:
            compStr = 'N\tTime (s)\tSim Time (s)\tTemperature (K)\t'
            compValStr = '{:.0f}\t{:.1e}\t\t{:.1f}\t\t{:.0f}\t\t'.format(i, modelTime, simTimeElapsed, self.pData.temperature[i])
            for a in range(self.numberOfElements):
                compStr += self.elements[a] + '\t'
                compValStr += '{:.4f}\t'.format(100*self.pData.composition[i,a])
            compValStr += '\n'
            print(compStr)
            print(compValStr)

        #Print status of each phase
        print('\tPhase\tPrec Density (#/m3)\tVolume Frac\tAvg Radius (m)\tDriving Force (J/mol)')
        for p in range(len(self.phases)):
            print('\t{}\t{:.3e}\t\t{:.4f}\t\t{:.4e}\t{:.4e}'.format(self.phases[p], self.pData.precipitateDensity[i,p], 100*self.pData.volFrac[i,p], self.pData.Ravg[i,p], self.pData.drivingForce[i,p]*self.precipitateParameters[p].volume.Vm))
        print('')

    def preProcess(self):
        '''
        Store array for non-derivative terms (which is everything except for the PBM models)

        We use these terms for the first step of the iterators (for Euler, this is all the steps)
            For RK4, these terms will be recalculated in dXdt
        '''
        self._currY = None
        return
    
    def _calculateDependentTerms(self, t, x):
        '''
        Gets all dependent terms (everything but PBM variables) that are needed to find dXdt

        Steps:
            1. Mass balance
            2. Driving force - must be done after mass balance to get the current matrix composition
            3. Growth rate - must be done after driving force since dG is needed in multicomponent systems
            4. Nucleation rate
            5. Nucleate radius - must be done after nucleation rate since derived classes can change nucleation rate

        For the first iteration, self._currY will be None from the preProcess function, in this case, we want
            to just grab the latest values to avoid double calculations
        '''
        self._processX(x)
        if self._currY is None:
            #print('start iteration')
            self._currY = self.pData.copySlice(self.pData.n)
        else:
            self._currY.time = np.array([t])
            self._currY.temperature = np.array([self.temperatureParameters(t)])
            self._currY = self._calcMassBalance(t, x, self._currY)
            self._currY = self._calcNucleationRateAlt(t, x, self._currY)
            self.growth, self._currY = self._growthRate(self._currY)

    def getdXdt(self, t, x):
        '''
        Gets dXdt as a list for each phase
        For the eulerian implementation, this is dn_i/dt for the bins in PBM for each phase

        This is partially kind of dumb to have getdXdt and _getdXdt, however:
            getdXdt is to be compatible with GenericModel, which requires only x and t
            _getdXdt is to account for current PrecipitationData and growth rate as defined in this model
        '''
        self._calculateDependentTerms(t, x)
        return self._getdXdt(t, x, self._currY, self.growth)
    
    def _getdXdt(self, t, x, Y : PrecipitationData, growth):
        raise NotImplementedError()
    
    def correctdXdt(self, dt, x, dXdt):
        return self._correctdXdt(dt, x, dXdt, self._currY, self.growth)
    
    def _correctdXdt(self, dt, x, dXdt, Y : PrecipitationData, growth):
        raise NotImplementedError()

    def postProcess(self, t, x):
        '''
        1) Updates internal arrays with new values of t and x
        2) Updates particle size distribution
        3) Updates coupled models
        4) Check stopping conditions
        5) Return new values and whether to stop the model
        '''
        self._calculateDependentTerms(t, x)
        self._appendArrays(self._currY)

        #Update particle size distribution (this includes adding bins, resizing bins, etc)
        #Should be agnostic of eulerian or lagrangian implementations
        self._updateParticleSizeDistribution(t, x)

        #Update coupled models
        self.updateCoupledModels()

        #Check stopping conditions
        orCondition = False
        andCondition = True
        numAndCondition = 0
        for i in range(len(self._stoppingConditions)):
            self._stoppingConditions[i].testCondition(self)
            if self._stopConditionMode[i]:
                orCondition = orCondition or self._stoppingConditions[i].isSatisfied()
            else:
                andCondition = andCondition and self._stoppingConditions[i].isSatisfied()
                numAndCondition += 1

        #If no and conditions, then andCondition will still be True, so set to False
        if numAndCondition == 0:
            andCondition = False

        stop = orCondition or andCondition

        return self.getCurrentX()[1], stop
    
    def _processX(self, x):
        raise NotImplementedError()
    
    def _calcMassBalance(self, t, x):
        raise NotImplementedError()
    
    def _updateParticleSizeDistribution(self, t, x):
        raise NotImplementedError()
    
    def _calcNucleationSites(self, t, x, p):
        raise NotImplementedError()
    
    def _growthRate(self):
        raise NotImplementedError()
    
    def _calcNucleationRateAlt(self, t, x, Y : PrecipitationData):
        xComp = np.squeeze(Y.composition[0])
        T = Y.temperature[0]
        for p in range(len(self.precipitateParameters)):
            precParams = self.precipitateParameters[p]

            # Compute driving force and precipitate composition (which helps with growth rate and impingement in multicomponent systems)
            # If driving force is negative, then we can skip the rest of the calculations (no nucleation barrier and no nucleation rate)
            aspectRatio = precParams.shapeFactor.aspectRatio(self.pData.Rcrit[self.pData.n, p])
            _, volDG, self._precBetaTemp[p] = nucfuncs.volumetricDrivingForce(self.therm, xComp, T, precParams, aspectRatio, self.removeCache)
            Y.drivingForce[0,p] = volDG
            if volDG < 0:
                continue

            # Critical Gibbs free energy and radius at nucleation barrier
            Rcrit, Gcrit = nucfuncs.nucleationBarrier(volDG, precParams, aspectRatio)
            
            # Impingement factor
            if self.therm._isBinary:
                if self.betaFuncType == 1:
                    beta = nucfuncs.betaBinary1(self.therm, xComp, T, Rcrit, self.matrixParameters, precParams, self.removeCache)
                else:
                    beta = nucfuncs.betaBinary2(self.therm, xComp, T, Rcrit, self.matrixParameters, precParams, Y.xEqAlpha[0], Y.xEqBeta[0], self.removeCache)
            else:
                beta = nucfuncs.betaMulti(self.therm, xComp, T, Rcrit, self.matrixParameters, precParams, self.removeCache, searchDir=self._precBetaTemp[p])
            
            # If impingement is 0, then skip rest of calculations (no nucleation rate)
            if beta == 0:
                continue

            # Zeldovich factor
            Z = nucfuncs.zeldovich(T, Rcrit, precParams)

            # Incubation time
            if self.temperatureParameters._isIsothermal:
                tau = nucfuncs.incubationTime(beta, Z, self.matrixParameters)
            else:
                tau = nucfuncs.incubationTimeNonIsothermal(Z, beta, t, T, self.pData.impingement[:,p], self.pData.time, self.pData.temperature, self.matrixParameters)
            
            # Nucleation rate
            nucRate = nucfuncs.nucleationRate(Z, beta, Gcrit, T, tau, time=t)
            nucRate *= self._calcNucleationSites(t, x, p)  # don't forget to add nucleation sites since we compare this to min nucleation rate

            # TODO: using 0.01 seems arbitrary here, is there a better way to do this?
            dt = t if self.pData.n == 0 else self.pData.time[self.pData.n] - self.pData.time[self.pData.n-1]
            if nucRate*dt >= self.constraints.minNucleateDensity and Rcrit >= self.precipitateParameters[p].Rmin:
                Rnuc = nucfuncs.nucleationRadius(T, Rcrit, precParams)
            else:
                Rnuc = 0

            # Store terms into PrecipitateData object
            Y.Rcrit[0,p] = Rcrit
            Y.Gcrit[0,p] = Gcrit
            Y.impingement[0,p] = beta
            Y.nucRate[0,p] = nucRate
            Y.Rnuc[0,p] = Rnuc

        return Y