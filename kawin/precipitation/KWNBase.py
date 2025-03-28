import numpy as np

from kawin.GenericModel import GenericModel
from kawin.thermo import GeneralThermodynamics
from kawin.precipitation.PrecipitationParameters import Constraints, PrecipitationData, MatrixParameters, PrecipitateParameters, TemperatureParameters
import kawin.precipitation.NucleationRate as nucfuncs

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
    def __init__(self, 
                 matrix: MatrixParameters,
                 precipitates: list[PrecipitateParameters], 
                 thermodynamics: GeneralThermodynamics,
                 temperature: TemperatureParameters, 
                 constraints: Constraints = None):
        super().__init__()

        self.constraints = constraints if constraints is not None else Constraints()
        self.temperatureParameters = TemperatureParameters(temperature)
        self.therm = thermodynamics
        self.removeCache = False

        if isinstance(precipitates, PrecipitateParameters):
            precipitates = [precipitates]
        self.precipitates = precipitates
        self.phases = np.array([p.phase for p in self.precipitates])

        self.matrix = matrix
        self.elements = self.matrix.solutes

        self.numberOfElements = len(self.elements)

        self.dTemp = 0
        self.iterationSinceTempChange = 0

        self._resetArrays()
        self._isSetup = False
        self._currY = None

        self.setBetaBinary()

        #Stopping conditions
        self.clearStoppingConditions()

        #Coupling models
        self.clearCouplingModels()

    def cacheCalculations(self, useCache: bool = False):
        self.removeCache = not useCache

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
        super().reset()
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
        self.data = PrecipitationData(self.phases, self.elements)

        #Temporary storage variables
        self._precBetaTemp = [None for _ in range(len(self.phases))]    #Composition of nucleate (found from driving force)

    def toDict(self):
        '''
        Converts precipitation data to dictionary
        '''
        data = self.data.toDict()
        return data
    
    def fromDict(self, data):
        '''
        Converts dictionary of data to precipitation data
        '''
        self.data.fromDict(data)
    
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
        self.data.appendToArrays(newVals)

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
        return self.precipitates[index].computeGibbsThomsonContribution(radius)

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
        
        for p in range(len(self.phases)):
            self.precipitates[p].nucleation.gbEnergy = self.matrix.GBenergy
            self.precipitates[p].validate()

        self.data.composition[0] = self.matrix.initComposition
        self.data.temperature[0] = self.temperatureParameters(self.data.time[0])
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
        i = self.data.n
        #For single element, we just print the composition as matrix comp in terms of the solute
        if self.numberOfElements == 1:
            print('N\tTime (s)\tSim Time (s)\tTemperature (K)\tMatrix Comp')
            print('{:.0f}\t{:.1e}\t\t{:.1f}\t\t{:.0f}\t\t{:.4f}\n'.format(i, modelTime, simTimeElapsed, self.data.temperature[i], 100*self.data.composition[i,0]))
        #For multicomponent systems, print each element
        else:
            compStr = 'N\tTime (s)\tSim Time (s)\tTemperature (K)\t'
            compValStr = '{:.0f}\t{:.1e}\t\t{:.1f}\t\t{:.0f}\t\t'.format(i, modelTime, simTimeElapsed, self.data.temperature[i])
            for a in range(self.numberOfElements):
                compStr += self.elements[a] + '\t'
                compValStr += '{:.4f}\t'.format(100*self.data.composition[i,a])
            compValStr += '\n'
            print(compStr)
            print(compValStr)

        #Print status of each phase
        print('\tPhase\tPrec Density (#/m3)\tVolume Frac\tAvg Radius (m)\tDriving Force (J/mol)')
        for p in range(len(self.phases)):
            print('\t{}\t{:.3e}\t\t{:.4f}\t\t{:.4e}\t{:.4e}'.format(self.phases[p], self.data.precipitateDensity[i,p], 100*self.data.volFrac[i,p], self.data.Ravg[i,p], self.data.drivingForce[i,p]*self.precipitates[p].volume.Vm))
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
            self._currY = self.data.copySlice(self.data.n)
        else:
            self._currY.time = np.array([t])
            self._currY.temperature = np.array([self.temperatureParameters(t)])
            self._currY = self._calcMassBalance(t, x, self._currY)
            self._currY = self._calcNucleationRate(t, x, self._currY)
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
        super().postProcess(t, x)
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

        return self.getCurrentX(), stop
    
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
    
    def _calcNucleationRate(self, t, x, Y : PrecipitationData):
        xComp = np.squeeze(Y.composition[0])
        T = Y.temperature[0]
        for p in range(len(self.precipitates)):
            precParams = self.precipitates[p]

            # Compute driving force and precipitate composition (which helps with growth rate and impingement in multicomponent systems)
            # If driving force is negative, then we can skip the rest of the calculations (no nucleation barrier and no nucleation rate)
            aspectRatio = precParams.shapeFactor.aspectRatio(self.data.Rcrit[self.data.n, p])
            _, volDG, self._precBetaTemp[p] = nucfuncs.volumetricDrivingForce(self.therm, xComp, T, precParams, aspectRatio, self.removeCache)
            Y.drivingForce[0,p] = volDG
            if volDG < 0:
                continue

            # Critical Gibbs free energy and radius at nucleation barrier
            Rcrit, Gcrit = nucfuncs.nucleationBarrier(volDG, precParams, aspectRatio)
            
            # Impingement factor
            if self.therm.numElements == 2:
                if self.betaFuncType == 1:
                    beta = nucfuncs.betaBinary1(self.therm, xComp, T, Rcrit, self.matrix, precParams, self.removeCache)
                else:
                    beta = nucfuncs.betaBinary2(self.therm, xComp, T, Rcrit, self.matrix, precParams, Y.xEqAlpha[0], Y.xEqBeta[0], self.removeCache)
            else:
                beta = nucfuncs.betaMulti(self.therm, xComp, T, Rcrit, self.matrix, precParams, self.removeCache, searchDir=self._precBetaTemp[p])
            
            # If impingement is 0, then skip rest of calculations (no nucleation rate)
            if beta == 0:
                continue

            # Zeldovich factor
            Z = nucfuncs.zeldovich(T, Rcrit, precParams)

            # Incubation time
            if self.temperatureParameters._isIsothermal:
                tau = nucfuncs.incubationTime(beta, Z, self.matrix)
            else:
                tau = nucfuncs.incubationTimeNonIsothermal(Z, beta, t, T, self.data.impingement[:,p], self.data.time, self.data.temperature, self.matrix)
            
            # Nucleation rate
            nucRate = nucfuncs.nucleationRate(Z, beta, Gcrit, T, tau, time=t)
            nucRate *= self._calcNucleationSites(t, x, p)  # don't forget to add nucleation sites since we compare this to min nucleation rate

            # TODO: using 0.01 seems arbitrary here, is there a better way to do this?
            dt = t if self.data.n == 0 else self.data.time[self.data.n] - self.data.time[self.data.n-1]
            if nucRate*dt >= self.constraints.minNucleateDensity and Rcrit >= self.precipitates[p].Rmin:
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