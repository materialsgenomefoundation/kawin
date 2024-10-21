from dataclasses import dataclass
from enum import Enum

import numpy as np
from kawin.precipitation.non_ideal.EffectiveDiffusion import EffectiveDiffusionFunctions
from kawin.precipitation.non_ideal.ShapeFactors import ShapeFactor
from kawin.precipitation.non_ideal.ElasticFactors import StrainEnergy
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
from kawin.GenericModel import GenericModel
from kawin.precipitation.PrecipitationParameters import Constraints

class VolumeParameter(Enum):
    MOLAR_VOLUME = 0
    ATOMIC_VOLUME = 1
    LATTICE_PARAMETER = 2

class PrecipitateBase(GenericModel):
    '''
    Base class for precipitation models

    Parameters
    ----------
    phases : list (optional)
        Precipitate phases (array of str)
        If only one phase is considered, the default is ['beta']
    elements : list (optional)
        Solute elements in system
        Note: order of elements must correspond to order of elements set in Thermodynamics module
                Also, the list here should just be the solutes while the Thermodynamics module needs also the parent element
        If binary system, then defualt is ['solute']
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

        #Default variables, these terms won't have to be set before simulation
        self.strainEnergy = [StrainEnergy() for i in self.phases]
        self.calculateAspectRatio = [False for i in self.phases]
        self.RdrivingForceLimit = np.zeros(len(self.phases), dtype=np.float32)
        self.shapeFactors = [ShapeFactor() for i in self.phases]
        self.theta = 2 * np.ones(len(self.phases), dtype=np.float32)
        self.effDiffFuncs = EffectiveDiffusionFunctions()
        self.effDiffDistance = self.effDiffFuncs.effectiveDiffusionDistance
        self.infinitePrecipitateDiffusion = [True for i in self.phases]
        self.dTemp = 0
        self.iterationSinceTempChange = 0
        self.GBenergy = 0.3     #J/m2
        self.parentPhases = [[] for i in self.phases]
        self.GB = [GBFactors() for p in self.phases]
        
        #Set other variables to None to throw errors if not set
        self.xInit = None
        self.Tparameters = None

        #Nucleation site density, it will default to dislocations with 5e12 /m2 density
        self._isNucleationSetup = False
        self.GBareaN0 = None
        self.GBedgeN0 = None
        self.GBcornerN0 = None
        self.dislocationN0 = None
        self.bulkN0 = None
        
        #Unit cell parameters
        self.aAlpha = None
        self.VaAlpha = None
        self.VmAlpha = None
        self.atomsPerCellAlpha = None
        self.atomsPerCellBeta = np.empty(len(self.phases), dtype=np.float32)
        self.VaBeta = np.empty(len(self.phases), dtype=np.float32)
        self.VmBeta = np.empty(len(self.phases), dtype=np.float32)
        self.Rmin = np.empty(len(self.phases), dtype=np.float32)
        
        #Free energy parameters
        self.gamma = np.empty(len(self.phases), dtype=np.float32)
        self.dG = [None for i in self.phases]
        self.interfacialComposition = [None for i in self.phases]

        #Beta function for nucleation rate
        if self.numberOfElements == 1:
            self._Beta = self._BetaBinary1
        else:
            self._Beta = self._BetaMulti
            self._betaFuncs = [None for p in phases]
            self._defaultBeta = 20

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
        self.xComp[0] = self.xInit
        self.dTemp = 0

        self._isSetup = False
        self._currY = None

        #Reset temperature array
        if np.isscalar(self.Tparameters):
            self.setTemperature(self.Tparameters)
        elif len(self.Tparameters) == 2:
            self.setTemperatureArray(*self.Tparameters)
        elif self.Tparameters is not None:
            self.setNonIsothermalTemperature(self.Tparameters)

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
        self.n = 0

        #Time
        self.time = np.zeros(1)

        #Temperature
        self.temperature = np.zeros(1)

        #Composition
        self.xComp = np.zeros((1, self.numberOfElements))                       #Matrix composition
        self.xEqAlpha = np.zeros((1, len(self.phases), self.numberOfElements))  #Equilibrium matrix composition
        self.xEqBeta = np.zeros((1, len(self.phases), self.numberOfElements))   #Equilibrium beta compostion

        #Nucleation
        self.dGs = np.zeros((1, len(self.phases)))                  #Driving force
        self.betas = np.zeros((1, len(self.phases)))                #Impingement rates (used for non-isothermal)
        self.incubationOffset = np.zeros(len(self.phases))          #Offset for incubation time (for non-isothermal precipitation)
        self.Gcrit = np.zeros((1, len(self.phases)))                #Height of nucleation barrier
        self.Rcrit = np.zeros((1, len(self.phases)))                #Critical radius
        self.Rad = np.zeros((1, len(self.phases)))                  #Radius of particles formed at each time step
        
        self.nucRate = np.zeros((1, len(self.phases)))              #Nucleation rate
        self.precipitateDensity = np.zeros((1, len(self.phases)))   #Number of nucleates
        
        #Average radius and precipitate fraction
        self.avgR = np.zeros((1, len(self.phases)))                 #Average radius
        self.avgAR = np.zeros((1, len(self.phases)))                #Mean aspect ratio
        self.betaFrac = np.zeros((1, len(self.phases)))             #Fraction of precipitate

        #Fconc - auxiliary array to store total solute composition of precipitates
        self.fConc = np.zeros((1, len(self.phases), self.numberOfElements))

        self._setEnum()
        self._packArrays()

        #Temporary storage variables
        self._precBetaTemp = [None for _ in range(len(self.phases))]    #Composition of nucleate (found from driving force)

    def _setEnum(self):
        '''
        Pseudo-enumeration

        This is just to keep a consistent list of IDs for each variable
        so we can grab the current values from varList
        '''
        self.TIME = 0
        self.TEMPERATURE = 1
        self.COMPOSITION = 2
        self.EQ_COMP_ALPHA = 3
        self.EQ_COMP_BETA = 4
        self.DRIVING_FORCE = 5
        self.IMPINGEMENT = 6
        self.G_CRIT = 7
        self.R_CRIT = 8
        self.R_NUC = 9
        self.NUC_RATE = 10
        self.PREC_DENS = 11
        self.R_AVG = 12
        self.AR_AVG = 13
        self.VOL_FRAC = 14
        self.FCONC = 15
        self.NUM_TERMS = 16

    def _packArrays(self):
        '''
        Create internal list of variables to solve for
        The "enumerators" in getEnum will serve as the indexing for this list
            Make sure the arrays in here and getEnum correspond to the same values
        '''
        self.varList = [
                self.time, 
                self.temperature, 
                self.xComp, 
                self.xEqAlpha, 
                self.xEqBeta,
                self.dGs, 
                self.betas, 
                self.Gcrit, 
                self.Rcrit, 
                self.Rad,
                self.nucRate, 
                self.precipitateDensity,
                self.avgR, 
                self.avgAR, 
                self.betaFrac,
                self.fConc
                ]

    def _getVarDict(self):
        '''
        Returns mapping of { variable name : attribute name } for saving
        The variable name will be the name in the .npz file
        '''
        saveDict = {
            'elements': 'elements',
            'phases': 'phases',
            'time': 'time',
            'temperature': 'temperature',
            'composition': 'xComp',
            'xEqAlpha': 'xEqAlpha',
            'xEqBeta': 'xEqBeta',
            'drivingForce': 'dGs',
            'impingement': 'betas',
            'Gcrit': 'Gcrit',
            'Rcrit': 'Rcrit',
            'nucRadius': 'Rad',
            'nucRate': 'nucRate',
            'precipitateDensity': 'precipitateDensity',
            'avgRadius': 'avgR',
            'avgAspectRatio': 'avgAR',
            'volFrac': 'betaFrac',
            'fConc': 'fConc',
        }
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
        self.time = np.append(self.time, newVals[self.TIME], axis=0)
        self.temperature = np.append(self.temperature, newVals[self.TEMPERATURE], axis=0)
        self.xComp = np.append(self.xComp, newVals[self.COMPOSITION], axis=0)
        self.xEqAlpha = np.append(self.xEqAlpha, newVals[self.EQ_COMP_ALPHA], axis=0)
        self.xEqBeta = np.append(self.xEqBeta, newVals[self.EQ_COMP_BETA], axis=0)
        self.dGs = np.append(self.dGs, newVals[self.DRIVING_FORCE], axis=0)
        self.betas = np.append(self.betas, newVals[self.IMPINGEMENT], axis=0)
        self.Gcrit = np.append(self.Gcrit, newVals[self.G_CRIT], axis=0)
        self.Rcrit = np.append(self.Rcrit, newVals[self.R_CRIT], axis=0)
        self.Rad = np.append(self.Rad, newVals[self.R_NUC], axis=0)
        self.nucRate = np.append(self.nucRate, newVals[self.NUC_RATE], axis=0)
        self.precipitateDensity = np.append(self.precipitateDensity, newVals[self.PREC_DENS], axis=0)
        self.avgR = np.append(self.avgR, newVals[self.R_AVG], axis=0)
        self.avgAR = np.append(self.avgAR, newVals[self.AR_AVG], axis=0)
        self.betaFrac = np.append(self.betaFrac, newVals[self.VOL_FRAC], axis=0)
        self.fConc = np.append(self.fConc, newVals[self.FCONC], axis=0)
        self._packArrays()
        self.n += 1

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
        if self.numberOfElements == 1:
            if functionType == 2:
                self.beta = self._BetaBinary2
            else:
                self.beta = self._BetaBinary1

    def setInitialComposition(self, xInit):
        '''
        Parameters
        
        xInit : float or array
            Initial composition of parent matrix phase in atomic fraction
            Use float for binary system and array of solutes for multicomponent systems
        '''
        self.xInit = xInit
        self.xComp[0] = xInit
        
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
        self.gamma[index] = gamma
        
    def resetAspectRatio(self, phase = None):
        '''
        Resets aspect ratio variables of defined phase to default

        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.shapeFactors[index].setSpherical()

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
        self.shapeFactors[index].setPrecipitateShape(precipitateShape, ratio)

    def _setVolume(self, value, valueType: VolumeParameter, atomsPerCell):
        '''
        Private function that returns Vm, Va, a, atomsPerCell given a VolumeParameter and atomsPerCell

        Parameters
        ----------
        value : float
            Value for volume parameters (lattice parameter, atomic (unit cell) volume or molar volume)
        valueType : VolumeParameter
            States what volume term that value is
        atomsPerCell : int
            Number of atoms in the unit cell
        '''
        if valueType == VolumeParameter.MOLAR_VOLUME:
            Vm = value
            Va = atomsPerCell * Vm / self.avo
            a = np.cbrt(Va)
        elif valueType == VolumeParameter.ATOMIC_VOLUME:
            Va = value
            Vm = Va * self.avo / atomsPerCell
            a = np.cbrt(Va)
        elif valueType == VolumeParameter.LATTICE_PARAMETER:
            a = value
            Va = a**3
            Vm = Va * self.avo / atomsPerCell
        return Vm, Va, a, atomsPerCell
    
    def setVolumeAlpha(self, value, valueType: VolumeParameter, atomsPerCell):
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
        self.VmAlpha, self.VaAlpha, self.aAlpha, self.atomsPerCellAlpha = self._setVolume(value, valueType, atomsPerCell)

    def setVolumeBeta(self, value, valueType: VolumeParameter, atomsPerCell, phase = None):
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
        self.VmBeta[index], self.VaBeta[index], _, self.atomsPerCellBeta[index] = self._setVolume(value, valueType, atomsPerCell)

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
        self.grainSize = grainSize * 1e-6
        self.grainAspectRatio = aspectRatio
        self.dislocationDensity = dislocationDensity

        self.bulkN0 = bulkN0
        self._isNucleationSetup = True

    def _getNucleationDensity(self):
        '''
        Calculates nucleation density
        This is separated from setting nucleation density to
            allow it to be called right before the simulation starts
        '''
        #Set bulk nucleation site to the number of solutes per unit volume
        #   This is the represent that any solute atom can be a nucleation site
        #NOTE: some texts will state the bulk nucleation sites to just be the number
        #       of lattice sites per unit volume. The justification for this would be 
        #       the solutes can diffuse around to any lattice site and nucleate there
        if self.bulkN0 is None:
            if self.numberOfElements == 1:
                self.bulkN0 = self.xComp[0] * (self.avo / self.VmAlpha)
            else:
                self.bulkN0 = np.amin(self.xComp[0,:]) * (self.avo / self.VmAlpha)

        self.dislocationN0 = self.dislocationDensity * (self.avo / self.VmAlpha)**(1/3)
        
        if self.grainSize != np.inf:
            #Number of lattice sites on grain boundaries (#/m3)
            if self.GBareaN0 is None:
                self.GBareaN0 = (6 * np.sqrt(1 + 2 * self.grainAspectRatio**2) + 1 + 2 * self.grainAspectRatio) / (4 * self.grainAspectRatio * self.grainSize)
                self.GBareaN0 *= (self.avo / self.VmAlpha)**(2/3)
            #Number of lattice sites on grain edges (#/m3)
            if self.GBedgeN0 is None:
                self.GBedgeN0 = 2 * (np.sqrt(2) + 2 * np.sqrt(1 + self.grainAspectRatio**2)) / (self.grainAspectRatio * self.grainSize**2)
                self.GBedgeN0 *= (self.avo / self.VmAlpha)**(1/3)
            #Number of lattice sites on grain corners (which is just the number of corners) (#/m3)
            if self.GBcornerN0 is None:
                self.GBcornerN0 = 12 / (self.grainAspectRatio * self.grainSize**3)
        else:
            self.GBareaN0 = 0
            self.GBedgeN0 = 0
            self.GBcornerN0 = 0
        
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

        self.GB[index].setNucleationType(site)
        
        if self.GB[index].nucleationSiteType != GBFactors.BULK and self.GB[index].nucleationSiteType != GBFactors.DISLOCATION:
            self.shapeFactors[index].setSpherical()
            
    def _setGBfactors(self):
        '''
        Calcualtes factors for bulk or grain boundary nucleation
        This is separated from setting the nucleation sites to allow
        it to be called right before simulation
        '''
        for p in range(len(self.phases)):
            self.GB[p].setFactors(self.GBenergy, self.gamma[p])
                    
    def _GBareaRemoval(self, p):
        '''
        Returns factor to multiply radius by to give the equivalent radius of circles representing the area of grain boundary removal

        Parameters
        ----------
        p : int
            Index for phase
        '''
        if self.GB[p].nucleationSiteType == GBFactors.BULK or self.GB[p].nucleationSiteType == GBFactors.DISLOCATION:
            return 1
        else:
            return np.sqrt(self.GB[p].gbRemoval / np.pi)
            
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
            self.parentPhases[index].append(self.phaseIndex(p))
           
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
        self.GBenergy = energy
        
    def setTheta(self, theta, phase = None):
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
        index = self.phaseIndex(phase)
        self.theta[index] = theta

    def setTemperature(self, temperature):
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
        self.Tparameters = temperature
        self.temperature[0] = self.getTemperature(0)
        if np.isscalar(temperature):
            self._incubation = self._incubationIsothermal
        else:
            self._incubation = self._incubationNonIsothermal

    def getTemperature(self, t):
        '''
        Gets temperature at time t

        Options:
            Options:
            temperature : float
                Returns temperature
            temperature : function
                Returns evaluated temperature function at time t
            temperature : [times, temps]
                If t < time[0] -> return first temperature
                If t > time[-1] -> return last temperature
                Else, find the two times that t is between and interpolate
        '''
        if np.isscalar(self.Tparameters):
            return self.Tparameters
        elif len(self.Tparameters) == 2:
            if t/3600 < self.Tparameters[0][0]:
                return self.Tparameters[1][0]
            for i in range(len(self.Tparameters[0])-1):
                if t/3600 >= self.Tparameters[0][i] and t/3600 < self.Tparameters[0][i+1]:
                    t0, tf, T0, Tf = self.Tparameters[0][i], self.Tparameters[0][i+1], self.Tparameters[1][i], self.Tparameters[1][i+1]
                    return (Tf - T0) / (tf - t0) * (t/3600 - t0) + T0
            return self.Tparameters[1][-1]
        elif self.Tparameters is not None:
            return self.Tparameters(t)
        else:
            return None
        
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
        self.strainEnergy[index] = strainEnergy
        self.calculateAspectRatio[index] = calculateAspectRatio

    def _setupStrainEnergyFactors(self):
        ''''
        For each phase, the strain energy calculation will be set to assume
        a spherical, cubic or ellipsoidal shape depending on the defined shape factors
        '''
        for i in range(len(self.phases)):
            self.strainEnergy[i].setup()
            if self.strainEnergy[i].type != StrainEnergy.CONSTANT:
                if self.shapeFactors[i].particleType == ShapeFactor.SPHERE:
                    self.strainEnergy[i].setSpherical()
                elif self.shapeFactors[i].particleType == ShapeFactor.CUBIC:
                    self.strainEnergy[i].setCuboidal()
                else:
                    self.strainEnergy[i].setEllipsoidal()

    def setDiffusivity(self, diffusivity):
        '''
        For binary systems only

        Parameters
        ----------
        diffusivity : function taking 
            Composition and temperature (K) and returning diffusivity (m2/s)
            Function must have inputs in this order: f(x, T)
        '''
        self.Diffusivity = diffusivity

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
            self.infinitePrecipitateDiffusion = [infinite for i in range(len(self.phases))]
        else:
            index = self.phaseIndex(phase)
            self.infinitePrecipitateDiffusion[index] = infinite

    def setThermodynamics(self, therm, phase = None, removeCache = False, addDiffusivity = True):
        '''
        Parameters
        ----------
        therm : Thermodynamics class
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        removeCache : bool (optional)
            Will not cache equilibrium results if True (defaults to False)
        addDiffusivity : bool (optional)
            For binary systems, will add diffusivity functions from the database if present
            Defaults to True
        '''
        index = self.phaseIndex(phase)
        self.dG[index] = lambda x, T, removeCache = removeCache: therm.getDrivingForce(x, T, precPhase=phase, removeCache = removeCache)
        
        if self.numberOfElements == 1:
            self.interfacialComposition[index] = lambda x, T: therm.getInterfacialComposition(x, T, precPhase=phase)
            if (therm.mobCallables is not None or therm.diffCallables is not None) and addDiffusivity:
                self.Diffusivity = lambda x, T, removeCache = removeCache: therm.getInterdiffusivity(x, T, removeCache = removeCache)
        else:
            self.interfacialComposition[index] = lambda x, T, dG, R, gExtra, removeCache = removeCache, searchDir = None: therm.getGrowthAndInterfacialComposition(x, T, dG, R, gExtra, precPhase=phase, removeCache = False, searchDir = searchDir)
            self._betaFuncs[index] = lambda x, T, removeCache = removeCache, searchDir = None: therm.impingementFactor(x, T, precPhase=phase, removeCache = False, searchDir = searchDir)

    def setSurrogate(self, surr, phase = None):
        '''
        Parameters
        ----------
        surr : Surrogate class
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.dG[index] = surr.getDrivingForce
        
        if self.numberOfElements == 1:
            self.interfacialComposition[index] = surr.getInterfacialComposition
        else:
            self.interfacialComposition[index] = surr.getGrowthAndInterfacialComposition
            self._betaFuncs[index] = surr.impingementFactor

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
        return self.VmBeta[index] * (self.strainEnergy[index].strainEnergy(self.shapeFactors[index].normalRadii(radius)) + 2 * self.shapeFactors[index].thermoFactor(radius) * self.gamma[index] / radius)

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
        self.effDiffDistance = self.effDiffFuncs.noDiffusionDistance if neglect else self.effDiffFuncs.effectiveDiffusionDistance

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
        
        if not self._isNucleationSetup:
            #Set nucleation density assuming grain size of 100 um and dislocation density of 5e12 m/m3 (Thermocalc default)
            print('Nucleation density not set.\nSetting nucleation density assuming grain size of {:.0f} um and dislocation density of {:.0e} #/m2'.format(100, 5e12))
            self.setNucleationDensity(100, 1, 5e12)
        for p in range(len(self.phases)):
            self.Rmin[p] = self.constraints.minRadius
        self._getNucleationDensity()
        self._setGBfactors()
        self._setupStrainEnergyFactors()
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
        i = len(self.time)-1
        #For single element, we just print the composition as matrix comp in terms of the solute
        if self.numberOfElements == 1:
            print('N\tTime (s)\tSim Time (s)\tTemperature (K)\tMatrix Comp')
            print('{:.0f}\t{:.1e}\t\t{:.1f}\t\t{:.0f}\t\t{:.4f}\n'.format(i, modelTime, simTimeElapsed, self.temperature[i], 100*self.xComp[i,0]))
        #For multicomponent systems, print each element
        else:
            compStr = 'N\tTime (s)\tSim Time (s)\tTemperature (K)\t'
            compValStr = '{:.0f}\t{:.1e}\t\t{:.1f}\t\t{:.0f}\t\t'.format(i, modelTime, simTimeElapsed, self.temperature[i])
            for a in range(self.numberOfElements):
                compStr += self.elements[a] + '\t'
                compValStr += '{:.4f}\t'.format(100*self.xComp[i,a])
            compValStr += '\n'
            print(compStr)
            print(compValStr)

        #Print status of each phase
        print('\tPhase\tPrec Density (#/m3)\tVolume Frac\tAvg Radius (m)\tDriving Force (J/mol)')
        for p in range(len(self.phases)):
            print('\t{}\t{:.3e}\t\t{:.4f}\t\t{:.4e}\t{:.4e}'.format(self.phases[p], self.precipitateDensity[i,p], 100*self.betaFrac[i,p], self.avgR[i,p], self.dGs[i,p]*self.VmBeta[p]))
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
            self._currY = [np.array([self.varList[i][self.n]]) for i in range(self.NUM_TERMS)]
        else:
            self._currY[self.TIME] = np.array([t])
            self._currY[self.TEMPERATURE] = np.array([self.getTemperature(t)])
            self._calcMassBalance(t, x)
            self._calcDrivingForce(t, x)
            self._growthRate()
            self._calcNucleationRate(t, x)
            self._setNucleateRadius(t)      #Must be done afterwards since derived classes can change nucRate
        #print(self._currY)

    def getdXdt(self, t, x):
        '''
        Gets dXdt as a list for each phase

        For the eulerian implementation, this is dn_i/dt for the bins in PBM for each phase
        '''
        self._calculateDependentTerms(t, x)
        dXdt = self._getdXdt(t, x)
        return dXdt

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
        return NotImplementedError()
    
    def _calcMassBalance(self, t, x):
        return NotImplementedError()
    
    def _getdXdt(self, t, x):
        return NotImplementedError()
    
    def _updateParticleSizeDistribution(self, t, x):
        return NotImplementedError()
    
    def _calcDrivingForce(self, t, x):
        '''
        Driving force is defined in terms of J/m3

        Calculation is dG_ch / V_m - dG_el
            dG_ch - chemical driving force
            V_m - molar volume
            dG_el - elastic strain energy (always reduces driving force)
                I guess there could be a case where it increases the driving force if
                the matrix is prestrained and the precipitate releases stress, but this should
                be handled in the ElasticFactors module

        If driving force is positive (precipitation is favorable)
            Calculate Rcrit and Gcrit based off the nucleation site type

        This will also calculate critical radius (Rcrit) and nucleation barrier (Gcrit)
        '''
        #Get variables
        dGs = np.zeros((1,len(self.phases)))
        Rcrit = np.zeros((1,len(self.phases)))
        Gcrit = np.zeros((1,len(self.phases)))
        if self.numberOfElements == 1:
            xComp = self._currY[self.COMPOSITION][0,0]
        else:
            xComp = self._currY[self.COMPOSITION][0]
        T = self._currY[self.TEMPERATURE][0]

        for p in range(len(self.phases)):
            dGs[0,p], self._precBetaTemp[p] = self.dG[p](xComp, T)
            dGs[0,p] /= self.VmBeta[p]
            dGs[0,p] -= self.strainEnergy[p].strainEnergy(self.shapeFactors[p].normalRadii(self.Rcrit[self.n, p]))
            if self.betaFrac[self.n, p] < 1 and dGs[0,p] >= 0:
                #Calculate critical radius
                #For bulk or dislocation nucleation sites, use previous critical radius to get aspect ratio
                if self.GB[p].nucleationSiteType == GBFactors.BULK or self.GB[p].nucleationSiteType == GBFactors.DISLOCATION:
                    Rcrit[0,p] = np.amax((2 * self.shapeFactors[p].thermoFactor(self.Rcrit[self.n, p]) * self.gamma[p] / dGs[0,p], self.Rmin[p]))
                    Gcrit[0,p] = (4 * np.pi / 3) * self.gamma[p] * Rcrit[0,p]**2

                #If nucleation is on a grain boundary, then use the critical radius as defined by the grain boundary type    
                else:
                    Rcrit[0,p] = np.amax((self.GB[p].Rcrit(dGs[0,p]), self.Rmin[p]))
                    Gcrit[0,p] = self.GB[p].Gcrit(dGs[0,p], Rcrit[0,p])

        self._currY[self.DRIVING_FORCE] = dGs
        self._currY[self.R_CRIT] = Rcrit
        self._currY[self.G_CRIT] = Gcrit

    def _calcNucleationRate(self, t, x):
        '''
        nucleation rate is defined as dn_nuc/dt = N_0 Z beta exp(-G/kBt) * exp(-tau/t)
        '''
        gCrit = self._currY[self.G_CRIT][0]
        T = self._currY[self.TEMPERATURE][0]
        dg = self._currY[self.DRIVING_FORCE][0]

        betas = np.zeros((1,len(self.phases)))
        nucRate = np.zeros((1,len(self.phases)))
        for p in range(len(self.phases)):
            #If driving force is negative, then nucleation rate is 0
            if dg[p] < 0:
                continue

            Z = self._Zeldovich(p)
            betas[0,p] = self._Beta(p)

            #If beta is 0, then nucRate is 0 and no need to do anymore calculation
            if betas[0,p] == 0:
                continue
            
            #Incubation time, either isothermal or nonisothermal
            incubation = self._incubation(t, p, Z, betas[0])
            if incubation > 1:
                incubation = 1

            nucRate[0,p] = Z * betas[0,p] * np.exp(-gCrit[p] / (self.kB * T)) * incubation

        self._currY[self.IMPINGEMENT] = betas
        self._currY[self.NUC_RATE] = nucRate

    def _Zeldovich(self, p):
        '''
        Zeldovich factor - probability that cluster at height of nucleation barrier will continue to grow
        '''
        rCrit = self._currY[self.R_CRIT][0]
        T = self._currY[self.TEMPERATURE][0]
        if rCrit[p] == 0:
            return 0
        return np.sqrt(3 * self.GB[p].volumeFactor / (4 * np.pi)) * self.VmBeta[p] * np.sqrt(self.gamma[p] / (self.kB * T)) / (2 * np.pi * self.avo * rCrit[p]**2)
        
    def _BetaBinary1(self, p):
        '''
        Impingement rate for binary systems using Perez et al
        '''
        rCrit = self._currY[self.R_CRIT][0]
        xComp = self._currY[self.COMPOSITION][0][0]
        T = self._currY[self.TEMPERATURE][0]
        return self.GB[p].areaFactor * rCrit[p]**2 * self.xComp[0] * self.Diffusivity(xComp, T) / self.aAlpha**4

    def _BetaBinary2(self, p):
        '''
        Impingement rate for binary systems taken from Thermocalc prisma documentation
        This will follow the same equation as with _BetaMulti; however, some simplications can be made based off the summation contraint
        '''
        xComp = self._currY[self.COMPOSITION][0][0]
        xEqAlpha = self._currY[self.EQ_COMP_ALPHA][0]
        xEqBeta = self._currY[self.EQ_COMP_BETA][0]
        rCrit = self._currY[self.R_CRIT][0]
        T = self._currY[self.TEMPERATURE][0]
        D = self.Diffusivity(xComp, T)
        Dfactor = (xEqBeta[p] - xEqAlpha[p])**2 / (xEqAlpha[p]*D) + (xEqBeta[p] - xEqAlpha[p])**2 / ((1 - xEqAlpha[p])*D)
        return self.GB[p].areaFactor * rCrit[p]**2 * (1/Dfactor) / self.aAlpha**4
            
    def _BetaMulti(self, p):
        '''
        Impingement rate for multicomponent systems
        '''
        if self._betaFuncs[p] is None:
            return self._defaultBeta
        else:
            xComp = self._currY[self.COMPOSITION][0]
            T = self._currY[self.TEMPERATURE][0]
            beta = self._betaFuncs[p](xComp, T, searchDir = self._precBetaTemp[p])
            if beta is None:
                return self.betas[p]
            else:
                rCrit = self._currY[self.R_CRIT][0]
                return (self.GB[p].areaFactor * rCrit[p]**2 / self.aAlpha**4) * beta

    def _incubationIsothermal(self, t, p, Z, betas):
        '''
        Incubation time for isothermal conditions
        '''
        tau = 1 / (self.theta[p] * (betas[p] * Z**2))
        return np.exp(-tau / t)
        
    def _incubationNonIsothermal(self, t, p, Z, betas):
        '''
        Incubation time for non-isothermal conditions
        This must match isothermal conditions if the temperature is constant

        Solve for tau by: integral(beta(t-t0)) from 0 to tau = 1/theta*Z(tau)^2

        Then it's exp(-tau/t) like the isothermal behavior
        '''
        T = self._currY[self.TEMPERATURE][0]
        startIndex = int(self.incubationOffset[p])
        LHS = 1 / (self.theta[p] * Z**2 * (T / self.temperature[startIndex:self.n+1]))

        RHS = np.cumsum(self.betas[startIndex+1:self.n+1,p] * (self.time[startIndex+1:self.n+1] - self.time[startIndex:self.n]))
        if len(RHS) == 0:
            RHS = self.betas[self.n,p] * (self.time[startIndex:] - self.time[startIndex])
        else:
            RHS = np.concatenate((RHS, [RHS[-1] + betas[p] * (t - self.time[startIndex])]))

        #Test for intersection
        diff = RHS - LHS
        signChange = np.sign(diff[:-1]) != np.sign(diff[1:])

        #If no intersection
        if not any(signChange):
            #If RHS > LHS, then intersection is at t = 0
            if diff[0] > 0:
                tau = 0
            #Else, RHS intersects LHS beyond simulation time
            #Extrapolate integral of RHS from last point to intersect LHS
            #integral(beta(t-t0)) from t0 to ti + beta_i * (tau - (ti - t0)) = 1 / theta * Z(tau+t0)^2
            else:
                tau = LHS[-1] / betas[p] - RHS[-1] / betas[p] + (t - self.time[startIndex])
        else:
            tau = self.time[startIndex:-1][signChange][0] - self.time[startIndex]

        return np.exp(-tau / (t - self.time[startIndex]))
    
    def _setNucleateRadius(self, t):
        '''
        Adds 1/2 * sqrt(kb T / pi gamma) to critical radius to ensure they grow when growth rates are calculated
        '''
        nucRate = self._currY[self.NUC_RATE][0]
        T = self._currY[self.TEMPERATURE][0]
        dt = t - self.time[self.n]
        Rcrit = self._currY[self.R_CRIT][0]
        Rad = np.zeros((1,len(self.phases)))
        for p in range(len(self.phases)):
            #If nucleates form, then calculate radius of precipitate
            #Radius is set slightly larger so precipitate
            dt = 0.01 if self.n == 0 else self.time[self.n] - self.time[self.n-1]
            if nucRate[p]*dt >= self.constraints.minNucleateDensity and Rcrit[p] >= self.Rmin[p]:
                Rad[0,p] = Rcrit[p] + 0.5 * np.sqrt(self.kB * T / (np.pi * self.gamma[p]))
            else:
                Rad[0,p] = 0

        self._currY[self.R_NUC] = Rad
