import numpy as np
import matplotlib.pyplot as plt
from kawin.precipitation.non_ideal.EffectiveDiffusion import EffectiveDiffusionFunctions
from kawin.precipitation.non_ideal.ShapeFactors import ShapeFactor
from kawin.precipitation.non_ideal.ElasticFactors import StrainEnergy
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
import copy
import time
import csv
from itertools import zip_longest
import traceback
from kawin.GenericModel import GenericModel
from enum import Enum

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
        If binary system, then defualt is ['solute']
    '''
    def __init__(self, phases = ['beta'], elements = ['solute']):
        self.elements = elements
        self.numberOfElements = len(elements)
        self.phases = np.array(phases)

        self._resetArrays()
        self.resetConstraints()
        self._isSetup = False

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

        if self.numberOfElements == 1:
            self._Beta = self._BetaBinary1
        else:
            self._Beta = self._BetaMulti
            self._betaFuncs = [None for p in phases]
            self._defaultBeta = 20

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

        #Reset temperature array
        if np.isscalar(self.Tparameters):
            self.setTemperature(self.Tparameters)
        elif len(self.Tparameters) == 2:
            self.setTemperatureArray(*self.Tparameters)
        elif self.Tparameters is not None:
            self.setNonIsothermalTemperature(self.Tparameters)

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
            (iterations)                     time
            (iterations, elements)           composition
            (iterations, phases, elements)   eq composition
            (iterations, phases)             Everything else
            This is intended for appending arrays to always be on the first axis
        '''
        self.n = 0

        #Time
        self.time = np.zeros(1)

        #Temperature
        self.temperature = np.zeros(1)

        #Composition
        self.xComp = np.zeros((1, self.numberOfElements))
        self.xEqAlpha = np.zeros((1, len(self.phases), self.numberOfElements))
        self.xEqBeta = np.zeros((1, len(self.phases), self.numberOfElements))

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

    def _setEnum(self):
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
        The "constants" will also act as the index to use when taking variables out of x
        '''
        self.varList = [
                self.time, self.temperature, self.xComp, self.xEqAlpha, self.xEqBeta,
                self.dGs, self.betas, self.Gcrit, self.Rcrit, self.Rad,
                self.nucRate, self.precipitateDensity,
                self.avgR, self.avgAR, self.betaFrac,
                self.fConc
                ]

    def _appendArrays(self, newVals):
        '''
        Appends new values to the variable list
        NOTE: newVals must correspond to the same order as _packArrays with first axis as 1
            Ex rCrit is (n, phases) so corresponding new value should be (1, phases)
        Since np append creates a new variable in memory, we have to reassign each term, then pack them into varList again
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
        self.minRadius = 3e-10
        self.maxTempChange = 1

        self.maxDTFraction = 1e-2
        self.minDTFraction = 1e-5

        #Constraints on maximum time step
        self.checkTemperature = True
        self.maxNonIsothermalDT = 1

        self.checkPSD = True
        self.maxDissolution = 1e-3

        self.checkRcrit = True
        self.maxRcritChange = 0.01

        self.checkNucleation = True
        self.maxNucleationRateChange = 0.5
        self.minNucleationRate = 1e-5

        self.checkVolumePre = True
        self.maxVolumeChange = 0.001
        
        self.checkCompositionPre = False
        self.maxCompositionChange = 0.001
        self.minComposition = 0

        self.minNucleateDensity = 1e-10

    def setConstraints(self, **kwargs):
        '''
        Sets constraints

        TODO: the following constraints are not implemented
            maxDTFraction
            maxRcritChange - this is somewhat implemented but disabled by default

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

        checkComposition - checks maximum change in composition (True)
        chekcCompositionPre - estimates maximum change in composition (False)
        maxCompositionChange - maximum change in composition in single time step (0.01)

        minNucleateDensity - minimum nucleate density to consider nucleation to have occurred (1e-5)
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def setBetaBinary(self, functionType = 1):
        '''
        Sets function for beta calculation in binary systems

        If using a multicomponent system, this function will not do anything

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
        self.VmAlpha, self.VaAlpha, self.aAlpha, self.atomsPerCellAlpha = self._setVolume(value, valueType, atomsPerCell)

    def setVolumeBeta(self, value, valueType: VolumeParameter, atomsPerCell, phase = None):
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
            if self.GBareaN0 is None:
                self.GBareaN0 = (6 * np.sqrt(1 + 2 * self.grainAspectRatio**2) + 1 + 2 * self.grainAspectRatio) / (4 * self.grainAspectRatio * self.grainSize)
                self.GBareaN0 *= (self.avo / self.VmAlpha)**(2/3)
            if self.GBedgeN0 is None:
                self.GBedgeN0 = 2 * (np.sqrt(2) + 2 * np.sqrt(1 + self.grainAspectRatio**2)) / (self.grainAspectRatio * self.grainSize**2)
                self.GBedgeN0 *= (self.avo / self.VmAlpha)**(1/3)
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
        self.Tparameters = temperature
        self.temperature[0] = self.getTemperature(0)
        if np.isscalar(temperature):
            self._incubation = self._incubationIsothermal
        else:
            self._incubation = self._incubationNonIsothermal

    def getTemperature(self, t):
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
        '''
        index = self.phaseIndex(phase)
        self.strainEnergy[index] = strainEnergy
        self.calculateAspectRatio[index] = calculateAspectRatio

    def _setupStrainEnergyFactors(self):
        #For each phase, the strain energy calculation will be set to assume
        # a spherical, cubic or ellipsoidal shape depending on the defined shape factors
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
        Parameters
        ----------
        diffusivity : function taking 
            Composition and temperature (K) and returning diffusivity (m2/s)
            Function must have inputs in this order: f(x, T)
                For multicomponent systems, x is an array
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
        self.dG[index] = lambda x, T, removeCache = removeCache: therm.getDrivingForce(x, T, precPhase=phase, training = removeCache)
        
        if self.numberOfElements == 1:
            self.interfacialComposition[index] = lambda x, T: therm.getInterfacialComposition(x, T, precPhase=phase)
            if (therm.mobCallables is not None or therm.diffCallables is not None) and addDiffusivity:
                self.Diffusivity = lambda x, T, removeCache = removeCache: therm.getInterdiffusivity(x, T, removeCache = removeCache)
        else:
            self.interfacialComposition[index] = lambda x, T, dG, R, gExtra, removeCache = removeCache: therm.getGrowthAndInterfacialComposition(x, T, dG, R, gExtra, precPhase=phase, training = False)
            self._betaFuncs[index] = lambda x, T, removeCache = removeCache: therm.impingementFactor(x, T, precPhase=phase, training = False)

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

    def addStoppingCondition(self, variable, condition, value, phase = None, element = None, mode = 'or'):
        '''
        Adds condition to stop simulation when condition is met

        Parameters
        ----------
        variable : str
            Variable to set condition for, options are 
                'Volume Fraction'
                'Average Radius'
                'Driving Force'
                'Nucleation Rate'
                'Precipitate Density'
        condition : str
            Operator for condition, options are
                'greater than' or '>'
                'less than' or '<'
        value : float
            Value for condition
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        element : str (optional)
            For 'Composition', element to consider for condition (defaults to first element in list)
        mode : str (optional)
            How the condition will be handled
                'or' (default) - at least one condition in this mode needs to be met before stopping
                'and' - all conditions in this mode need to be met before stopping
                    This will also record the times each condition is met

        Example
        model.addStoppingCondition('Volume Fraction', '>', 0.002, 'beta')
            will add a condition to stop simulation when the volume fraction of the 'beta'
            phase becomes greater than 0.002
        '''
        index = self.phaseIndex(phase)

        if self._stoppingConditions is None:
            self._stoppingConditions = []
            self.stopConditionTimes = []
            self._stopConditionMode = []

        standardLabels = {
            'Volume Fraction': 'betaFrac',
            'Average Radius': 'avgR',
            'Driving Force': 'dGs',
            'Nucleation Rate': 'nucRate',
            'Precipitate Density': 'precipitateDensity',
        }
        otherLabels = ['Composition']

        if variable in standardLabels:
            if 'greater' in condition or '>' in condition:
                cond = lambda self, i, p = index, var=standardLabels[variable] : getattr(self, var)[p,i] > value
            elif 'less' in condition or '<' in condition:
                cond = lambda self, i, p = index, var=standardLabels[variable] : getattr(self, var)[p,i] < value
        else:
            if variable == 'Composition':
                eIndex = 0 if element is None else self.elements.index(element)
                if 'greater' in condition or '>' in condition:
                    if self.numberOfElements > 1:
                        cond = lambda self, i, e = eIndex, var='xComp' : getattr(self, var)[i, e] > value
                    else:
                        cond = lambda self, i, var='xComp' : getattr(self, var)[i] > value
                elif 'less' in condition or '<' in condition:
                    if self.numberOfElements > 1:
                        cond = lambda self, i, e = eIndex, var='xComp' : getattr(self, var)[i, e] < value
                    else:
                        cond = lambda self, i, var='xComp' : getattr(self, var)[i] < value

        self._stoppingConditions.append(cond)
        self.stopConditionTimes.append(-1)
        if mode == 'and':
            self._stopConditionMode.append(False)
        else:
            self._stopConditionMode.append(True)

    def clearStoppingConditions(self):
        '''
        Clears all stopping conditions
        '''
        self._stoppingConditions = None
        self.stopConditionTimes = None
        self._stopConditionMode = None

    def setup(self):
        '''
        Sets up hidden parameters before solving
        Here it's just the nucleation density and the grain boundary nucleation factors
        '''
        if self._isSetup:
            return
        
        if not self._isNucleationSetup:
            #Set nucleation density assuming grain size of 100 um and dislocation density of 5e12 m/m3 (Thermocalc default)
            print('Nucleation density not set.\nSetting nucleation density assuming grain size of {:.0f} um and dislocation density of {:.0e} #/m2'.format(100, 5e12))
            self.setNucleationDensity(100, 1, 5e12)
        for p in range(len(self.phases)):
            self.Rmin[p] = self.minRadius
        self._getNucleationDensity()
        self._setGBfactors()
        self._setupStrainEnergyFactors()
        self._isSetup = True

    def printStatus(self, iteration, simTimeElapsed):
        '''
        Prints various terms at latest step
        '''
        i = len(self.time)-1
        if self.numberOfElements == 1:
            print('N\tTime (s)\tTemperature (K)\tMatrix Comp')
            print('{:.0f}\t{:.1e}\t\t{:.0f}\t\t{:.4f}\n'.format(i, self.time[i], self.temperature[i], 100*self.xComp[i,0]))
        else:
            compStr = 'N\tTime (s)\tTemperature (K)\t'
            compValStr = '{:.0f}\t{:.1e}\t\t{:.0f}\t\t'.format(i, self.time[i], self.temperature[i])
            for a in range(self.numberOfElements):
                compStr += self.elements[a] + '\t'
                compValStr += '{:.4f}\t'.format(100*self.xComp[i,a])
            compValStr += '\n'
            print(compStr)
            print(compValStr)
        print('\tPhase\tPrec Density (#/m3)\tVolume Frac\tAvg Radius (m)\tDriving Force (J/mol)')
        for p in range(len(self.phases)):
            print('\t{}\t{:.3e}\t\t{:.4f}\t\t{:.4e}\t{:.4e}'.format(self.phases[p], self.precipitateDensity[i,p], 100*self.betaFrac[i,p], self.avgR[i,p], self.dGs[i,p]*self.VmBeta[p]))
        print('')

    def getCurrentX(self):
        return self.time[self.n], [self.PBM[p].PSD for p in range(len(self.phases))]

    def preProcess(self):
        '''
        Store array for non-derivative terms (which is everything except for the PBM models)

        We use these terms for the first step of the iterators (for Euler, this is all the steps)
            For RK4, these terms will be recalculated in dXdt
        '''
        self._currY = None
        self._firstIt = True
        return
    
    def _calculateDependentTerms(self, t, x):
        '''
        Gets all dependent terms (everything but PBM variables) that are needed to find dXdt

        For the first iteration, self._currX will be None from the preProcess function, in this case, we want
            to just grab the latest values
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
        self._calculateDependentTerms(t, x)
        dXdt = self._getdXdt(t, x)
        if self._firstIt:
            self._firstIt = False
        return dXdt

    def postProcess(self, t, x):
        self._calculateDependentTerms(t, x)
        self._appendArrays(self._currY)

        #Update particle size distribution (this includes adding bins, resizing bins, etc)
        #Should be agnostic of eulerian or lagrangian implementations
        self._updateParticleSizeDistribution(t, x)

        return self.getCurrentX()
    
    def _processX(self, x):
        return NotImplementedError()
    
    def _calcMassBalance(self, t, x):
        return NotImplementedError()
    
    def _getdXdt(self, t, x):
        return NotImplementedError()
    
    def _updateParticleSizeDistribution(self, t, x):
        return NotImplementedError()
    
    def _calcDrivingForce(self, t, x):
        dGs = np.zeros((1,len(self.phases)))
        Rcrit = np.zeros((1,len(self.phases)))
        Gcrit = np.zeros((1,len(self.phases)))
        if self.numberOfElements == 1:
            xComp = self._currY[self.COMPOSITION][0,0]
        else:
            xComp = self._currY[self.COMPOSITION][0]
        T = self._currY[self.TEMPERATURE][0]
        for p in range(len(self.phases)):
            dGs[0,p], _ = self.dG[p](xComp, T)
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
        gCrit = self._currY[self.G_CRIT][0]
        T = self._currY[self.TEMPERATURE][0]
        betas = np.zeros((1,len(self.phases)))
        nucRate = np.zeros((1,len(self.phases)))
        for p in range(len(self.phases)):
            Z = self._Zeldovich(p)
            betas[0,p] = self._Beta(p)
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
            beta = self._betaFuncs[p](xComp, T)
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

        Solve for integral(beta(t-t0)) from 0 to tau = 1/theta*Z(tau)^2
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
            if nucRate[p]*dt >= self.minNucleateDensity and Rcrit[p] >= self.Rmin[p]:
                Rad[0,p] = Rcrit[p] + 0.5 * np.sqrt(self.kB * T / (np.pi * self.gamma[p]))
            else:
                Rad[0,p] = 0

        self._currY[self.R_NUC] = Rad
