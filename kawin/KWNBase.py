import numpy as np
import matplotlib.pyplot as plt
from kawin.EffectiveDiffusion import effectiveDiffusionDistance, noDiffusionDistance
from kawin.ShapeFactors import ShapeFactor
from kawin.ElasticFactors import StrainEnergy
from kawin.GrainBoundaries import GBFactors
import copy
import time

class PrecipitateBase:
    '''
    Base class for precipitation models
    Note: currently only the Euler implementation is available, but
        other implementations are planned to be added
    
    Parameters
    ----------
    t0 : float
        Initial time in seconds
    tf : float
        Final time in seconds
    steps : int
        Number of time steps
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
    def __init__(self, t0, tf, steps, phases = ['beta'], linearTimeSpacing = False, elements = ['solute']):
        #Store input parameters
        self.initialSteps = int(steps)      #Initial number of steps for when model is reset
        self.steps = int(steps)             #This includes the number of steps added when adaptive time stepping is enabled
        self.t0 = t0
        self.tf = tf
        self.phases = phases
        self.linearTimeSpacing = linearTimeSpacing

        #Change t0 to finite value if logarithmic time spacing
        #New t0 will be tf / 1e6
        if self.t0 <= 0 and self.linearTimeSpacing == False:
            self.t0 = self.tf / 1e6
            print('Warning: Cannot use 0 as an initial time when using logarithmic time spacing')
            print('\tSetting t0 to {:.3e}'.format(self.t0))
        
        #Time variables
        self._timeIncrementCheck = self._noCheckDT

        #Predefined constraints, these can be set if they make the simulation unstable
        self._defaultConstraints()
            
        #Composition array
        self.elements = elements
        self.numberOfElements = len(elements)
        
        #All other arrays
        self._resetArrays()
        
        #Constants
        self.Rg = 8.314                 #J/mol-K
        self.avo = 6.022e23             #/mol
        self.kB = self.Rg / self.avo    #J/K
        
        #Default variables, these terms won't have to be set before simulation
        self.strainEnergy = [StrainEnergy() for i in self.phases]
        self.RdrivingForceLimit = np.zeros(len(self.phases), dtype=np.float32)
        self.shapeFactors = [ShapeFactor() for i in self.phases]
        self.theta = 2 * np.ones(len(self.phases), dtype=np.float32)
        self.effDiffDistance = effectiveDiffusionDistance
        self.infinitePrecipitateDiffusion = [True for i in self.phases]
        self.dTemp = 0
        self.GBenergy = 0.3     #J/m2
        self.parentPhases = [[] for i in self.phases]
        self.GB = [GBFactors() for p in self.phases]
        
        #Set other variables to None to throw errors if not set
        self.xInit = None
        self.T = None
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
        self.atomsPerCellBeta = np.empty(len(self.phases), dtype=np.float32)
        self.VaBeta = np.empty(len(self.phases), dtype=np.float32)
        self.VmBeta = np.empty(len(self.phases), dtype=np.float32)
        self.Rmin = np.empty(len(self.phases), dtype=np.float32)
        
        #Free energy parameters
        self.gamma = np.empty(len(self.phases), dtype=np.float32)
        self.dG = [None for i in self.phases]
        self.interfacialComposition = [None for i in self.phases]

        if self.numberOfElements == 1:
            self._Beta = self._BetaBinary
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
        return 0 if phase is None else self.phases.index(phase)
        
    def reset(self):
        '''
        Resets simulation results
        This does not reset the model parameters
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
        self.steps = self.initialSteps
        self.time = np.linspace(self.t0, self.tf, self.steps) if self.linearTimeSpacing else np.logspace(np.log10(self.t0), np.log10(self.tf), self.steps)

        if self.numberOfElements == 1:
            self.xComp = np.zeros(self.steps)
        else:
            self.xComp = np.zeros((self.steps, self.numberOfElements))
            
        self.Rcrit = np.zeros((len(self.phases), self.steps))                #Critical radius
        self.Gcrit = np.zeros((len(self.phases), self.steps))                #Height of nucleation barrier
        self.Rad = np.zeros((len(self.phases), self.steps))                  #Radius of particles formed at each time step
        self.avgR = np.zeros((len(self.phases), self.steps))                 #Average radius
        self.avgAR = np.zeros((len(self.phases), self.steps))                #Mean aspect ratio
        self.betaFrac = np.zeros((len(self.phases), self.steps))             #Fraction of precipitate
        
        self.nucRate = np.zeros((len(self.phases), self.steps))              #Nucleation rate
        self.precipitateDensity = np.zeros((len(self.phases), self.steps))  #Number of nucleates
        
        self.dGs = np.zeros((len(self.phases), self.steps))                  #Driving force
        self.BetaSum = np.zeros(len(self.phases))                            #Integral of beta for each nucleation site type (probability that nucleate will grow past nucleation barrier)
        self.incubationOffset = np.zeros(len(self.phases))                   #Offset for incubation time (temporary fix for non-isothermal precipitation)
        self.incubationSum = np.zeros(len(self.phases))                      #Sum of incubation time

        self.prevFConc = np.zeros((2, len(self.phases), self.numberOfElements))    #Sum of precipitate composition for mass balance

    def save(self, filename, compressed = False):
        '''
        Save results into a numpy .npz format

        TODO: possibly add support to save to CSV

        Parameters
        ----------
        filename : str
        compressed : bool
            If true, will save compressed .npz format
        '''
        variables = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements', \
            'time', 'xComp', 'Rcrit', 'Gcrit', 'Rad', 'avgR', 'avgAR', 'betaFrac', 'nucRate', 'precipitateDensity', 'dGs']
        vDict = {v: getattr(self, v) for v in variables}
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
        PrecipitateBase object
            Note: this will only contain model outputs which can be used for plotting
        '''
        data = np.load(filename)
        setupVars = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements']
        model = PrecipitateBase(data['t0'], data['tf'], data['steps'], data['phases'], data['linearTimeSpacing'], data['elements'])
        for d in data:
            if d not in setupVars:
                setattr(model, d, data[d])
        return model
        
    def _divideTimestep(self, i, dt):
        '''
        Adds a new time step between t_i-1 and t_i, with new time being t_i-1 + dt

        Parameters
        ----------
        i : int
        dt : float
            Note: this must be smaller than t_i - t_i-1
        '''
        self.steps += 1

        if self.numberOfElements == 1:
            self.xComp = np.append(self.xComp, 0)
        else:
            self.xComp = np.append(self.xComp, np.zeros((1, self.numberOfElements)), axis=0)

        #Add new element to each variable
        self.Rcrit = np.append(self.Rcrit, np.zeros((len(self.phases), 1)), axis=1)
        self.Gcrit = np.append(self.Gcrit, np.zeros((len(self.phases), 1)), axis=1)
        self.Rad = np.append(self.Rad, np.zeros((len(self.phases), 1)), axis=1)
        self.avgR = np.append(self.avgR, np.zeros((len(self.phases), 1)), axis=1)
        self.avgAR = np.append(self.avgAR, np.zeros((len(self.phases), 1)), axis=1)
        self.betaFrac = np.append(self.betaFrac, np.zeros((len(self.phases), 1)), axis=1)
        self.nucRate = np.append(self.nucRate, np.zeros((len(self.phases), 1)), axis=1)
        self.precipitateDensity = np.append(self.precipitateDensity, np.zeros((len(self.phases), 1)), axis=1)
        self.dGs = np.append(self.dGs, np.zeros((len(self.phases), 1)), axis=1)

        prevDT = self.time[i] - self.time[i-1]
        self.time = np.insert(self.time, i, self.time[i-1] + dt)

        ratio = dt / prevDT
        self.T = np.insert(self.T, i, ratio * self.T[i-1] + (1-ratio) * self.T[i])

    def adaptiveTimeStepping(self, adaptive = True):
        '''
        Sets if adaptive time stepping is used

        Parameters
        ----------
        adaptive : bool (optional)
            Defaults to True
        '''
        if adaptive:
            self._timeIncrementCheck = self._checkDT
        else:
            self._timeIncrementCheck = self._noCheckDT

    def _defaultConstraints(self):
        '''
        Default values for contraints
        '''
        self.minRadius = 5e-10
        self.maxTempChange = 10
        self.maxDTFraction = 0.1
        self.minDTFraction = 1e-5
        self.maxDissolution = 0.01
        self.maxRcritChange = 0.01
        self.maxNucleationRateChange = 0.5
        self.maxVolumeChange = 0.001
        self.maxNonIsothermalDT = 1
        self.maxCompositionChange = 0.01

    def setConstraints(self, **kwargs):
        '''
        Sets constraints

        TODO: the following constraints are not implemented
            maxDTFraction
            minDTFraction
            maxDissolution
            maxRcritChange - will need to account for special case that driving force becomes negative
            maxNonIsothermalDT

        Possible constraints:
        ---------------------
        minRadius - minimum radius to be considered a precipitate - default is 5e-10 m^3
        maxTempChange - maximum temperature change before lookup table is updated (only for Euler in binary case) - default is 10 K
        maxDTFraction - maximum time increment allowed as a fraction of total simulation time - default is 0.1
        minDTFraction - minimum time increment allowed as a fraction of total simulation time - default is 1e-5
        maxDissolution - maximum volume fraction of precipitates allowed to dissolve in a single time step - default is 0.01
        maxRcritChange - maximum change in critical radius (as a fraction) per single time step - default is 0.01
        maxNucleationRateChange - maximum change in nucleation rate (on log scale) per single time step - default is 0.5
        maxVolumeChange - maximum absolute value that volume fraction can change per single time step - default is 0.001
        maxNonIsothermalDT - maximum time step when temperature is changing - default is 1 s
        maxCompositionChange - maximum change in composition in single time step - default is 0.01
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)
        
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

    def setSpherical(self, phase = None):
        '''
        Sets precipitate shape to spherical for defined phase

        Parameters
        ----------
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.shapeFactors[index].setSpherical()
        
    def setAspectRatioNeedle(self, ratio, phase = None):
        '''
        Consider specified precipitate phase as needle-shaped
        with specified aspect ratio

        Parameters
        ----------
        ratio : float or function
            Aspect ratio of needle-shaped precipitate
            If float, must be greater than 1
            If function, must take in radius as input and output float greater than 1
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.shapeFactors[index].setNeedleShape(ratio)
        
    def setAspectRatioPlate(self, ratio, phase = None):
        '''
        Consider specified precipitate phase as plate-shaped
        with specified aspect ratio

        Parameters
        ----------
        ratio : float or function
            Aspect ratio of needle-shaped precipitate
            If float, must be greater than 1
            If function, must take in radius as input and output float greater than 1
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.shapeFactors[index].setPlateShape(ratio)
        
    def setAspectRatioCuboidal(self, ratio, phase = None):
        '''
        Consider specified precipitate phase as cuboidal-shaped
        with specified aspect ratio

        TODO: add cuboidal factor
            Currently, I think this considers that the cuboidal factor is 1

        Parameters
        ----------
        ratio : float or function
            Aspect ratio of needle-shaped precipitate
            If float, must be greater than 1
            If function, must take in radius as input and output float greater than 1
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.shapeFactors[index].setCuboidalShape(ratio)
    
    def setVmAlpha(self, Vm, atomsPerCell):
        '''
        Molar volume for parent phase
        
        Parameters
        ----------
        Vm : float
            Molar volume (m3 / mol)
        atomsPerCell : int
            Number of atoms in a unit cell
        '''
        self.VmAlpha = Vm
        self.VaAlpha = atomsPerCell * self.VmAlpha / self.avo
        self.aAlpha = np.cbrt(self.VaAlpha)
        self.atomsPerCellAlpha = atomsPerCell
        
    def setVaAlpha(self, Va, atomsPerCell):
        '''
        Unit cell volume for parent phase
        
        Parameters
        ----------
        Va : float
            Unit cell volume (m3 / unit cell)
        atomsPerCell : int
            Number of atoms in a unit cell
        '''
        self.VaAlpha = Va
        self.VmAlpha = self.VaAlpha * self.avo / atomsPerCell
        self.aAlpha = np.cbrt(Va)
        self.atomsPerCellAlpha = atomsPerCell
        
    def setUnitCellAlpha(self, a, atomsPerCell):
        '''
        Lattice parameter for parent phase (assuming cubic unit cell)
        
        Parameters
        ----------
        a : float
            Lattice constant (m)
        atomsPerCell : int
            Number of atoms in a unit cell
        '''
        self.aAlpha = a
        self.VaAlpha = a**3
        self.VmAlpha = self.VaAlpha * self.avo / atomsPerCell
        self.atomsPerCellAlpha = atomsPerCell
        
    def setVmBeta(self, Vm, atomsPerCell, phase = None):
        '''
        Molar volume for precipitate phase
        
        Parameters
        ----------
        Vm : float
            Molar volume (m3 / mol)
        atomsPerCell : int
            Number of atoms in a unit cell
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.VmBeta[index] = Vm
        self.VaBeta[index] = atomsPerCell * self.VmBeta[index] / self.avo
        self.atomsPerCellBeta[index] = atomsPerCell
        self.Rmin[index] = self.minRadius
        
    def setVaBeta(self, Va, atomsPerCell, phase = None):
        '''
        Unit cell volume for precipitate phase
        
        Parameters
        ----------
        Va : float
            Unit cell volume (m3 / unit cell)
        atomsPerCell : int
            Number of atoms in a unit cell
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.VaBeta[index] = Va
        self.VmBeta[index] = self.VaBeta[index] * self.avo / atomsPerCell
        self.atomsPerCellBeta[index] = atomsPerCell
        self.Rmin[index] = self.minRadius
        
    def setUnitCellBeta(self, a, atomsPerCell, phase = None):
        '''
        Lattice parameter for precipitate phase (assuming cubic unit cell)
        
        Parameters
        ----------
        a : float
            Latice parameter (m)
        atomsPerCell : int
            Number of atoms in a unit cell
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.VaBeta[index] = a**3
        self.VmBeta[index] = self.VaBeta[index] * self.avo / atomsPerCell
        self.atomsPerCellBeta[index] = atomsPerCell
        self.Rmin[index] = self.minRadius

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
        '''
        Sets isothermal temperature
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        ''' 
        #Store parameter in case model is reset
        self.Tparameters = temperature

        self.T = np.full(self.steps, temperature, dtype=np.float32)
        self._incubation = self._incubationIsothermal
        
    def setNonIsothermalTemperature(self, temperatureFunction):
        '''
        Sets temperature as a function of time
        
        Parameters
        ----------
        temperatureFunction : function 
            Takes in time and returns temperature in K
        '''
        #Store parameter in case model is reset
        self.Tparameters = temperatureFunction

        self.T = np.array([temperatureFunction(t) for t in self.time])
        
        if len(np.unique(self.T) == 1):
            self._incubation = self._incubationIsothermal
        else:
            self._incubation = self._incubationNonIsothermal
        
    def setTemperatureArray(self, times, temperatures):
        '''
        Sets temperature as a function of time interpolating between the inputted times and temperatures
        
        Parameters
        ----------
        times : list
            Time in hours for when the corresponding temperature is reached
        temperatures : list
            Temperatures in K to be reached at corresponding times
        '''
        #Store parameter in case model is reset
        self.Tparameters = (times, temperatures)

        self.T = np.full(self.steps, temperatures[0])
        for i in range(1, len(times)):
            self.T[(self.time < 3600*times[i]) & (self.time >= 3600*times[i-1])] = (temperatures[i] - temperatures[i-1]) / (3600 * (times[i] - times[i-1])) * (self.time[(self.time < 3600*times[i]) & (self.time >= 3600*times[i-1])] - 3600 * times[i-1]) + temperatures[i-1]
        self.T[self.time >= 3600*times[-1]] = temperatures[-1]
        
        if len(np.unique(self.T) == 1):
            self._incubation = self._incubationIsothermal
        else:
            self._incubation = self._incubationNonIsothermal

    def setStrainEnergy(self, strainEnergy, phase = None):
        '''
        Sets strain energy class to precipitate
        '''
        index = self.phaseIndex(phase)
        self.strainEnergy[index] = strainEnergy

    def _setupStrainEnergyFactors(self):
        #For each phase, the strain energy calculation will be set to assume
        # a spherical, cubic or ellipsoidal shape depending on the defined shape factors
        for i in range(len(self.phases)):
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
        
    def setDrivingForce(self, drivingForce, phase = None):
        '''
        Parameters
        ----------
        drivingForce : function
            Taking in composition (at. fraction) and temperature (K) and return driving force (J/mol)
                f(x, T) = dg, where x is float for binary and array for multicomponent
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.dG[index] = drivingForce
        
    def setInterfacialComposition(self, composition, phase = None):
        '''
        Parameters
        ----------
        composition : function
            Takes in temperature (K) and excess free energy (J/mol) and 
            returns a tuple of (matrix composition, precipitate composition)
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)

        The excess free energy term will be taken as the interfacial curvature and elastic energy contributions.
        This will be a positive value, so the function should ensure that the excess free energy to reduce the driving force
        
        If equilibrium cannot be solved, then the function should return (None, None) or (-1, -1)
        '''
        index = self.phaseIndex(phase)
        self.interfacialComposition[index] = composition

    def setThermodynamics(self, therm, phase = None):
        '''
        Parameters
        ----------
        therm : Thermodynamics class
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.dG[index] = lambda x, T: therm.getDrivingForce(x, T, precPhase=phase)
        
        if self.numberOfElements == 1:
            self.interfacialComposition[index] = lambda x, T: therm.getInterfacialComposition(x, T, precPhase=phase)
        else:
            self.interfacialComposition[index] = lambda x, T, dG, R, gExtra: therm.getGrowthAndInterfacialComposition(x, T, dG, R, gExtra, precPhase=phase)
            self._betaFuncs[index] = lambda x, T: therm.impingementFactor(x, T, precPhase=phase)

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
        if neglect:
            self.effDiffDistance = noDiffusionDistance
        else:
            self.effDiffDistance = effectiveDiffusionDistance

    def printModelParameters(self):
        '''
        Prints the model parameters
        '''
        print('Temperature (K):               {:.3f}'.format(self.T[0]))
        print('Initial Composition (at%):     {:.3f}'.format(100*self.xInit))
        print('Molar Volume (m3):             {:.3e}'.format(self.VmAlpha))

        for p in range(len(self.phases)):
            print('Phase: {}'.format(self.phases[p]))
            print('\tMolar Volume (m3):             {:.3e}'.format(self.VmBeta[p]))
            print('\tInterfacial Energy (J/m2):     {:.3f}'.format(self.gamma[p]))
            print('\tMinimum Radius (m):            {:.3e}'.format(self.Rmin[p]))

    def setup(self):
        '''
        Sets up hidden parameters before solving
        Here it's just the nucleation density and the grain boundary nucleation factors
        '''
        if not self._isNucleationSetup:
            #Set nucleation density assuming grain size of 100 um and dislocation density of 5e12 m/m3 (Thermocalc default)
            print('Nucleation density not set.\nSetting nucleation density assuming grain size of {:.0f} um and dislocation density of {:.0e} #/m2'.format(100, 5e12))
            self.setNucleationDensity(100, 1, 5e12)
        self._getNucleationDensity()
        self._setGBfactors()
        self._setupStrainEnergyFactors()

    def _printOutput(self, i):
        '''
        Prints various terms at step i
        '''
        if self.numberOfElements == 1:
            print('N\tTime (s)\tTemperature (K)\tMatrix Comp')
            print('{:.0f}\t{:.1e}\t\t{:.0f}\t\t{:.4f}\n'.format(i, self.time[i], self.T[i], 100*self.xComp[i]))
        else:
            compStr = 'N\tTime (s)\tTemperature (K)'
            compValStr = '{:.0f}\t{:.1e}\t\t{:.0f}\t'.format(i, self.time[i], self.T[i])
            for a in range(self.numberOfElements):
                compStr += self.elements[a] + '\t'
                compValStr += '{:.4f}\t'.format(100*self.xComp[i,a])
            compValStr += '\n'
            print(compStr)
            print(compValStr)
        print('\tPhase\tPrec Density (#/m3)\tVolume Frac\tAvg Radius (m)')
        for p in range(len(self.phases)):
            print('\t{}\t{:.3e}\t{:.4f}\t\t{:.4e}'.format(self.phases[p], self.precipitateDensity[p,i], 100*self.betaFrac[p,i], self.avgR[p,i]))
        print('')
                
    def solve(self, verbose = False, vIt = 1000):
        '''
        Solves the KWN model between initial and final time
        
        Note: _calculateNucleationRate, _calculatePSD and _printOutput will need to be implemented in the child classes

        Parameters
        ----------
        verbose : bool (optional)
            Whether to print current simulation terms every couple iterations
            Defaults to False
        vIt : int (optional)
            If verbose is True, vIt is how many iterations will pass before printing an output
            Defaults to 1000
        '''
        self.setup()

        t0 = time.time()
        
        #While loop since number of steps may change with adaptive time stepping
        i = 1
        while i < self.steps:
            self._iterate(i)

            #Print current variables
            if i % vIt == 0 and verbose:
                self._printOutput(i)
            
            i += 1

        t1 = time.time()
        if verbose:
            print('Finished in {:.3f} seconds.'.format(t1 - t0))

    def _iterate(self, i):
        '''
        Blank iteration function to be implemented in other classes
        '''
        pass

    def _nucleationRate(self, p, i, dt):
        '''
        Calculates nucleation rate at current timestep (normalized to number of nucleation sites)
        This step is general to all systems except for how self._Beta is defined
        
        Parameters
        ----------
        p : int
            Phase index (int)
        i : int
            Current iteration
        dt : float
            Current time increment
        '''
        #Most equations in literature take the driving force to be positive
        #Although this really only changes the calculation of Rcrit since Z and beta is dG^2
        self.dGs[p, i], _ = self.dG[p](self.xComp[i-1], self.T[i])
        self.dGs[p, i] /= self.VmBeta[p]

        #Temporary, may add way to solve for aspect ratio by minimizing free energy between surface and elastic energy
        self.dGs[p, i] -= self.strainEnergy[p].strainEnergy(self.shapeFactors[p].normalRadii(self.Rcrit[p, i-1]))

        if self.dGs[p, i] < 0:
            return self._noDrivingForce(p, i)

        #Only do this if there is some parent phase left (brute force solution for to avoid numerical errors)
        #if self.betaFrac[p, i-1] < 1 and self.xComp[i-1] > 0:
        if self.betaFrac[p, i-1] < 1:
            
            #Calculate critical radius
            #For bulk or dislocation nucleation sites, the precipitate can be any shape,
            # so we need to solve for the critical radius in case the aspect ratio is not constant
            if self.GB[p].nucleationSiteType == GBFactors.BULK or self.GB[p].nucleationSiteType == GBFactors.DISLOCATION:
                self.Rcrit[p, i] = self.shapeFactors[p].findRcrit(2 * self.gamma[p] / self.dGs[p, i], 20 * self.gamma[p] / self.dGs[p, i])
                if self.Rcrit[p, i] < self.Rmin[p]:
                    self.Rcrit[p, i] = self.Rmin[p]
                
                self.Gcrit[p, i] = (4 * np.pi / 3) * self.gamma[p] * self.Rcrit[p, i]**2

            #If nucleation is on a grain boundary, then use the critical radius as defined by the grain boundary type    
            else:
                self.Rcrit[p, i] = self.GB[p].Rcrit(self.dGs[p, i])
                if self.Rcrit[p, i] < self.Rmin[p]:
                    self.Rcrit[p, i] = self.Rmin[p]
                    
                self.Gcrit[p, i] = self.GB[p].Gcrit(self.dGs[p, i], self.Rcrit[p, i])

            #Calculate nucleation rate
            Z = self._Zeldovich(p, i)
            beta = self._Beta(p, i)

            self.BetaSum[p] += beta * dt

            #I find just calculating tau assuming steady state seems to work better
            #I think it's because we're integrating the incubation term, where the thermocalc method of finding tau may already account for it
            tau = 1 / (self.theta[p] * (np.pi * beta * Z**2))
                
            #If tau becomes super large, then add delay to the incubation time
            if tau > 10 * self.time[-1] and self.incubationOffset[p] == 0:
                self.incubationOffset[p] = self.time[i-1]
                
            #Incubation time, either isothermal or nonisothermal
            self.incubationSum[p] = self._incubation(tau, p, i)
            if self.incubationSum[p] > 1:
                self.incubationSum[p] = 1
            
            return Z * beta * np.exp(-self.Gcrit[p, i] / (self.kB * self.T[i])) * self.incubationSum[p]

        else:
            return self._noDrivingForce(p, i)
            
    def _noCheckDT(self, i):
        '''
        Default time increment function if not implemented
        '''
        pass

    def _checkDT(self, i):
        '''
        Default time increment function if implement (which is no implementation)
        '''
        pass

    def _noDrivingForce(self, p, i):
        '''
        Set everything to 0 if there is no driving force for precipitation
        '''
        #self.dGs[p, i] = 0
        self.Rcrit[p, i] = 0
        self.incubationOffset[p] = self.time[i-1]
        return 0

    def _Zeldovich(self, p, i):
        return np.sqrt(3 * self.GB[p].volumeFactor / (4 * np.pi)) * self.VaBeta[p] * np.sqrt(self.gamma[p] / (self.kB * self.T[i])) / (2 * np.pi * self.Rcrit[p,i]**2)
        
    def _BetaBinary(self, p, i):
        return self.GB[p].areaFactor * self.Rcrit[p,i]**2 * self.xComp[0] * self.Diffusivity(self.xComp[i], self.T[i]) / self.aAlpha**4
            
    def _BetaMulti(self, p, i):
        if self._betaFuncs[p] is None:
            return self._defaultBeta
        else:
            #print((self.GBfactors[p,1] * self.Rcrit[p, i]**2 / self.aAlpha**4))
            return (self.GB[p].areaFactor * self.Rcrit[p, i]**2 / self.aAlpha**4) * self._betaFuncs[p](self.xComp[i-1], self.T[i-1])

    def _incubationIsothermal(self, tau, p, i):
        '''
        Incubation time for isothermal conditions
        '''
        return np.exp(-tau / self.time[i])
        
    def _incubationNonIsothermal(self, tau, p, i):
        '''
        Incubation time for non-isothermal conditions
        This must match isothermal conditions if the temperature is constant
        '''
        return np.exp(-tau / (self.time[i] - self.incubationOffset[p]))
    
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
                'Temperature' and 'Composition'

                Note: for multi-phase simulations, adding the word 'Total' will
                    sum the variable for all phases. Without the word 'Total', the variable
                    for each phase will be plotted separately
                    
        bounds : tuple (optional)
            Limits on the x-axis (float, float) or None (default, this will set bounds to (initial time, final time))
        *args, **kwargs - extra arguments for plotting
        '''
        if bounds is None:
            if self.t0 == 0:
                bounds = [1e-5 * self.tf, self.tf]
            else:
                bounds = [self.t0, self.tf]
            
        axes.set_xlabel('Time (s)')
        axes.set_xlim(bounds)

        labels = {
            'Volume Fraction': 'Volume Fraction',
            'Total Volume Fraction': 'Volume Fraction',
            'Critical Radius': 'Critical Radius (m)',
            'Average Radius': 'Average Radius (m)',
            'Volume Average Radius': 'Volume Average Radius (m)',
            'Total Average Radius': 'Average Radius (m)',
            'Total Volume Average Radius': 'Volume Average Radius (m)',
            'Aspect Ratio': 'Mean Aspect Ratio',
            'Total Aspect Ratio': 'Mean Aspect Ratio',
            'Driving Force': 'Driving Force (J/m$^3$)',
            'Nucleation Rate': 'Nucleation Rate (#/m$^3$-s)',
            'Total Nucleation Rate': 'Nucleation Rate (#/m$^3$-s)',
            'Precipitate Density': 'Precipitate Density (#/m$^3$)',
            'Total Precipitate Density': 'Precipitate Density (#/m$^3$)',
            'Temperature': 'Temperature (K)',
            'Composition': 'Matrix Composition (at.%)',
        }

        totalVariables = ['Total Volume Fraction', 'Total Average Radius', 'Total Aspect Ratio', 'Total Nucleation Rate', 'Total Precipitate Density']
        singleVariables = ['Volume Fraction', 'Critical Radius', 'Average Radius', 'Aspect Ratio', 'Driving Force', 'Nucleation Rate', 'Precipitate Density']

        if variable == 'Temperature':
            axes.semilogx(self.time, self.T, *args, **kwargs)
            axes.set_ylabel(labels[variable])

        elif variable == 'Composition':
            if self.numberOfElements == 1:
                axes.semilogx(self.time, self.xComp, *args, **kwargs)
                axes.set_ylabel('Matrix Composition (at.% ' + self.elements[0] + ')')
            else:
                for i in range(self.numberOfElements):
                    axes.semilogx(self.time, self.xComp[:,i], label=self.elements[i], *args, **kwargs)
                axes.legend(self.elements)
                axes.set_ylabel(labels[variable])
            yRange = [np.amin(self.xComp), np.amax(self.xComp)]
            axes.set_ylim([yRange[0] - 0.1 * (yRange[1] - yRange[0]), yRange[1] + 0.1 * (yRange[1] - yRange[0])])

        elif variable in singleVariables:
            if variable == 'Volume Fraction':
                plotVariable = self.betaFrac
            elif variable == 'Critical Radius':
                plotVariable = self.Rcrit
            elif variable == 'Average Radius':
                plotVariable = self.avgR
            elif variable == 'Volume Average Radius':
                plotVariable = np.cbrt(self.betaFrac / self.precipitateDensity)
            elif variable == 'Aspect Ratio':
                plotVariable = self.avgAR
            elif variable == 'Driving Force':
                plotVariable = self.dGs
            elif variable == 'Nucleation Rate':
                plotVariable = self.nucRate
            elif variable == 'Precipitate Density':
                plotVariable = self.precipitateDensity

            if (len(self.phases)) == 1:
                axes.semilogx(self.time, plotVariable[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    axes.semilogx(self.time, plotVariable[p], label=self.phases[p], *args, **kwargs)
                axes.legend()
            axes.set_ylabel(labels[variable])
            axes.set_ylim([0, 1.1 * np.amax(plotVariable)])

        elif variable in totalVariables:
            if variable == 'Total Volume Fraction':
                plotVariable = np.sum(self.betaFrac, axis=0)
            elif variable == 'Total Average Radius':
                totalN = np.sum(self.precipitateDensity, axis=0)
                totalN[totalN == 0] = 1
                totalR = np.sum(self.avgR * self.precipitateDensity, axis=0)
                plotVariable = totalR / totalN
            elif variable == 'Total Volume Average Radius':
                totalN = np.sum(self.precipitateDensity, axis=0)
                totalN[totalN == 0] = 1
                totalVol = np.sum(self.betaFrac, axis=0)
                plotVariable = np.cbrt(totalVol / totalN)
            elif variable == 'Total Aspect Ratio':
                totalN = np.sum(self.precipitateDensity, axis=0)
                totalN[totalN == 0] = 1
                totalAR = np.sum(self.avgAR * self.precipitateDensity, axis=0)
                plotVariable = totalAR / totalN
            elif variable == 'Total Nucleation Rate':
                plotVariable = np.sum(self.nucRate, axis=0)
            elif variable == 'Total Precipitate Density':
                plotVariable = np.sum(self.precipitateDensity, axis=0)

            axes.semilogx(self.time, plotVariable, *args, **kwargs)
            axes.set_ylabel(labels[variable])
            axes.set_ylim(bottom=0)
