import numpy as np
import matplotlib.pyplot as plt
from kawin.EffectiveDiffusion import EffectiveDiffusionFunctions
from kawin.ShapeFactors import ShapeFactor
from kawin.ElasticFactors import StrainEnergy
from kawin.GrainBoundaries import GBFactors
import copy
import time
import csv
from itertools import zip_longest

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
        self.adaptiveTimeStepping(True)

        #Predefined constraints, these can be set if they make the simulation unstable
        self._defaultConstraints()

        #Stopping conditions
        self.clearStoppingConditions()
            
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
        self.steps = self.initialSteps
        self.time = np.linspace(self.t0, self.tf, self.steps) if self.linearTimeSpacing else np.logspace(np.log10(self.t0), np.log10(self.tf), self.steps)

        if self.numberOfElements == 1:
            self.xComp = np.zeros(self.steps)                                #Current composition of matrix phase
            self.xEqAlpha = np.zeros((len(self.phases), self.steps))         #Equilibrium composition of matrix phase with respect to each precipitate phase
            self.xEqBeta = np.zeros((len(self.phases), self.steps))          #Equilibrium composition of precipitate phases
        else:
            self.xComp = np.zeros((self.steps, self.numberOfElements))
            self.xEqAlpha = np.zeros((len(self.phases), self.steps, self.numberOfElements))
            self.xEqBeta = np.zeros((len(self.phases), self.steps, self.numberOfElements))
            
        self.Rcrit = np.zeros((len(self.phases), self.steps))                #Critical radius
        self.Gcrit = np.zeros((len(self.phases), self.steps))                #Height of nucleation barrier
        self.Rad = np.zeros((len(self.phases), self.steps))                  #Radius of particles formed at each time step
        self.avgR = np.zeros((len(self.phases), self.steps))                 #Average radius
        self.avgAR = np.zeros((len(self.phases), self.steps))                #Mean aspect ratio
        self.betaFrac = np.zeros((len(self.phases), self.steps))             #Fraction of precipitate
        
        self.nucRate = np.zeros((len(self.phases), self.steps))              #Nucleation rate
        self.precipitateDensity = np.zeros((len(self.phases), self.steps))   #Number of nucleates
        
        self.dGs = np.zeros((len(self.phases), self.steps))                  #Driving force
        self.betas = np.zeros((len(self.phases), self.steps))                #Impingement rates (used for non-isothermal)
        self.incubationOffset = np.zeros(len(self.phases))                   #Offset for incubation time (for non-isothermal precipitation)
        self.incubationSum = np.zeros(len(self.phases))                      #Sum of incubation time

        self.prevFConc = np.zeros((2, len(self.phases), self.numberOfElements))    #Sum of precipitate composition for mass balance

    def save(self, filename, compressed = False, toCSV = False):
        '''
        Save results into a numpy .npz or .csv format

        Parameters
        ----------
        filename : str
        compressed : bool
            If true, will save compressed .npz format
        toCSV : bool
            If true, will save to .csv
        '''
        variables = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements', \
            'time', 'xComp', 'Rcrit', 'Gcrit', 'Rad', 'avgR', 'avgAR', 'betaFrac', 'nucRate', 'precipitateDensity', 'dGs', 'xEqAlpha', 'xEqBeta']
        vDict = {v: getattr(self, v) for v in variables}
        
        if toCSV:
            vDict['t0'] = np.array([vDict['t0']])
            vDict['tf'] = np.array([vDict['tf']])
            vDict['steps'] = np.array([vDict['steps']])
            vDict['linearTimeSpacing'] = np.array([vDict['linearTimeSpacing']])
            if self.numberOfElements == 2:
                vDict['xComp'] = vDict['xComp'].T
            arrays = []
            headers = []
            for v in vDict:
                vDict[v] = np.array(vDict[v])
                if len(vDict[v].shape) == 2:
                    for i in range(len(vDict[v])):
                        arrays.append(vDict[v][i])
                        headers.append(v + str(i))
                        if v == 'xComp':
                            headers.append(v + '_' + self.elements[i])
                        else:
                            headers.append(v + '_' + self.phases[i])
                elif v == 'xEqAlpha' or v == 'xEqBeta':
                    for i in range(len(self.phases)):
                        for j in range(self.numberOfElements):
                            arrays.append(vDict[v][i,:,j])
                            headers.append(v + '_' + self.phases[i] + '_' + self.elements[j])
                else:
                    arrays.append(vDict[v])
                    headers.append(v)
            rows = zip_longest(*arrays, fillvalue='')
            if '.csv' not in filename.lower():
                filename = filename + '.csv'
            with open(filename, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
                csv.writer(f).writerows(rows)
        else:
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
        setupVars = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements']
        if '.np' in filename.lower():
            data = np.load(filename)
            model = PrecipitateBase(data['t0'], data['tf'], data['steps'], data['phases'], data['linearTimeSpacing'], data['elements'])
            for d in data:
                if d not in setupVars:
                    setattr(model, d, data[d])
        elif '.csv' in filename.lower():
            with open(filename, 'r') as csvFile:
                data = csv.reader(csvFile, delimiter=',')
                i = 0
                headers = []
                columns = {}
                #Grab all columns
                for row in data:
                    if i == 0:
                        headers = row
                        columns = {h: [] for h in headers}
                    else:
                        for j in range(len(row)):
                            if row[j] != '':
                                columns[headers[j]].append(row[j])
                    i += 1

                t0, tf, steps, phases, elements = float(columns['t0'][0]), float(columns['tf'][0]), int(columns['steps'][0]), columns['phases'], columns['elements']
                linearTimeSpacing = True if columns['linearTimeSpacing'][0] == 'True' else False
                model = PrecipitateBase(t0, tf, steps, phases, linearTimeSpacing, elements)

                restOfVariables = ['time', 'xComp', 'Rcrit', 'Gcrit', 'Rad', 'avgR', 'avgAR', 'betaFrac', 'nucRate', 'precipitateDensity', 'dGs', 'xEqAlpha', 'xEqBeta']
                restOfColumns = {v: [] for v in restOfVariables}
                for d in columns:
                    if d not in setupVars:
                        if d == 'time':
                            restOfColumns[d] = np.array(columns[d], dtype='float')
                        elif d == 'xComp':
                            if model.numberOfElements == 1:
                                restOfColumns[d] = np.array(columns[d], dtype='float')
                            else:
                                restOfColumns['xComp'].append(columns[d], dtype='float')
                        else:
                            selectedVar = ''
                            for r in restOfVariables:
                                if r in d:
                                    selectedVar = r
                            restOfColumns[selectedVar].append(np.array(columns[d], dtype='float'))
                for d in restOfColumns:
                    restOfColumns[d] = np.array(restOfColumns[d])
                    setattr(model, d, restOfColumns[d])

                #For multicomponent systems, adjust as necessary such that number of elements will be the last axis
                if model.numberOfElements > 1:
                    model.xComp = model.xComp.T
                    if len(model.phases) == 1:
                        model.xEqAlpha = np.expand_dims(model.xEqAlpha, 0)
                        model.xEqBeta = np.expand_dims(model.xEqBeta, 0)
                    else:
                        model.xEqAlpha = np.reshape(model.xEqAlpha, ((len(model.phases), model.numberOfElements, len(model.time))))
                        model.xEqBeta = np.reshape(model.xEqBeta, ((len(model.phases), model.numberOfElements, len(model.time))))
                    model.xEqAlpha = np.transpose(model.xEqAlpha, (0, 2, 1))
                    model.xEqBeta = np.transpose(model.xEqBeta, (0, 2, 1))
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
            self.xEqAlpha = np.append(self.xEqAlpha, np.zeros((len(self.phases), 1)), axis=1)
            self.xEqBeta = np.append(self.xEqBeta, np.zeros((len(self.phases), 1)), axis=1)
        else:
            self.xComp = np.append(self.xComp, np.zeros((1, self.numberOfElements)), axis=0)
            self.xEqAlpha = np.append(self.xEqAlpha, np.zeros((len(self.phases), 1, self.numberOfElements)), axis=1)
            self.xEqBeta = np.append(self.xEqBeta, np.zeros((len(self.phases), 1, self.numberOfElements)), axis=1)

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
        self.betas = np.append(self.betas, np.zeros((len(self.phases), 1)), axis=1)

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
            #self._postTimeIncrementCheck = self._postCheckDT
            self._postTimeIncrementCheck = self._noPostCheckDT
        else:
            self._timeIncrementCheck = self._noCheckDT
            self._postTimeIncrementCheck = self._noPostCheckDT

    def _calculateDT(self, i, fraction):
        '''
        Calculates DT as a fraction of the total simulation time
        '''
        if self.linearTimeSpacing:
            dt = fraction*(self.tf - self.t0)
        else:
            dt = self.time[i] * (np.exp(fraction*np.log(self.tf / self.t0)) - 1)
        return dt

    def _defaultConstraints(self):
        '''
        Default values for contraints
        '''
        self.minRadius = 3e-10
        self.maxTempChange = 1

        self.maxDTFraction = 1e-2
        self.minDTFraction = 1e-5

        self.checkTemperature = True
        self.maxNonIsothermalDT = 1

        self.checkPSD = True
        self.maxDissolution = 0.01

        self.checkRcrit = True
        self.maxRcritChange = 0.01

        self.checkNucleation = True
        self.maxNucleationRateChange = 0.5
        self.minNucleationRate = 1e-5

        self.checkVolumePre = True
        self.checkVolumePost = False
        self.maxVolumeChange = 0.001
        
        self.checkComposition = False
        self.checkCompositionPre = False
        self.maxCompositionChange = 0.001
        self.minComposition = 0

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
        
    def setAspectRatioNeedle(self, ratio=1, phase = None):
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
        
    def setAspectRatioPlate(self, ratio=1, phase = None):
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
        
    def setAspectRatioCuboidal(self, ratio=1, phase = None):
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
        
        if len(np.unique(self.T)) == 1:
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
        
        if len(np.unique(self.T)) == 1:
            self._incubation = self._incubationIsothermal
        else:
            self._incubation = self._incubationNonIsothermal

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

    def setThermodynamics(self, therm, phase = None, removeCache = False):
        '''
        Parameters
        ----------
        therm : Thermodynamics class
        phase : str (optional)
            Phase to consider (defaults to first precipitate in list)
        '''
        index = self.phaseIndex(phase)
        self.dG[index] = lambda x, T, removeCache = removeCache: therm.getDrivingForce(x, T, precPhase=phase, training = removeCache)
        
        if self.numberOfElements == 1:
            self.interfacialComposition[index] = lambda x, T: therm.getInterfacialComposition(x, T, precPhase=phase)
            if therm.mobCallables is not None or therm.diffCallables is not None:
                self.Diffusivity = lambda x, T, removeCache = removeCache: therm.getInterdiffusivity(x, T, removeCache = removeCache)
        else:
            self.interfacialComposition[index] = lambda x, T, dG, R, gExtra, removeCache = removeCache: therm.getGrowthAndInterfacialComposition(x, T, dG, R, gExtra, precPhase=phase, training = removeCache)
            self._betaFuncs[index] = lambda x, T, removeCache = removeCache: therm.impingementFactor(x, T, precPhase=phase, training = removeCache)

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
            self.effDiffDistance = self.effDiffFuncs.noDiffusionDistance
        else:
            self.effDiffDistance = self.effDiffFuncs.effectiveDiffusionDistance

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
        for p in range(len(self.phases)):
            self.Rmin[p] = self.minRadius
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
            compStr = 'N\tTime (s)\tTemperature (K)\t'
            compValStr = '{:.0f}\t{:.1e}\t\t{:.0f}\t\t'.format(i, self.time[i], self.T[i])
            for a in range(self.numberOfElements):
                compStr += self.elements[a] + '\t'
                compValStr += '{:.4f}\t'.format(100*self.xComp[i,a])
            compValStr += '\n'
            print(compStr)
            print(compValStr)
        print('\tPhase\tPrec Density (#/m3)\tVolume Frac\tAvg Radius (m)\tDriving Force (J/mol)')
        for p in range(len(self.phases)):
            print('\t{}\t{:.3e}\t\t{:.4f}\t\t{:.4e}\t{:.4e}'.format(self.phases[p], self.precipitateDensity[p,i], 100*self.betaFrac[p,i], self.avgR[p,i], self.dGs[p,i]*self.VmBeta[p]))
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
        stopCondition = False
        while i < self.steps and not stopCondition:
            self._iterate(i)

            #Apply stopping condition
            if self._stoppingConditions is not None:
                andConditions = True
                numberOfAndConditions = 0
                orConditions = False
                for s in range(len(self._stoppingConditions)):
                    #Record time if stopping condition is met
                    conditionResult = self._stoppingConditions[s](self, i)
                    if conditionResult and self.stopConditionTimes[s] == -1:
                        self.stopConditionTimes[s] = self.time[i]

                    #If condition mode is 'or'
                    if self._stopConditionMode[s]:
                        orConditions = orConditions or conditionResult
                    #If condition mode is 'and'
                    else:
                        andConditions = andConditions and conditionResult
                        numberOfAndConditions += 1

                #If there are no 'and' conditions, andConditions will be True
                #Set to False so andConditions will not stop the model unneccesarily
                if numberOfAndConditions == 0:
                    andConditions = False

                stopCondition = andConditions or orConditions

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

    def _nucleationRate(self, p, i):
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

        #Add strain energy for spherical shape, use previous critical radius
        #This should still be correct even if the interfacial energy dominates at small radii since the aspect ratio may be near 1
        self.dGs[p, i] -= self.strainEnergy[p].strainEnergy(self.shapeFactors[p].normalRadii(self.Rcrit[p, i-1]))

        if self.dGs[p, i] < 0:
            return self._noDrivingForce(p, i)

        #Only do this if there is some parent phase left (brute force solution for to avoid numerical errors)
        if self.betaFrac[p, i-1] < 1:

            #Calculate critical radius
            #For bulk or dislocation nucleation sites, use previous critical radius to get aspect ratio
            if self.GB[p].nucleationSiteType == GBFactors.BULK or self.GB[p].nucleationSiteType == GBFactors.DISLOCATION:
                self.Rcrit[p, i] = 2 * self.shapeFactors[p].thermoFactor(self.Rcrit[p, i-1]) * self.gamma[p] / self.dGs[p, i]
                #self.Rcrit[p, i] = 2 * self.gamma[p] / self.dGs[p, i]
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
            self.betas[p,i] = self._Beta(p, i)
            if self.betas[p,i] == 0:
                return self._noDrivingForce(p, i)

            #Incubation time, either isothermal or nonisothermal
            self.incubationSum[p] = self._incubation(Z, p, i)
            if self.incubationSum[p] > 1:
                self.incubationSum[p] = 1

            return Z * self.betas[p,i] * np.exp(-self.Gcrit[p, i] / (self.kB * self.T[i])) * self.incubationSum[p]

        else:
            return self._noDrivingForce(p, i)
            
    def _noCheckDT(self, i):
        '''
        Default for no adaptive time stepping
        '''
        pass

    def _noPostCheckDT(self, i):
        '''
        Default for no adaptive time stepping
        '''
        pass

    def _checkDT(self, i):
        '''
        Default time increment function if implement (which is no implementation)
        '''
        pass

    def _postCheckDT(self, i):
        '''
        Default time increment function if implement (which is no implementation)
        '''
        pass

    def _noDrivingForce(self, p, i):
        '''
        Set everything to 0 if there is no driving force for precipitation
        '''
        self.Rcrit[p, i] = 0
        self.incubationOffset[p] = np.amax([i-1, 0])
        return 0

    def _nucleateFreeEnergy(self, Rsph, p, i):
        volContribution = 4/3 * np.pi * Rsph**3 * (self.dGs[p,i] + self.strainEnergy[p].strainEnergy(self.shapeFactors[p].normalRadii(Rsph)))
        areaContribution = 4 * np.pi * self.gamma[p] * Rsph**2 * self.shapeFactors[p].thermoFactor(Rsph)
        return -volContribution + areaContribution

    def _Zeldovich(self, p, i):
        return np.sqrt(3 * self.GB[p].volumeFactor / (4 * np.pi)) * self.VmBeta[p] * np.sqrt(self.gamma[p] / (self.kB * self.T[i])) / (2 * np.pi * self.avo * self.Rcrit[p,i]**2)
        
    def _BetaBinary(self, p, i):
        return self.GB[p].areaFactor * self.Rcrit[p,i]**2 * self.xComp[0] * self.Diffusivity(self.xComp[i], self.T[i]) / self.aAlpha**4
            
    def _BetaMulti(self, p, i):
        if self._betaFuncs[p] is None:
            return self._defaultBeta
        else:
            beta = self._betaFuncs[p](self.xComp[i-1], self.T[i-1])
            if beta is None:
                return self.betas[p,i-1]
            else:
                return (self.GB[p].areaFactor * self.Rcrit[p, i]**2 / self.aAlpha**4) * beta

    def _incubationIsothermal(self, Z, p, i):
        '''
        Incubation time for isothermal conditions
        '''
        tau = 1 / (self.theta[p] * (self.betas[p,i] * Z**2))
        return np.exp(-tau / self.time[i])
        
    def _incubationNonIsothermal(self, Z, p, i):
        '''
        Incubation time for non-isothermal conditions
        This must match isothermal conditions if the temperature is constant

        Solve for integral(beta(t-t0)) from 0 to tau = 1/theta*Z(tau)^2
        '''
        LHS = 1 / (self.theta[p] * Z**2 * (self.T[i] / self.T[int(self.incubationOffset[p]):]))

        RHS = np.cumsum(self.betas[p,int(self.incubationOffset[p])+1:i] * (self.time[int(self.incubationOffset[p])+1:i] - self.time[int(self.incubationOffset[p]):i-1]))
        if len(RHS) == 0:
            RHS = self.betas[p,i] * (self.time[int(self.incubationOffset[p]):] - self.time[int(self.incubationOffset[p])])
        else:
            RHS = np.concatenate((RHS, RHS[-1] + self.betas[p,i] * (self.time[i-1:] - self.time[int(self.incubationOffset[p])])))

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
                tau = LHS[-1] / self.betas[p,i] - RHS[-1] / self.betas[p,i] + (self.time[i] - self.time[int(self.incubationOffset[p])])
        else:
            tau = self.time[int(self.incubationOffset[p]):-1][signChange][0] - self.time[int(self.incubationOffset[p])]

        return np.exp(-tau / (self.time[i] - self.time[int(self.incubationOffset[p])]))

    def _setNucleateRadius(self, i):
        for p in range(len(self.phases)):
            #If nucleates form, then calculate radius of precipitate
            #Radius is set slightly larger so preciptate 
            if self.nucRate[p,i]*(self.time[i]-self.time[i-1]) >= 1 and self.Rcrit[p, i] >= self.Rmin[p]:
                self.Rad[p, i] = self.Rcrit[p, i] + 0.5 * np.sqrt(self.kB * self.T[i] / (np.pi * self.gamma[p]))
            else:
                self.Rad[p, i] = 0

    def getTimeAxis(self, timeUnits='s', bounds=None):
        timeScale = 1
        timeLabel = 'Time (s)'
        if 'min' in timeUnits:
            timeScale = 1/60
            timeLabel = 'Time (min)'
        if 'h' in timeUnits:
            timeScale = 1/3600
            timeLabel = 'Time (hrs)'

        if bounds is None:
            if self.t0 == 0:
                bounds = [timeScale * 1e-5 * self.tf, timeScale * self.tf]
            else:
                bounds = [timeScale * self.t0, timeScale * self.tf]

        return timeScale, timeLabel, bounds
        
    
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
                'Temperature' and 'Composition'

                Note: for multi-phase simulations, adding the word 'Total' will
                    sum the variable for all phases. Without the word 'Total', the variable
                    for each phase will be plotted separately
                    
        bounds : tuple (optional)
            Limits on the x-axis (float, float) or None (default, this will set bounds to (initial time, final time))
        timeUnits : str (optional)
            Plot time dependent variables per seconds ('s'), minutes ('m') or hours ('h')
        radius : str (optional)
            For non-spherical precipitates, plot the Average Radius by the -
                Equivalent spherical radius ('spherical')
                Short axis ('short')
                Long axis ('long')
            Note: Total Average Radius and Volume Average Radius will still use the equivalent spherical radius
        *args, **kwargs - extra arguments for plotting
        '''
        timeScale, timeLabel, bounds = self.getTimeAxis(timeUnits, bounds)

        axes.set_xlabel(timeLabel)
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
            'Eq Composition Alpha': 'Matrix Composition (at.%)',
            'Eq Composition Beta': 'Matrix Composition (at.%)',
            'Supersaturation': 'Supersaturation',
            'Eq Volume Fraction': 'Volume Fraction'
        }

        totalVariables = ['Total Volume Fraction', 'Total Average Radius', 'Total Aspect Ratio', \
                            'Total Nucleation Rate', 'Total Precipitate Density']
        singleVariables = ['Volume Fraction', 'Critical Radius', 'Average Radius', 'Aspect Ratio', \
                            'Driving Force', 'Nucleation Rate', 'Precipitate Density']
        eqCompositions = ['Eq Composition Alpha', 'Eq Composition Beta']
        saturations = ['Supersaturation', 'Eq Volume Fraction']

        if variable == 'Temperature':
            axes.semilogx(timeScale * self.time, self.T, *args, **kwargs)
            axes.set_ylabel(labels[variable])

        elif variable == 'Composition':
            if self.numberOfElements == 1:
                axes.semilogx(timeScale * self.time, self.xComp, *args, **kwargs)
                axes.set_ylabel('Matrix Composition (at.% ' + self.elements[0] + ')')
            else:
                for i in range(self.numberOfElements):
                    #Keep color consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                    if 'color' in kwargs:
                        axes.semilogx(timeScale * self.time, self.xComp[:,i], label=self.elements[i], *args, **kwargs)
                    else:
                        axes.semilogx(timeScale * self.time, self.xComp[:,i], label=self.elements[i], color='C'+str(i), *args, **kwargs)
                axes.legend(self.elements)
                axes.set_ylabel(labels[variable])
            yRange = [np.amin(self.xComp), np.amax(self.xComp)]
            axes.set_ylim([yRange[0] - 0.1 * (yRange[1] - yRange[0]), yRange[1] + 0.1 * (yRange[1] - yRange[0])])

        elif variable in eqCompositions:
            if variable == 'Eq Composition Alpha':
                plotVariable = self.xEqAlpha
            elif variable == 'Eq Composition Beta':
                plotVariable = self.xEqBeta

            if len(self.phases) == 1:
                if self.numberOfElements == 1:
                    axes.semilogx(timeScale * self.time, plotVariable[0], *args, **kwargs)
                    axes.set_ylabel('Matrix Composition (at.% ' + self.elements[0] + ')')
                else:
                    for i in range(self.numberOfElements):
                        #Keep color consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                        if 'color' in kwargs:
                            axes.semilogx(timeScale * self.time, plotVariable[0,:,i], label=self.elements[i]+'_Eq', *args, **kwargs)
                        else:
                            axes.semilogx(timeScale * self.time, plotVariable[0,:,i], label=self.elements[i]+'_Eq', color='C'+str(i), *args, **kwargs)
                    axes.legend()
                    axes.set_ylabel(labels[variable])
            else:
                if self.numberOfElements == 1:
                    for p in range(len(self.phases)):
                        #Keep color somewhat consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                        if 'color' in kwargs:
                            axes.semilogx(timeScale * self.time, plotVariable[p], label=self.phases[p]+'_Eq', *args, **kwargs)
                        else:
                            axes.semilogx(timeScale * self.time, plotVariable[p], label=self.phases[p]+'_Eq', color='C'+str(p), *args, **kwargs)
                    axes.legend()
                    axes.set_ylabel('Matrix Composition (at.% ' + self.elements[0] + ')')
                else:
                    cIndex = 0
                    for p in range(len(self.phases)):
                        for i in range(self.numberOfElements):
                            #Keep color somewhat consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                            if 'color' in kwargs:
                                axes.semilogx(timeScale * self.time, plotVariable[p,:,i], label=self.phases[p]+'_'+self.elements[i]+'_Eq', *args, **kwargs)
                            else:
                                axes.semilogx(timeScale * self.time, plotVariable[p,:,i], label=self.phases[p]+'_'+self.elements[i]+'_Eq', color='C'+str(cIndex), *args, **kwargs)
                            cIndex += 1
                    axes.legend()
                    axes.set_ylabel(labels[variable])

        elif variable in saturations:
            #Since supersaturation is calculated in respect to the tie-line, it is the same for each element
            #Thus only a single element is needed
            plotVariable = np.zeros(self.betaFrac.shape)
            for p in range(len(self.phases)):
                if self.numberOfElements == 1:
                    if variable == 'Eq Volume Fraction':
                        num = self.xComp[0] - self.xEqAlpha[p]
                    else:
                        num = self.xComp - self.xEqAlpha[p]
                    den = self.xEqBeta[p] - self.xEqAlpha[p]
                else:
                    if variable == 'Eq Volume Fraction':
                        num = self.xComp[0,0] - self.xEqAlpha[p,:,0]
                    else:
                        num = self.xComp[:,0] - self.xEqAlpha[p,:,0]
                    den = self.xEqBeta[p,:,0] - self.xEqAlpha[p,:,0]
                #If precipitate is unstable, both xEqAlpha and xEqBeta are set to 0
                #For these cases, change the values of numerator and denominator so that supersaturation is 0 instead of undefined
                num[den == 0] = 0
                den[den == 0] = 1
                plotVariable[p] = num / den
            
            if len(self.phases) == 1:
                axes.semilogx(timeScale * self.time, plotVariable[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    if 'color' in kwargs:
                        axes.semilogx(timeScale * self.time, plotVariable[p], label=self.phases[p], *args, **kwargs)
                    else:
                        axes.semilogx(timeScale * self.time, plotVariable[p], label=self.phases[p], color='C'+str(p), *args, **kwargs)
                axes.legend()
            axes.set_ylabel(labels[variable])

        elif variable in singleVariables:
            if variable == 'Volume Fraction':
                plotVariable = self.betaFrac
            elif variable == 'Critical Radius':
                plotVariable = self.Rcrit
            elif variable == 'Average Radius':
                plotVariable = self.avgR
                for p in range(len(self.phases)):
                    if self.GB[p].nucleationSiteType == self.GB[p].BULK or self.GB[p].nucleationSiteType == self.GB[p].DISLOCATION:
                        if radius != 'spherical':
                            plotVariable[p] /= self.shapeFactors[p].eqRadiusFactor(self.avgR[p])
                        if radius == 'long':
                            plotVariable[p] *= self.avgAR[p]
                    else:
                        plotVariable[p] *= self._GBareaRemoval(p)

            elif variable == 'Volume Average Radius':
                plotVariable = np.cbrt(self.betaFrac / self.precipitateDensity / (4/3*np.pi))
            elif variable == 'Aspect Ratio':
                plotVariable = self.avgAR
            elif variable == 'Driving Force':
                plotVariable = self.dGs
            elif variable == 'Nucleation Rate':
                plotVariable = self.nucRate
            elif variable == 'Precipitate Density':
                plotVariable = self.precipitateDensity

            if (len(self.phases)) == 1:
                axes.semilogx(timeScale * self.time, plotVariable[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    axes.semilogx(timeScale * self.time, plotVariable[p], label=self.phases[p], color='C'+str(p), *args, **kwargs)
                axes.legend()
            axes.set_ylabel(labels[variable])
            yb = 1 if variable == 'Aspect Ratio' else 0
            axes.set_ylim([yb, 1.1 * np.amax(plotVariable)])

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

            axes.semilogx(timeScale * self.time, plotVariable, *args, **kwargs)
            axes.set_ylabel(labels[variable])
            yb = 1 if variable == 'Total Aspect Ratio' else 0
            axes.set_ylim(bottom=yb)
