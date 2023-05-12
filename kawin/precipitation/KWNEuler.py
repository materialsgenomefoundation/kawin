import numpy as np
from kawin.precipitation.KWNBase import PrecipitateBase
from kawin.precipitation.PopulationBalance import PopulationBalanceModel
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
import copy
import csv
from itertools import zip_longest
import time

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
        #Initialize base class
        super().__init__(t0, tf, steps, phases, linearTimeSpacing, elements)

        if self.numberOfElements == 1:
            self._growthRate = self._growthRateBinary
            self._Beta = self._BetaBinary1
        else:
            self._growthRate = self._growthRateMulti
            self._Beta = self._BetaMulti

        #Additional outputs
        self.additionalFunctions = []
        self.additionalFunctionNames = []
        self.additionalOutputs = None

        #Coupling functions - this can be used to integrate a model simutaneously within the KWN model
        self.couplingFunctions = []
        self.couplingFunctionNames = []

    def _resetArrays(self):
        '''
        Resets and initializes arrays for all variables

        In addition to PrecipitateBase, the equilibrium aspect ratio area and population balance models are created here
        '''
        super()._resetArrays()
        self.PBM = [PopulationBalanceModel() for p in self.phases]

        #Index of particle size classes which below, precipitates are unstable
        self.RdrivingForceIndex = np.zeros(len(self.phases), dtype=np.int32)

        #Aspect ratio
        self.eqAspectRatio = [[] for p in self.phases]

    def reset(self):
        '''
        Resets model results
        '''
        super().reset()

        #Bounds of the bins in PSD
        for i in range(len(self.phases)):
            self.PBM[i].reset()

        #Resets PSD outputs
        self._setupAdditionalOutputs()

    def save(self, filename, compressed = False, toCSV = False):
        '''
        Save results into a numpy .npz format

        Parameters
        ----------
        filename : str
        compressed : bool
            If true, will save compressed .npz format
        toCSV : bool
            If true, wil save to .csv
        '''
        variables = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements', \
            'time', 'xComp', 'Rcrit', 'Gcrit', 'Rad', 'avgR', 'avgAR', 'betaFrac', 'nucRate', 'precipitateDensity', 'dGs', 'xEqAlpha', 'xEqBeta']
        vDict = {v: getattr(self, v) for v in variables}
        if self.additionalOutputs is not None:
            vDict['additionalOutputs'] = self.additionalOutputs
            if not toCSV:
                vDict['additionalFunctionNames'] = self.additionalFunctionNames
        for p in range(len(self.phases)):
            vDict['PSDdata_'+self.phases[p]] = [self.PBM[p].min, self.PBM[p].max, self.PBM[p].bins]
            vDict['PSDsize_' + self.phases[p]] = self.PBM[p].PSDsize
            vDict['PSD_' + self.phases[p]] = self.PBM[p].PSD
            vDict['PSDbounds_' + self.phases[p]] = self.PBM[p].PSDbounds
            vDict['eqAspectRatio_' + self.phases[p]] = self.eqAspectRatio[p]

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
                        if v == 'xComp':
                            headers.append(v + '_' + self.elements[i])
                        else:
                            headers.append(v + '_' + self.phases[i])
                elif v == 'xEqAlpha' or v == 'xEqBeta':
                    for i in range(len(self.phases)):
                        for j in range(self.numberOfElements):
                            arrays.append(vDict[v][i,:,j])
                            headers.append(v + '_' + self.phases[i] + '_' + self.elements[j])
                elif v == 'additionalOutputs':
                    for i in range(len(self.phases)):
                        for j in range(len(self.additionalFunctionNames)):
                            arrays.append(vDict[v][i,:,j])
                            headers.append(v + '_' + self.phases[i] + '_' + self.additionalFunctionNames[j])
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
                #np.savez_compressed(filename, **vDict, allow_pickle=True)
            else:
                np.savez(filename, **vDict)
                #np.savez(filename, **vDict, allow_pickle=True)

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
        setupVars = ['t0', 'tf', 'steps', 'phases', 'linearTimeSpacing', 'elements']
        if '.np' in filename.lower():
            data = np.load(filename, allow_pickle=True)
            
            #Input arbitrary values for PSD parameters (rMin, rMax, bins) since this will be changed shortly after
            model = PrecipitateModel(data['t0'], data['tf'], data['steps'], data['phases'], data['linearTimeSpacing'], data['elements'])
            for p in range(len(model.phases)):
                PSDvars = ['PSDdata_' + model.phases[p], 'PSD_' + model.phases[p], 'PSDsize_' + model.phases[p], 'eqAspectRatio_' + model.phases[p], 'PSDbounds_' + model.phases[p]]
                #For back compatibility
                if PSDvars[0] not in data:
                    PSDvars = ['PSDdata' + str(p), 'PSD' + str(p), 'PSDsize' + str(p), 'eqAspectRatio' + str(p), 'PSDbounds' + str(p)]
                setupVars = np.concatenate((setupVars, PSDvars))
                model.PBM[p] = PopulationBalanceModel(data[PSDvars[0]][0], data[PSDvars[0]][1], int(data[PSDvars[0]][2]), True)
                model.PBM[p].PSD = data[PSDvars[1]]
                model.PBM[p].PSDsize = data[PSDvars[2]]
                model.eqAspectRatio[p] = data[PSDvars[3]]
                model.PBM[p].PSDbounds = data[PSDvars[4]]
            for d in data:
                if d not in setupVars:
                    setattr(model, d, data[d])
            if 'additionalOutputs' not in data:
                model.additionalOutputs = None
                model.additionalFunctions = []
                model.additionalFunctionNames = []
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
                model = PrecipitateModel(t0, tf, steps, phases, linearTimeSpacing, elements)

                for p in range(len(model.phases)):
                    PSDvars = ['PSDdata_' + model.phases[p], 'PSD_' + model.phases[p], 'PSDsize_' + model.phases[p], 'eqAspectRatio_' + model.phases[p], 'PSDbounds_' + model.phases[p]]
                    #For back compatibility
                    if PSDvars[0] not in columns:
                        PSDvars = ['PSDdata' + str(p), 'PSD' + str(p), 'PSDsize' + str(p), 'eqAspectRatio' + str(p), 'PSDbounds' + str(p)]
                    setupVars = np.concatenate((setupVars, PSDvars))
                    model.PBM[p] = PopulationBalanceModel(float(columns[PSDvars[0]][0]), float(columns[PSDvars[0]][1]), int(float(columns[PSDvars[0]][2])), True)
                    model.PBM[p].PSD = np.array(columns[PSDvars[1]], dtype='float')
                    model.PBM[p].PSDsize = np.array(columns[PSDvars[2]], dtype='float')
                    model.eqAspectRatio[p] = np.array(columns[PSDvars[3]], dtype='float')
                    model.PBM[p].PSDbounds = np.array(columns[PSDvars[4]], dtype='float')

                restOfVariables = ['time', 'xComp', 'Rcrit', 'Gcrit', 'Rad', 'avgR', 'avgAR', 'betaFrac', 'nucRate', 'precipitateDensity', 'dGs', 'xEqAlpha', 'xEqBeta', 'additionalOutputs']
                restOfColumns = {v: [] for v in restOfVariables}
                additionalFunctionNames = []
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
                            if selectedVar == 'additionalOutputs':
                                additionalFunctionNames.append(d[18:])
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

                #If additional outputs exists, then reshape array to (phase, iterations, functions)
                if len(additionalFunctionNames) > 0:
                    numberOfFunctions = int(len(additionalFunctionNames) / len(model.phases))
                    model.additionalOutputs = np.reshape(model.additionalOutputs, (len(model.phases), numberOfFunctions, len(model.time)))
                    model.additionalOutputs = np.transpose(model.additionalOutputs, (0, 2, 1))
                    model.additionalFunctionNames = []
                    for i in range(numberOfFunctions):
                        model.additionalFunctionNames.append(additionalFunctionNames[i][len(model.phases[0])+1:])
                    model.additionalFunctionNames = np.array(model.additionalFunctionNames)

        return model

    def _divideTimestep(self, i, dt):
        '''
        Divides timestep at iteration i
        '''
        super()._divideTimestep(i, dt)

        if len(self.additionalFunctions) > 0:
            self.additionalOutputs = np.append(self.additionalOutputs, np.zeros((len(self.phases), 1, len(self.additionalFunctions))), axis=1)

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

    def setPSDrecording(self, record = True, dtype = np.float32, phase = 'all'):
        '''
        Sets recording parameters for PSD of specified phase

        Parameters
        ----------
        record : bool (optional)
            Whether to record PSD, defaults to True
        dtype : numpy data type (optional)
            Data type to record PSD in, defaults to np.float32
        phase : str (optional)
            Precipitate phase to record for
            Defaults to 'all', which will apply to all precipitate phases
        '''
        if phase is None or phase == 'all':
            for p in self.phases:
                index = self.phaseIndex(p)
                if record:
                    self.PBM[index].enableRecording(dtype)
                else:
                    self.PBM[index].disableRecording(dtype)
        else:
            index = self.phaseIndex(phase)
            if record:
                self.PBM[index].enableRecording(dtype)
            else:
                self.PBM[index].disableRecording(dtype)

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

    def addCouplingFunction(self, name, f):
        '''
        Adds a coupling function to integrate along the KWN model
        This is done at the end of the iteration

        Parameter
        ---------
        name : str
            Name of the function
        f : function
            Takes in model, dt and iteration, and returns a value
        '''
        if name in self.couplingFunctionNames:
            i = 1
            name = name + '_{}'.format(i)
            while name in self.couplingFunctionNames:
                i += 1
                name = name[:-2]
                name = name + '_{}'.format(i)
            print('Warning: Function \'{}\' has already been set, this function will be stored as \'{}\''.format(name[:-2], name))

        self.couplingFunctions.append(f)
        self.couplingFunctionNames = np.append(self.couplingFunctionNames, name)

    def _calculateAdditionalCoupling(self, dt, i):
        '''
        Calculates additional coupling
        '''
        for f in range(len(self.couplingFunctions)):
            self.couplingFunctions[f](self, dt, i)

    def addAdditionalOutput(self, name, f):
        '''
        Creates output based off PSD

        Parameters
        ----------
        name : str
            Name of the function
        f : function
            Takes in model, phase index and iteration index and returns a value
        '''
        if name in self.additionalFunctionNames:
            i = 1
            name = name + '_{}'.format(i)
            while name in self.additionalFunctionNames:
                i += 1
                name = name[:-2]
                name = name + '_{}'.format(i)
            print('Warning: Function \'{}\' has already been set, this function will be stored as \'{}\''.format(name[:-2], name))
            
        self.additionalFunctions.append(f)
        self.additionalFunctionNames = np.append(self.additionalFunctionNames, name)

    def _setupAdditionalOutputs(self):
        '''
        Function to setup PSD output arrays, will be used in setup and reset functions
        '''
        #Resets PSD outputs
        if len(self.additionalFunctions) > 0:
            self.additionalOutputs = np.zeros((len(self.phases), self.steps, len(self.additionalFunctions)))

    def _calculateAdditionalOutputs(self, i):
        '''
        Calculates additional PSD functions
        '''
        for f in range(len(self.additionalFunctions)):
            for p in range(len(self.phases)):
                self.additionalOutputs[p, i, f] = self.additionalFunctions[f](self, p, i)

    def getAdditionalOutput(self, name):
        '''
        Gets additional output by name

        Parameters
        ----------
        name : str
            Name of function used for the additional output

        Returns
        -------
        (p, N) array for the output for each phase
        '''
        if name in self.additionalFunctionNames:
            index, = np.where(self.additionalFunctionNames == name)
            return self.additionalOutputs[:, :, index[0]]

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

            self.PSDXalpha[p], self.PSDXbeta[p] = self.interfacialComposition[p](self.T[i], self.particleGibbs(self.PBM[p].PSDbounds, self.phases[p]))
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
            
    def setup(self):
        '''
        Sets up additional variables in addition to PrecipitateBase

        Sets up additional outputs, population balance models, equilibrium aspect ratio and equilibrium compositions
        '''
        super().setup()

        self._setupAdditionalOutputs()

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

    def _iterate(self, i):
        '''
        Iteration function
        '''
        #Nucleation and growth rate are independent of time increment
        #They can be calculated first and used to determine the time increment for numerical stability
        self._nucleate(i)
        self._setNucleateRadius(i)
        self._growthRate(i)
        self._timeIncrementCheck(i)

        #Backup variables in case size classes on PSD changes
        self.growthBackup = copy.copy(self.growth)
        self.PSDXalphaBackup = copy.copy(self.PSDXalpha)
        self.PSDXbetaBackup = copy.copy(self.PSDXbeta)
        self.eqAspectRatioBackup = copy.copy(self.eqAspectRatio)
        self.RdrivingForceIndexBackup = copy.copy(self.RdrivingForceIndex)
        self.RdrivingForceLimitBackup = copy.copy(self.RdrivingForceLimit)

        postDTCheck = False
        while not postDTCheck:
            dt = self.time[i] - self.time[i-1]
            self._calculatePSD(i, dt)
            self._massBalance(i)

            if i < self.steps - 1:
                postDTCheck = self._postTimeIncrementCheck(i)
            else:
                postDTCheck = True

        #Calculate additional coupling
        self._calculateAdditionalCoupling(dt, i)

        #Calculate additional PSD function
        self._calculateAdditionalOutputs(i)

        #Record PSD, this will only do stuff if PopulationBalanceModel._record is True
        for p in range(len(self.phases)):
            self.PBM[p].record(self.time[i])

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
        dt = self._calculateDT(i-1, self.maxDTFraction)
        dtAll = [dt]

        if self.checkPSD:
            if self.T[i] == self.T[i-1]:
                dtPBM = [self.PBM[p].getDTEuler(dt, self.growth[p], self.maxDissolution, self.RdrivingForceIndex[p]) for p in range(len(self.phases))]
            else:
                dtPBM = [dt]
            dt = np.amin(np.concatenate(([dt], dtPBM)))
            dtAll.append(dt)

        if i > 1:
            dtPrev = self.time[i-1] - self.time[i-2]
        else:
            dtPrev = dt

        #Nucleation rate constraint
        if self.checkNucleation:
            dtNuc = dt * np.ones(len(self.phases)+1)
            for p in range(len(self.phases)):
                if self.nucRate[p,i] > self.minNucleationRate and self.nucRate[p,i-1] > self.minNucleationRate and self.nucRate[p,i-1] != self.nucRate[p,i]:
                    dtNuc[p] = self.maxNucleationRateChange * dtPrev / np.abs(np.log10(self.nucRate[p,i-1] / self.nucRate[p,i]))
            dt = np.amin(dtNuc)
            dtAll.append(dt)

        #Temperature change constraint
        if self.checkTemperature:
            Tchange = self.T[i] - self.T[i-1]
            dtTemp = dt
            if Tchange > self.maxNonIsothermalDT:
                dtTemp = self.maxNonIsothermalDT * (self.time[i] - self.time[i-1]) / Tchange
                dt = np.amin([dt, dtTemp])

        if self.checkRcrit:
            dtRad = dt * np.ones(len(self.phases)+1)
            if not all((self.Rcrit[:,i-1] == 0) & (self.Rcrit[:,i] - self.Rcrit[:,i-1] == 0) & (self.dGs[:,i] <= 0)):
                indices = (self.Rcrit[:,i-1] > 0) & (self.Rcrit[:,i] - self.Rcrit[:,i-1] != 0) & (self.dGs[:,i] > 0)
                dtRad[:-1][indices] = self.maxRcritChange * dtPrev / np.abs((self.Rcrit[:,i][indices] - self.Rcrit[:,i-1][indices]) / self.Rcrit[:,i-1][indices])
            dt = np.amin(dtRad)
            dtAll.append(dt)

        if self.checkVolumePre or self.checkCompositionPre:
            dV = np.zeros(len(self.phases))
            for p in range(len(self.phases)):
                #Calculate estimate volume change based off growth rate and nucleated particles
                #TODO: account for non-spherical precipitates
                dVi = self.PBM[p].PSD * self.PBM[p].PSDsize**2 * 0.5 * (self.growth[p][1:] + self.growth[p][:-1])
                dVi[dVi < 0] = 0
                dV = self.VmAlpha / self.VmBeta[p] * (self.GB[p].areaFactor * np.sum(dVi) + self.GB[p].volumeFactor * self.nucRate[p,i] * self.Rad[p,i]**3)

            if self.checkVolumePre:
                dtVol = dt * np.ones(len(self.phases) + 1)
                for p in range(len(self.phases)):
                    if dV != 0:
                        dtVol[p] = self.maxVolumeChange / (2 * np.abs(dV))
                #if not all((self.Rad[:,i]**3*self.nucRate[:,i] > 1e-30)):
                #    indices = (self.Rad[:,i]**3*self.nucRate[:,i] > 1e-30)
                #    dtVol[:-1][indices] = self.maxVolumeChange / (10 * (4*np.pi*self.Rad[:,i][indices]**3*self.nucRate[:,i][indices]/3))
                dt = np.amin(dtVol)
                dtAll.append(dt)

            if self.checkCompositionPre:
                dtComp = dt * np.ones(self.numberOfElements + 1)
                fvsum = np.sum(self.betaFrac[:,i-1])
                xbavg = np.zeros(self.numberOfElements)
                if self.numberOfElements == 1:
                    xbavg[0] = 0 if fvsum == 0 else (self.xComp[0] - self.xComp[i-1] * (1 - fvsum)) / fvsum
                    dxadt = (self.xComp[i-1] - xbavg) * np.sum(dV) / (1 - fvsum)
                else:
                    for e in range(self.numberOfElements):
                        xbavg[e] = 0 if fvsum == 0 else (self.xComp[0,e] - self.xComp[i-1,e] * (1 - fvsum)) / fvsum
                    dxadt = (self.xComp[i-1,:] - xbavg) * np.sum(dV) / (1 - fvsum)
                dxadt[dxadt == 0] = self.maxCompositionChange / (2 * dt)
                dtComp[:self.numberOfElements] = self.maxCompositionChange / (2 * dxadt)
                    
                dt = np.amin(dtComp)
                dtAll.append(dt)

        #Minimum dt is the lower of the minimum allowed time increment or the time to the next pre-defined increment
        minDT = self._calculateDT(i-1, self.minDTFraction)
        dt = np.amax([dt, minDT])

        #Override time increment with the predefined time steps
        #This prevents the next time increment from becoming 0 or negative
        dt = np.amin([dt, self.time[i] - self.time[i-1]])
        
        if dt < self.time[i] - self.time[i-1]:
            #print(dtAll)
            self._divideTimestep(i, dt)

    def _noPostCheckDT(self, i):
        '''
        Function if no adaptive time stepping is used, no need to do anything in this function
        '''
        return True

    def _postCheckDT(self, i):
        '''
        CURRENTLY UNUSED AND MAY BE REMOVED LATER

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

        if self.checkVolumePost:
            volChange = np.abs(self.betaFrac[:,i] - self.betaFrac[:,i-1])
            #If current volume fraction is 0, then ignore (either precipitation has not occured or precipitates has dissolved)
            volChange[self.betaFrac[:,i] == 0] = 0
            volCheck = np.amax(volChange) < self.maxVolumeChange
        else:
            volCheck = True

        if self.checkComposition:
            if self.numberOfElements == 1:
                compCheck = (np.abs(self.xComp[i] - self.xComp[i-1]) < self.maxCompositionChange) & (self.xComp[i] > 0)
            else:
                compCheck = (np.amax(np.abs(self.xComp[i,:] - self.xComp[i-1,:])) < self.maxCompositionChange) & (np.amin(self.xComp[i,:] > self.minComposition))
        else:
            compCheck = True

        checks = [volCheck, compCheck]

        #If any test fails, then reset iteration and divide time increment
        if not all(checks):
            dt = (self.time[i] - self.time[i-1]) / 2
            minDT = self._calculateDT(i-1, self.minDTFraction)

            #If proposed time increment is smaller than the minimum allowed increment, then skip the checks
            if dt < minDT:
                return True

            #Only revert changes to variables that aren't stored per iteration
            #Variables related to nucleation are not dependent on the time increment
            #Variables related to the particle size distribution (composition, volume fraction, etc)
            # will be overridden if the time increment changes            
            self.prevFConc[0] = copy.copy(self.prevFConc[1])

            for p in range(len(self.phases)):
                self.PBM[p].revert()
            self.growth = copy.copy(self.growthBackup)
            self.PSDXalpha = copy.copy(self.PSDXalphaBackup)
            self.PSDXbeta = copy.copy(self.PSDXbetaBackup)
            self.eqAspectRatio = copy.copy(self.eqAspectRatioBackup)
            self.RdrivingForceIndex = copy.copy(self.RdrivingForceIndexBackup)
            self.RdrivingForceLimit = copy.copy(self.RdrivingForceLimitBackup)

            self._divideTimestep(i, dt)

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

    def _calculatePSD(self, i, dt):
        '''
        Updates the PSD using the population balance model from coarsening and nucleation rate
        This also updates the fraction of precipitates, matrix composition and average radius
        '''
        for p in range(len(self.phases)):
            #Backup PSD for time increment checks
            #Also backup PSDXbeta for precipitate composition with no diffusion
            self.PBM[p].createBackup()
            self._prevPSDXbeta = copy.copy(self.PSDXbeta)

            change1, newIndices = self.PBM[p].UpdateEuler(dt, self.growth[p])
            change2 = self.PBM[p].Nucleate(self.nucRate[p, i] * dt, self.Rad[p, i])
            if change1 or change2:
                #Add aspect ratio, do this before growth rate and interfacial composition since those are dependent on this
                if self.calculateAspectRatio[p]:
                    self.eqAspectRatio[p] = self.strainEnergy[p].eqAR_bySearch(self.PBM[p].PSDbounds, self.gamma[p], self.shapeFactors[p])
                else:
                    self.eqAspectRatio[p] = self.shapeFactors[p].aspectRatio(self.PBM[p].PSDbounds)

                self.growth[p] = np.zeros(len(self.PBM[p].PSDbounds))
                if self.numberOfElements == 1:
                    if newIndices is None:
                        #This is very slow to do
                        self.createLookup(i)
                    else:
                        self.PSDXalpha[p] = np.concatenate((self.PSDXalpha[p], np.zeros(self.PBM[p].bins+1 - len(self.PSDXalpha[p]))))
                        self.PSDXbeta[p] = np.concatenate((self.PSDXbeta[p], np.zeros(self.PBM[p].bins+1 - len(self.PSDXbeta[p]))))
                        self.PSDXalpha[p][newIndices:], self.PSDXbeta[p][newIndices:] = self.interfacialComposition[p](self.T[i-1], self.particleGibbs(self.PBM[p].PSDbounds[newIndices:], self.phases[p]))
                    self.growth[p] = self._singleGrowthBinary(i, p)
                else:
                    self.growth[p] = self._singleGrowthMulti(i, p)
            
            #Set negative frequencies in PSD to 0
            #Also set any less than the minimum possible radius to be 0
            self.PBM[p].PSD[:self.RdrivingForceIndex[p]] = 0
            self.PBM[p].PSD[self.PBM[p].PSDsize < self.minRadius] = 0

    def _massBalance(self, i):
        '''
        Updates matrix composition and volume fraction of precipitates
        '''
        fBeta = np.zeros(len(self.phases))
        if self.numberOfElements == 1:
            fConc = np.zeros(len(self.phases))
        else:
            fConc = np.zeros((len(self.phases), self.numberOfElements))

        for p in range(len(self.phases)):
            #Sum up particles and average for particles
            Ntot = self.PBM[p].ZeroMoment()
            RadSum = self.PBM[p].Moment(order=1)
            ARsum = self.PBM[p].WeightedMoment(0, self.shapeFactors[p].aspectRatio(self.PBM[p].PSDsize))
            fBeta[p] = self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].ThirdMoment()

            if self.numberOfElements == 1:
                if self.infinitePrecipitateDiffusion[p]:
                    fConc[p] = self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].WeightedMoment(3, 0.5 * (self.PSDXbeta[p][:-1] + self.PSDXbeta[p][1:]))
                else:
                    y = self.VmAlpha / self.VmBeta[p] * self.GB[p].areaFactor * np.sum(self.PBM[p]._prevPSDbounds[1:]**2 * self.PBM[p]._fv[1:] * self._prevPSDXbeta[p][1:] * (self.PBM[p]._prevPSDbounds[1:] - self.PBM[p]._prevPSDbounds[:-1]))
                    fConc[p] = self.prevFConc[0,p,0] + y
                self.prevFConc[1,p,0] = copy.copy(self.prevFConc[0,p,0])
                self.prevFConc[0,p,0] = fConc[p]
            else:
                if self.infinitePrecipitateDiffusion[p]:
                    for a in range(self.numberOfElements):
                        fConc[p,a] = self.VmAlpha / self.VmBeta[p] * self.GB[p].volumeFactor * self.PBM[p].WeightedMoment(3, 0.5 * (self.PSDXbeta[p][:-1,a] + self.PSDXbeta[p][1:,a]))
                else:
                    for a in range(self.numberOfElements):
                        y = self.VmAlpha / self.VmBeta[p] * self.GB[p].areaFactor * np.sum(self.PBM[p]._prevPSDbounds[1:]**2 * self.PBM[p]._fv[1:] * self._prevPSDXbeta[p][1:,a] * (self.PBM[p]._prevPSDbounds[1:] - self.PBM[p]._prevPSDbounds[:-1]))
                        fConc[p,a] = self.prevFConc[0,p,a] + y
                self.prevFConc[1,p] = copy.copy(self.prevFConc[0,p])
                self.prevFConc[0,p] = fConc[p]

            #Average radius and precipitate density
            if Ntot > 0:
                self.avgR[p, i] = RadSum / Ntot
                self.precipitateDensity[p, i] = Ntot
                self.avgAR[p, i] = ARsum / Ntot
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
                self.xComp[i][self.xComp[i] < 0] = self.minComposition
            else:
                self.xComp[i] = np.zeros(self.numberOfElements)

    def _singleGrowthBinary(self, i, p):
        '''
        Calculates growth rate for a single phase
        This is separated from _growthRateBinary since it's used in _calculatePSD

        Matrix/precipitate composition are not calculated here since it's
        already calculated in createLookup
        '''
        growthRate = np.zeros(self.PBM[p].bins + 1)
        #If no precipitates are stable, don't calculate growth rate and set PSD to 0
        #This should represent dissolution of the precipitates
        if self.RdrivingForceIndex[p]+1 < len(self.PSDXalpha[p]):
            superSaturation = (self.xComp[i-1] - self.PSDXalpha[p]) / (self.VmAlpha * self.PSDXbeta[p] / self.VmBeta[p] - self.PSDXalpha[p])
            growthRate = self.shapeFactors[p].kineticFactor(self.PBM[p].PSDbounds) * self.Diffusivity(self.xComp[i-1], self.T[i]) * superSaturation / (self.effDiffDistance(superSaturation) * self.PBM[p].PSDbounds)
        else:
            self.PBM[p].PSD = np.zeros(self.PBM[p].bins)

        return growthRate

    
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
            growthRate.append(self._singleGrowthBinary(i, p))
            
        self.growth = growthRate

    def _singleGrowthMulti(self, i, p):
        '''
        Calculates growth rate for a single phase
        This is separated from _growthRateMulti since it's used in _calculatePSD

        This will also calculate the matrix/precipitate composition 
        for the radius in the PSD as well as equilibrium (infinite radius)
        '''
        growth, xAlpha, xBeta, xEqAlpha, xEqBeta = self.interfacialComposition[p](self.xComp[i-1], self.T[i], self.dGs[p,i-1] * self.VmBeta[p], self.PBM[p].PSDbounds, self.particleGibbs(phase=self.phases[p]))

        #If two-phase equilibrium not found, two possibilities - precipitates are unstable or equilibrium calculations didn't converge
        if growth is None:
            #If driving force is negative, then precipitates are unstable
            if self.dGs[p,i] < 0:
                #Completely reset the PBM, including bounds and number of bins
                #In case nucleation occurs again, the PBM will be at a good length scale
                self.PBM[p].reset()
                self.PSDXalpha[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.PSDXbeta[p] = np.zeros((self.PBM[p].bins + 1, self.numberOfElements))
                self.xEqAlpha[p,i] = np.zeros(self.numberOfElements)
                self.xEqBeta[p,i] = np.zeros(self.numberOfElements)
                return np.zeros(self.PBM[p].bins + 1)
            #Else, equilibrium did not converge and just use previous values
            #Only the growth rate needs to be updated, since all other terms are previous
            #Also revert the PSD in case this function was called to adjust for the new PSD bins
            else:
                self.PBM[p].revert()
                return self.growth[p]
        else:
            #Update interfacial composition for each precipitate size
            self.PSDXalpha[p] = xAlpha
            self.PSDXbeta[p] = xBeta
            self.xEqAlpha[p,i] = xEqAlpha
            self.xEqBeta[p,i] = xEqBeta

            #Add shape factor to growth rate - will need to add effective diffusion distance as well
            return self.shapeFactors[p].kineticFactor(self.PBM[p].PSDbounds) * growth
    
    def _growthRateMulti(self, i):
        '''
        Determines current growth rate of all particle size classes in a multicomponent system
        '''
        growthRate = []
        for p in range(len(self.phases)):
            growthRate.append(self._singleGrowthMulti(i, p))
        self.growth = growthRate

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
        sizeDistributionVariables = ['Size Distribution', 'Size Distribution Curve', 'Size Distribution KDE', 'Size Distribution Density']
        compositionVariables = ['Interfacial Composition Alpha', 'Interfacial Composition Beta']

        scale = []
        for p in range(len(self.phases)):
            if self.GB[p].nucleationSiteType == self.GB[p].BULK or self.GB[p].nucleationSiteType == self.GB[p].DISLOCATION:
                if radius == 'spherical':
                    scale.append(self._GBareaRemoval(p) * np.ones(len(self.PBM[p].PSDbounds)))
                else:
                    scale.append(1/self.shapeFactors[p].eqRadiusFactor(self.PBM[p].PSDbounds))
                    if radius == 'long':
                        scale.append(self.shapeFactors[p].aspectRatio(self.PBM[p].PSDbounds) / self.shapeFactors[p].eqRadiusFactor(self.PBM[p].PSDbounds))
            else:
                scale.append(self._GBareaRemoval(p) * np.ones(len(self.PBM[p].PSDbounds)))

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
                getattr(self.PBM[0], functionName)(axes, scale=scale[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    getattr(self.PBM[p], functionName)(axes, label=self.phases[p], scale=scale[p], *args, **kwargs)
                axes.legend()
            axes.set_xlabel('Radius (m)')
            axes.set_ylabel(ylabel)
            axes.set_xlim([0, np.amax([pb.max for pb in self.PBM])])
            if variable == 'Size Distribution Density':
                axes.set_ylim([0, 1.1*np.amax(np.concatenate(([np.amax(pb.PSD/(pb.PSDbounds[1:] - pb.PSDbounds[:-1])) for pb in self.PBM], [1])))])
            else:
                axes.set_ylim([0, 1.1*np.amax(np.concatenate(([np.amax(pb.PSD) for pb in self.PBM], [1])))])

        elif variable == 'Cumulative Size Distribution':
            ylabel = 'CDF'
            if len(self.phases) == 1:
                self.PBM[0].PlotCDF(axes, scale=scale[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    self.PBM[p].PlotCDF(axes, label=self.phases[p], scale=scale[p], *args, **kwargs)
                axes.legend()
            axes.set_xlabel('Radius (m)')
            axes.set_ylabel(ylabel)
            axes.set_xlim([0, np.amax([pb.max for pb in self.PBM])])

        elif variable == 'Aspect Ratio Distribution':
            if len(self.phases) == 1:
                axes.plot(self.PBM[0].PSDbounds * np.interp(self.PBM[p].PSDbounds, self.PBM[0].PSDbounds, scale[0]), self.eqAspectRatio[0], *args, **kwargs)
            else:
                for p in range(len(self.phases)):
                    axes.plot(self.PBM[p].PSDbounds * np.interp(self.PBM[p].PSDbounds, self.PBM[p].PSDbounds, scale[p]), self.eqAspectRatio[p], label=self.phases[p], *args, **kwargs)
                axes.legend()
            axes.set_xlim([0, np.amax(self.PBM[p].PSDbounds * np.interp(self.PBM[p].PSDbounds, self.PBM[p].PSDbounds, scale[p]))])
            axes.set_ylim(bottom=1)
            axes.set_xlabel('Radius (m)')
            axes.set_ylabel('Aspect ratio distribution')
            
        else:
            super().plot(axes, variable, bounds, timeUnits, radius, *args, **kwargs)

        