import numpy as np
from kawin.GenericModel import GenericModel
from kawin.diffusion.DiffusionParameters import TemperatureParameters, DiffusionConstraints, HashTable


class DiffusionModel(GenericModel):
    '''
    Class for defining a 1-dimensional mesh

    Parameters
    ----------
    mesh: AbstractMesh
    elements : list of str
        Elements in system (first element will be assumed as the reference element)
    phases : list of str
        Number of phases in the system
    thermodynamics: GeneralThermodynamics
    constraints: DiffusionConstraints
    '''
    def __init__(self, mesh, elements, phases, 
                 thermodynamics = None,
                 temperatureParameters = None, 
                 constraints = None,
                 record = False):
        super().__init__()
        if isinstance(phases, str):
            phases = [phases]
        self.allElements, self.elements = elements, elements[1:]
        self.phases = phases
        self.therm = None

        self.mesh = mesh
        if self.mesh.numResponses != len(self.elements):
            raise ValueError("Mesh dimensions must match independent elements")

        self.temperatureParameters = temperatureParameters if temperatureParameters is not None else TemperatureParameters()
        self.constraints = constraints if constraints is not None else DiffusionConstraints()
        self.therm = thermodynamics
        self.hashTable = HashTable()

        self.reset()

        if record:
            self.enableRecording()
        else:
            self.disableRecording()
            self._recordedX = None
            self._recordedTime = None

    def reset(self):
        '''
        Resets model

        This involves clearing any caches in the Thermodynamics object and this model
        as well as resetting the composition and phase profiles
        '''
        super().reset()

        if self.therm is not None:
            self.therm.clearCache()
        
        self.isSetup = False

    def toDict(self):
        '''
        Converts diffusion data to dictionary
        '''
        data = {
            'finalTime': self.t,
            'finalX': self.x,
            'recordX': self._recordedX,
            'recordTime': self._recordedTime
        }
        return data

    def fromDict(self, data):
        '''
        Converts dictionary of data to diffusion data
        '''
        self.t = data['finalTime']
        self.x = data['finalX']
        self._recordedX = data['recordX']
        self._recordedTime = data['recordTime']
    
    def setHashSensitivity(self, s):
        '''
        If composition caching is used, this sets the composition precision (in sig figs) when creating a hash

        Ex. if s = 2, then a composition of [0.3334, 0.3333, 0.3333] will be converted to [0.33, 0.33, 0.33] when hashing
            While this can lead to faster compute times if new compositions has the same hash, this can also lower fidelity of
            the calculations since a range of compositions could correspond to the same moblity values

        Parameters
        ----------
        s : int
            Number of sig figs when creating composition hash
        '''
        self.hashTable.setHashSensitivity(s)

    def useCache(self, use):
        '''
        Whether to cache compositions

        Parameters
        ----------
        use : bool
        '''
        self.hashTable.enableCaching(use)

    def clearCache(self):
        '''
        Clears the composition cache
        '''
        self.hashTable.clearCache()

    def enableRecording(self):
        '''
        Enables recording of composition and phase
        '''
        self._record = True
        self._recordedX = np.zeros((1, len(self.elements), self.N))
        self._recordedTime = np.zeros(1)

    def disableRecording(self):
        '''
        Disables recording
        '''
        self._record = False

    def removeRecordedData(self):
        '''
        Removes recorded data
        '''
        self._recordedX = None
        self._recordedTime = None

    def record(self, time):
        '''
        Adds current mesh data to recorded arrays
        '''
        if self._record:
            if time > 0:
                self._recordedX = np.pad(self._recordedX, ((0, 1), (0, 0), (0, 0)))
                self._recordedTime = np.pad(self._recordedTime, (0, 1))

            self._recordedX[-1] = self.x
            self._recordedTime[-1] = time

    def setMeshtoRecordedTime(self, time):
        '''
        From recorded values, interpolated at time to get composition and phase fraction
        '''
        if self._record:
            if time < self._recordedTime[0]:
                print('Input time is lower than smallest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[0]))
                self.x = self._recordedX[0]
            elif time > self._recordedTime[-1]:
                print('Input time is larger than longest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[-1]))
                self.x = self._recordedX[-1]
            else:
                uind = np.argmax(self._recordedTime > time)
                lind = uind - 1

                ux, utime = self._recordedX[uind], self._recordedTime[uind]
                lx, ltime = self._recordedX[lind], self._recordedTime[lind]

                self.x = (ux - lx) * (time - ltime) / (utime - ltime) + lx

    def _getElementIndex(self, element = None):
        '''
        Gets index of element in self.elements

        Parameters
        ----------
        element : str
            Specified element, will return first element if None
        '''
        if element is None:
            return 0
        else:
            return self.elements.index(element)

    def _getPhaseIndex(self, phase = None):
        '''
        Gets index of phase in self.phases

        Parameters
        ----------
        phase : str
            Specified phase, will return first phase if None
        '''
        if phase is None:
            return 0
        else:
            return self.phases.index(phase)

    def setup(self):
        '''
        General setup function for all diffusion models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if not self.isSetup:
            if self.therm is not None:
                self.therm.clearCache()

        self.mesh.validateCompositions(len(self.allElements), self.constraints.minComposition)
        self.isSetup = True
        self.record(self.currentTime) #Record at t = 0

    def getFluxes(self, t, xCurr):
        '''
        Computes fluxes from mesh and diffusivity-response pairs
        NOTE: this directly returns the mesh outputs, so format of fluxes will correspond to the mesh
        '''
        pairs = self._getPairs(t, xCurr)
        return self.mesh.computeFluxes(pairs)

    def getdXdt(self, t, xCurr):
        '''
        Computes dXdt from mesh and diffusivity-respones pairs
        '''
        pairs = self._getPairs(t, xCurr)
        return [self.mesh.computedXdt(pairs)]
    
    def printHeader(self):
        print('Iteration\tSim Time (h)\tRun time (s)')

    def printStatus(self, iteration, modelTime, simTimeElapsed):
        # Convert time to hours
        super().printStatus(iteration, modelTime/3600, simTimeElapsed)

    def getCurrentX(self):
        return [self.mesh.y]
    
    def postProcess(self, time, x):
        '''
        Stores new x and t
        Records new values if recording is enabled
        '''
        super().postProcess(time, x)
        #self.t = time
        #self.x = x[0]
        #self.x = np.clip(self.x, self.constraints.minComposition, 1-self.constraints.minComposition)
        #self.record(self.t)
        self.mesh.y = x[0]
        self.updateCoupledModels()
        return self.getCurrentX(), False
    
    def flattenX(self, X):
        '''
        np.hstack does not flatten a 2D array, so we have to overload this function
            By itself, this doesn't actually affect the solver/iterator, but when coupled with other models,
            it becomes an issue

        This will convert the 2D array X to a 1D array by reshaping to 1D array of len(# elements * # nodes)
        '''
        return np.reshape(X[0], (np.prod(X[0].shape)))
    
    def unflattenX(self, X_flat, X_ref):
        '''
        Reshape X_flat to original shape
        '''
        return [np.reshape(X_flat, X_ref[0].shape)]
