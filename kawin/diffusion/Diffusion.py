import numpy as np
from kawin.GenericModel import GenericModel
from kawin.thermo import GeneralThermodynamics
from kawin.diffusion.mesh import AbstractMesh
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
    def __init__(self, mesh: AbstractMesh, elements: list[str], phases: list[str], 
                 thermodynamics: GeneralThermodynamics,
                 temperature: TemperatureParameters, 
                 constraints: DiffusionConstraints = None,
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

        self.temperatureParameters = TemperatureParameters(temperature)
        self.constraints = constraints if constraints is not None else DiffusionConstraints()
        self.therm = thermodynamics
        self.hashTable = HashTable()

        self.reset()
        
        self._recordBatch = 1000
        if record:
            self.enableRecording(record)
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
            'finalTime': self.currentTime,
            'finalX': self.mesh.flattenResponse(self.mesh.y),
            'recordInterval': self._record,
            'recordIndex': self._recordIndex,
            'recordX': self._recordedX,
            'recordTime': self._recordedTime
        }
        return data

    def fromDict(self, data):
        '''
        Converts dictionary of data to diffusion data
        '''
        self.currentTime = data['finalTime']
        self.mesh.y = self.mesh.unflattenResponse(data['finalX'])
        self._record = data['recordInterval']
        self._recordIndex = data['recordIndex']
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

    def enableRecording(self, record = 1):
        '''
        Enables recording of composition and phase
        '''
        if isinstance(record, bool):
            record = 1
        self._record = record
        self.resetRecordedData()

    def disableRecording(self):
        '''
        Disables recording
        '''
        self._record = -1
        self.resetRecordedData()

    def resetRecordedData(self):
        '''
        Resets arrays storing response variables over time
        '''
        flatY = self.mesh.flattenResponse(self.mesh.y)
        # Initial size will be (batch size, N, e)
        self._recordedX = np.zeros((self._recordBatch, *flatY.shape))
        self._recordedTime = np.zeros(self._recordBatch)
        self._recordIndex = 0

    def record(self, time):
        '''
        Adds current mesh data to recorded arrays
        '''
        if self._record > 0:
            # we record every N iterations (where N is self._record)
            if self._recordIndex % self._record == 0:
                row = int(self._recordIndex / self._record)

                # Pad a batch of empty rows to recordedX and recordedTime if we reached the last row
                if row >= self._recordedTime.shape[0]:
                    self._recordedX = np.pad(self._recordedX, ((0, self._recordBatch), (0, 0), (0, 0)))
                    self._recordedTime = np.pad(self._recordedTime, (0, self._recordBatch))

                # Add new data to current row
                self._recordedX[row] = self.mesh.flattenResponse(self.mesh.y)
                self._recordedTime[row] = time

            self._recordIndex += 1

    def setMeshtoRecordedTime(self, time):
        '''
        From recorded values, interpolated at time to get composition and phase fraction
        '''
        if self._record > 0:
            if time < self._recordedTime[0]:
                print('Input time is lower than smallest recorded time, setting data to t = {:.3e}'.format(self._recordedTime[0]))
                self.mesh.y = self.mesh.unflattenResponse(self._recordedX[0])
            elif time > self._recordedTime[-1]:
                print('Input time is larger than longest recorded time, setting data to t = {:.3e}'.format(self._recordedTime[-1]))
                self.mesh.y = self.mesh.unflattenResponse(self._recordedX[-1])
            else:
                uind = np.argmax(self._recordedTime > time)
                lind = uind - 1

                ux, utime = self._recordedX[uind], self._recordedTime[uind]
                lx, ltime = self._recordedX[lind], self._recordedTime[lind]

                flatY = (ux - lx) * (time - ltime) / (utime - ltime) + lx
                self.mesh.y = self.mesh.unflattenResponse(flatY)

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
        
    def _validateMeshComposition(self):
        '''
        Checks that composition along mesh will be between min/max limits (set by DiffusionConstraints.minComposition)
        Since the mesh can store y in an arbitrary shape, we use the flatten array and unflatten it to store back in the mesh
        '''
        yFlat = self.mesh.flattenResponse(self.mesh.y)
        ySum = np.sum(yFlat, axis=1)
        if np.any(ySum > 1) or np.any(ySum < 0):
            raise Exception('Some compositions sum up to below 0 or above 1')
        # We a scaled minimum composition to account for multicomponent system
        # So for a 3 component system, if 2 components at at the min comp, then the third is as 1-2*minComp
        scaledMinComp = len(self.allElements)*self.constraints.minComposition
        yFlat[yFlat > scaledMinComp] = yFlat[yFlat > scaledMinComp] - scaledMinComp
        yFlat[yFlat < scaledMinComp] = self.constraints.minComposition
        self.mesh.y = self.mesh.unflattenResponse(yFlat)

    def setup(self):
        '''
        General setup function for all diffusion models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if not self.isSetup:
            if self.therm is not None:
                self.therm.clearCache()

        self._validateMeshComposition()
        self.isSetup = True
        self.record(self.currentTime) #Record at t = 0

    def _getPairs(self, t, xCurr):
        '''
        Returns diffusivity-response pairs for diffusive fluxes
        '''
        raise NotImplementedError()

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
