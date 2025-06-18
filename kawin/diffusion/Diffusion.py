import numpy as np
from kawin.GenericModel import GenericModel
from kawin.thermo import GeneralThermodynamics
from kawin.thermo.Mobility import x_to_u_frac, expand_x_frac, u_to_x_frac, expand_u_frac, interstitials
from kawin.diffusion.mesh import AbstractMesh, MeshData
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

        self.data = MeshData(self.mesh, record)
        self.reset()

        self._validateMeshComposition()

    def reset(self):
        '''
        Resets model

        This involves clearing any caches in the Thermodynamics object and this model
        as well as resetting the composition and phase profiles
        '''
        super().reset()
        if self.therm is not None:
            self.therm.clearCache()

        self.data.reset()
        self.data.record(0, self.mesh.flattenResponse(self.mesh.y))
        
        self.isSetup = False

    def toDict(self):
        '''
        Converts diffusion data to dictionary
        '''
        data = {
            'x': self.data._y,
            'time': self.data._time,
            'interval': self.data.recordInterval,
            'index': self.data.N,
        }
        return data

    def fromDict(self, data):
        '''
        Converts dictionary of data to diffusion data
        '''
        self.data.recordInterval = data['interval']
        self.data.N = data['index']
        self.data._y = data['x']
        self.data._time = data['time']
        self.data.currentY = self.data._y[-1]
        self.data.currentTime = self.data._time[-1]
        self.currentTime = self.data.currentTime
    
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

        # Convert composition to u-fraction and store into mesh
        yFull = expand_x_frac(yFlat)
        uFull = x_to_u_frac(yFull, self.allElements, interstitials, return_usum=False)
        self.data.setResponseAtN(uFull[:,1:])
        self.data.currentY = uFull[:,1:]

    def setup(self):
        '''
        General setup function for all diffusion models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if not self.isSetup:
            if self.therm is not None:
                self.therm.clearCache()

        self.isSetup = True

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
        return [self.mesh.flattenResponse(self.mesh.computedXdt(pairs))]
    
    def printHeader(self):
        print('Iteration\tSim Time (h)\tRun time (s)')

    def printStatus(self, iteration, modelTime, simTimeElapsed):
        # Convert time to hours
        super().printStatus(iteration, modelTime/3600, simTimeElapsed)

    def getCurrentX(self):
        # This is a little inefficient, but the model should take in a shape of (N,e) regardless of the mesh
        return [self.data.currentY]
    
    def postProcess(self, time, x):
        '''
        Stores new x and t
        Records new values if recording is enabled
        '''
        super().postProcess(time, x)
        # Bound u-fraction to not be invalid values
        #   For substitutional - u-fraction = (0, 1)
        #   For interstitial   - u-fraction = (0, inf)
        for i, e in enumerate(self.elements):
            if e not in interstitials:
                x[0][:,i] = np.clip(x[0][:,i], self.constraints.minComposition, 1-self.constraints.minComposition)
            else:
                x[0][:,i] = np.clip(x[0][:,i], self.constraints.minComposition, None)
        self.data.record(time, x[0])
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
    
    def postSolve(self):
        self.data.finalize()

    def getCompositions(self, time = -1):
        '''
        Returns composition of nodes in mesh as (N,e)
        Since the diffusion model stores everything in terms of u-fraction
        this is a useful way to get compositions back
        We return composition in (N,e), but the mesh can be used to convert
        this shape back to the internal mesh shape

        Returns
        -------
        composition: np.ndarray (N,e)
            N is number of nodes in mesh and e is the full list of elements (including dependent)
        '''
        u_flat = self.data.y(time)
        u_ext = expand_u_frac(u_flat, self.allElements, interstitials)
        return u_to_x_frac(u_ext, self.allElements, interstitials)