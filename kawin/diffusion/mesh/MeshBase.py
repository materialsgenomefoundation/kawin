'''
Defines a generic mesh that diffusion models can use. This will allow for
different coordinate systems or even different numerical schemes to be used
without changes to the diffusion models themselves

Basic idea:
    Fick's first law: J = -D*dy/dz
    Fick's second law: dc/dt = -dJ/dz

    The mesh will hold y (response variable) and z (spatial variable)
        A pair of y, z defines a node (e.g. y_i, z_i)
    The diffusion model will supply the diffusivity as D = f(y,z)

    In more general terms, we will define the flux as J_j = -sum(D_i * dy_i/dz)
    For single phase diffusion, this will account for the diffusion matrix 
        J_k = -sum_j(D^n_kj * dx_j/dz)
    For homogenization model, this will account for both the homogenization function and ideal entropy contribution
        J_k = -M_K * dmu_k/dz + -eps*R*T*M_k/x_k * dx_k/dz
    Thus, the diffusion model will supply a pair of D and y
        E.g. for single phase diffusion - (D^n_k1, x_1), (D^n_k2, x_2), ...
             for homogenization         - (M_k, mu_k), (eps*R*T*M_k/x_k, x_k)

    Finally, since D is computed at the node but is to represent the middle of two nodes, 
    we will have the diffusion model supply an averaging function for D:
        Arithmetic mean - D_avg = 1/2*(D_i + D_j)
        Geometric mean  - D_avg = sqrt(D_i * D_j)
        Harmonic mean   - D_avg = (1/2 * (D_i^-1 + D_j^-1))^-1

    TODO: I want an averaging function that can account for both log scale and negative numbers, which are two not so friendly terms
'''
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, Protocol
import numpy as np

DiffusionPair = namedtuple('DiffusionPair', ['diffusivity', 'response', 'averageFunction', 'atNodeFunction'], defaults=[None, None, None, None])

def noChangeAtNode(Ds):
    '''
    Default function for at node diffusivity
    '''
    return Ds

def arithmeticMean(Ds):
    '''
    Arithmetic mean - D_avg = 1/2*(D_i + D_j)
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    '''
    return np.average(Ds, axis=0)

def geometricMean(Ds):
    '''
    Geometric mean - D_avg = sqrt(D_i * D_j)
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    Note: this is only valid when D_i and D_j are the same sign
    '''
    return np.power(np.prod(Ds, axis=0), 1/len(Ds))

def logMean(Ds):
    '''
    Log mean - D_avg = exp(1/2*(log(D_i)+log(D_j)))
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    Note: this is only valid for positive D
    '''
    return np.exp(np.average(np.log(Ds), axis=0))

def harmonicMean(Ds):
    '''
    Harmonic mean - D_avg = (1/2 * (D_i^-1 + D_j^-1))^-1
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    Note: in the limit of D_i or D_j -> 0, the mean is 0. In practice, this is undefined, so we 
          convert all inf/nan to 0
    '''
    Davg = np.power(np.sum(np.power(Ds, -1), axis=0), -1)
    Davg[~np.isfinite(Davg)] = 0
    return Davg

def _getResponseVariables(responses: Union[int, list[str]]) -> tuple[int, list[str]]:
    '''
    This returns the number of response variables and names for each variable

    Parameters
    ----------
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length
    '''
    if np.issubdtype(np.array(responses).dtype, int):
        return responses, [f"R{i}" for i in range(responses)]
    # If dims is a list of str, then dims is the length of array
    if hasattr(responses, "__len__"):
        return len(responses), [str(r) for r in responses]
    else:
        raise ValueError("dims must be int or list[str]")
    
def _getResponseIndex(responseVar: Union[int, str], responses):
        '''
        Given the response variable, return the corresponding index
        '''
        numResponses = len(responses)
        # If integer, then index is just the response variable
        if np.issubdtype(np.array(responseVar).dtype, int):
            if responseVar < 0 or responseVar >= numResponses:
                raise ValueError(f"Response index must be [0, {numResponses-1}]")
            return responseVar
        # If string, then get the index in the self.responses list
        else:
            if responseVar not in responses:
                raise ValueError(f"responseVar must be one of the following: {responses}")
            else:
                return responses.index(responseVar)
            
def _formatSpatial(z):
    z = np.atleast_2d(z)
    if z.shape[0] == 1:
        return z.T
    else:
        return z
    
class BoundaryCondition(ABC):
    @abstractmethod
    def setInitialResponse(self, y):
        raise NotImplementedError()

    @abstractmethod
    def adjustFluxes(self, fluxes):
        raise NotImplementedError()

    @abstractmethod
    def adjustdXdt(self, dXdt):
        raise NotImplementedError()

# TODO: consider removing this since it doesn't really do anything other than store
#       a list that the user can create themselves

class ProfileFunction(Protocol):
    def __call__(self, z: np.ndarray) -> np.ndarray:
        '''
        Function that takes in a list of coordinates and returns compositions at
        the corresponding coordinates

        Parameters
        ----------
        z: np.ndarray
            Array of floats with shape of (N,d)
            where N is number of points and d is number of dimensions
    
        Returns
        -------
        np.ndarray of shape (N,r) where N is number of points and
        r is number of response variables

        Note that the number of response variables can be specific to the
        profile function. It does not have to correspond to the responses in
        the mesh as long as the ProfileBuilder knows what response variables
        the ProfileFunction corresponds to
        '''
        ...

class ProfileBuilder:
    '''
    Stores build steps to construct a response profile in a mesh

    Note that all steps are additive to the response profile

    TODO: consider removing this since it doesn't really do anything other
          than store a list that the user can create themselves

    Parameters
    ----------
    steps: list[tuple(callable, int|str | list[int|str])] (optional)
        List of build steps to initialize
    '''
    def __init__(self, steps = []):
        self.buildSteps = []
        for step in steps:
            self.addBuildStep(step[0], step[1])

    def addBuildStep(self, func: ProfileFunction, responseVar: int|str|list[int|str] = 0):
        '''
        Adds a build step to construct the initial profile

        Parameters
        ----------
        func: ProfileFunction
            Function that takes in a set of coordinates and returns the profile values
            at coordinates
        responseVar: int or str or list of int or str (optional)
            Response variable(s) that func will return
            This does not have to match exactly to the mesh variables, but only needs
            to be a subset
        '''
        self.buildSteps.append((func, np.atleast_1d(responseVar)))

    def clearBuildSteps(self):
        '''
        Removes all build steps
        '''
        self.buildSteps.clear()

class ConstantProfile(ProfileFunction):
    '''
    Constant value across profile
    '''
    def __init__(self, value):
        self.value = np.atleast_1d(value)

    def __call__(self, z):
        return np.squeeze(self.value[np.newaxis,:]*np.ones((_formatSpatial(z).shape[0], self.value.shape[0])))
    
class DiracDeltaProfile(ProfileFunction):
    '''
    Zero at all values except at z
    Note: if a mesh does not have a node exactly at z, it will choose the closest node
    '''
    def __init__(self, z, value):
        self.value = np.atleast_1d(value)
        self.z = np.atleast_1d(z)

    def __call__(self, z):
        z = _formatSpatial(z)
        y = np.zeros((z.shape[0], self.value.shape[0]))
        r = np.sqrt(np.sum((z-self.z[np.newaxis,:])**2, axis=1))
        nearestIndex = np.argmin(r)
        y[nearestIndex,:] = self.value
        return np.squeeze(y)
    
class GaussianProfile(ProfileFunction):
    '''
    Gaussian function centered around z with standard deviation of sigma
    Scaled to maxValue
    '''
    def __init__(self, z, sigma, maxValue):
        self.maxValue = np.atleast_1d(maxValue)
        self.z = np.atleast_1d(z)
        
        if np.isscalar(sigma):
            self.sigma = sigma*np.ones(len(self.z))
        else:
            self.sigma = np.atleast_1d(sigma)

    def __call__(self, z):
        z = _formatSpatial(z)
        r2 = (z-self.z[np.newaxis,:])**2
        y = np.exp(-np.sum(r2 / self.sigma[np.newaxis,:]**2, axis=1))
        # maximum value of y will be 1, so we can just multiply the max value
        y = y[:,np.newaxis]*self.maxValue[np.newaxis,:]
        return np.squeeze(y)
    
class BoundedRectangleProfile(ProfileFunction):
    '''
    Defines rectangle with lower and upper corners
    innerValue corresponds to value inside the rectangle while
    outerValue corresponds to value outside the rectangle (defaults to 0)
    '''
    def __init__(self, lowerZ, upperZ, innerValue, outerValue = 0):
        self.iv = np.atleast_1d(innerValue)
        if outerValue == 0:
            self.ov = np.zeros(self.iv.shape)
        else:
            self.ov = np.atleast_1d(outerValue)
        self.lz = np.atleast_1d(lowerZ)
        self.uz = np.atleast_1d(upperZ)

    def __call__(self, z):
        z = _formatSpatial(z)
        y = self.ov[np.newaxis,:]*np.ones((z.shape[0], self.ov.shape[0]))
        indices = (self.lz[0] < z[:,0]) & (z[:,0] <= self.uz[0])
        for i in range(1, len(self.lz)):
            indices &= (self.lz[i] < z[:,i]) & (z[:,i] <= self.uz[i])
        y[indices,:] = self.iv
        return np.squeeze(y)
    
class BoundedEllipseProfile(ProfileFunction):
    '''
    Defines ellipse centered at z with radii r
    innerValue corresponds to value inside the rectangle while
    outerValue corresponds to value outside the rectangle (defaults to 0)
    '''
    def __init__(self, z, r, innerValue, outerValue = 0):
        self.iv = np.atleast_1d(innerValue)
        self.ov = np.atleast_1d(outerValue)
        self.z = np.atleast_1d(z)
        if np.isscalar(r):
            self.r = r*np.ones(len(self.z))
        else:
            self.r = np.atleast_1d(r)

    def __call__(self, z):
        z = _formatSpatial(z)
        y = self.ov[np.newaxis,:]*np.ones((z.shape[0], self.ov.shape[0]))
        r2 = np.sum((z-self.z[np.newaxis,:])**2 / self.r[np.newaxis,:]**2, axis=1)
        y[r2 < 1,:] = self.iv
        return y

class AbstractMesh (ABC):
    '''
    Abstract mesh class that defines basic methods to be used in a diffusion model

    Parameters
    ----------
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length

    Attributes
    ----------
    numResponses: int
        Number of response variables
    responses: list[str]
        Names of response variables
    N: int
        Total number of nodes
    dims: int
        Number of dimensions in z
    y: np.ndarray
        Initial values of response variables
        This is intended not to change in a model. Time evolution
        of response variables should be stored in MeshData
    z: np.ndarray
        Values of spatial coordinates corresponding to y
    dz: float
        In general, the smallest distance between two nodes

    Default assumptions in this mesh:
        y has shape [N,e] - e is number of responses
        z has shape [N,d] - d is number of dimensiosn
    '''
    def __init__(self, responses):
        self.numResponses, self.responses = _getResponseVariables(responses)
        
        self.N = None
        self.dims = None
        self.y = None
        self.z = None
        self.dz = None
    
    def setResponseProfile(self, profileBuilder: ProfileBuilder, boundaryConditions: BoundaryCondition = None):
        '''
        Creates initial profile on mesh based off a series of build steps
        '''
        yFlat = np.zeros(self.flattenResponse(self.y).shape)
        flatZ = self.flattenSpatial(self.z)
        for step, responseVars in profileBuilder.buildSteps:
            yStep = np.atleast_2d(step(flatZ))
            if yStep.shape[0] == 1:
                yStep = yStep.T

            for i, rv in enumerate(responseVars):
                index = _getResponseIndex(rv, self.responses)
                yFlat[:,index] += yStep[:,i]

        self.y = self.unflattenResponse(yFlat)
        if boundaryConditions is not None:
            boundaryConditions.setInitialResponse(self.y)

    @abstractmethod
    def computeFluxes(self, pairs):
        raise NotImplementedError()
    
    @abstractmethod
    def computedXdt(self, pairs: list[DiffusionPair]):
        '''
        Given list of diffusivity and response pairs, compute dX/dt from diffusion fluxes

        Parameters
        ----------
        pairs: list[DiffusionPair]
            Tuple will contain (diffusivity, response, averaging function, at node function)
            Note that the diffusivity and response will be in the form of [N,e] (corresponding
            to getDiffusivityCoordinates and getResponseCoordinates), which may not be the 
            shape of self.y. This function is responsible for accounting for this difference
            and the shape of dxdt and y should be the same

        Returns
        -------
        dXdt: np.ndarray
            Must be same shape as y
        '''
        raise NotImplementedError()
    
    def flattenResponse(self, y, numResponses = None):
        '''
        Converts y [internal shape, ...] to yFlat [N, e, ...]

        Parameters
        ----------
        y: np.ndarray
            Input y response in internal mesh shape with 1 dimension corresponding to responses
        numResponses: int (optional)
            Number of responses in y, defaults to total number of responses in mesh

        Returns
        -------
        yFlat: np.ndarray
            Shape of [N, e] with N being the total number of nodes and e being the number of responses
        '''
        return y
    
    def flattenSpatial(self, z):
        '''
        Converts z [internal shape, ...] to zFlat [N, d, ...]

        Parameters
        ----------
        z: np.ndarray
            Input z response in internal mesh shape with 1 dimension 
            corresponding to the number of spatial dimensions

        Returns
        -------
        zFlat: np.ndarray
            Shape of [N, d] with N being the total number of nodes and d being the number of spatial dimensions
        '''
        return z
    
    def unflattenResponse(self, yFlat, numResponses = None):
        '''
        Converts yFlat [N, e, ...] to y [internal shape, ...]

        Parameters
        ----------
        yFlat: np.ndarray
            Shape of [N, e] with N being the total number of nodes and e being the number of responses
        numResponses: int (optional)
            Number of responses in y, defaults to total number of responses in mesh

        Returns
        -------
        y: np.ndarray
            Input y response in internal mesh shape with 1 dimension corresponding to responses
        '''
        return yFlat
    
    def unflattenSpatial(self, zFlat):
        '''
        Converts zFlat [N, d, ...] to z [internal shape, ...]

        Parameters
        ----------
        zFlat: np.ndarray
            Shape of [N, d] with N being the total number of nodes and d being the number of spatial dimensions

        Returns
        -------
        z: np.ndarray
            Input z response in internal mesh shape with 1 dimension 
            corresponding to the number of spatial dimensions
        '''
        return zFlat

    def getResponseCoordinates(self, y):
        '''Returns y as [N,e] and z as [N,D] arrays to compute response terms'''
        return self.flattenResponse(y), self.flattenSpatial(self.z)
    
    def getDiffusivityCoordinates(self, y):
        '''
        Returns y as [N,e] and z as [N,D] arrays to compute diffusivity terms
        We decouple this from the response coordinates to allow for different ways
        of computing D^(1/2)
            a) By computing D^0 and D^1 and averaging
            b) By computing D^(1/2) at (y^1/2, z^1/2)
        By default, this will use option a since its an easier implementation, but the user can override this to use option b
        '''
        return self.getResponseCoordinates(y)
    
    def _diffusiveFlux(self, D, rHigh, rLow, dz):
        '''Flux = -D (r1 - r0) / dz'''
        return -D * (rHigh - rLow) / dz
    
class FiniteVolumeGrid(AbstractMesh):
    '''
    Functions for the finite volume method on a rectangular grid
    '''
    def __init__(self, responses, zlims, Ns, dims):
        super().__init__(responses)
        self.dims = dims
        self.zlim = zlims
        self.Ns = Ns
        self.N = np.prod(Ns)
        self.y = np.zeros((*self.Ns, self.numResponses))
        self.defineZCoordinates()
        
    @abstractmethod
    def defineZCoordinates(self):
        raise NotImplementedError()
    
    def flattenResponse(self, y, numResponses = None):
        '''
        Converts y [n x m x ..., e, ...] to yFlat [N, e, ...]
        '''
        numResponses = self.numResponses if numResponses is None else numResponses
        shape = y.shape     # shape is (n,m,...,e,...)
        return np.reshape(y, (self.N, numResponses, *shape[len(self.Ns)+1:]))
    
    def flattenSpatial(self, z):
        '''
        Converts z [n x m x ..., d, ...] to zFlat [N, d, ...]
        '''
        shape = z.shape     # shape is (n,m,...,e,...)
        return np.reshape(z, (self.N, self.dims, *shape[len(self.Ns)+1:]))
    
    def unflattenResponse(self, yFlat, numResponses = None):
        '''
        Converts yFlat [N, e, ...] to y [n x m x ..., e, ...]
        '''
        numResponses = self.numResponses if numResponses is None else numResponses
        shape = yFlat.shape     # shape is (N,e,...)
        return np.reshape(yFlat, (*self.Ns, numResponses, *shape[2:]))
    
    def unflattenSpatial(self, zFlat):
        '''
        Converts zFlat [N, e, ...] to z [n x m x ..., e, ...]
        '''
        shape = zFlat.shape     # shape is (N,e,...)
        return np.reshape(zFlat, (*self.Ns, self.dims, *shape[2:]))
    
class MeshData:
    '''
    Stores time and response variables for a mesh

    Parameters
    ----------
    mesh : AbstractMesh
        Mesh to take dimensions of response variables from
    record : bool | int
        Record interval or whether to record
        If False, only the current state of the mesh will be stored
        If int, then every n iterations will be stored
    '''
    def __init__(self, mesh: AbstractMesh, record: bool | int = False):
        if isinstance(record, bool):
            if record:
                self.recordInterval = 1
            else:
                self.recordInterval = -1
        else:
            self.recordInterval = record

        self.batchSize = 1000
        self.yShape = mesh.flattenResponse(mesh.y).shape
        self.reset()

    def reset(self):
        '''
        Resets arrays
        '''
        self._y = np.zeros((self.batchSize, *self.yShape))
        self._time = np.zeros(self.batchSize)
        self.currentIndex = 0
        self.currentY = self._y[0]
        self.currentTime = self._time[0]
        self.N = 0
        
    def record(self, time, y, force = False):
        '''
        Stores current state of time and response variables
        Response variables should be in the flattened state
            This is the same format as used in GenericModel
            But for the mesh, we must call flattenResponse
        '''
        if self.recordInterval > 0:
            if self.currentIndex % self.recordInterval == 0 or force:
                self.N = int(self.currentIndex / self.recordInterval)

                # Path a batch of empty rows to y and time if we reached the last row
                if self.N >= self._time.shape[0]:
                    self._y = np.pad(self._y, ((0, self.batchSize), (0, 0), (0, 0)))
                    self._time = np.pad(self._time, (0, self.batchSize))

                self._y[self.N] = y
                self._time[self.N] = time

            self.currentIndex += 1
        else:
            self._y[self.N] = y
            self._time[self.N] = time

        self.currentY = y
        self.currentTime = time

    def setResponseAtN(self, y, N = -1):
        '''
        Sets response profile at index N
        If N is not supplied, then this sets at the current index
        '''
        if N < 0:
            N = self.N
        self._y[N] = y

    def finalize(self):
        '''
        Removes extra padding
        '''
        self.record(self.currentTime, self.currentY, force=True)
        self._y = self._y[:self.N+1]
        self._time = self._time[:self.N+1]

    def y(self, time = None):
        '''
        Returns reponse variable at time

        If recording is disabled, then this will return the current state
        '''
        # If time is not supplied, then return current state of response
        if time is None:
            return self._y[self.N]
        
        # If time is supplied, then interpolate if recording, else return current state
        if self.recordInterval > 0:
            if time < self._time[0]:
                print('Input time is lower than smallest recorded time, returning data at t = {:.3e}'.format(self._time[0]))
                return self._y[0]
            elif time > self._time[-1]:
                print('Input time is larger than longest recorded time, return data at t = {:.3e}'.format(self._time[-1]))
                return self._y[-1]
            else:
                uind = np.argmax(self._time > time)
                lind = uind - 1

                ux, utime = self._y[uind], self._time[uind]
                lx, ltime = self._y[lind], self._time[lind]

                flatY = (ux - lx) * (time - ltime) / (utime - ltime) + lx
                return flatY
        else:
            print('Recording is disabled. Returning current state.')
            return self._y[0]

    
    
    
