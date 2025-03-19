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
from typing import Union
import numpy as np

def arithmeticMean(Ds):
    '''
    Arithmetic mean - D_avg = 1/2*(D_i + D_j)
    '''
    return np.average(Ds, axis=0)

def geometricMean(Ds):
    '''
    Geometric mean - D_avg = sqrt(D_i * D_j)
    '''
    return np.power(np.prod(Ds, axis=0), 1/len(Ds))

def logMean(Ds):
    return np.exp(np.average(np.log(Ds), axis=0))

def harmonicMean(Ds):
    '''
    Harmonic mean - D_avg = (1/2 * (D_i^-1 + D_j^-1))^-1
    '''
    return np.power(np.sum(np.power(Ds, -1), axis=0), -1)

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

    
class ProfileBuilder:
    def __init__(self):
        self.buildSteps = []

    def addBuildStep(self, func, responseVar=0):
        self.buildSteps.append((func, np.atleast_1d(responseVar)))

class ConstantProfile:
    def __init__(self, value):
        self.value = np.atleast_1d(value)

    def __call__(self, z):
        return np.squeeze(self.value[np.newaxis,:]*np.ones((_formatSpatial(z).shape[0], self.value.shape[0])))
    
class DiracDeltaProfile:
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
    
class GaussianProfile:
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
    
class BoundedRectangleProfile:
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
    
class BoundedEllipseProfile:
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

    Attributes
    ----------
    numResponses: int
        Number of response variables
    responses: list[str]
        Names of response variables
    y: np.ndarray
        Values of response variables
    z: np.ndarray
        Values of spatial coordinates corresponding to y

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
    def computedXdt(self, pairs):
        '''
        Given list of diffusivity and response pairs, compute dX/dt from diffusion fluxes

        Parameters
        ----------
        Pairs is a list of tuples
            Tuple will contain (diffusivity, response, averaging function)
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
        Converts y [internal shape] to yFlat [N, e]

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
        Converts z [internal shape] to zFlat [N, d]

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
        Converts yFlat [N, e] to y [internal shape]

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
        Converts zFlat [N, e] to z [internal shape]

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
        self.N = Ns
        self.Ntot = np.prod(Ns)
        self.y = np.zeros((*self.N, self.numResponses))
        self.defineZCoordinates()
        
    @abstractmethod
    def defineZCoordinates(self):
        raise NotImplementedError()

    def validateCompositions(self, numElements, minComposition):
        '''
        Checks that initial composition is between [0, 1]
        '''
        ysum = np.sum(self.y, axis=self.dims)
        if np.any(ysum > 1):
            print('Compositions add up to above 1 between z = [{:.3e}, {:.3e}]'.format(np.amin(self.z[ysum>1]), np.amax(self.z[ysum>1])))
            raise Exception('Some compositions sum up to above 1')
        self.y[self.y > minComposition] = self.y[self.y > minComposition] - numElements*minComposition
        self.y[self.y < minComposition] = minComposition
    
    def flattenResponse(self, y, numResponses = None):
        '''
        Converts y [internal shape] to yFlat [N, e]
        '''
        numResponses = self.numResponses if numResponses is None else numResponses
        return np.reshape(y, (self.Ntot, numResponses))
    
    def flattenSpatial(self, z):
        '''
        Converts z [internal shape] to zFlat [N, d]
        '''
        return np.reshape(z, (self.Ntot, self.dims))
    
    def unflattenResponse(self, yFlat, numResponses = None):
        '''
        Converts yFlat [N, e] to y [internal shape]
        '''
        numResponses = self.numResponses if numResponses is None else numResponses
        return np.reshape(yFlat, (*self.N, numResponses))
    
    def unflattenSpatial(self, zFlat):
        '''
        Converts zFlat [N, e] to z [internal shape]
        '''
        return np.reshape(zFlat, (*self.N, self.dims))

    

    
    
