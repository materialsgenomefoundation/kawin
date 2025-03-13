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

class BoundaryCondition:
    def adjustFluxes(self, fluxes):
        pass

    def adjustdXdt(self, dXdt):
        pass

class PeriodicBoundary1D(BoundaryCondition):
    '''
    This doesn't do anything, but rather is used in the Mesh to compute the fluxes accordingly
    Unfortunately, we can't just wrap the fluxes around and instead have to wrap the response variable
    around. Which means that we would have to have knowledge on how to compute the flux
    '''
    pass

class MixedBoundary1D(BoundaryCondition):
    '''
    Currently, left and right boundary conditions both are defined here
    We'll need a way to define boundary conditions at any edge, which will
    especially be the case if we add a 2D mesh
    '''
    NEUMANN = 0
    DIRICHLET = 1

    def __init__(self, dims):
        self.dims = dims
        self.LBCtype = MixedBoundary1D.NEUMANN * np.ones(dims)
        self.LBCvalue = np.zeros(dims)
        self.RBCtype = MixedBoundary1D.NEUMANN * np.ones(dims)
        self.RBCvalue = np.zeros(dims)

    def adjustFluxes(self, fluxes):
        for d in range(self.dims):
            if self.LBCtype[d] == MixedBoundary1D.NEUMANN:
                fluxes[0,d] = self.LBCvalue[d]
            if self.RBCtype[d] == MixedBoundary1D.NEUMANN:
                fluxes[-1,d] = self.RBCvalue[d]

    def adjustdXdt(self, dXdt):
        for d in range(self.dims):
            if self.LBCtype[d] == MixedBoundary1D.DIRICHLET:
                dXdt[0,d] = 0
            if self.RBCtype[d] == MixedBoundary1D.DIRICHLET:
                dXdt[-1,d] = 0

class MeshBase (ABC):
    def __init__(self, dims):
        self.dims = dims
        self._dt = 0
    
    @abstractmethod
    def computedXdt(self, pairs):
        '''
        Pairs is a list of tuples
            Tuple will contain (diffusivity, response, averaging function)
            Note that the diffusivity and response will be in the form of [N,e] (corresponding
            to getDiffusivityCoordinates and getResponseCoordinates), which may not be the 
            shape of self.y. This function is responsible for accounting for this difference
            and the shape of dxdt and y should be the same
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def validateCompositions(self, minComposition):
        '''
        Checks that initial composition is between [0, 1]
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def getDiffusivityCoordinates(self, y):
        '''
        Returns y as [N,e] and z as [N,D] arrays to compute diffusivity terms
        We decouple this from the response coordinates to allow for different ways
        of computing D^(1/2)
            - By computing D^0 and D^1 and averaging
            - By computing D^(1/2) at (y^1/2, z^1/2)
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def getResponseCoordinates(self, y):
        '''
        Returns y as [N,e] and z as [N,D] arrays to compute response terms
        '''
        raise NotImplementedError()
    
    @property
    def dt(self):
        '''
        Returns a good time step for iteration
        '''
        return self._dt
    

    
    
