from abc import ABC, abstractmethod
import numpy as np
from kawin.diffusion.mesh.MeshBase import MeshBase, arithmeticMean, MixedBoundary1D, PeriodicBoundary1D

class FiniteVolumeMidPointCalculator(ABC):
    '''
    In the finite volume method, we need to compute diffusivity at the
    edges of out nodes, but the response variables are always at the cell centers

    This class is intended to provide functionality of how to we approach computing
    diffusivities at the edges
    '''
    @staticmethod
    @abstractmethod
    def getDiffusivityCoordinates(self, *args, **kwargs):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def getDMid(self, *args, **kwargs):
        raise NotImplementedError()

class FVM1DMidpoint(FiniteVolumeMidPointCalculator):
    @staticmethod
    def getDiffusivityCoordinates(y, z, zEdge, isPeriodic = False, *args, **kwargs):
        # If computing diffusivity at the midpoint, then take midpoint of y
        # zmid will be zEdge starting at the first index (this corresponds to ymid)
        ymid = (y[:-1] + y[1:]) / 2
        if isPeriodic:
            yend = (y[0] + y[-1]) / 2
            return np.concatenate((ymid, [yend])), zEdge[1:]
        else:
            return ymid, zEdge[1:-1]
        
    @staticmethod
    def getDMid(D, isPeriodic = False, *args, **kwargs):
        if isPeriodic:
            return D[:-1], D[-1]
        else:
            return D, None
        
class FVM1DEdge(FiniteVolumeMidPointCalculator):
    @staticmethod
    def getDiffusivityCoordinates(y, z, zEdge, *args, **kwargs):
        return y, z
    
    @staticmethod
    def getDMid(D, isPeriodic = False, avgFunc = arithmeticMean, *args, **kwargs):
        D = avgFunc([D[:-1], D[1:]])
        Dend = avgFunc([D[0], D[1]]) if isPeriodic else None
        return D, Dend

class FiniteVolume1D(MeshBase):
    '''
    1D finite volume mesh

    Node will be a volume with thickness dz
        y - value at node center
        z - spatial coordinate at node center
        zEdge - spatial coordinate at node edge

    TODO: clean up branch-y code for periodic boundary conditions and computeMidpoint option
    '''
    def __init__(self, zlim, N, dims, boundaryConditions = None, computeMidpoint = False):
        super().__init__(dims)
        if boundaryConditions is None:
            boundaryConditions = MixedBoundary1D(self.dims)
        self.boundaryConditions = boundaryConditions

        self.zlim = zlim
        self.N = N

        self.y = np.zeros((N, dims))
        self.zEdge = np.linspace(zlim[0], zlim[1], N+1)
        self.z = 0.5*(self.zEdge[1:] + self.zEdge[:-1])
        self.dz = self.z[1] - self.z[0]
        self.computeAtMidpoint(computeMidpoint)

    def computeAtMidpoint(self, compute):
        if compute:
            self.midPointCalculator = FVM1DMidpoint()
        else:
            self.midPointCalculator = FVM1DEdge()

    def _fluxTodXdt(self, fluxes):
        raise NotImplementedError()
        
    def _fluxCalculation(self, D, rHigh, rLow):
        return -D * (rHigh - rLow) / self.dz

    def _computeFluxes(self, pairs):
        fluxes = np.zeros((self.N+1, self.dims))
        Dmax = 0
        isPeriodic = isinstance(self.boundaryConditions, PeriodicBoundary1D)
        for p in pairs:
            D, r = p[0], p[1]
            avgFunc = p[2] if len(p) == 3 else arithmeticMean
            Dmid, Dend = self.midPointCalculator.getDMid(D, isPeriodic=isPeriodic, avgFunc=avgFunc)
            fluxes[1:-1] += self._fluxCalculation(Dmid, r[1:], r[:-1])

            Dmax = np.amax([Dmax, np.amax(np.abs(Dmid))])

            if isPeriodic:
                endFlux = self._fluxCalculation(Dend, r[0], r[-1])
                fluxes[0] += endFlux
                fluxes[-1] += endFlux
                Dmax = np.amax([Dmax, np.amax(np.abs(Dend))])
        self._dt = 0.4 * self.dz**2 / Dmax

        return fluxes
    
    def computedXdt(self, pairs):
        fluxes = self._computeFluxes(pairs)
        self.boundaryConditions.adjustFluxes(fluxes)
        dXdt = self._fluxTodXdt(fluxes)
        self.boundaryConditions.adjustdXdt(dXdt)
        return dXdt
    
    def validateCompositions(self, numElements, minComposition):
        '''
        Note: this is only for validating compositions, if we're using this mesh for other variables, then this is not needed
        '''
        ysum = np.sum(self.y, axis=1)
        if any(ysum > 1):
            print('Compositions add up to above 1 between z = [{:.3e}, {:.3e}]'.format(np.amin(self.z[ysum>1]), np.amax(self.z[ysum>1])))
            raise Exception('Some compositions sum up to above 1')
        self.y[self.y > minComposition] = self.y[self.y > minComposition] - numElements*minComposition
        self.y[self.y < minComposition] = minComposition
    
    def getDiffusivityCoordinates(self, y):
        isPeriodic = isinstance(self.boundaryConditions, PeriodicBoundary1D)
        return self.midPointCalculator.getDiffusivityCoordinates(y, self.z, self.zEdge, isPeriodic=isPeriodic)
    
    def getResponseCoordinates(self, y):
        return y, self.z
    
class Cartesian1D(FiniteVolume1D):
    def _fluxTodXdt(self, fluxes):
        return -(fluxes[1:] - fluxes[:-1]) / self.dz
    
class Cylindrical1D(FiniteVolume1D):
    def __init__(self, rlim, N, dims, boundaryConditions = None, computeMidpoint = False):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        if isinstance(boundaryConditions, PeriodicBoundary1D):
            raise ValueError('Periodic boundary conditions are not-defined on cylindrical coordinates')
        
        super().__init__(rlim, N, dims, boundaryConditions, computeMidpoint)

    def _fluxTodXdt(self, fluxes):
        fr = fluxes*self.zEdge[:,np.newaxis]
        return -(fr[1:] - fr[:-1]) / self.z[:,np.newaxis] / self.dz
    
class Spherical1D(FiniteVolume1D):
    def __init__(self, rlim, N, dims, boundaryConditions = None, computeMidpoint = False):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        if isinstance(boundaryConditions, PeriodicBoundary1D):
            raise ValueError('Periodic boundary conditions are not-defined on spherical coordinates')
        
        super().__init__(rlim, N, dims, boundaryConditions, computeMidpoint)

    def _fluxTodXdt(self, fluxes):
        fr = fluxes*self.zEdge[:,np.newaxis]**2
        return -(fr[1:] - fr[:-1]) / self.z[:,np.newaxis]**2 / self.dz