from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from kawin.diffusion.mesh.MeshBase import FiniteVolumeGrid, BoundaryCondition, arithmeticMean, _getResponseVariables, _getResponseIndex, _formatSpatial

class PeriodicBoundary1D(BoundaryCondition):
    '''
    This doesn't do anything, but rather is used in the Mesh to compute the fluxes accordingly
    Unfortunately, we can't just wrap the fluxes around and instead have to wrap the response variable
    around. Which means that we would have to have knowledge on how to compute the flux (which is done in the mesh)
    '''
    def setInitialResponse(self, y):
        pass

    def adjustFluxes(self, fluxes):
        pass

    def adjustdXdt(self, dXdt):
        pass

class MixedBoundary1D(BoundaryCondition):
    '''
    Currently, left and right boundary conditions both are defined here
    We'll need a way to define boundary conditions at any edge, which will
    especially be the case if we add a 2D mesh
    '''
    NEUMANN = 0
    DIRICHLET = 1

    def __init__(self, responses):
        self.numResponses, self.responses = _getResponseVariables(responses)
        self.LBCtype = MixedBoundary1D.NEUMANN * np.ones(self.numResponses)
        self.LBCvalue = np.zeros(self.numResponses)
        self.RBCtype = MixedBoundary1D.NEUMANN * np.ones(self.numResponses)
        self.RBCvalue = np.zeros(self.numResponses)

    def _setBC(self, responseVar, bcType, value, bcTypeArray, bcValueArray):
        '''Sets boundary condition to given side (left or right)'''
        index = _getResponseIndex(responseVar, self.responses)

        setBC = True
        if isinstance(bcType, str):
            if bcType.upper() == 'FLUX' or bcType.upper() == 'NEUMANN':
                bcTypeArray[index] = MixedBoundary1D.NEUMANN
            elif bcType.upper() == 'COMPOSITION' or bcType.upper() == 'DIRICHLET':
                bcTypeArray[index] = MixedBoundary1D.DIRICHLET
            else:
                setBC = False
        else:
            if bcType == MixedBoundary1D.NEUMANN:
                bcTypeArray[index] = MixedBoundary1D.NEUMANN
            elif bcType == MixedBoundary1D.DIRICHLET:
                bcTypeArray[index] = MixedBoundary1D.DIRICHLET
            else:
                setBC = False
        if not setBC:
            raise ValueError(f"bcType must be one of the following [flux, neumann, composition, dirichlet, MixedBoundary1D.NEUMANN, MixedBoundary1D.DIRICHLET]")
        bcValueArray[index] = value

    def setLBC(self, responseVar, bcType, value):
        '''
        Left boundary condition

        Parameters
        ----------
        responseVar: int | str
            Response variable name or index
        bcType: str
            For flux/Neumann condition, it could be [flux, neumman, MixedBoundary1D.NEUMANN]
            For composition/dirichlet condition, it could be [composition, dirichlet, MixedBoundary1D.DIRICHLET]
        value: float
            Value of boundary condition
        '''
        self._setBC(responseVar, bcType, value, self.LBCtype, self.LBCvalue)

    def setRBC(self, responseVar, bcType, value):
        '''
        Left boundary condition

        Parameters
        ----------
        responseVar: int | str
            Response variable name or index
        bcType: str
            For flux/Neumann condition, it could be [flux, neumman, MixedBoundary1D.NEUMANN]
            For composition/dirichlet condition, it could be [composition, dirichlet, MixedBoundary1D.DIRICHLET]
        value: float
            Value of boundary condition
        '''
        self._setBC(responseVar, bcType, value, self.RBCtype, self.RBCvalue)

    def setInitialResponse(self, y):
        '''
        Sets dirichlet boundary conditions onto y
        '''
        for d in range(self.numResponses):
            if self.LBCtype[d] == MixedBoundary1D.DIRICHLET:
                y[0,d] = self.LBCvalue[d]
            if self.RBCtype[d] == MixedBoundary1D.DIRICHLET:
                y[-1,d] = self.RBCvalue[d]

    def adjustFluxes(self, fluxes):
        '''
        Adjust flux values using boundary conditions

        For Neumann boundary conditions, this overrides the flux with the BC value
        '''
        for d in range(self.numResponses):
            if self.LBCtype[d] == MixedBoundary1D.NEUMANN:
                fluxes[0,d] = self.LBCvalue[d]
            if self.RBCtype[d] == MixedBoundary1D.NEUMANN:
                fluxes[-1,d] = self.RBCvalue[d]

    def adjustdXdt(self, dXdt):
        '''
        Adjust dx/dt using boundary conditions
        For Dirichlet boundary conditions, this sets dx/dt to 0
        '''
        for d in range(self.numResponses):
            if self.LBCtype[d] == MixedBoundary1D.DIRICHLET:
                dXdt[0,d] = 0
            if self.RBCtype[d] == MixedBoundary1D.DIRICHLET:
                dXdt[-1,d] = 0

class StepProfile1D:
    def __init__(self, z, leftValue, rightValue):
        self.lv = np.atleast_1d(leftValue)
        self.rv = np.atleast_1d(rightValue)
        self.z = z

    def __call__(self, z):
        z = _formatSpatial(z)[:,0]
        y = self.lv[np.newaxis,:]*np.ones((z.shape[0], self.lv.shape[0]))
        y[z >= self.z,:] = self.rv
        return np.squeeze(y)
    
class LinearProfile1D:
    def __init__(self, leftZ, leftValue, rightZ, rightValue, lowerLeftValue = None, upperRightValue = None):
        self.lv = np.atleast_1d(leftValue)
        self.lz = leftZ
        self.rv = np.atleast_1d(rightValue)
        self.rz = rightZ
        self.llv = self.lv if lowerLeftValue is None else np.atleast_1d(lowerLeftValue)
        self.urv = self.rv if upperRightValue is None else np.atleast_1d(upperRightValue)

    def __call__(self, z):
        z = _formatSpatial(z)[:,0]
        y = np.zeros((z.shape[0], self.lv.shape[0]))
        y[z <= self.lz,:] = self.llv
        midIndices = (self.lz < z) & (z <= self.rz)
        for i in range(self.lv.shape[0]):
            y[midIndices,i] = np.interp(z[midIndices], [self.lz, self.rz], [self.lv[i], self.rv[i]])
        y[self.rz < z,:] = self.urv
        return np.squeeze(y)
    
class ExperimentalProfile1D:
    def __init__(self, z, values, left=None, right=None):
        values = np.atleast_2d(values)
        if values.shape[0] == 1:
            values = values.T
        z = np.array(z)
        sortIndices = np.argsort(z)
        self.values = values[sortIndices]
        self.z = z[sortIndices]
        self.left = left if left is None else np.atleast_1d(left)
        self.right = right if right is None else np.atleast_1d(right)

    def __call__(self, z):
        z = _formatSpatial(z)[:,0]
        y = np.zeros((z.shape[0], self.values.shape[1]))
        for i in range(self.values.shape[1]):
            y[:,i] = np.interp(z, self.z, self.values[:,i], left=self.left, right=self.right)
        return np.squeeze(y)

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
        '''
        This returns the mid point of y and z (which is zEdge)
        Note: first element in ymid is second element of zEdge
        For periodic conditions, we also include y value between the first and last node
        '''
        ymid = (y[:-1] + y[1:]) / 2
        if isPeriodic:
            yend = (y[0] + y[-1]) / 2
            return np.concatenate((ymid, [yend])), zEdge[1:]
        else:
            return ymid, zEdge[1:-1]
        
    @staticmethod
    def getDMid(D, isPeriodic = False, *args, **kwargs):
        '''For periodic conditions, the last D will correspond to between the first and last node'''
        if isPeriodic:
            return D[:-1], D[-1]
        else:
            return D, None
        
class FVM1DEdge(FiniteVolumeMidPointCalculator):
    @staticmethod
    def getDiffusivityCoordinates(y, z, zEdge, *args, **kwargs):
        '''Return y and z since we compute diffusivity at these coordinates'''
        return y, z
    
    @staticmethod
    def getDMid(D, isPeriodic = False, avgFunc = arithmeticMean, *args, **kwargs):
        '''For periodic conditions, we average the first and last D'''
        D = avgFunc([D[:-1], D[1:]])
        Dend = avgFunc([D[0], D[-1]]) if isPeriodic else None
        return D, Dend

class FiniteVolume1D(FiniteVolumeGrid):
    '''
    1D finite volume mesh

    Cell will be a volume with thickness dz
        y - value at node center
        z - spatial coordinate at node center
        zEdge - spatial coordinate at node edge

    Parameters
    ----------
    zlim: list[float]
        Left and right boundary position of mesh
    N: int
        Number of cells
    responses: int | list[str]
        Response variables
    boundaryConditions: BoundaryCondition1D
        Boundary conditions on mesh
    computeMidpoint: bool
        Whether to compute diffusivity at average midpoint composition (True) or
        average diffusivities computed on nearby cell centers (False)
    '''
    def __init__(self, responses, zlim, N, computeMidpoint = False):
        super().__init__(responses, [zlim], [N], 1)
        self.computeAtMidpoint(computeMidpoint)

    def defineZCoordinates(self):
        self.zEdge = np.reshape(np.linspace(self.zlim[0][0], self.zlim[0][1], self.N[0]+1), (self.N[0]+1,1))
        self.z = 0.5*(self.zEdge[1:,:] + self.zEdge[:-1,:])
        self.dz = [self.z[1] - self.z[0]]

    def setResponseProfile(self, profileBuilder, boundaryConditions = None):
        if boundaryConditions is None:
            boundaryConditions = MixedBoundary1D(self.numResponses)
        self.boundaryConditions = boundaryConditions
        super().setResponseProfile(profileBuilder, self.boundaryConditions)

    def computeAtMidpoint(self, compute: bool):
        '''
        Sets midpoint calculator

        Parameters
        ----------
        compute: bool
            If True, diffusivity is computed at average response at node edge
            If False, diffusivity is the average diffusivity computed at the neighboring node centers
        '''
        if compute:
            self.midPointCalculator = FVM1DMidpoint()
        else:
            self.midPointCalculator = FVM1DEdge()

    def _fluxTodXdt(self, fluxes):
        '''Given fluxes, compute dx/dt. This depends on coordinate system'''
        raise NotImplementedError()

    def computeFluxes(self, pairs):
        '''Compute fluxes from (diffusivity, response) pairs on a 1D FVM mesh'''
        fluxes = np.zeros((self.N[0]+1, self.numResponses))
        isPeriodic = isinstance(self.boundaryConditions, PeriodicBoundary1D)
        for p in pairs:
            # compute fluxes for all but first and last edge (these are handled by boundary condition)
            D, r = p[0], p[1]
            avgFunc = p[2] if len(p) == 3 else arithmeticMean
            Dmid, Dend = self.midPointCalculator.getDMid(D, isPeriodic=isPeriodic, avgFunc=avgFunc)
            fluxes[1:-1] += self._diffusiveFlux(Dmid, r[1:], r[:-1], self.dz[0])

            # if periodic, then wrap first cell to the end, and first/last flux will be the same
            if isPeriodic:
                endFlux = self._diffusiveFlux(Dend, r[0], r[-1], self.dz[0])
                fluxes[0] += endFlux
                fluxes[-1] += endFlux

        return fluxes
    
    def computedXdt(self, pairs):
        '''Computes fluxes, correct fluxes for boundary conditions, compute dx/dt and correct dx/dt for boundary conditions'''
        fluxes = self.computeFluxes(pairs)
        self.boundaryConditions.adjustFluxes(fluxes)
        dXdt = self._fluxTodXdt(fluxes)
        self.boundaryConditions.adjustdXdt(dXdt)
        return dXdt
    
    def getDiffusivityCoordinates(self, y):
        '''Return y and z to compute diffusivities at'''
        isPeriodic = isinstance(self.boundaryConditions, PeriodicBoundary1D)
        return self.midPointCalculator.getDiffusivityCoordinates(y, self.z, self.zEdge, isPeriodic=isPeriodic)
    
class Cartesian1D(FiniteVolume1D):
    def _fluxTodXdt(self, fluxes):
        '''For cartesian: dx/dt = -dJ/dz'''
        return -(fluxes[1:] - fluxes[:-1]) / self.dz[0]
    
class Cylindrical1D(FiniteVolume1D):
    def __init__(self, responses, rlim, N, computeMidpoint = False):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        super().__init__(responses, rlim, N, computeMidpoint)

    def setResponseProfile(self, profileBuilder, boundaryConditions=None):
        if isinstance(boundaryConditions, PeriodicBoundary1D):
            raise ValueError('Periodic boundary conditions are not-defined on cylindrical coordinates')
        return super().setResponseProfile(profileBuilder, boundaryConditions)

    def _fluxTodXdt(self, fluxes):
        '''For cylindrical: dx/dt = -1/r dJ/dz'''
        fr = fluxes*self.zEdge
        return -(fr[1:] - fr[:-1]) / self.z / self.dz[0]
    
class Spherical1D(FiniteVolume1D):
    def __init__(self, responses, rlim, N, boundaryConditions = None, computeMidpoint = False):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        
        super().__init__(responses, rlim, N, computeMidpoint)

    def setResponseProfile(self, profileBuilder, boundaryConditions=None):
        if isinstance(boundaryConditions, PeriodicBoundary1D):
            raise ValueError('Periodic boundary conditions are not-defined on spherical coordinates')
        return super().setResponseProfile(profileBuilder, boundaryConditions)

    def _fluxTodXdt(self, fluxes):
        '''For spherical: dx/dt = -1/r^2 dJ/dz'''
        fr = fluxes*self.zEdge**2
        return -(fr[1:] - fr[:-1]) / self.z**2 / self.dz[0]