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
'''
import numpy as np

def arithmeticMean(D1, D2):
    '''
    Arithmetic mean - D_avg = 1/2*(D_i + D_j)
    '''
    return 1/2 * (D1 + D2)

def geometricMean(D1, D2):
    return np.sqrt(D1*D2)

def harmonicMean(D1, D2):
    return 1 / (1/D1 + 1/D2)

class BoundaryCondition:
    def adjustFluxes(self, fluxes):
        pass

    def adjustdXdt(self, dXdt):
        pass

class PeriodicBoundary(BoundaryCondition):
    '''
    This doesn't do anything, but rather is used in the Mesh to compute the fluxes accordingly
    Unfortunately, we can't just wrap the fluxes around and instead have to wrap the response variable
    around. Which means that we would have to have knowledge on how to compute the flux
    '''
    pass

class MixedBoundary(BoundaryCondition):
    '''
    Currently, left and right boundary conditions both are defined here
    We'll need a way to define boundary conditions at any edge, which will
    especially be the case if we add a 2D mesh
    '''
    NEUMANN = 0
    DIRICHLET = 1

    def __init__(self, dims):
        self.dims = dims
        self.LBCtype = MixedBoundary.NEUMANN * np.ones(dims)
        self.LBCvalue = np.zeros(dims)
        self.RBCtype = MixedBoundary.NEUMANN * np.ones(dims)
        self.RBCvalue = np.zeros(dims)

    def adjustFluxes(self, fluxes):
        for d in range(self.dims):
            if self.LBCtype[d] == MixedBoundary.NEUMANN:
                fluxes[0,d] = self.LBCvalue[d]
            if self.RBCtype[d] == MixedBoundary.NEUMANN:
                fluxes[-1,d] = self.RBCvalue[d]

    def adjustdXdt(self, dXdt):
        for d in range(self.dims):
            if self.LBCtype[d] == MixedBoundary.DIRICHLET:
                dXdt[0,d] = 0
            if self.RBCtype[d] == MixedBoundary.DIRICHLET:
                dXdt[-1,d] = 0

class MeshBase:
    def __init__(self, dims, boundaryConditions = None):
        self.dims = dims
        if boundaryConditions is None:
            boundaryConditions = MixedBoundary(self.dims)
        self.boundaryConditions = boundaryConditions
        self._dt = 0
    
    def computedXdt(self, pairs):
        '''
        Pairs is a list of tuples
            Tuple will contain (diffusivity, response, averaging function)
        '''
        raise NotImplementedError()
    
    @property
    def dt(self):
        return self._dt
    
class FiniteVolume1D(MeshBase):
    '''
    1D finite volume mesh

    Node will be a volume with thickness dz
        y - value at node center
        z - spatial coordinate at node center
        zEdge - spatial coordinate at node edge
    '''
    def __init__(self, zlim, N, dims, boundaryConditions = None):
        super().__init__(dims, boundaryConditions)
        self.zlim = zlim
        self.N = N

        self.y = np.zeros((N, dims))
        self.zEdge = np.linspace(zlim[0], zlim[1], N+1)
        self.z = 0.5*(self.zEdge[1:] + self.zEdge[:-1])
        self.dz = self.z[1] - self.z[0]

    def _computeFluxes(self, pairs):
        fluxes = np.zeros((self.N+1, self.dims))
        for p in pairs:
            D, r = p[0], p[1]
            func = p[2] if len(p) == 3 else arithmeticMean
            Dmid = func(D[:-1], D[1:])
            fluxes[1:-1] -= Dmid * (r[1:] - r[:-1]) / self.dz

            Dmax = np.amax(np.abs(Dmid))

            # TODO: I would like to have this be done in PeriodicBoundary, but not
            # too sure how, especially if we want to extend to 2D systems
            if isinstance(self.boundaryConditions, PeriodicBoundary):
                Dend = func(D[0], D[1])
                fluxes[0] -= Dend * (r[0] - r[-1]) / self.dz
                fluxes[-1] -= Dend * (r[0] - r[-1]) / self.dz
                Dmax = np.amax([Dmax, np.amax(np.abs(Dend))])
        self._dt = 0.4 * self.dz**2 / Dmax

        self.boundaryConditions.adjustFluxes(fluxes)
        return fluxes
    
class Cartesian1D(FiniteVolume1D):
    def computedXdt(self, pairs):
        fluxes = self._computeFluxes(pairs)
        dXdt = -(fluxes[1:] - fluxes[:-1]) / self.dz
        self.boundaryConditions.adjustdXdt(dXdt)
        return dXdt
    
class Cylindrical1D(FiniteVolume1D):
    def __init__(self, rlim, N, dims, boundaryConditions = None):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        super().__init__(rlim, N, dims, boundaryConditions)

    def computedXdt(self, pairs):
        fluxes = self._computeFluxes(pairs)
        fr = fluxes*self.zEdge[:,np.newaxis]
        dXdt = -(fr[1:] - fr[:-1]) / self.z[:,np.newaxis] / self.dz
        self.boundaryConditions.adjustdXdt(dXdt)
        return dXdt
    
class Spherical1D(FiniteVolume1D):
    def __init__(self, rlim, N, dims, boundaryConditions = None):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        super().__init__(rlim, N, dims, boundaryConditions)

    def computedXdt(self, pairs):
        fluxes = self._computeFluxes(pairs)
        fr = fluxes*self.zEdge[:,np.newaxis]**2
        dXdt = -(fr[1:] - fr[:-1]) / self.z[:,np.newaxis]**2 / self.dz
        self.boundaryConditions.adjustdXdt(dXdt)
        return dXdt
    
    
