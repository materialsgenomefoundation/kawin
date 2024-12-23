import numpy as np

from kawin.Constants import GAS_CONSTANT
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.thermo.Mobility import mobility_from_composition_set, interstitials, x_to_u_frac
from kawin.diffusion.DiffusionParameters import HomogenizationParameters, computeHomogenizationFunction
import copy

class HomogenizationModel(DiffusionModel): 
    def __init__(self, zlim, N, elements, phases, 
                 thermodynamics = None,
                 temperatureParameters = None, 
                 boundaryConditions = None,
                 compositionProfile = None,
                 constraints = None,
                 homogenizationParameters = None,
                 record = True):
        super().__init__(zlim=zlim, N=N, elements=elements, phases=phases, 
                         thermodynamics=thermodynamics,
                         temperatureParameters=temperatureParameters, 
                         boundaryConditions=boundaryConditions, 
                         compositionProfile=compositionProfile, 
                         constraints=constraints, 
                         record=record)
        self.homogenizationParameters = homogenizationParameters if homogenizationParameters is not None else HomogenizationParameters()

    def setMobilityFunction(self, function):
        self.homogenizationParameters.setHomogenizationFunction(function)

    def setLabyrinthFactor(self, n):
        self.homogenizationParameters.setLabyrinthFactor(n)

    def setMobilityPostProcessFunction(self, function, functionArgs = None):
        self.homogenizationParameters.setPostProcessFunction(function, functionArgs)

    def setIdealEps(self, eps):
        self.homogenizationParameters.eps = eps
    
    def _getFluxes(self, t, x_curr):
        '''
        Return fluxes and time interval for the current iteration

        Steps:
            1. Get average mobility from homogenization function. Interpolate to get mobility (M) at cell boundaries
            2. Interpolate composition to get composition (x) at cell boundaries
            3. Calculate chemical potential gradient (dmu/dz) at cell boundaries
            4. Calculate composition gradient (dx/dz) at cell boundaries
            5. Calculate homogenization flux = -M / dmu/dz
            6. Calculate ideal contribution = -eps * M*R*T / x * dx/dz
            7. Apply boundary conditions for fluxes at ends of mesh
                If fixed flux condition (Neumann) - then use the flux defined in the condition
                If fixed composition condition (Dirichlet) - then use nearby flux (this will keep the composition fixed after apply the fluxes)
        '''
        x = x_curr[0]
        T = self.temperatureParameters(self.z, t)

        avg_mob, mu = computeHomogenizationFunction(self.therm, x.T, T, self.homogenizationParameters, self.hashTable)
        avg_mob = avg_mob.T
        mu = mu.T

        #Get average mobility between nodes
        log_mob = np.log(avg_mob)
        avg_mob = np.exp(0.5*(log_mob[:,1:] + log_mob[:,:-1]))

        #Composition between nodes
        x_full = np.concatenate(([1-np.sum(x, axis=0)], x), axis=0)
        u_frac = x_to_u_frac(x_full.T, self.allElements, interstitials).T
        avgU = 0.5 * (u_frac[:,1:] + u_frac[:,:-1])

        #Chemical potential gradient
        dmudz = (mu[:,1:] - mu[:,:-1]) / self.dz

        #Composition gradient (we need to calculate gradient for reference element)
        dudz = (u_frac[:,1:] - u_frac[:,:-1]) / self.dz

        # J = -M * dmu/dz
        # Ideal contribution: J_id = -eps * M*R*T / x * dx/dz
        fluxes = np.zeros((len(self.elements)+1, self.N-1))
        fluxes = -avg_mob * dmudz
        nonzeroComp = avgU != 0
        Tmid = (T[1:] + T[:-1]) / 2
        Tmidfull = Tmid[np.newaxis,:]
        for i in range(fluxes.shape[0]-1):
            Tmidfull = np.concatenate((Tmidfull, Tmid[np.newaxis,:]), axis=0)
        fluxes[nonzeroComp] += -self.homogenizationParameters.eps * avg_mob[nonzeroComp] * GAS_CONSTANT * Tmidfull[nonzeroComp] * dudz[nonzeroComp] / avgU[nonzeroComp]

        #Flux in a volume fixed frame: J_vi = J_i - x_i * sum(J_j)
        vfluxes = np.zeros((len(self.elements), self.N+1))
        vfluxes[:,1:-1] = fluxes[1:,:] - avgU[1:,:] * np.sum([fluxes[i] for i in range(len(self.allElements)) if self.allElements[i] not in interstitials], axis=0)

        #Boundary conditions
        self.boundaryConditions.applyBoundaryConditionsToFluxes(self.elements, vfluxes)

        return vfluxes

    def getFluxes(self):
        '''
        Return fluxes and time interval for the current iteration
        '''
        vfluxes = self._getFluxes(self.t, [self.x])
        dJ = np.abs(vfluxes[:,1:] - vfluxes[:,:-1]) / self.dz
        dt = self.constraints.maxCompositionChange / np.amax(dJ[dJ!=0])
        return vfluxes, dt
    
    def getDt(self, dXdt):
        '''
        Time increment
        This is done by finding the time interval such that the composition
            change caused by the fluxes will be lower than self.maxCompositionChange
        '''
        return self.constraints.maxCompositionChange / np.amax(np.abs(dXdt[0][dXdt[0]!=0]))