import numpy as np

from kawin.Constants import GAS_CONSTANT
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.thermo.Mobility import interstitials, x_to_u_frac
from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
from kawin.diffusion.mesh.MeshBase import arithmeticMean, harmonicMean

class HomogenizationModel(DiffusionModel): 
    def __init__(self, mesh, elements, phases, 
                 thermodynamics = None,
                 temperatureParameters = None, 
                 constraints = None,
                 homogenizationParameters = None,
                 record = False):
        super().__init__(mesh=mesh, elements=elements, phases=phases, 
                         thermodynamics=thermodynamics,
                         temperatureParameters=temperatureParameters,  
                         constraints=constraints,
                         record=record)
        self.homogenizationParameters = homogenizationParameters if homogenizationParameters is not None else HomogenizationParameters()
    
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
    
    def getdXdt(self, t, xCurr):
        '''
        dXdt is defined as -dJ/dz
        '''
        x = xCurr[0]
        yD, zD = self.mesh.getDiffusivityCoordinates(x)
        yR, zR = self.mesh.getResponseCoordinates(x)

        # temp is (N,e)
        tempD = self.temperatureParameters(zD, t)
        tempR = self.temperatureParameters(zR, t)
        # mob and mu are (N,e)
        mobD, muD = computeHomogenizationFunction(self.therm, yD, tempD, self.homogenizationParameters, self.hashTable)
        mobR, muR = computeHomogenizationFunction(self.therm, yR, tempR, self.homogenizationParameters, self.hashTable)

        # Full composition
        # x_full = (N,e+1), u_full = (N,e+1), u_term = (N,e+1,e+1)
        x_fullD = np.concatenate((1-np.sum(yD, axis=1)[:,np.newaxis], yD), axis=1)
        u_fullD = x_to_u_frac(x_fullD, self.allElements, interstitials)
        u_termD = (np.eye(len(self.allElements))[np.newaxis,:,:] - u_fullD[:,:,np.newaxis])

        x_fullR = np.concatenate((1-np.sum(yR, axis=1)[:,np.newaxis], yR), axis=1)
        u_fullR = x_to_u_frac(x_fullR, self.allElements, interstitials)
        u_termR = (np.eye(len(self.allElements))[np.newaxis,:,:] - u_fullR[:,:,np.newaxis])

        # mob_term and ideal_term are (N,e+1,e+1)
        # We do this for volume fixed frame of reference
        # For J^v_k = sum((\delta_jk - x_k) J_j), the dimensions are ordered: (nodes, k, j)
        mob_termD = u_termD*mobD[:,np.newaxis,:]
        ideal_termD = mob_termD * self.homogenizationParameters.eps * GAS_CONSTANT * tempD[:,np.newaxis,np.newaxis]
        pairs = []
        # Since volume fixed frame leads to 1 dependent component (which we take as the first)
        # we don't need to take the 1st row of mob_term and ideal_term
        for i in range(len(self.allElements)):
            pairs.append((mob_termD[:,1:,i], np.tile([muR[:,i]], (len(self.elements), 1)).T, arithmeticMean))
            pairs.append((ideal_termD[:,1:,i], np.tile([u_fullR[:,i]], (len(self.elements), 1)).T, harmonicMean))

        dxdt = self.mesh.computedXdt(pairs)
        return [dxdt]
