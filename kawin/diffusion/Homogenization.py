import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.thermo.Mobility import mobility_from_composition_set, interstitials, x_to_u_frac
from kawin.diffusion.DiffusionProperties import compute_mobility, wiener_lower, wiener_upper, hashin_shtrikman_lower, hashin_shtrikman_upper, labyrinth
import copy

class HomogenizationModel(DiffusionModel):
    def __init__(self, zlim, N, elements = ['A', 'B'], phases = ['alpha'], record = True):
        super().__init__(zlim, N, elements, phases, record)

        self.mobilityFunction = wiener_upper
        self.defaultMob = 0
        self.eps = 0.05

        self.sortIndices = np.argsort(self.allElements)
        self.unsortIndices = np.argsort(self.sortIndices)
        self.labFactor = 1

    def reset(self):
        '''
        Resets model

        This also includes chemical potential and pycalphad CompositionSets for each node
        '''
        super().reset()
        self.mu = np.zeros((len(self.elements)+1, self.N))
        self.compSets = [None for _ in range(self.N)]

    def setMobilityFunction(self, function):
        '''
        Sets averaging function to use for mobility

        Default mobility value should be that a phase of unknown mobility will be ignored for average mobility calcs

        Parameters
        ----------
        function : str
            Options - 'upper wiener', 'lower wiener', 'upper hashin-shtrikman', 'lower hashin-strikman', 'labyrinth'
        '''
        #np.finfo(dtype).max - largest representable value
        #np.finfo(dtype).tiny - smallest positive usable value
        if 'upper' in function and 'wiener' in function:
            self.mobilityFunction = wiener_upper
        elif 'lower' in function and 'wiener' in function:
            self.mobilityFunction = wiener_lower
        elif 'upper' in function and 'hashin' in function:
            self.mobilityFunction = hashin_shtrikman_upper
        elif 'lower' in function and 'hashin' in function:
            self.mobilityFunction = hashin_shtrikman_lower
        elif 'lab' in function:
            self.mobilityFunction = labyrinth

    def setLabyrinthFactor(self, n):
        '''
        Labyrinth factor

        Parameters
        ----------
        n : int
            Either 1 or 2
            Note: n = 1 will the same as the weiner upper bounds
        '''
        self.labFactor = np.clip(n, 1, 2)

    def setup(self):
        '''
        Sets up model

        This also includes getting the CompositionSets for each node
        '''
        super().setup()
        self.updateCompSets(self.x)

    def updateCompSets(self, xarray):
        '''
        Updates the array of CompositionSets

        If an equilibrium calculation is already done for a given composition, 
        the CompositionSet will be taken out of the hash table

        Otherwise, a new equilibrium calculation will be performed

        Parameters
        ----------
        xarray : (e-1, N) array
            Composition for each node
            e is number of elements
            N is number of nodes

        Returns
        -------
        parray : (p, N) array
            Phase fractions for each node
            p is number of phases
        '''
        mob, phase_fracs, chemical_potentials = compute_mobility(self.therm, xarray.T, self.T, self.hashTable, self._getHash)
        self.p = phase_fracs.T
        self.mu = chemical_potentials.T
        self.mob = mob.transpose((1,2,0))
    
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

        TODO: If using RK4, I believe the phase fraction will be from the last step of the RK4 iteration. May not make sense to do that
        '''
        x = x_curr[0]
        self.T = self.Tfunc(self.z, t)
        self.updateCompSets(x)
        #self.p = self.updateCompSets(x)

        #Get average mobility between nodes
        avgMob = self.mobilityFunction(self.mob.transpose(2,0,1), self.p.T, labyrinth_factor = self.labFactor).T
        #avgMob = 0.5 * (avgMob[:,1:] + avgMob[:,:-1])
        logMob = np.log(avgMob)
        avgMob = np.exp(0.5*(logMob[:,1:] + logMob[:,:-1]))

        #Composition between nodes
        avgX = 0.5 * (x[:,1:] + x[:,:-1])
        avgX = np.concatenate(([1-np.sum(avgX, axis=0)], avgX), axis=0)

        #Chemical potential gradient
        dmudz = (self.mu[:,1:] - self.mu[:,:-1]) / self.dz

        #Composition gradient (we need to calculate gradient for reference element)
        dxdz = (x[:,1:] - x[:,:-1]) / self.dz
        dxdz = np.concatenate(([0-np.sum(dxdz, axis=0)], dxdz), axis=0)

        # J = -M * dmu/dz
        # Ideal contribution: J_id = -eps * M*R*T / x * dx/dz
        fluxes = np.zeros((len(self.elements)+1, self.N-1))
        fluxes = -avgMob * dmudz
        nonzeroComp = avgX != 0
        Tmid = (self.T[1:] + self.T[:-1]) / 2
        Tmidfull = Tmid[np.newaxis,:]
        for i in range(fluxes.shape[0]-1):
            Tmidfull = np.concatenate((Tmidfull, Tmid[np.newaxis,:]), axis=0)
        fluxes[nonzeroComp] += -self.eps * avgMob[nonzeroComp] * 8.314 * Tmidfull[nonzeroComp] * dxdz[nonzeroComp] / avgX[nonzeroComp]

        #Flux in a volume fixed frame: J_vi = J_i - x_i * sum(J_j)
        vfluxes = np.zeros((len(self.elements), self.N+1))
        vfluxes[:,1:-1] = fluxes[1:,:] - avgX[1:,:] * np.sum(fluxes, axis=0)

        #Boundary conditions
        for e in range(len(self.elements)):
            vfluxes[e,0] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else vfluxes[e,1]
            vfluxes[e,-1] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else vfluxes[e,-2]

        return vfluxes

    def getFluxes(self):
        '''
        Return fluxes and time interval for the current iteration
        '''
        vfluxes = self._getFluxes(self.t, [self.x])
        dJ = np.abs(vfluxes[:,1:] - vfluxes[:,:-1]) / self.dz
        dt = self.maxCompositionChange / np.amax(dJ[dJ!=0])
        return vfluxes, dt
    
    def getDt(self, dXdt):
        '''
        Time increment
        This is done by finding the time interval such that the composition
            change caused by the fluxes will be lower than self.maxCompositionChange
        '''
        return self.maxCompositionChange / np.amax(np.abs(dXdt[0][dXdt[0]!=0]))