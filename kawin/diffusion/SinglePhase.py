import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel

class SinglePhaseModel(DiffusionModel):
    def _getFluxes(self, t, x_curr):
        '''
        Private function that gets fluxes at the boundary of each nodes given an array of compositions and current time

        Steps:
            1. Get diffusivity from cell centers using cell compositions
            2. Interpolate diffusivity to get diffusivity (D) at cell boundaries
            3. Calculate fluxes from concentration gradient (dx/dz) and interpolated diffusivity = -D * dx/dz
            4. Apply boundary conditions for fluxes at ends of mesh
                If fixed flux condition (Neumann) - then use the flux defined in the condition
                If fixed composition condition (Dirichlet) - then use nearby flux (this will keep the composition fixed after apply the fluxes)
            5. Store dt (from von Neumann analysis) for later

        Returns
        -------
        fluxes : (e-1, n+1) array of floats
            e - number of elements including reference element
            n - number of nodes
        dt : float
            Maximum calculated time interval for numerical stability
        '''
        #Calculate diffusivity at cell centers
        x = x_curr[0]
        T = self.parameters.temperature(self.z, t)
        d = np.zeros(self.N) if len(self.elements) == 1 else np.zeros((self.N, len(self.elements), len(self.elements)))
        for i in range(self.N):
            inter_diff = self.parameters.hash_table.retrieveFromHashTable(x[:,i], T[i])
            if inter_diff is None:
                inter_diff = self.therm.getInterdiffusivity(x[:,i], T[i], phase=self.phases[0])
                self.parameters.hash_table.addToHashTable(x[:,i], T[i], inter_diff)
            d[i] = inter_diff
        
        #Get diffusivity and composition gradient at cell boundaries
        dmid = (d[1:] + d[:-1]) / 2
        dxdz = (x[:,1:] - x[:,:-1]) / self.dz

        #Fluxes = -D * dx/dz
        fluxes = np.zeros((len(self.elements), self.N+1))
        if len(self.elements) == 1:
            fluxes[0,1:-1] = -dmid * dxdz
        else:
            dxdz = np.expand_dims(dxdz, axis=0)
            fluxes[:,1:-1] = -np.matmul(dmid, np.transpose(dxdz, (2,1,0)))[:,:,0].T

        #Boundary condition
        self.parameters.boundary_conditions.apply_boundary_conditions_to_fluxes(fluxes)

        #Time step from von Neumann analysis (using 0.4 instead of 0.5 to be safe)
        self._currdt = 0.4 * self.dz**2 / np.amax(np.abs(dmid))

        return fluxes

    def getFluxes(self):
        '''
        Gets fluxes at the boundary of each nodes

        This calls the private _getFluxes method with the internal current x and t

        Returns
        -------
        fluxes : (e-1, n+1) array of floats
            e - number of elements including reference element
            n - number of nodes
        dt : float
            Maximum calculated time interval for numerical stability
        '''
        fluxes = self._getFluxes(self.t, [self.x])
        dt = self._currdt
        return fluxes, dt
    
    def getDt(self, dXdt):
        '''
        Returns dt that was calculated from _getFluxes
        This prevents double calculation of the diffusivity just to get a time step
        '''
        return self._currdt