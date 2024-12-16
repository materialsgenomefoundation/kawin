import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.Mesh import geometricMean

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
        T = self.Tfunc(self.z, t)
        if len(self.elements) == 1:
            d = np.zeros(self.N)
        else:
            d = np.zeros((self.N, len(self.elements), len(self.elements)))
        if self.cache:
            for i in range(self.N):
                hashValue = self._getHash(x[i], T[i])
                if hashValue not in self.hashTable:
                    self.hashTable[hashValue] = self.therm.getInterdiffusivity(x[i], T[i], phase=self.phases[0])
                d[i] = self.hashTable[hashValue]
        else:
            d = self.therm.getInterdiffusivity(x.T, T, phase=self.phases[0])
        
        #Get diffusivity and composition gradient at cell boundaries
        dmid = (d[1:] + d[:-1]) / 2
        dxdz = (x[1:] - x[:-1]) / self.dz

        #Fluxes = -D * dx/dz
        #fluxes = np.zeros((len(self.elements), self.N+1))
        fluxes = np.zeros((self.N+1, len(self.elements)))
        if len(self.elements) == 1:
            fluxes[1:-1] = -dmid[:,np.newaxis] * dxdz
        else:
            dxdz = np.expand_dims(dxdz, axis=2)
            fluxes[1:-1] = -np.matmul(dmid, dxdz)[:,:,0]

        #Boundary condition
        for e in range(len(self.elements)):
            fluxes[0,e] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else fluxes[1,e]
            fluxes[-1,e] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else fluxes[-2,e]

        #Time step from von Neumann analysis (using 0.4 instead of 0.5 to be safe)
        self._currdt = 0.4 * self.dz**2 / np.amax(np.abs(dmid))

        return fluxes
    
    # def getdXdt(self, t, x):
    #     #Calculate diffusivity at cell centers
    #     x = x[0].T
    #     T = self.Tfunc(self.z, t)
    #     if len(self.elements) == 1:
    #         d = np.zeros(self.N)
    #     else:
    #         d = np.zeros((self.N, len(self.elements), len(self.elements)))
    #     if self.cache:
    #         for i in range(self.N):
    #             hashValue = self._getHash(x[:,i], T[i])
    #             if hashValue not in self.hashTable:
    #                 self.hashTable[hashValue] = self.therm.getInterdiffusivity(x[:,i], T[i], phase=self.phases[0])
    #             d[i] = self.hashTable[hashValue]
    #     else:
    #         d = self.therm.getInterdiffusivity(x.T, T, phase=self.phases[0])

    #     dmid = geometricMean(d[1:], d[:-1])
    #     #self._currdt = 0.4 * self.mesh.dz**2 / np.amax(dmid)

    #     pairs = []
    #     if len(self.elements) == 1:
    #         pairs.append((d, self.mesh.y, geometricMean))
    #     else:
    #         for i in range(len(self.elements)):
    #             pairs.append((d[:,i,:], np.tile([self.mesh.y[:,i]], (len(self.elements), 1)).T, geometricMean))
    #     dxdt = self.mesh.computedXdt(pairs)
    #     return [dxdt]

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
        fluxes = self._getFluxes(self.currentTime, [self.x])
        dt = self._currdt
        return fluxes, dt
    
    def getDt(self, dXdt):
        '''
        Returns dt that was calculated from _getFluxes
        This prevents double calculation of the diffusivity just to get a time step
        '''
        #return self.mesh.dt
        return self._currdt