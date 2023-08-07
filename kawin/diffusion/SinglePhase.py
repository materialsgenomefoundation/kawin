import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel

class SinglePhaseModel(DiffusionModel):
    def _getFluxes(self, t, x_curr):
        '''
        Private function that gets fluxes at the boundary of each nodes given an array of compositions and current time

        When used in the iterator, calculation of the first step in the iteration will be done twice, once for dt and once for flux
            However, if using the cache, this shouldn't be much of an issue since we'll just take from the hash table

        Returns
        -------
        fluxes : (e-1, n+1) array of floats
            e - number of elements including reference element
            n - number of nodes
        dt : float
            Maximum calculated time interval for numerical stability
        '''
        x = x_curr[0]
        xMid = (x[:,1:] + x[:,:-1]) / 2
        T = self.Tfunc((self.z[1:]+self.z[:-1])/2, t)
        if len(self.elements) == 1:
            d = np.zeros(self.N-1)
        else:
            d = np.zeros((self.N-1, len(self.elements), len(self.elements)))
        if self.cache:
            for i in range(self.N-1):
                hashValue = self._getHash(xMid[:,i], T[i])
                if hashValue not in self.hashTable:
                    self.hashTable[hashValue] = self.therm.getInterdiffusivity(xMid[:,i], T[i], phase=self.phases[0])
                d[i] = self.hashTable[hashValue]
        else:
            d = self.therm.getInterdiffusivity(xMid.T, T, phase=self.phases[0])
        
        dxdz = (x[:,1:] - x[:,:-1]) / self.dz
        fluxes = np.zeros((len(self.elements), self.N+1))
        if len(self.elements) == 1:
            fluxes[0,1:-1] = -d * dxdz
        else:
            dxdz = np.expand_dims(dxdz, axis=0)
            fluxes[:,1:-1] = -np.matmul(d, np.transpose(dxdz, (2,1,0)))[:,:,0].T
        for e in range(len(self.elements)):
            fluxes[e,0] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else fluxes[e,1]
            fluxes[e,-1] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else fluxes[e,-2]

        self._currdt = 0.4 * self.dz**2 / np.amax(np.abs(d))

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
        return self._currdt
    
    def getdXdt(self, t, x):
        fluxes = self._getFluxes(t, x)
        return [-(fluxes[:,1:] - fluxes[:,:-1])/self.dz]
    
    def preProcess(self):
        return
    
    def postProcess(self, time, x):
        self.t = time
        self.x = x[0]
        self.record(self.t)
        return self.getCurrentX()