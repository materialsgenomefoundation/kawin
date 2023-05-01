import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel

class SinglePhaseModel(DiffusionModel):
    def getFluxes(self):
        '''
        Gets fluxes at the boundary of each nodes

        Returns
        -------
        fluxes : (e-1, n+1) array of floats
            e - number of elements including reference element
            n - number of nodes
        dt : float
            Maximum calculated time interval for numerical stability
        '''
        xMid = (self.x[:,1:] + self.x[:,:-1]) / 2
        self.T = self.Tfunc((self.z[1:]+self.z[:-1])/2, self.t)

        if len(self.elements) == 1:
            d = np.zeros(self.N-1)
        else:
            d = np.zeros((self.N-1, len(self.elements), len(self.elements)))
        if self.cache:
            for i in range(self.N-1):
                hashValue = self._getHash(xMid[:,i], self.T[i])
                if hashValue not in self.hashTable:
                    self.hashTable[hashValue] = self.therm.getInterdiffusivity(xMid[:,i], self.T[i], phase=self.phases[0])
                d[i] = self.hashTable[hashValue]
        else:
            d = self.therm.getInterdiffusivity(xMid.T, self.T, phase=self.phases[0])

        dxdz = (self.x[:,1:] - self.x[:,:-1]) / self.dz
        fluxes = np.zeros((len(self.elements), self.N+1))
        if len(self.elements) == 1:
            fluxes[0,1:-1] = -d * dxdz
        else:
            dxdz = np.expand_dims(dxdz, axis=0)
            fluxes[:,1:-1] = -np.matmul(d, np.transpose(dxdz, (2,1,0)))[:,:,0].T
        for e in range(len(self.elements)):
            fluxes[e,0] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else fluxes[e,1]
            fluxes[e,-1] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else fluxes[e,-2]

        dt = 0.4 * self.dz**2 / np.amax(np.abs(d))

        return fluxes, dt