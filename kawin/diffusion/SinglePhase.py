import numpy as np
from kawin.thermo.Mobility import u_to_x_frac, expand_u_frac, interstitials
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh.MeshBase import DiffusionPair, arithmeticMean

class SinglePhaseModel(DiffusionModel):
    def _getPairs(self, t, xCurr):
        '''
        Compute diffusivity-response pairs

        J^n_k = -sum(D^n_jk dx_j/dz)
        dx_k/dt = -dJ^n_k/dz

        For x_k, a pair would comprise of (D^n_jk, x_j)
        '''
        # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
        u = expand_u_frac(xCurr[0], self.allElements, interstitials)
        x = u_to_x_frac(u, self.allElements, interstitials)[:,1:]
        x = self.mesh.unflattenResponse(x)
        yD, zD = self.mesh.getDiffusivityCoordinates(x)
        yR, zR = self.mesh.getResponseCoordinates(x)

        T = self.temperatureParameters(zD, t)
        numElements = self.mesh.numResponses
        N = len(yD)
        d = np.zeros(N) if numElements == 1 else np.zeros((N, numElements, numElements))
        for i in range(N):
            inter_diff = self.hashTable.retrieveFromHashTable(yD[i], T[i])
            if inter_diff is None:
                inter_diff = self.therm.getInterdiffusivity(yD[i], T[i], phase=self.phases[0])
                self.hashTable.addToHashTable(yD[i], T[i], inter_diff)
            d[i] = inter_diff

        pairs = []
        # For binary systems, we only have 1 independent component
        if numElements == 1:
            pairs.append(DiffusionPair(
                diffusivity=d[:,np.newaxis], 
                response=yR, 
                averageFunction=arithmeticMean))
        else:
            # For 2+ independent component, we want to tile x_j from (1,N) -> (e,N) -> (N,e)
            for i in range(len(self.elements)):
                pairs.append(DiffusionPair(
                    diffusivity=d[:,:,i], 
                    response=np.tile([yR[:,i]], (numElements, 1)).T, 
                    averageFunction=arithmeticMean
                    ))
                #pairs.append((d[:,:,i], np.tile([yR[:,i]], (numElements, 1)).T, arithmeticMean))
        self._currdt = 0.4 * self.mesh.dz**2 / np.amax(d) / self.mesh.dims
        return pairs
    
    def getDt(self, dXdt):
        '''
        Returns dt that was calculated from _getPairs using von-Neumann stability
        This prevents double calculation of the diffusivity just to get a time step
        '''
        return self._currdt