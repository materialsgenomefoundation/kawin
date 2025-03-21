import numpy as np

from kawin.Constants import GAS_CONSTANT
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.thermo.Mobility import interstitials, x_to_u_frac
from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
from kawin.diffusion.mesh.MeshBase import arithmeticMean, harmonicMean

class HomogenizationModel(DiffusionModel): 
    def __init__(self, mesh, elements, phases, 
                 thermodynamics = None,
                 temperature = None, 
                 constraints = None,
                 homogenizationParameters = None,
                 record = False):
        super().__init__(mesh=mesh, elements=elements, phases=phases, 
                         thermodynamics=thermodynamics,
                         temperature=temperature,  
                         constraints=constraints,
                         record=record)
        self.homogenizationParameters = homogenizationParameters if homogenizationParameters is not None else HomogenizationParameters()
    
    def _getPairs(self, t, xCurr):
        '''
        Compute diffusivity-response pairs

        J_k = -\Gamma_k d\mu_k/dz - \eps*RT*\Gamma_k/u_k du_k/dz
        J^n_k = -sum(\delta_jk - u_k) J_j
        dx_k/dt = -dJ^n_k/dz

        For x_k, a pair would comprise of:
            (\delta_jk - u_k) \Gamma_j, \mu_k
            (\delta_jk - u_k) \eps*RT*\Gamma_k/u_k, u_k
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

        return pairs
    
    def getDt(self, dXdt):
        '''
        Time increment
        This is done by finding the time interval such that the composition
            change caused by the fluxes will be lower than self.maxCompositionChange
        '''
        return self.constraints.maxCompositionChange / np.amax(np.abs(dXdt[0][dXdt[0]!=0]))
