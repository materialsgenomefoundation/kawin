import numpy as np

from kawin.Constants import GAS_CONSTANT
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.thermo.Mobility import interstitials, x_to_u_frac, u_to_x_frac, expand_u_frac
from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
from kawin.diffusion.mesh.MeshBase import DiffusionPair, arithmeticMean, harmonicMean, logMean

def _homogenizationMean(Ds):
    '''
    Average homogenization flux
    Ds should be in the shape of (m x N x y x 2)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        2 - (transformation to volume fixed frame, homogenization term (\Gamma))
            transformation is in composition, so use arithmetic average
            homogenization term is in terms of mobility, so use log average (mobility is always positive)
    '''
    v = arithmeticMean([D[...,0] for D in Ds])
    m = logMean([D[...,1] for D in Ds])
    return v * m

def _idealMean(Ds):
    '''
    Average homogenization flux
    Ds should be in the shape of (m x N x y x 4)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        4 - (transformation to volume fixed frame, temperature, homogenization term (\Gamma), inverse u term)
            transformation is in composition, so use arithmetic average
            temperature (+eps*R) is linear, so use arithmetic average
            homogenization term is in terms of mobility, so use log average
            inverse u term is inverse composition, so use harmonic mean (composition should always be > 0 based off constraints in diffusion model)
    '''
    v = arithmeticMean([D[...,0] for D in Ds])
    t = arithmeticMean([D[...,1] for D in Ds])
    m = logMean([D[...,2] for D in Ds])
    invU = harmonicMean([D[...,3] for D in Ds])
    #print(v.shape, t.shape, m.shape, invU.shape)
    return v * t * m * invU

def _atNodeProduct(Ds):
    return np.prod(Ds, axis=-1)

class HomogenizationModel(DiffusionModel): 
    def __init__(self, mesh, elements, phases, 
                 thermodynamics = None,
                 temperature = None, 
                 homogenizationParameters = None,
                 constraints = None,
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
        # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
        # TODO: this feels inefficient to convert u->x for mobility, then back to u for flux calculation
        u = expand_u_frac(xCurr[0], self.allElements, interstitials)
        x = u_to_x_frac(u, self.allElements, interstitials)[:,1:]
        x = self.mesh.unflattenResponse(x)
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
        # u_term is the (\delta_jk - u_k), which converts from a lattice fixed frame to a volume fixed frame
        u_termD = (np.eye(len(self.allElements))[np.newaxis,:,:] - u_fullD[:,:,np.newaxis])
        # When converting to a volume fixed frame, only the substitutional (or volume contributing) elements
        # contribute to the corrected flux. To account for interstitials, we replace the column
        # corresponding to the interstitial (which at this point is [u_A, u_B, 1-u_I, u_D] where I is interstital)
        # to be [0, 0, 1, 0]. So the column is 0 except for the interstitial row
        for i,e in enumerate(self.allElements):
            if e in interstitials:
                u_termD[:,:,i] = 0
                u_termD[:,i,i] = 1

        x_fullR = np.concatenate((1-np.sum(yR, axis=1)[:,np.newaxis], yR), axis=1)
        u_fullR = x_to_u_frac(x_fullR, self.allElements, interstitials)

        nEle = len(self.allElements)
        # mobility matrix is repeated among rows
        mobD_matrix = np.repeat(mobD[:,np.newaxis,:], nEle, axis=1)
        # We'll group eps*R with T here
        T_matrix = np.tile(tempD[:,np.newaxis,np.newaxis], (1, nEle, nEle)) * self.homogenizationParameters.eps * GAS_CONSTANT
        # inverse composition and response is repeated among rows
        invU_matrix = np.repeat(1/u_fullD[:,np.newaxis,:], nEle, axis=1)
        mu_matrix = np.repeat(muR[:,np.newaxis,:], nEle, axis=1)
        uR_matrix = np.repeat(u_fullR[:,np.newaxis,:], nEle, axis=1)

        pairs = []
        # Since volume fixed frame leads to 1 dependent component (which we take as the first)
        # we don't need to take the 1st row of mob_term and ideal_term
        for i in range(len(self.allElements)):
            # homogenization contribution - (vol transform * \Gamma) * dmu/dz
            pairs.append(DiffusionPair(
                diffusivity=np.transpose(np.array([u_termD[:,1:,i], mobD_matrix[:,1:,i]]), axes=(1,2,0)),
                response=mu_matrix[:,1:,i],
                averageFunction=_homogenizationMean,
                atNodeFunction=_atNodeProduct
            ))
            # ideal contribution - (vol transform * eps*R*T * \Gamma / u) * du/dz
            pairs.append(DiffusionPair(
                diffusivity=np.transpose(np.array([u_termD[:,1:,i], T_matrix[:,1:,i], mobD_matrix[:,1:,i], invU_matrix[:,1:,i]]), axes=(1,2,0)),
                response=uR_matrix[:,1:,i],
                averageFunction=_idealMean,
                atNodeFunction=_atNodeProduct
            ))
        return pairs
    
    def getDt(self, dXdt):
        '''
        Time increment
        This is done by finding the time interval such that the composition
            change caused by the fluxes will be lower than self.maxCompositionChange
        '''
        return self.constraints.maxCompositionChange / np.amax(np.abs(dXdt[0][dXdt[0]!=0]))
