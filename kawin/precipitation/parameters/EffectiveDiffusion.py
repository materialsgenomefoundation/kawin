import numpy as np
import scipy.special as ssp

class EffectiveDiffusionFunctions:
    '''
    Stores variables and functions to calculate effective diffusion distance
    '''
    def __init__(self, isEnabled=True):
        self.setupInterpolation()
        self.isEnabled = isEnabled

    def setupInterpolation(self, n = 250, lmin = -5, lmax = 1):
        '''
        Sets up interpolation points for effective diffusion distance

        Parameters
        ----------
        n : int
            Number of points to interpolate between, two endpoints will be added
            at supersaturation of 0 and 1
        lmin : float
            Minimum lambda value in log scale
        lmax : float
            Maximum lambda value in log scale
        '''
        lamInterp = np.logspace(lmin, lmax, n)
        self.ohmInterp = 2*lamInterp**2 - 2*lamInterp**3 * np.sqrt(np.pi) * np.exp(lamInterp**2) * ssp.erfc(lamInterp)
        self.effDiffInterp = self.ohmInterp / (2 * lamInterp**2)
        self.ohmInterp = np.concatenate(([0], self.ohmInterp, [1]))
        self.effDiffInterp = np.concatenate(([1], self.effDiffInterp, [0]))

    def effectiveDiffusionDistance(self, supersaturation):
        '''
        Effective diffusion distance given supersaturation
        This gives a constant to be multiplied by the particle radius
        
        The effective diffusion distance is given by
        eps = Q/2*lambda
        
        Where Q is the super saturation and lambda is given by
        Q = 2*lambda^2 - 2*lambda^3 * sqrt(pi) * exp(lambda^2) * (1 - erf(lambda))
        
        Parameters
        ----------
        supersaturation : float or array of floats
            (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition
        
        Returns
        -------
        Effective diffusion distance (eps) to be used as (eps*R) in the growth rate equation
        '''
        return np.interp(supersaturation, self.ohmInterp, self.effDiffInterp)

    def lambdaLow(self, supersaturation):
        '''
        Lambda when Q approaches 0
        This is done to prevent precision errors when multiplying exp*(1-erf)
        '''
        return np.sqrt(supersaturation / 2)

    def lambdaHigh(self, supersaturation):
        '''
        Lambda when Q approaches 1
        This is done to prevent precision errors when multiplying exp*(1-erf)
        '''
        return np.sqrt(3 / (2 * (1 - supersaturation)))

    def effectiveDiffusionDistanceApprox(self, supersaturation):
        '''
        Effective diffusion distance given supersaturation
        This gives a constant to be multiplied by the particle radius
        
        When Q approaches 1, this equation oscillates due to floating point errors
        Therefore, interpolation is performed between the two limits (Q -> 0 and Q -> 1)

        This may be removed in later versions
        
        Parameters
        ----------
        supersaturation : float or array of floats
            (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition
        
        Returns
        -------
        Effective diffusion distance (eps) to be used as (eps*R) in the growth rate equation
        '''
        #Interpolation constant
        a = 1.2

        supersaturation = np.atleast_1d(supersaturation)
        diff = np.zeros(len(supersaturation))
        indices = (supersaturation >= 0) & (supersaturation < 1)
        diff[supersaturation >= 1] = 0
        diff[supersaturation < 0] = 1
        
        lam = (1 - supersaturation[indices]**a) * self.lambdaLow(supersaturation[indices]) + (supersaturation[indices]**a) * self.lambdaHigh(supersaturation[indices])
        diff[indices] = supersaturation[indices] / (2 * lam**2)
        
        return np.squeeze(diff)
        
    def noDiffusionDistance(self, supersaturation):
        '''
        If neglecting the effective diffusion distance, return 1

        Parameters
        ----------
        supersaturation : float or array of floats
            (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition
        
        Returns
        -------
        1 or array of 1s
        '''
        supersaturation = np.atleast_1d(supersaturation)
        return np.squeeze(np.ones(supersaturation.shape, dtype=np.float64))
    
    def __call__(self, supersaturation):
        if self.isEnabled:
            return self.effectiveDiffusionDistance(supersaturation)
        else:
            return self.noDiffusionDistance(supersaturation)