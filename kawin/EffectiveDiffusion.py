import numpy as np

def lambdaLow(supersaturation):
    # Lambda when Q approaches 0
    # This is done to prevent precision errors when multiplying exp*(1-erf)
    return np.sqrt(supersaturation / 2)

def lambdaHigh(supersaturation):
    # Lambda when Q approaches 1
    # This is done to prevent precision errors when multiplying exp*(1-erf)
    return np.sqrt(3 / (2 * (1 - supersaturation)))


def effectiveDiffusionDistance(supersaturation):
    '''
    Effective diffusion distance given supersaturation
    This gives a constant to be multiplied by the particle radius
    
    The effective diffusion distance is given by
    eps = Q/2*lambda
    
    Where Q is the super saturation and lambda is given by
    Q = 2*lambda^2 - 2*lambda^3 * sqrt(pi) * exp(lambda^2) * (1 - erf(lambda))
    
    When Q approaches 1, this equation oscillates due to floating point errors
    Therefore, interpolation is performed between the two limits (Q -> 0 and Q -> 1)
    
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
    
    if hasattr(supersaturation, '__len__'):
        diff = np.zeros(len(supersaturation))
        indices = (supersaturation >= 0) & (supersaturation < 1)
        diff[supersaturation >= 1] = 0
        diff[supersaturation < 0] = 1
        
        lam = (1 - supersaturation[indices]**a) * lambdaLow(supersaturation[indices]) + (supersaturation[indices]**a) * lambdaHigh(supersaturation[indices])
        diff[indices] = supersaturation[indices] / (2 * lam**2)
        
        return diff
        
    else:
        if supersaturation < 0:
            return 1
        
        if supersaturation >= 1:
            return 0
            
        lam = (1 - supersaturation**a) * lambdaLow(supersaturation) + (supersaturation**a) * lambdaHigh(supersaturation)

        return supersaturation / (2 * lam**2)
    
def noDiffusionDistance(supersaturation):
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
    if hasattr(supersaturation, '__len__'):
        return np.ones(len(supersaturation), dtype=np.float32)
    else:
        return 1