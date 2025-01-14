import numpy as np

def _process_x(x, numElements):
    '''
    Processes x to always be an array for len(elements) - 1
    If x in len(elements), then we assume that the first item is the solute
    '''
    x = np.atleast_1d(x)
    if len(x) == numElements:
        x = x[1:]
    return x

def _process_xT_arrays(x, T, isBinary):
    '''
    Converts x, T to np.array with proper dimensions

    Shape of x is (N,1) if binary or (N,e-1) if multicomponent
    Shape of T is (N,)

    If len(x) == 1 and len(T) > 1, then this will repeat x to have the same length
    as T. Ex. if x is (1,3) and T is (5,), then resulting arrays will be (5,3) and (5,)
    Same for is len(T) == 1 and len(x) > 1: if x is (5,3) and T is (1,), then arrays will be (5,3) and (5,)
    '''
    x = np.atleast_2d(x)
    #For binary, make sure x is (N,1) and not (1,N)
    #If we don't check for this, two things can occur:
    #  If we never transpose, then an input of (N,) will result in (1,N)
    #  If we always transpose, then calling this function twice will transpose any (N,1) arrays to (1,N)
    if isBinary and x.shape[1] != 1:
        x = x.T
    T = np.atleast_1d(T)
    if len(x) != len(T):
        if len(x) == 1:
            x = np.repeat(x, len(T), axis=0)
        elif len(T) == 1:
            T = np.repeat(T, len(x), axis=0)
        else:
            raise ValueError(f'Length of x ({len(x)}) and T ({len(T)}) arrays are incompatible. They must be either equal, or either x or T should have length of 1.')
    return x, T

def _process_TG_arrays(T, gExtra):
    '''
    Converts T and gExtra to np.array with proper dimensions
    Both T and gExtra will be (N,)
    If len(T) or len(gExtra) is 1, then we repeat the array to be the same size as the other
    '''
    T = np.atleast_1d(T)
    gExtra = np.atleast_1d(gExtra)

    if len(T) != len(gExtra):
        if len(T) == 1:
            T = np.repeat(T, len(gExtra), axis=0)
        elif len(gExtra) == 1:
            gExtra = np.repeat(gExtra, len(T), axis=0)
        else:
            raise ValueError(f'Length of T ({len(T)}) and gExtra ({len(gExtra)}) arrays are incompatible. They must be either equal, or either T or gExtra should have length of 1.')
    return T, gExtra

def _getMatrixPhase(phases, phase = None):
    '''
    kawin assumes that the first phase will be the matrix phase
    Of course, this may not be true when we're working in models such as the 
    homogenization model, so this assumption can be overridden using the 'phase' parameter
    '''
    return phases[0] if phase is None else phase
    
def _getPrecipitatePhase(phases, phase = None):
    '''
    For a two-phase system, kawin assumes the second phase is the precipitate phase
    For multi-phase system, the 'phase' parameter is used to access different precipitate phases
    '''
    return phases[1] if phase is None else phase