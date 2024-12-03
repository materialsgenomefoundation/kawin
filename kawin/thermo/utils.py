import numpy as np

def _process_xT_arrays(x, T, isBinary):
    '''
    Converts x, T to np.array

    Shape of x is (N,1) if binary or (N,e-1) if multicomponent
    Shape of T is (N,)

    If len(x) == 1 and len(T) > 1, then this will repeat x to have the same length
    as T. Ex. if x is (1,3) and T is (5,), then resulting arrays will be (5,3) and (5,)
    Same for is len(T) == 1 and len(x) > 1: if x is (5,3) and T is (1,), then arrays will be (5,3) and (5,)
    '''
    #print(x)
    x = np.atleast_2d(x)
    if isBinary:
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
        return phases[0] if phase is None else phase
    
def _getPrecipitatePhase(phases, phase = None):
    return phases[1] if phase is None else phase