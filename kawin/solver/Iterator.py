import numpy as np
import copy

class Iterator:
    '''
    Abstract iterator to be implemented by different iteration schemes

    Available schemes:
        Explicit euler
        4th order Runga-Kutta
    '''
    def __init__(self):
        return
    
    def iterate(self, f, t, X_old, dtfunc, dtmin, dtmax, correctdXdt):
        '''
        Function to iterate X by dt

        This will be incremented by derived solvers

        Parameters
        ----------
        f : function
            dX/dt - function taking in time and X and returning dX/dt
        t : float
            Current time
        X_old : list of arrays
            X at time t
        dtfunc : function
            Takes in dXdt and return a suitable time step (float)
        dtmin : float
            Minimum time step (absolute)
        dtmax : float
            Maximum time step (absolute)
        correctdXdt : function
            Takes in dt, X and dXdt and modifies dXdt, returns nothing

        Returns
        -------
        X_new : unformatted list of floats
            New values of X in format of X_old
        dt : float
            Time step
        '''
        raise NotImplementedError()
    
    def _flatten(self, X):
        '''
        Since can be a list of arrays, we want to convert it to a 1D array to easily to operations

        TODO - this should be compatible with arrays of any dimensions. Currently, this will only work on a list of 1D arrays
            This is okay for now since none of the models uses arrays more than 1D

        Parameters
        ----------
        X : list of arrays

        Returns
        -------
        X_flat : 1D numpy array
        '''
        return np.hstack(X)
    
    def _unflatten(self, X_flat, X_ref):
        '''
        Converts flattened X array to original list of arrays

        Parameters
        ----------
        X_flat : 1D numpy array
            Flattened array
        X_ref : list of arrays
            Template to convert X_flat to

        Returns
        -------
        X_new : unflattened list in the same format as X_ref
        '''
        #Not sure if this is the most efficient way, but we can't assume how the nested list in X_ref is structured
        X_new = copy.copy(X_ref)
        n = 0
        for i in range(len(X_new)):
            if len(X_new[i].shape) == 0:
                X_new[i] = X_flat[n]
                n += 1
            else:
                arrLen = np.product(X_new[i].shape)
                X_new[i] = np.reshape(X_flat[n:n+arrLen], X_new[i].shape)
                n += arrLen
        return X_new