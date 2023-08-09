import numpy as np
import copy

class Iterator:
    def __init__(self):
        return
    
    def iterate(self, f, t, X_old, dtfunc, dtmin, dtmax):
        '''
        Function to iterate X by dt

        This will be incremented by derived solvers

        Parameters
        ----------
        f : function
            dX/dt - function taking in time and X and returning dX/dt
        X_old : list of arrays
            X at time t
        t : float
            Current time
        dt : float
            Time increment

        Returns X_new
        '''
        raise NotImplementedError()
    
    def _flatten(self, X):
        '''
        Since can be a list of arrays, we want to convert it to a 1D array to easily to operations

        TODO - this should be compatible with arrays of any dimensions. Currently, this will only work on a list of 1D arrays

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
        '''
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