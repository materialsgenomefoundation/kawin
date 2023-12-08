from kawin.solver.Iterator import Iterator
import numpy as np

class RK4Iterator (Iterator):
    def __init__(self):
        super().__init__()

    def iterate(self, f, t, X_old, dtfunc, dtmin, dtmax, correctdXdt):
        '''
        Function to iterate X by dt

        This will be incremented by derived solvers

        Parameters
        ----------
        f : function
            dX/dt - function taking in time and returning dX/dt
        X_old : tuple of arrays
            X at time t
        t : float
            Current time
        dt : float
            Time increment

        Returns X_new
        '''
        X_flat = self._flatten(X_old)
        dXdt = f(t, X_old)
        dt = dtfunc(dXdt)
        if dt < dtmin:
            dt = dtmin
        if dt > dtmax:
            dt = dtmax
        
        k1 = dXdt
        dXdtsum = self._flatten(k1)
        correctdXdt(dt/2, X_old, k1)
        X_k1 = self._unflatten(X_flat + dt/2 * self._flatten(k1), X_old)

        k2 = f(t+dt/2, X_k1)
        dXdtsum += 2*self._flatten(k2)
        correctdXdt(dt/2, X_old, k2)
        X_k2 = self._unflatten(X_flat + dt/2 * self._flatten(k2), X_old)

        k3 = f(t+dt/2, X_k2)
        dXdtsum += 2*self._flatten(k3)
        correctdXdt(dt, X_old, k3)
        X_k3 = self._unflatten(X_flat + dt * self._flatten(k3), X_old)

        k4 = f(t+dt, X_k3)
        dXdtsum += self._flatten(k4)
        dXdtsum /= 6
        dXdtsum = self._unflatten(dXdtsum, X_old)
        correctdXdt(dt, X_old, dXdtsum)

        return self._unflatten(X_flat + dt * self._flatten(dXdtsum), X_old), dt
    
    
        