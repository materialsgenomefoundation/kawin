from kawin.solver.Iterator import Iterator

class RK4Iterator (Iterator):
    def __init__(self):
        super().__init__()

    def iterate(self, f, t, X_old, dtfunc, dtmin, dtmax):
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
        k1 = self._flatten(dXdt)
        k2 = self._flatten(f(t+dt/2, self._unflatten(X_flat + dt*k1/2, X_old)))
        k3 = self._flatten(f(t+dt/2, self._unflatten(X_flat + dt*k2/2, X_old)))
        k4 = self._flatten(f(t+dt, self._unflatten(X_flat + dt*k3, X_old)))
        return self._unflatten(X_flat + dt/6 * (k1 + 2*k2 + 2*k3 + k4), X_old), dt
    
    
        