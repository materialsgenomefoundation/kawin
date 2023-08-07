from kawin.solver.Iterator import Iterator

class ExplicitEulerIterator(Iterator):
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
        dXdt = f(t, X_old)
        dt = dtfunc(dXdt)
        if dt < dtmin:
            dt = dtmin
        if dt > dtmax:
            dt = dtmax
        return self._unflatten(self._flatten(X_old)+self._flatten(dXdt)*dt, X_old), dt
        