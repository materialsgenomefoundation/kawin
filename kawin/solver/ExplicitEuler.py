from kawin.solver.Iterator import Iterator

class ExplicitEulerIterator(Iterator):
    def __init__(self):
        super().__init__()

    def iterate(self, f, t, X_old, dt):
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
        dXdt = self._flatten(f(t, X_old))
        return self._unflatten(self._flatten(X_old)+dXdt*dt, X_old)
        