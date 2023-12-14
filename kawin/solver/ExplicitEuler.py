from kawin.solver.Iterator import Iterator

class ExplicitEulerIterator(Iterator):
    '''
    Explicit euler iteration scheme

    Defined by:
    dXdt = f(t, X_n)
    X_n+1 = X_n + f(t, X_n) * dt

    Steps:
        1. Calculate dXdt from t and X_old
        2. Calculate suitable dt
        3. Correct dXdt from new value of dt
        4. Return X_old + dXdt*dt and dt
    '''
    def __init__(self):
        super().__init__()

    def iterate(self, f, t, X_old, dtfunc, dtmin, dtmax, correctdXdt):
        '''
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
            Time step, important if modified from dtfunc
        '''
        dXdt = f(t, X_old)
        dt = dtfunc(dXdt)
        if dt < dtmin:
            dt = dtmin
        if dt > dtmax:
            dt = dtmax
        correctdXdt(dt, X_old, dXdt)
        return self._unflatten(self._flatten(X_old)+self._flatten(dXdt)*dt, X_old), dt
        