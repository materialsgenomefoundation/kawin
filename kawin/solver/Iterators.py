'''
Built-in iterators

Currently, this is explicit-euler and 4th order runga kutta
'''

def _startIteration(t, X_old, f, dtfunc, dtmin, dtmax):
    '''
    Helper function to start the iteration by calculating dXdt at X_old and dt
        This is just to avoid some repeated code

    Parameters correspond to the same parameters as in the iterator functions

    Parameters
    ----------
    t : float
    X_old : list of arrays
    f : function
    dtfunc : function
    dtmin : float
    dtmax : float
    '''
    dXdt = f(t, X_old)
    dt = dtfunc(dXdt)
    if dt < dtmin:
        dt = dtmin
    if dt > dtmax:
        dt = dtmax
    return dt, dXdt

def ExplicitEulerIterator(f, t, X_old, dtfunc, dtmin, dtmax, correctdXdt, flattenX, unflattenX):
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
    dt, dXdt = _startIteration(t, X_old, f, dtfunc, dtmin, dtmax)
    correctdXdt(dt, X_old, dXdt)
    return unflattenX(flattenX(X_old)+flattenX(dXdt)*dt, X_old), dt

def RK4Iterator(f, t, X_old, dtfunc, dtmin, dtmax, correctdXdt, flattenX, unflattenX):
    '''
    4th order Runga Kutta iteration scheme

    Defined by:
    dXdt = f(t, X_n)
    k1 = f(t, X_n)
    k2 = f(t + dt/2, X_n + k1 * dt/2)
    k3 = f(t + dt/2, X_n + k2 * dt/2)
    k4 = f(t + dt, X_n, k3 * dt)
    X_n+1 = X_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt

    Steps:
        1. Calculate dXdt from t and X_old
        2. Calculate suitable dt
        3. For each k_i, calculate k_i, add to dXdtsum, then correct k_i to calculate k_i+1
        4. Correct dXdtsum
        5. Return X_old + dXdtsum*dt and dt

    I compared correcting k_i before added to dXdtsum (since that would be the actual values of dXdt at f(t,X) that we would use), 
    but versus summing the uncorrected k_i, then correcting the average, but it didn't seem to make a difference. I'm doing the latter
    since it'll be a bit faster since we don't have to correct k_2 and k_3 twice for dt and dt/2

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
    dt, dXdt = _startIteration(t, X_old, f, dtfunc, dtmin, dtmax)

    X_flat = flattenX(X_old)
    
    k1 = dXdt
    dXdtsum = flattenX(k1)
    correctdXdt(dt/2, X_old, k1)
    X_k1 = unflattenX(X_flat + dt/2 * flattenX(k1), X_old)

    k2 = f(t+dt/2, X_k1)
    dXdtsum += 2*flattenX(k2)
    correctdXdt(dt/2, X_old, k2)
    X_k2 = unflattenX(X_flat + dt/2 * flattenX(k2), X_old)

    k3 = f(t+dt/2, X_k2)
    dXdtsum += 2*flattenX(k3)
    correctdXdt(dt, X_old, k3)
    X_k3 = unflattenX(X_flat + dt * flattenX(k3), X_old)

    k4 = f(t+dt, X_k3)
    dXdtsum += flattenX(k4)
    dXdtsum /= 6
    dXdtsum = unflattenX(dXdtsum, X_old)
    correctdXdt(dt, X_old, dXdtsum)

    return unflattenX(X_flat + dt * flattenX(dXdtsum), X_old), dt