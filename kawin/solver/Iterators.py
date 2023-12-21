'''
Built-in iterators

Currently, this is explicit-euler and 4th order runga kutta
''' 
def ExplicitEulerIterator(f, t, X_old, updateX):
    '''
    Explicit euler iteration scheme

    Defined by:
        dXdt = f(t, X_n)
        X_n+1 = X_n + f(t, X_n) * dt

    Parameters
    ----------
    f : function
        dX/dt - function taking in time and X and returning dX/dt
    t : float
        Current time
    X_old : list of arrays
        X at time t
    updateX : function
        Helper function to handle any correction to dxdt
        Takes in X_old, dxdt, dt and returns X_new

    Returns
    -------
    X_new : unformatted list of floats
        New values of X in format of X_old
    dt : float
        Time step
    '''
    dxdt, dt = f(t, X_old, True)
    return updateX(X_old, dxdt, dt), dt

def RK4Iterator(f, t, X_old, updateX):
    '''
    4th order Runga Kutta iteration scheme

    Defined by:
        k1 = f(t, X_n)
        k2 = f(t + dt/2, X_n + k1 * dt/2)
        k3 = f(t + dt/2, X_n + k2 * dt/2)
        k4 = f(t + dt, X_n, k3 * dt)
        X_n+1 = X_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt

    Parameters
    ----------
    f : function
        dX/dt - function taking in time and X and returning dX/dt
    t : float
        Current time
    X_old : list of arrays
        X at time t
    updateX : function
        Helper function to handle any correction to dxdt
        Takes in X_old, dxdt, dt and returns X_new

    Returns
    -------
    X_new : unformatted list of floats
        New values of X in format of X_old
    dt : float
        Time step, important if modified from dtfunc
    '''
    dxdt, dt = f(t, X_old, True)

    k1 = dxdt
    dxdtsum = k1
    X_k1 = updateX(X_old, k1, dt/2)

    k2 = f(t, X_k1)
    dxdtsum += 2*k2
    X_k2 = updateX(X_old, k2, dt/2)

    k3 = f(t, X_k2)
    dxdtsum += 2*k3
    X_k3 = updateX(X_old, k3, dt)

    k4 = f(t, X_k3)
    dxdtsum += k4

    return updateX(X_old, dxdtsum/6, dt), dt