import time
from kawin.solver.Iterators import explicitEulerIterator, rk4Iterator

class DESolver:
    '''
    Generic class for ODE/PDE solvers

    Generalization - coupled ODEs or PDEs (bunch of coupled ODEs) can be stated as dX/dt = f(X, t)

    Parameters
    ----------
    iterator : Iterator function
        Defines what iteration scheme to use
    defaultDt : float (defaults to 0.1)
        Default time increment if no function is implement to estimate a good time increment
    minDtFrac : float (defaults to 1e-8)
        Minimum time step as a fraction of simulation time
    maxDtFrac : float (defaults to 1)
        Maximum time step as a fraction of simulation time
    '''
    def __init__(self, iterator = rk4Iterator, defaultDT = 0.1, minDtFrac = 1e-8, maxDtFrac = 1):
        self.dtmin = minDtFrac       #Min and max dt fraction of simulation time
        self.dtmax = maxDtFrac
        self.dt = defaultDT

        self.setFunctions(self.defaultPreProcess, self.defaultPostProcess, self.defaultPrintHeader, self.defaultPrintStatus)
        self.iterator = iterator

    def setFunctions(self, preProcess = None, postProcess = None, printHeader = None, printStatus = None):
        '''
        Sets functions before solving

        If any of these are not defined, then the corresponding function will be the default defined here
        '''
        self.preProcess = self.preProcess if preProcess is None else preProcess
        self.postProcess = self.postProcess if postProcess is None else postProcess
        self.printHeader = self.printHeader if printHeader is None else printHeader
        self.printStatus = self.printStatus if printStatus is None else printStatus

    def setdXdtFunctions(self, f, correctdXdt, getDt, flattenX, unflattenX):
        self._f = f
        self._correctdXdt = correctdXdt
        self._getDt = getDt
        self._flattenX = flattenX
        self._unflattenX = unflattenX

    def defaultDtFunc(self, dXdt):
        '''
        Returns the default time increment
        '''
        return self.dt
    
    def defaultPreProcess(self):
        '''
        Default pre-processing function before an iteration
        '''
        return
    
    def defaultPostProcess(self, currTime, X_new):
        '''
        Default post-processing function after an iteration
        '''
        return X_new, False
    
    def defaultPrintHeader(self):
        '''
        Default print function before solving
        '''
        return
    
    def defaultPrintStatus(self, iteration, modeltime, simTimeElapsed):
        '''
        Default print function for when n iterations passed and verbose is true
        '''
        return
    
    def correctdXdtNotImplemented(self, dt, x, dXdt):
        '''
        Default function to correct dXdt
        '''
        pass

    def flattenXNotImplemented(self, X):
        '''
        Default flattenX function, which assumes X is in the correct format
        '''
        return X
    
    def unflattenXNotImplemented(self, X_flat, X_ref):
        '''
        Default unflattenX function which assumes X is in the correct format
        '''
        return X_flat
    
    def _getdXdt(self, t, x, getDt = False):
        '''
        Wrapper around getdXdt which will handle the following:
            Handle flattening/unfalttening the x and dx/dt arrays
            Calculate dt if not supplied

        The API for the iterator will be that all arrays are 1D np.arrays where operators will be trivial

        Parameters
        ----------
        t : float
            Time
        x : 1D np.array
            Model values
        getDt : bool
            Will calculate dt if True
        '''
        unflatX = self._unflattenX(x, self._X0)
        dXdt = self._f(t, unflatX)
        if getDt:
            dt = self._getDt(dXdt)
            dt = dt if dt > self._dtmin else self._dtmin
            dt = dt if dt < self._dtmax else self._dtmax
            return self._flattenX(dXdt), dt
        else:
            return self._flattenX(dXdt)
        
    def _updateX(self, x, dxdt, dt):
        '''
        Helper function that hides the correctdXdt function

        The API for the iterator will be that all arrays are 1D np.arrays where operators will be trivial

        Parameters
        ----------
        x : 1D np.array
            Model values
        dxdt : 1D np.array
            Derivatives at x
        dt : float
            Time step
        '''
        unflatdxdt = self._unflattenX(dxdt, self._X0)
        self._correctdXdt(dt, self._X0, unflatdxdt)
        return x + self._flattenX(unflatdxdt)*dt
    
    def solve(self, t0, X0, tf, verbose = False, vIt = 10):
        '''
        Solves dX/dt over a time increment
        This will be the main function that a model will use

        Steps during each iteration
            1. Print status if vIt iterations passed
            2. preProcess
            3. Iterate
            4. Update current time
            5. postProcess

        Parameters
        ----------
        f : function
            dX/dt - function taking in time and returning dX/dt
        t0 : float
            Starting time
        X0 : list of arrays
            X at time t
        tf : float
            Final time
        verbose: bool (defaults to False)
            Whether to print status
        vIt : integer (defaults to 10)
            Number of iterations to print status
        '''
        if verbose:
            self.printHeader()

        self._dtmin = self.dtmin * (tf - t0)
        self._dtmax = self.dtmax * (tf - t0)
        currTime = t0
        i = 0
        timeStart = time.time()
        stop = False
        while currTime < tf and not stop:
            if verbose and i % vIt == 0:
                timeFinish = time.time()
                self.printStatus(i, currTime, timeFinish - timeStart)

            self.preProcess()
            #Limit dtmax to remaining time if it's larger
            if self._dtmax > tf - currTime:
                self._dtmax = tf - currTime

            #Store X0 as a reference variable for _unflattenX
            #We have to do this per iteration since the shape of X0 can change during postProcess
            #    This is especially true for the population balance model with adaptive bins
            #The iterator also returns the flat array of X, so we need to unflatten it afterwards here
            self._X0 = X0
            X0_flat, dt = self.iterator(self._getdXdt, currTime, self._flattenX(X0), self._updateX)
            X0 = self._unflattenX(X0_flat, self._X0)
            
            currTime += dt
            X0, stop = self.postProcess(currTime, X0)
            i += 1

        if verbose:
            if stop:
                print('Stopping condition met. Ending simulation early.')
                
            timeFinish = time.time()
            self.printStatus(i, currTime, timeFinish - timeStart)