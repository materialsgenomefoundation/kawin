from kawin.solver.Iterators import ExplicitEulerIterator, RK4Iterator
from enum import Enum
import time

class SolverType(Enum):
    EXPLICITEULER = 0
    RK4 = 1

class DESolver:
    '''
    Generic class for ODE/PDE solvers

    Generalization - coupled ODEs or PDEs (bunch of coupled ODEs) can be stated as dX/dt = f(X, t)

    Parameters
    ----------
    iterator : SolverType or Iterator
        Defines what iteration scheme to use
    defaultDt : float (defaults to 0.1)
        Default time increment if no function is implement to estimate a good time increment
    minDtFrac : float (defaults to 1e-8)
        Minimum time step as a fraction of simulation time
    maxDtFrac : float (defaults to 1)
        Maximum time step as a fraction of simulation time
    '''
    def __init__(self, iterator = SolverType.RK4, defaultDT = 0.1, minDtFrac = 1e-8, maxDtFrac = 1):
        self.dtmin = minDtFrac       #Min and max dt fraction of simulation time
        self.dtmax = maxDtFrac
        self.dt = defaultDT

        self.setFunctions(self.defaultPreProcess, self.defaultPostProcess, self.defaultPrintHeader, self.defaultPrintStatus, self.defaultDtFunc)

        self.setIterator(iterator)

    def setIterator(self, iterator):
        '''
        Parameters
        ----------
        iterator : SolverType or Iterator
            Defines what iteration scheme to use
        '''
        if iterator == SolverType.EXPLICITEULER:
            self.iterator = ExplicitEulerIterator
        elif iterator == SolverType.RK4:
            self.iterator = RK4Iterator
        else:
            self.iterator = iterator

    def setFunctions(self, preProcess = None, postProcess = None, printHeader = None, printStatus = None, getDt = None):
        '''
        Sets functions before solving

        If any of these are not defined, then the corresponding function will be the default defined here
            Except for getDt (which returns defaultDt), the other functions will do nothing
        '''
        self.preProcess = self.preProcess if preProcess is None else preProcess
        self.postProcess = self.postProcess if postProcess is None else postProcess
        self.printHeader = self.printHeader if printHeader is None else printHeader
        self.printStatus = self.printStatus if printStatus is None else printStatus
        self.getDt = self.getDt if getDt is None else getDt

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
    
    def solve(self, f, t0, X0, tf, verbose = False, vIt = 10, correctdXdtFunc = None, flattenXFunc = None, unflattenXFunc = None):
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
        correctdXdtFunc : function (defaults to None)
            Corrects dXdt (takes in t, x and dXdt) and returns nothing
            Should this be here or in setFunctions? It only gets called in Iterator.iterator
        '''
        if verbose:
            self.printHeader()

        #Use default function if correctdXdtFunc, flattenXFunc or unflattenXFunc is not supplied
        correctdXdtFunc = self.correctdXdtNotImplemented if correctdXdtFunc is None else correctdXdtFunc
        flattenXFunc = self.flattenXNotImplemented if flattenXFunc is None else flattenXFunc
        unflattenXFunc = self.unflattenXNotImplemented if unflattenXFunc is None else unflattenXFunc

        dtmin = self.dtmin * (tf - t0)
        dtmax = self.dtmax * (tf - t0)
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
            if dtmax > tf - currTime:
                dtmax = tf - currTime
            #X0, dt = self.iterator.iterate(f, currTime, X0, self.getDt, dtmin, dtmax, correctdXdtFunc)
            X0, dt = self.iterator(f, currTime, X0, self.getDt, dtmin, dtmax, correctdXdtFunc, flattenXFunc, unflattenXFunc)
            currTime += dt
            X0, stop = self.postProcess(currTime, X0)
            i += 1

        if verbose:
            if stop:
                print('Stopping condition met. Ending simulation early.')
                
            timeFinish = time.time()
            self.printStatus(i, currTime, timeFinish - timeStart)
