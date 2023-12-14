from kawin.solver.ExplicitEuler import ExplicitEulerIterator
from kawin.solver.RK4 import RK4Iterator
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
        self.getDt = self.defaultDtFunc
        self.preProcess = self.defaultPreProcess
        self.printHeader = self.defaultPrintHeader
        self.printStatus = self.defaultPrintStatus
        self.postProcess = self.defaultPostProcess
        self.setIterator(iterator)

    def setIterator(self, iterator):
        '''
        Parameters
        ----------
        iterator : SolverType or Iterator
            Defines what iteration scheme to use
        '''
        if iterator == SolverType.EXPLICITEULER:
            self.iterator = ExplicitEulerIterator()
        elif iterator == SolverType.RK4:
            self.iterator = RK4Iterator()
        else:
            self.iterator = iterator

    def setFunctions(self, preProcess = None, postProcess = None, printHeader = None, printStatus = None, getDt = None):
        '''
        Sets functions before solving

        If any of these are not defined, then the corresponding function will be the default defined here
            Except for getDt (which returns defaultDt), the other functions will do nothing
        '''
        if preProcess is not None:
            self.preProcess = preProcess
        if postProcess is not None:
            self.postProcess = postProcess
        if printHeader is not None:
            self.printHeader = printHeader
        if printStatus is not None:
            self.printStatus = printStatus
        if getDt is not None:
            self.getDt = getDt

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
    
    def defaultPrintStatus(self, iteration, simTimeElapsed):
        '''
        Default print function for when n iterations passed and verbose is true
        '''
        return
    
    def correctdXdtNotImplemented(self, dt, x, dXdt):
        '''
        Default function to correct dXdt
        '''
        pass
    
    def solve(self, f, t0, X0, tf, verbose = False, vIt = 10, correctdXdtFunc = None):
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

        #Use default function if correctdXdtFunc is not supplied
        if correctdXdtFunc is None:
            correctdXdtFunc = self.correctdXdtNotImplemented

        dtmin = self.dtmin * (tf - t0)
        dtmax = self.dtmax * (tf - t0)
        currTime = t0
        i = 0
        timeStart = time.time()
        stop = False
        while currTime < tf and not stop:
            if verbose and i % vIt == 0:
                timeFinish = time.time()
                self.printStatus(i, timeFinish - timeStart)

            self.preProcess()
            X0, dt = self.iterator.iterate(f, currTime, X0, self.getDt, dtmin, dtmax, correctdXdtFunc)
            currTime += dt
            X0, stop = self.postProcess(currTime, X0)

            i += 1

        if verbose:
            if stop:
                print('Stopping condition met. Ending simulation early.')
                
            timeFinish = time.time()
            self.printStatus(i, timeFinish - timeStart)
