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
    defaultDt : float (optional)
        Default time increment if no function is implement to estimate a good time increment
    '''
    def __init__(self, iterator = SolverType.RK4, defaultDT = 0.1):
        self.dtmin = 1e-8       #Fraction of simulation time
        self.dtmax = 1
        self.dt = defaultDT
        self.getDt = self.defaultDtFunc
        self.preProcess = self.defaultPreProcess
        self.printStatus = self.defaultPrintStatus
        self.postProcess = self.defaultPostProcess
        self.setIterator(iterator)

    def setIterator(self, iterator):
        if iterator == SolverType.EXPLICITEULER:
            self.iterator = ExplicitEulerIterator()
        else:
            self.iterator = RK4Iterator()

    def setFunctions(self, preProcess = None, postProcess = None, printStatus = None, getDt = None):
        if preProcess is not None:
            self.preProcess = preProcess
        if postProcess is not None:
            self.postProcess = postProcess
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
        Default post-processing function before an iteration
        '''
        return
    
    def defaultPrintStatus(self, iteration, simTimeElapsed):
        '''
        Default print function for when n iterations passed and verbose is true
        '''
        return
    
    def solve(self, f, t0, X0, tf, verbose = False, vIt = 10):
        '''
        Solves dX/dt over a time increment
        This will be the main function that a model will use

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
        '''
        currTime = t0
        i = 0
        timeStart = time.time()
        stop = False
        while currTime < tf and not stop:
            if verbose and i % vIt == 0:
                timeFinish = time.time()
                self.printStatus(i, timeFinish - timeStart)

            self.preProcess()
            dtmin = self.dtmin * (tf - t0)
            dtmax = self.dtmax * (tf - t0)
            X0, dt = self.iterator.iterate(f, currTime, X0, self.getDt, dtmin, dtmax)
            currTime += dt
            X0, stop = self.postProcess(currTime, X0)

            i += 1

        if verbose:
            if stop:
                print('Stopping condition met. Ending simulation early.')
                
            timeFinish = time.time()
            self.printStatus(i, timeFinish - timeStart)
