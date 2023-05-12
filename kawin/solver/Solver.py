from kawin.solver.ExplicitEuler import ExplicitEulerIterator
from kawin.solver.RK4 import RK4Iterator

class DESolver:
    '''
    Generic class for ODE/PDE solvers

    Generalization - coupled ODEs or PDEs (bunch of coupled ODEs) can be stated as dX/dt = f(X, t)

    Parameters
    ----------
    defaultDt : float (optional)
        Default time increment if no function is implement to estimate a good time increment
    '''
    def __init__(self, iterator = 'RK4', defaultDt = 0.1):
        self.dt = defaultDt
        self.getDt = self.defaultDtFunc
        self.preProcess = self.defaultPreProcess
        self.postProcess = self.defaultPostProcess
        self.setIterator(iterator)

    def setIterator(self, iterator):
        if 'explicit' in iterator and 'euler' in iterator:
            self.iterator = ExplicitEulerIterator()
        else:
            self.iterator = RK4Iterator()

    def setDtFunc(self, func):
        '''
        Sets getDt function. This assumes that the class will handled any internal variables necessary

        Parameters
        ----------
        func : function returning scalar float
            Given the current state of the class, return the optimal time increment
        '''
        self.getDt = func

    def defaultDtFunc(self):
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
    
    def solve(self, f, t0, X0, tf):
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
        while currTime < tf:
            self.preProcess()
            dt = self.getDt()
            if currTime + dt >= tf:
                dt = tf - currTime
            X0 = self.iterator.iterate(f, currTime, X0, dt)
            currTime += dt
            self.postProcess(currTime, X0)
