from kawin.solver.Solver import SolverType, DESolver
import numpy as np
from typing import List
import copy

class GenericModel:
    '''
    Abstract model that new models can inherit from to interface with the Solver
    
    The model is intended to be defined by an ordinary differential equation or a set of them
    The differential equations are defined by dX/dt = f(t,X)
        Where t is time and X is the set of time-dependent variables at time t

    Required functions to be implemented:
        getCurrentX(self)              - should return time and all time-dependent variables
        getdXdt(self, t, x)            - should return all time-dependent derivatives
        getDt(self, dXdt)              - should return a suitable time step
        

    Functions that can be implemented but not necessary:
        _getVarDict(self) - returns a dictionary of {variable name : member name}
        _addExtraSaveVariables(self, saveDict) - adds to saveDict additional variables to save
        _loadExtraVariables(self, data) - loads additional data to model

        setup(self) - ran before solver is called
        correctdXdt(self, dt, x, dXdt) - does not need to return anything, but should modify dXdt
        preProcess(self) - preprocessing before each iteration
        postProcess(self, time, x) - postprocessing after each iteration
        printHeader(self) - initial output statements before solver is called
        printStatus(self, iteration, modelTime, simTimeElapsed) - output states made after n iterations
    '''
    def __init__(self):
        self.clearCouplingModels()

    def toDict(self):
        '''
        Creates a dictionary data set of the following:
            - this will only save the data that was solved for and not model parameters

        TODO: eventually support saving model parameters. This is a bit tough with all the nested parameters right now
        '''
        return {}
    
    @classmethod
    def fromDict(self, data):
        pass
    
    def save(self, filename: str):
        '''
        Saves model data into file

        1. Store model attributes into saveDict using mapping defined from _getVarDict
        2. Add extra variables to saveDict if needed
        3. Save data into .npz format

        Parameters
        ----------
        filename : str
            File name to save to
        compressed : bool (defaults to True)
            Whether to save in compressed format
        '''
        data = self.toDict()
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez_compressed(filename, **data)

    def load(self, filename: str):
        if not filename.endswith('.npz'):
            filename += '.npz'
        data = np.load(filename)
        self.fromDict(dict(data))

    def addCouplingModel(self, model):
        '''
        Adds a coupling model to the KWN model

        These will be updated after each iteration with the new values of the model

        Parameters
        ----------
        model : object
            Must have a function called updateCoupledModel that takes in a KWNBase or KWNEuler object
        '''
        self.couplingModels.append(model)

    def clearCouplingModels(self):
        '''
        Clears list of coupling models

        Note - this will not reset the coupling models, just removes them from the list
        '''
        self.couplingModels = []

    def updateCoupledModels(self):
        '''
        Updates coupled models with current values
        '''
        for cm in self.couplingModels:
            cm.updateCoupledModel(self)

    def setup(self):
        '''
        Sets up model before being solved

        This is the first thing that is called when the solve function is called

        Note: this will be called each time the solve function called, so if setup only needs to
              be called once, then make sure there's a check in the model implementation to prevent
              setup from being called more than once
        '''
        pass

    def getCurrentX(self):
        '''
        Gets values of time-dependent variables at current time

        The required format of X is not strict as long as it matches dXdt
        Example: if X is a nested list of [[a, b], c], then dXdt should be [[da/dt, db/dt], dc/dt]

        Note: X should only be for variables that are solved by dX/dt = f(t,X)
        Variables that can be computed directly from X should be calculated in the preProcess or postProcess functions

        Returns
        -------
        t : current time of model
        X : unformatted list of floats
        '''
        raise NotImplementedError()

    def getDt(self, dXdt):
        '''
        Gets suitable time step based off dXdt

        Parameters
        ----------
        dXdt : unformated list of floats
            Time derivatives that may be used to find dt

        Returns
        -------
        dt : float
        '''
        raise NotImplementedError()
    
    def getdXdt(self, t, x):
        '''
        Gets dXdt from current time and X

        Parameters
        ----------
        t : float
            Current time
        x : unformated list of floats
            Current values of time-dependent variables
        
        Returns
        -------
        dXdt : unformated list of floats
            Must be in same format as x
        '''
        raise NotImplementedError()

    def correctdXdt(self, dt, x, dXdt):
        '''
        Intended for cases where dXdt can only be corrected once dt is known
            For example, the time derivatives in the population balance model in PrecipitateModel needs to be
                adjusted to avoid negative bins, but this can only be done once dt is known

        If dXdt can be corrected without knowing dt, then it is recommended to be done during the getdXdt function

        No return value, dXdt is to be modified directly
        '''
        pass
    
    def preProcess(self):
        '''
        Performs any pre-processing before an iteration. This may include some calculations or storing temporary variables
        '''
        pass
    
    def postProcess(self, time, x):
        '''
        Post processing done after an iteration

        This should at least involve storing the new values of time and X
        But this can also include additional calculations or return a signal to stop simulations

        Parameters
        ----------
        time : float
            New time
        x : unformatted list of floats
            New values of X

        Returns
        -------
        x : unformatted list of floats
            This is in case X was modified in postProcess
        stop : bool
            If the simulation needs to end early (ex. a stopping condition is met), then return True to stop solving
        '''
        return x, False

    def printHeader(self):
        '''
        First output to be printed when solve is called

        verbose must be True when calling solve
        '''
        print('Iteration\tSim Time(s)\tRun Time(s)')

    def printStatus(self, iteration, modelTime, simTimeElapsed):
        '''
        Output to be printed after n iterations (defined by vIt in solve)

        verbose must be True when calling solve
        '''
        print('{}\t\t{:.1e}\t\t{:.1f}'.format(iteration, modelTime, simTimeElapsed))

    def setTimeInfo(self, currTime, simTime):
        '''
        Store time variables for starting, final and delta time

        This is sometimes useful for determining the time step
        '''
        self.deltaTime = simTime
        self.initialTime = currTime
        self.finalTime = currTime+simTime

    def flattenX(self, X):
        '''
        Since X can be a nested list of values or arrays (or anything),
        we want some instructions for the solver and Iterator for how to convert X
        to a 1D array

        By default, we'll assume X is a list of either floats or 1D arrays

        For more complex nesting, this function should be overloaded

        Parameters
        ----------
        X : list of arrays

        Returns
        -------
        X_flat : 1D numpy array
        '''
        return np.hstack(X)
    
    def unflattenX(self, X_flat, X_ref):
        '''
        Converts flattened X array to original nested X

        Parameters
        ----------
        X_flat : 1D numpy array
            Flattened array
        X_ref : list of arrays
            Template to convert X_flat to

        Returns
        -------
        X_new : unflattened list in the same format as X_ref
        '''
        #Not sure if this is the most efficient way, but we can't assume how the nested list in X_ref is structured
        #This should be a shallow copy though, so maybe it's fine
        X_new = copy.copy(X_ref)
        n = 0
        for i in range(len(X_new)):
            #We can't be sure that X_new[i] a python scalar or numpy scalar, so we'll convert to an np.ndarray first
            if len(np.array(X_new[i]).shape) == 0:
                X_new[i] = X_flat[n]
                n += 1
            else:
                arrLen = np.prod(np.array(X_new[i]).shape)
                X_new[i] = np.reshape(X_flat[n:n+arrLen], np.array(X_new[i]).shape)
                n += arrLen
        return X_new

    def solve(self, simTime, solverType = SolverType.RK4, verbose=False, vIt=10, minDtFrac = 1e-8, maxDtFrac = 1):
        '''
        Solves model using the DESolver

        Steps:
            1. Call setup
            2. Create DESolver object and set necessary functions
            3. Get current values of t and X
            4. Solve from current t to t+simTime

        Parameters
        ----------
        simTime : float
            Simulation time (as a delta from current time)
        solverType : SolverType or Iterator (defaults to SolverType.RK4)
            Defines what iteration scheme to use
        verbose : bool (defaults to False)
            Outputs status if true
        vIt : integer (defaults to 10)
            Number of iterations before printing status
        minDtFrac : float (defaults to 1e-8)
            Minimum dt as fraction of simulation time
        maxDtFrac : float (defaults to 1)
            Maximum dt as fraction of simulation time
        '''
        self.setup()

        solver = DESolver(solverType, minDtFrac = minDtFrac, maxDtFrac = maxDtFrac)
        solver.setFunctions(preProcess=self.preProcess, postProcess=self.postProcess, printHeader=self.printHeader, printStatus=self.printStatus)
        solver.setdXdtFunctions(self.getdXdt, self.correctdXdt, self.getDt, self.flattenX, self.unflattenX)
        
        t, X0 = self.getCurrentX()
        self.setTimeInfo(t, simTime)
        solver.solve(self.initialTime, X0, self.finalTime, verbose, vIt)
        #solver.solve(self.getdXdt, self.initialTime, X0, self.finalTime, verbose, vIt, self.correctdXdt, self.flattenX, self.unflattenX)

class Coupler(GenericModel):
    '''
    Class for coupling multiple GenericModel objects together

    Note:
        coupleddXdt, coupledPreProcess and coupledPostProcess aren't really necessary since
        tighter coupling can also be done by overloading the getdXdt, preProcess and/or postProcess
        functions and calling the method of the Coupler before anything else
        Ex. tighter coupling can be done by
            a)  Overloading coupleddXdt as
                    def coupleddXdt(self, dXdt):
                        ===
                        Modify dXdt here
                        ===
            
            b)  Overriding getdXdt as
                    def getdXdt(self, t, x):
                        dXdt = super().getdXdt(t, x)
                        ---
                        modify dXdt here
                        ---
                        return dXdt

    Parameters
    ----------
    models : List[GenericModel]
        List of models to be solved
    '''
    def __init__(self, models : List[GenericModel]):
        self.models = models

        #Internal time to record
        #We have the option to solve a model for a given amount of time before coupling it
        #  to another model, which would make each model have a different internal time
        #  Thus, we'll record time here as well representing the time during the coupling
        self.time = np.zeros(1)

    def setup(self):
        '''
        Sets up each model
        '''
        super().setup()
        for m in self.models:
            m.setup()

    def setTimeInfo(self, currTime, simTime):
        '''
        Sets time info for the CoupledModel class and each model
        '''
        super().setTimeInfo(currTime, simTime)
        for m in self.models:
            m.setTimeInfo(currTime, simTime)

    def flattenX(self, X):
        '''
        Instructions for converting X to 1D array

        We grab the flattened x array of each model and concatenate them
            Thus we don't have to care about the structure of x in each model as
            long as the model itself has the instructions to flatten its x array

        Also record the length of each flattened x in each model so we know what
        indices to used for unflattening
        '''
        X_new = []
        for m, xsub in zip(self.models, X):
            xsub_new = m.flattenX(xsub)
            X_new.append(xsub_new)
        self._sizeRef = [len(xi) for xi in X_new]
        return np.concatenate(X_new)

    def unflattenX(self, X_flat, X_ref):
        '''
        Instructions for converting X_flat to list of x of each model

        We take the subset of X_flat corresponding to each model and unflatten it
        based off the instructions in the model. Then we just return a list containing
        each unflattened x
        '''
        X_new = []
        ind = 0
        for m, s, x_refsub in zip(self.models, self._sizeRef, X_ref):
            xi_new = m.unflattenX(X_flat[ind:ind+s], x_refsub)
            X_new.append(xi_new)
            ind += s
        return X_new

    def getCurrentX(self):
        '''
        Get current time and x for each model
        '''
        xs = []
        for m in self.models:
            _, x = m.getCurrentX()
            xs.append(x)

        return self.time[-1], xs
    
    def getDt(self, dXdt):
        '''
        Get the minimum dt out of all models
        '''
        dts = []
        for m, dxdtsub in zip(self.models, dXdt):
            dts.append(m.getDt(dxdtsub))
        return np.amin(dts)
    
    def getdXdt(self, t, x):
        '''
        Get dXdt for each model
        '''
        dxdts = []
        for m, xsub in zip(self.models, x):
            dxdts.append(m.getdXdt(t, xsub))
        self.coupledXdt(t, x, dxdts)
        return dxdts
    
    def correctdXdt(self, dt, x, dXdt):
        '''
        Corrects dXdt for each model

        Note - dXdt has to be modified since we don't return dXdt in this function
            Since dXdt here is composed of a nested list of dXdts of each model, these
            will be passed by reference
        '''
        for m, xsub, dxdtsub in zip(self.models, x, dXdt):
            m.correctdXdt(dt, xsub, dxdtsub)

    def preProcess(self):
        '''
        Pre process on each model
        '''
        for m in self.models:
            m.preProcess()
        self.couplePreProcess()

    def postProcess(self, time, x):
        '''
        Post process on each model and records new time
        '''
        xNew = []
        stop = False
        for m, xsub in zip(self.models, x):
            xnew_sub, s = m.postProcess(time, xsub)
            stop = stop or s
            xNew.append(xnew_sub)
        self.time = np.append(self.time, time)
        self.couplePostProcess()
        return xNew, stop
    
    def coupledXdt(self, t, x, dXdt):
        '''
        Empty function where inherited classes can do extra operations on
        the time derivatives each models or between models
        '''
        return
    
    def couplePreProcess(self):
        '''
        Empty function where inherited classes can do extra operations on
        each models or between models for an iteration
        '''
        return
    
    def couplePostProcess(self):
        '''
        Empty function where inherited classes can do extra operations on
        each models or between models after an iteration
        '''
        return



        