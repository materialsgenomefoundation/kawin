from kawin.solver.Solver import SolverType, DESolver
import numpy as np

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
        printStatus(self, iteration, simTimeElapsed) - output states made after n iterations
    '''
    def __init__(self):
        self.verbose = False

    def _getVarDict(self):
        '''
        Returns variable dictionary mapping variable name to internal member name

        This is used to when saving the model into a npz format, where the member names
        will be replaced with the variable names defined by this dictionary
        '''
        return {}
    
    def _addExtraSaveVariables(self, saveDict):
        '''
        Adds extra variables to the save dictionary that are not covered by the variable dictionary
            The variable dictionary only cover members that can be retrieved from getattr, so
            this function is used to save data if it is from another class that itself is an attribute
        
        Parameters
        ----------
        saveDict : dictionary { str : np.ndarray }
            Dictionary to add data to
        '''
        return
    
    def _loadExtraVariables(self, data):
        '''
        Loads extra variables in data not covered by the variable dictionary

        Parameters
        ----------
        data : dictionary { str : np.ndarray }
            Dictionary to read data from
        '''
        return
    
    def save(self, filename, compressed = True):
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
        varDict = self._getVarDict()
        saveDict = {}
        for var in varDict:
            saveDict[var] = getattr(self, varDict[var])
        self._addExtraSaveVariables(saveDict)
        if compressed:
            np.savez_compressed(filename, **saveDict)
        else:
            np.savez(filename, **saveDict)

    def _loadData(self, data):
        '''
        Loads data taken from .npz file into model

        1. Sets attributes using mapping defined from _getVarDict
        2. Loads extra variables using _loadExtraVariables

        Parameters
        ----------
        data : dictionary { str : np.ndarray }
            Data to load from
        '''
        varDict = self._getVarDict()
        for var in varDict:
            setattr(self, varDict[var], data[var])
        self._loadExtraVariables(data)

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
        pass

    def printStatus(self, iteration, simTimeElapsed):
        '''
        Output to be printed after n iterations (defined by vIt in solve)

        verbose must be True when calling solve
        '''
        pass

    def setTimeInfo(self, currTime, simTime):
        '''
        Store time variables for starting, final and delta time

        This is sometimes useful for determining the time step
        '''
        self.deltaTime = simTime
        self.startTime = currTime
        self.finalTime = currTime+simTime

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
        solver.setFunctions(preProcess=self.preProcess, postProcess=self.postProcess, printHeader=self.printHeader, printStatus=self.printStatus, getDt=self.getDt)
        
        t, X0 = self.getCurrentX()
        self.setTimeInfo(t, simTime)
        solver.solve(self.getdXdt, self.startTime, X0, self.finalTime, verbose, vIt, self.correctdXdt)