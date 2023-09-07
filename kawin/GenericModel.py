from kawin.solver.Solver import SolverType, DESolver
import numpy as np

class GenericModel:
    def __init__(self):
        self.verbose = False

    def _getVarDict(self):
        return {}
    
    def _addExtraSaveVariables(self, saveDict):
        return

    def _loadModel(self, data):
        return
    
    def _loadExtraVariables(self, data):
        return
    
    def save(self, filename, compressed = True):
        varDict = self._getVarDict()
        saveDict = {}
        for var in varDict:
            saveDict[var] = getattr(self, varDict[var])
        self._addExtraSaveVariables(saveDict)
        print(saveDict.keys())
        if compressed:
            np.savez_compressed(filename, **saveDict)
        else:
            np.savez(filename, **saveDict)

    def _loadData(self, data):
        varDict = self._getVarDict()
        for var in varDict:
            setattr(self, varDict[var], data[var])
        self._loadExtraVariables(data)

    def setup(self):
        pass

    def record(self, t):
        pass

    def headerStr(self):
        pass

    def getCurrentX(self):
        raise NotImplementedError()

    def getDt(self, dXdt):
        raise NotImplementedError()
    
    def getdXdt(self, x):
        raise NotImplementedError()
    
    def preProcess(self):
        pass
    
    def postProcess(self, time, x):
        pass

    def printStatus(self):
        pass

    def solve(self, simTime, solverType = SolverType.RK4, verbose=False, vIt=10):
        '''
        Solves model using the DESolver
        '''
        self.setup()
        if verbose:
            print('Iteration\tSim Time (h)\tRun time (s)')

        solver = DESolver(solverType)
        solver.setFunctions(preProcess=self.preProcess, postProcess=self.postProcess, printStatus=self.printStatus, getDt=self.getDt)

        t, X0 = self.getCurrentX()
        self.deltaTime = simTime - t
        solver.solve(self.getdXdt, t, X0, t+simTime, verbose, vIt)