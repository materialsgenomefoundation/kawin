import numpy as np
import matplotlib.pyplot as plt
from kawin.precipitation import PrecipitateModel
from kawin.precipitation.StoppingConditions import PrecipitationStoppingCondition

class TTPCalculator:
    '''
    Time-temperature-precipitation

    Parameters
    ----------
    model : PrecipitateModel
    stopConds : list of PrecipitateStoppingConditions
        Stopping conditions to store times when these conditions are reached
        Model will continue to solve until the max time is reached or all conditions are satisfied
    '''
    def __init__(self, model : PrecipitateModel, stopConds : list[PrecipitationStoppingCondition]):
        self.model = model
        self.stopConds = stopConds
        self._maxTime = 0
        self.transformationTimes = None

        #Add stopping conditions to model
        #NOTE: this clears any previous stopping conditions
        self.model.clearStoppingConditions()
        for j in range(len(stopConds)):
            self.model.addStoppingCondition(self.stopConds[j], 'and')

    def _getStopTime(self, T):
        '''
        Internal function to get times for each stopping conditions at a single temperature

        Parameters
        ----------
        T : float
            Temperature
        '''
        self.model.reset()
        self.model.setTemperature(T)
        self.model.solve(self._maxTime, verbose = True, vIt = 1000)

        values = np.zeros(len(self.stopConds))
        for j in range(len(self.stopConds)):
            values[j] = self.stopConds[j].satisfiedTime()

        return values
    
    def calculateTTP(self, Tlow, Thigh, Tsteps, maxTime, pool = None):
        '''
        Calculates TTP diagram between Tlow and Thigh

        Parameters
        ----------
        Tlow : float
            Lower temperature range
        Thigh : float
            Upper temperature range
        Tsteps : int
            Number of temperatures between Tlow and Thigh to evaluate
        maxTime : float
            Maximum simulation time
            If the model reaches the max time before all stopping conditions are met, it will stop prematurely
                and any unsatisfied stopping conditions will be recorded as -1
        pool : None or multiprocessing pool
            If None, each temperature will be evaluated in serial
            If a pool, must have a map function
                Possible options: 
                    multiprocessing.Pool - (mac and unix only)
                    pathos.multiprocessing.ProcessingPool - (windows, mac and unix)
                    dask.Client - (windows, mac and unix)
        '''
        self.transformationTimes = np.zeros((Tsteps, len(self.stopConds)))
        self._maxTime = maxTime
        self.temperatures = np.linspace(Tlow, Thigh, Tsteps)

        if pool is None:
            outputs = list(map(self._getStopTime, self.temperatures))
        else:
            outputs = list(pool.map(self._getStopTime, self.temperatures))

        for i in range(len(self.temperatures)):
            for j in range(len(self.stopConds)):
                self.transformationTimes[i,j] = outputs[i][j]

    def plot(self, ax, labels, xlim = [1, 1e6]):
        '''
        Plots TTP diagram

        Parameters
        ----------
        ax : Matplotlib axes object
        labels : list of str
            Labels for each stopping condition
        xlim : list of float
            x-axis limits
            Plotting will be set on log scale, so lower limits will be set to be non-zero
        '''
        for i in range(len(self.stopConds)):
            indices = self.transformationTimes[:,i] != -1
            plt.plot(self.transformationTimes[indices,i], self.temperatures[indices], label=labels[i])
        ax.legend()
        if xlim[0] == 0:
            xlim[0] = 1e-3
        plt.xlim(xlim)
        plt.xlabel('Time (s)')
        plt.xscale('log')
        plt.ylabel('Temperature (K)')
        plt.show()