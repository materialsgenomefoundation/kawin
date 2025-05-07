import numpy as np
from kawin.PlotUtils import _get_axis, _adjust_kwargs
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
        if isinstance(stopConds, PrecipitationStoppingCondition):
            stopConds = [stopConds]
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
        self.model.temperatureParameters.setIsothermalTemperature(T)
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

def plotTTP(ttp: TTPCalculator, ax=None, *args, **kwargs):
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
    ax = _get_axis(ax)
    for i in range(len(ttp.stopConds)):
        indices = ttp.transformationTimes[:,i] != -1
        plot_kwargs = _adjust_kwargs(ttp.stopConds[i].name, {'label': ttp.stopConds[i].label}, **kwargs)
        ax.plot(ttp.transformationTimes[indices,i], ttp.temperatures[indices], *args, **plot_kwargs)
    ax.set_xlabel('Time (s)')
    ax.set_xscale('log')
    ax.set_ylabel('Temperature (K)')
    ax.legend()
    return ax