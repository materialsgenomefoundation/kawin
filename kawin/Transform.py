import numpy as np

class TTTDiagram:
    '''
    Class for creating time-temperature-transformation (TTT) diagrams

    Parameters
    ----------
    model : PrecipitateModel object
        All necessary parameters in the model must be already defined
    '''
    def __init__(self, model):
        self.model = model

    def calculateTTT(self, phaseFractions, Tlow, Thigh, nT):
        '''
        Calculates TTT diagram

        Parameters
        ----------
        phaseFractions : float or list
            All phase fractions to look for when model runs
        Tlow : float
            Lower temperature
        Thigh : float
            Upper temperature
        nT : int
            Number of temperature intervals to run the precipitate model at
        '''
        phaseFractions = np.array(phaseFractions)
        #If scalar, then make into array
        if (type(phaseFractions) == np.ndarray and phaseFractions.ndim == 0):
            phaseFractions = np.array([phaseFractions])
        self.phaseFractions = phaseFractions

        self.Tlow = Tlow
        self.Thigh = Thigh
        self.nT = int(nT)
        self.Tarray = np.linspace(Tlow, Thigh, nT)

        self.times = {p: np.zeros((nT, len(self.phaseFractions))) for p in self.model.phases}


        for i in range(len(self.Tarray)):
            print('Solving model at T = {:.3f}'.format(self.Tarray[i]))

            #Set current temperature
            self.model.setTemperature(self.Tarray[i])

            #Reset model and clear stopping conditions
            #This won't clear the temperature since it's an input
            self.model.reset()
            self.model.clearStoppingConditions()
            
            #Add stopping condition for all phase fractions and all phases
            for j in range(len(self.phaseFractions)):
                for k in range(len(self.model.phases)):
                    self.model.addStoppingCondition('Volume Fraction', '>', self.phaseFractions[j], phase=self.model.phases[k], mode='and')

            #Solve model
            self.model.solve()

            #Store times when conditions are met
            for j in range(len(self.phaseFractions)):
                for k in range(len(self.model.phases)):
                    index = j * len(self.model.phases) + k
                    self.times[self.model.phases[k]][i,j] = self.model.stopConditionTimes[index]

    def plot(self, ax, timeUnits = 's'):
        '''
        Plots TTT diagram

        Parameters
        ----------
        ax - Axis object
            Axis to plot on
        timeUnits - str
            Units to plot time in
        '''
        timeScale = 1
        timeLabel = 'Time (s)'
        if 'min' in timeUnits:
            timeScale = 1/60
            timeLabel = 'Time (min)'
        if 'h' in timeUnits:
            timeScale = 1/3600
            timeLabel = 'Time (hrs)'

        for j in range(len(self.phaseFractions)):
            for k in range(len(self.model.phases)):
                subIndices = self.times[self.model.phases[k]][:,j] > -1
                ax.semilogx(timeScale * self.times[self.model.phases[k]][subIndices,j], self.Tarray[subIndices], label=self.model.phases[k] + '_' + str(self.phaseFractions[j]))
        ax.legend()

        #Set x limits to nearest power of 10
        minT = np.log10(np.amin([self.times[p][self.times[p] != -1] for p in self.model.phases]))
        maxT = np.log10(np.amax([self.times[p][self.times[p] != -1] for p in self.model.phases]))
        ax.set_xlim([np.power(10, np.floor(minT)), np.power(10, np.ceil(maxT))])
        ax.set_ylabel('Temperature (K)')
        ax.set_xlabel(timeLabel)