import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import copy

class PopulationBalanceModel:
    '''
    Class for handling particle size distributions (PSD)
        This include time evolution, moments and graphing

    Parameters
    ----------
    cMin : float (optional)
        Lower bound of PSD (defaults at 0)
    cMax : float (optional)
        Upper bound of PSD (defaults at 100)
    bins : int (optional)
        Number of bins (defaults at 100)
    linearSpacing : bool (optional)
        If True (default), will store PSD on a linear scale
        Else, will store PSD on a logarithmic scale
    '''
    def __init__(self, cMin = 0, cMax = 100, bins = 100, linearSpacing = True):
        self.originalBins = bins
        self.bins = bins
        self.min = cMin
        self.max = cMax
        self.linear = linearSpacing
        
        self.reset()

        #Hidden variable for use in KWNEuler when determining composition assuming no diffusion in precipitate
        #Represents d(PSD)/dr * growth rate * dt
        #I would like this variable to be in KWNEuler, but this way is much easier
        self._fv = np.zeros(self.bins + 1)

        #Hidden variable for use in KWNEuler when adaptive time stepping is enabled
        #This allows for PSD to revert to its previous value if a time constraint is not met
        self._prevPSD = np.zeros(self.bins)
        
    def reset(self):
        '''
        Resets the PSD to 0
        This will remove any size classes that were added since initialization
        '''
        self.bins = self.originalBins
        if self.linear:
            self.PSDbounds = np.linspace(self.min, self.max, self.bins+1)
            self.PSDsize = 0.5 * (self.PSDbounds[:-1] + self.PSDbounds[1:])
        else:
            self.PSDbounds = np.logspace(self.min, self.max, self.bins+1)
            self.PSDsize = np.power(10, 0.5 * (np.log(self.PSDbounds[1:]) + np.log(self.PSDbounds[:-1])))
            
        self.PSD = np.zeros(self.bins)

    def revert(self):
        self.PSD = copy.copy(self._prevPSD)

    def changeSizeClasses(self, cMin, cMax, bins, linearSpacing = True, resetPSD = True):
        '''
        Changes the size classes and resets the PSD
        
        Parameters
        ----------
        min : float
            Lower bound of PSD
        cMax : float
            Upper bound of PSD
        bins : int
            Number of bins
        linearSpacing : bool (optional)
            If True (default), will store PSD on a linear scale
            Else, will store PSD on a logarithmic scale
        resetPSD : bool (optional)
            Whether to reset the PSD (defaults to True)
        '''
        self.bins = bins
        self.min = cMin
        self.max = cMax
        self.linear = linearSpacing
        
        if linearSpacing:
            self.PSDbounds = np.linspace(cMin, cMax, bins+1)
            self.PSDsize = 0.5 * (self.PSDbounds[:-1] + self.PSDbounds[1:])
        else:
            self.PSDbounds = np.logspace(cMin, cMax, bins+1)
            self.PSDsize = np.power(10, 0.5 * (np.log(self.PSDbounds[1:] * self.PSDbounds[:-1])))
            
        if resetPSD:
            self.reset()

    def addSizeClass(self):
        '''
        Adds an additional size class to end of distribution
        '''
        self.bins += 1
        self.PSDbounds = np.append(self.PSDbounds, 0)
        self.PSDsize = np.append(self.PSDsize, 0)
        self.PSD = np.append(self.PSD, 0)

        if self.linear:
            self.max += (self.PSDbounds[1] - self.PSDbounds[0])
            self.PSDbounds[-1] = self.max
            self.PSDsize[-1] = 0.5 * (self.PSDbounds[-1] + self.PSDbounds[-2])
        else:
            self.max = np.power(10, np.log(self.max * self.PSDbounds[1] / self.PSDbounds[0]))
            self.PSDbounds[-1] = self.max
            self.PSDsize[-1] = np.power(10, 0.5 * np.log(self.PSDbounds[-1] * self.PSDbounds[-2]))
        
    def LoadDistribution(self, data):
        '''
        Creates a particle size distribution from a set of data
        
        Parameters
        ----------
        data : array of floats
            Array of data to be inserted into PSD
        '''
        self.PSD, self.PSDbounds = np.histogram(data, self.PSDbounds)
        self.PSD = self.PSD.astype('float')
        
    def Normalize(self):
        '''
        Normalizes the PSD to 1
        '''
        sum = self.WeightedMoment(0, self.PSDbounds[1:] - self.PSDbounds[:-1])
        self.PSD /= sum
            
    def NormalizeToMoment(self, order = 0):
        '''
        Normalizes the PSD with so that the moment of specified order will be 1
        
        Parameters
        ----------
        order : int (optional)
            Order of moment that PSD will be normalized to (defaults to 0)
            Using zeroth order will normalize the PSD so the sum of PSD will be 1
        '''
        sum = self.Moment(order)
        self.PSD /= sum
        
    def Update(self, dt, flux, externalRates = None):
        '''
        Updates PSD given the flux and any external contributions

        Change in the amount of particles in a given size class = d(G*n)/dx
        Where G is flux of size class, n is number of particles in size class and dx is range of size class
        
        Parameters
        ----------
        dt : float
            Time increment
        flux : array
            Growth rate of each particle size class
            Array size must be (bins + 1) since this operates on bounds of size classes
        externalRates : array (optional)
            Additional contribution of particles (nucleation or dissolution)
            Array size must be (bins) since this operates on the individual size classes
        '''
        #Store current PSD
        self._prevPSD = copy.copy(self.PSD)

        netFlux = np.zeros(self.bins + 1)
        netFlux[0] = 0 if flux[0] > 0 else flux[0] * dt * self.PSD[0] / (self.PSDbounds[1] - self.PSDbounds[0])
        netFlux[-1] = 0 if flux[-1] < 0 else flux[-1] * dt * self.PSD[-1] / (self.PSDbounds[-1] - self.PSDbounds[-2])

        #If flux is going from size class n to n-1, then use size class n (flux <= 0)
        indices = flux[1:-1] <= 0
        netFlux[1:-1][indices] = dt * flux[1:-1][indices] * self.PSD[1:][indices] / (self.PSDbounds[2:] - self.PSDbounds[1:-1])[indices]
        under = (netFlux[1:-1] < -self.PSD[1:]) & indices
        netFlux[1:-1][under] = -self.PSD[1:][under]
        
        #if flux is going from size class n-1 to n, then use size class n-1 (flux > 0)
        indices = ~indices
        netFlux[1:-1][indices] = dt * flux[1:-1][indices] * self.PSD[:-1][indices] / (self.PSDbounds[1:-1] - self.PSDbounds[:-2])[indices]
        over = (netFlux[1:-1] > self.PSD[:-1]) & indices
        netFlux[1:-1][over] = self.PSD[:-1][over]

        self._fv = netFlux
        
        self.PSD += netFlux[:-1] - netFlux[1:]
        if externalRates is not None:
            self.PSD += externalRates
            
        #Set negative frequencies to 0
        self.PSD[self.PSD < 1] = 0

    def UpdateMomentConserved(self, dt, flux, externalRates = None, order = 1):
        '''
        Updates PSD given the flux and any external contributions

        Change in the amount of particles in a given size class = d(G*n^m)/dx
        Where G is flux of size class, n is number of particles in size class, m is moment order and dx is range of size class
        
        Parameters
        ----------
        dt : float
            Time increment
        flux : array
            Growth rate of each particle size class
            Array size must be (bins + 1) since this operates on bounds of size classes
        externalRates : array (optional)
            Additional contribution of particles (nucleation or dissolution)
            Array size must be (bins) since this operates on the individual size classes
        order : int (optional)
            Order of moment to conserve (defaults to 1, which is the same as the Update method)
        '''
        #Store current PSD
        self._prevPSD = copy.copy(self.PSD)
        
        netFlux = np.zeros(self.bins + 1)
        netFlux[0] = 0 if flux[0] > 0 else flux[0] * dt * self.PSD[0] * self.PSDsize[0]**order / (self.PSDbounds[1] - self.PSDbounds[0])
        netFlux[-1] = 0 if flux[-1] < 0 else flux[-1] * dt * self.PSD[-1] * self.PSDsize[-1]**order / (self.PSDbounds[-1] - self.PSDbounds[-2])

        #Note - this is not tested yet
        #If flux is going from size class n to n-1, then use size class n (flux <= 0)
        indices = flux[1:-1] <= 0
        netFlux[1:-1][indices] = dt * flux[1:-1][indices] * self.PSD[1:][indices] * self.PSDsize[1:][indices]**order / (self.PSDbounds[2:] - self.PSDbounds[1:-1])[indices]
        under = (netFlux[1:-1] < -self.PSD[1:] * self.PSDsize[1:]**order) & indices
        netFlux[1:-1][under] = -self.PSD[1:][under] * self.PSDsize[1:][under]**order
        
        #if flux is going from size class n-1 to n, then use size class n-1 (flux > 0)
        indices = ~indices
        netFlux[1:-1][indices] = dt * flux[1:-1][indices] * self.PSD[:-1][indices] * self.PSDsize[:-1][indices]**order / (self.PSDbounds[1:-1] - self.PSDbounds[:-2])[indices]
        over = (netFlux[1:-1] > self.PSD[:-1] * self.PSDsize[:-1]**order) & indices
        netFlux[1:-1][over] = self.PSD[:-1][over] * self.PSDsize[1:][over]**order

        self._fv = netFlux
        
        self.PSD += (netFlux[:-1] - netFlux[1:]) / self.PSDsize**order
        if externalRates is not None:
            self.PSD += externalRates
            
        #Set negative frequencies to 0
        self.PSD[self.PSD < 0] = 0
        
    def Moment(self, order):
        '''
        Moment of specified order

        Parameters
        ----------
        order : int
            Order of moment
        '''
        return np.sum(self.PSD * self.PSDsize**order)

    def CumulativeMoment(self, order):
        '''
        Cumulative distribution using moment of specified order

        Parameters
        ----------
        order : int
            Order of moment
        '''
        return np.cumsum(self.PSD * self.PSDsize**order)
        
    def WeightedMoment(self, order, weights):
        '''
        Weighted moment of specified order

        Parameters
        ----------
        order : int
            Order of moment
        weights : array
            Weights to apply to each size class
            Array size of (bins)
        '''
        return np.sum(self.PSD * self.PSDsize**order * weights)

    def CumulativeWeightedMoment(self, order, weights):
        '''
        Weighted moment of specified order

        Parameters
        ----------
        order : int
            Order of moment
        weights : array
            Weights to apply to each size class
            Array size of (bins)
        '''
        return np.cumsum(self.PSD * self.PSDsize**order * weights)

    def ZeroMoment(self):
        '''
        Sum of the PSD
        '''
        return np.sum(self.PSD)
    
    def FirstMoment(self):
        '''
        Length weighted moment
        '''
        return np.sum(self.PSD * self.PSDsize)
        
    def SecondMoment(self):
        '''
        Area weighted moment
        '''
        return np.sum(self.PSD * self.PSDsize**2)
        
    def ThirdMoment(self):
        '''
        Volume weighted moment
        '''
        return np.sum(self.PSD * self.PSDsize**3)
        
    def PlotCurve(self, axes, fill = False, logX = False, logY = False, scale = 1, *args, **kwargs):
        '''
        Plots the PSD as a curve
        
        Parameters
        ----------
        axes : Axes
            Axis to plot on
        fill : bool (optional)
            Will fill area between PSD curve and x-axis (defaults to False)
        logX : bool (optional)
            Whether to set x-axis on log scale (defaults to False)
        logY : bool (optional)
            Whether to set y-axis on log scale (defaults to False)
        scale : float (optional)
            Scale factor for x-axis (defaults to 1)
            Note: this is for grain boundary nucleation where the
                reported precipitate radius differs from the radius
                determined by precipitate curvature
        *args, **kwargs - extra arguments for plotting
        '''
        if fill:
            axes.fill_between(self.PSDsize * scale, self.PSD, np.zeros(len(self.PSD)), *args, **kwargs)
        else:
            axes.plot(self.PSDsize * scale, self.PSD, *args, **kwargs)
        self.setAxes(axes, logX, logY) 

    def PlotDistributionDensity(self, axes, fill = False, logX = False, logY = False, scale = 1, *args, **kwargs):
        '''
        Plots the distribution density as a curve
        Defined as N_i / (R_i+1/2 - R_i-1/2)
        
        Parameters
        ----------
        axes : Axes
            Axis to plot on
        fill : bool (optional)
            Will fill area between PSD curve and x-axis (defaults to False)
        logX : bool (optional)
            Whether to set x-axis on log scale (defaults to False)
        logY : bool (optional)
            Whether to set y-axis on log scale (defaults to False)
        scale : float (optional)
            Scale factor for x-axis (defaults to 1)
            Note: this is for grain boundary nucleation where the
                reported precipitate radius differs from the radius
                determined by precipitate curvature
        *args, **kwargs - extra arguments for plotting
        '''
        if fill:
            axes.fill_between(self.PSDsize * scale, self.PSD, np.zeros(len(self.PSD)), *args, **kwargs)
        else:
            axes.plot(self.PSDsize * scale, self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1]), *args, **kwargs)
        
        #Set x-limits
        if logX:
            if self.min == 0:
                axes.set_xlim([self.PSDbounds[1], self.max])
            else:
                axes.set_xlim([self.min, self.max])
            axes.set_xscale('log')
        else:
            axes.set_xlim([self.min, self.max]) 

        #Set y-limits
        if logY:
            axes.set_ylim([1e-1, np.amax([1.1 * np.max(self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1])), 1])])
            axes.set_yscale('log')
        else:
            axes.set_ylim([0, np.amax([1.1 * np.max(self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1])), 1])])

    def PlotKDE(self, axes, bw_method = None, fill = False, logX = False, logY = False, scale = 1, *args, **kwargs):
        '''
        Plots the kernel density estimation (KDE)
        
        Parameters
        ----------
        axes : Axes
            Axis to plot on
        bw_method : str (optional)
            Method to estimate bandwidth ('scott', 'silverman' or a scalar)
            Defaults to scipy's default for KDE
        fill : bool (optional)
            Will fill area between PSD curve and x-axis (defaults to False)
        logX : bool (optional)
            Whether to set x-axis on log scale (defaults to False)
        logY : bool (optional)
            Whether to set y-axis on log scale (defaults to False)
        scale : float (optional)
            Scale factor for x-axis (defaults to 1)
            Note: this is for grain boundary nucleation where the
                reported precipitate radius differs from the radius
                determined by precipitate curvature
        *args, **kwargs - extra arguments for plotting
        '''
        kernel = sts.gaussian_kde(self.PSDsize, bw_method = bw_method, weights = self.PSD)
        x = np.linspace(self.min, self.max, 1000) if self.linear else np.logspace(self.min, self.max, 1000)    
        y = kernel(x) * self.ZeroMoment() * (self.PSDbounds[1] - self.PSDbounds[0])
        
        if fill:
            axes.fill_between(x * scale, y, np.zeros(len(y)), *args, **kwargs)
        else:
            axes.plot(x * scale, y, *args, **kwargs)
        self.setAxes(axes, logX, logY) 
            
    def PlotHistogram(self, axes, outline = 'outline bins', fill = True, logX = False, logY = False, scale = 1, *args, **kwargs):
        '''
        Plots the PSD as a histogram
        
        Parameters
        ----------
        axes : Axes
            Axis to plot on
        outline : str (optional)
            How to outline the bins ('no outline', 'outline bins', 'outline top')
            Defaults to 'outline bins'
        fill : bool (optional)
            Will fill area between PSD curve and x-axis (defaults to False)
        logX : bool (optional)
            Whether to set x-axis on log scale (defaults to False)
        logY : bool (optional)
            Whether to set y-axis on log scale (defaults to False)
        scale : float (optional)
            Scale factor for x-axis (defaults to 1)
            Note: this is for grain boundary nucleation where the
                reported precipitate radius differs from the radius
                determined by precipitate curvature
        *args, **kwargs - extra arguments for plotting
        '''
        if outline == 'outline bins':
            xCoord, yCoord = np.zeros(1 + 3 * self.bins), np.zeros(1 + 3 * self.bins)
            xCoord[0], xCoord[1::3], xCoord[2::3], xCoord[3::3] = self.PSDbounds[0], self.PSDbounds[:-1], self.PSDbounds[1:], self.PSDbounds[1:]
            yCoord[1::3], yCoord[2::3] = self.PSD, self.PSD
        else:
            xCoord, yCoord = np.zeros(1 + 2 * self.bins), np.zeros(1 + 2 * self.bins)
            xCoord[0], xCoord[1::2], xCoord[2::2] = self.PSDbounds[0], self.PSDbounds[:-1], self.PSDbounds[1:]
            yCoord[1::2], yCoord[2::2] = self.PSD, self.PSD

        if outline != 'no outline':
            axes.plot(xCoord * scale, yCoord, *args, **kwargs)
            if fill:
                axes.fill_between(xCoord * scale, yCoord, np.zeros(len(yCoord)), alpha=0.3, *args, **kwargs)
        else:
            axes.fill_between(xCoord * scale, yCoord, np.zeros(len(yCoord)), *args, **kwargs)
        self.setAxes(axes, logX, logY)

    def PlotCDF(self, axes, logX = False, scale = 1, order = 0, *args, **kwargs):
        '''
        Plots cumulative size distribution
        
        Parameters
        ----------
        axes : Axes
            Axis to plot on
        logX : bool (optional)
            Whether to set x-axis on log scale (defaults to False)
        scale : float (optional)
            Scale factor for x-axis (defaults to 1)
            Note: this is for grain boundary nucleation where the
                reported precipitate radius differs from the radius
                determined by precipitate curvature
        order : int (optional)
            Moment of specified order
        *args, **kwargs - extra arguments for plotting
        '''
        axes.plot(self.PSDsize * scale, self.CumulativeMoment(order) / self.Moment(order), *args, **kwargs)
        self.setAxes(axes, logX, False) 
        axes.set_ylim([0, 1])
        
    def setAxes(self, axes, logX = False, logY = False): 
        '''
        Sets x- and y-axis to linear or log scale
        
        Parameters
        ----------
        axes : Axis
            Axis to plot on
        logX : bool (optional)
            Whether to set x-axis on log scale (defaults to False)
        logY : bool (optional)
            Whether to set y-axis on log scale (defaults to False)
        '''    
        if logX:
            if self.min == 0:
                axes.set_xlim([self.PSDbounds[1], self.max])
            else:
                axes.set_xlim([self.min, self.max])
            axes.set_xscale('log')
        else:
            axes.set_xlim([self.min, self.max])

        #Don't set y limits if the PSD is empty
        if any(self.PSD > 0): 
            if logY:
                axes.set_ylim([1e-1, np.amax([1.1 * np.max(self.PSD), 1])])
                axes.set_yscale('log')
            else:
                axes.set_ylim([0, np.amax([1.1 * np.max(self.PSD), 1])])
        
        axes.set_ylabel('Frequency')
            
                    
        
        