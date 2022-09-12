import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import copy

class PopulationBalanceModel:
    '''
    Class for handling particle size distributions (PSD)
        This include time evolution, moments and graphing

    NOTE: more formally, the PSD is defined such that the number of particles of size R in a volume
        is the integral of the PSD from R-1/2 to R+1/2
        For the discrete PSD, n_i = PSD_i * (R+1/2 - R-1/2)
        Here, we just store the PSD as the number of particles (n_i), but the formulation for 
        growth rates and moments are the same since the factor (R+1/2 - R-1/2) cancels out

    Parameters
    ----------
    cMin : float (optional)
        Lower bound of PSD (defaults at 1e-10)
    cMax : float (optional)
        Upper bound of PSD (defaults at 1e-9)
    bins : int (optional)
        Initial number of bins (defaults at 150)
    minBins : int (optional)
        Minimum number of bins over an order of magnitude (defaults at 100)
    maxBins : int (optional)
        Maximum number of bins over an order of magnitude (defaults at 200)

    Attributes
    ----------
    originalMin : float
        Minimum bin size initially set
    min : float
        Current minimum bin size (this will almost always be equal to originalMin)
    originalMax : float
        Maximum bin size initially set
    max : float
        Current maximum bin size
    bins : int
        Default number of bins
    minBins : int
        Minimum number of allowed bins
    maxBins : int
        Maximum number of allowed bins
    PSD : array
        Particle size distribution
    PSDsize : array
        Average radius of each PSD size class
    PSDbounds : array
        Radius at the bounds of each size class - length of array is len(PSD)+1
    '''
    def __init__(self, cMin = 1e-10, cMax = 1e-9, bins = 150, minBins = 100, maxBins = 200):
        self.originalMin = cMin
        self.originalMax = np.amax([10*self.originalMin, cMax])
        self.min = self.originalMin
        self.max = self.originalMax

        self.originalBins = bins
        self.setBinConstraints(bins, minBins, maxBins)
        
        self.reset()

        self._adaptiveBinSize = True

    def setAdaptiveBinSize(self, adaptive):
        '''
        For Euler implementation, sets whether to change the bin size when 
        the number of filled bins > maxBins or < minBins

        If False, the bins will still change if nucleated particles are greater than the max bin size
        and bins will still be added when the last bins starts to fill (although this will not change the bin size)
        '''
        self._adaptiveBinSize = adaptive

    def setBinConstraints(self, bins = 150, minBins = 100, maxBins = 200):
        '''
        Sets constraints for minimum and maxinum number of bins over an order of magnitude

        All bins will be overridden to an even number
        Minimum number of bins will be overridden to be at most half of maximum bins
        Default bins will be overridden to be halfway between min and max bins if out of range
        '''
        self.minBins = minBins
        self.maxBins = maxBins
        self.bins = bins

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

    def LoadDistributionFunction(self, function):
        '''
        Creates a particle size distribution from a function

        Parameters
        ----------
        function : function
            Takes in R and returns density
        '''
        self.PSD = function(self.PSDsize)

    def createBackup(self):
        '''
        Stores current PSD and PSDbounds
        '''
        self._prevPSD = copy.copy(self.PSD)
        self._prevPSDbounds = copy.copy(self.PSDbounds)

    def revert(self):
        '''
        Reverts to previous PSD and PSDbounds
        '''
        self.PSD = copy.copy(self._prevPSD)
        self.PSDbounds = copy.copy(self._prevPSDbounds)
        self.PSDsize = 0.5 * (self.PSDbounds[1:] + self.PSDbounds[:-1])
        self.bins = len(self.PSD)
        self.min, self.max = self.PSDbounds[0], self.PSDbounds[-1]

    def reset(self, resetBounds = True):
        '''
        Resets the PSD to 0
        This will remove any size classes that were added since initialization
        '''
        if resetBounds:
            self.min = self.originalMin
            self.max = self.originalMax
            self.bins = self.originalBins
        self.PSDbounds = np.linspace(self.min, self.max, self.bins+1)
        self.PSDsize = 0.5 * (self.PSDbounds[:-1] + self.PSDbounds[1:])
            
        self.PSD = np.zeros(self.bins)

        #Hidden variable for use in KWNEuler when determining composition assuming no diffusion in precipitate
        #Represents d(PSD)/dr * growth rate * dt
        #I would like this variable to be in KWNEuler, but this way is much easier
        self._fv = np.zeros(self.bins + 1)

        #Hidden variable for use in KWNEuler when adaptive time stepping is enabled
        #This allows for PSD to revert to its previous value if a time constraint is not met
        self._prevPSD = np.zeros(self.bins)
        self._prevPSDbounds = np.zeros(self.bins+1)

    def changeSizeClasses(self, cMin, cMax, bins = None, resetPSD = False):
        '''
        Changes the size classes and resets the PSD
        
        Parameters
        ----------
        cMin : float
            Lower bound of PSD
        cMax : float
            Upper bound of PSD
        bins : int
            Number of bins
        resetPSD : bool (optional)
            Whether to reset the PSD (defaults to False)
        '''
        self.bins = self.bins if bins is None else bins
        self.min = cMin
        self.max = np.amax([10*self.min, cMax])

        if resetPSD:
            self.reset()
        else:
            oldV = self.ThirdMoment()
            distDen = self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1])
            rOld = 0.5 * (self.PSDbounds[1:] + self.PSDbounds[:-1])
            self.reset(False)
            self.PSD = np.interp(self.PSDsize, rOld, distDen) * (self.PSDbounds[1:] - self.PSDbounds[:-1])
            newV = self.ThirdMoment()
            if newV != 0:
                self.PSD *= oldV / newV
            else:
                self.PSD = np.zeros(self.bins)

    def addSizeClasses(self, bins = 1):
        '''
        Adds an additional size class to end of distribution

        Parameters
        ----------
        bins : int
            Number of bins to add
        '''
        self.bins += bins
        self.PSD = np.append(self.PSD, np.zeros(bins))

        self.max += bins * (self.PSDbounds[1] - self.PSDbounds[0])
        self.PSDbounds = np.linspace(self.min, self.max, self.bins+1)
        self.PSDsize = 0.5 * (self.PSDbounds[:-1] + self.PSDbounds[1:])

    def getDTEuler(self, currDT, growth, maxDissolution, startIndex):
        '''
        Calculates time interval for Euler implementation
            dt < dR / (2 * growth rate)
        This ensures that at most, only half of particles in one size class can go to another

        Parameters
        ----------
        currDT : float
            Current time interval, will be returned if it's smaller than what's given by the contraint
        growth : array of floats
            Growth rate, must have lenth of bins+1 (or length of PSDbounds)
        maxDissolution : float
            Maximum volume allowed to dissolve
        startIndex : int
            First index to look at for growth rate, all indices below startIndex will be ignored
        '''
        dissFrac = maxDissolution * self.ThirdMoment()
        dissIndex = np.amax([np.argmax(self.CumulativeMoment(3) > dissFrac), startIndex])
        growthFilter = growth[dissIndex:-1][self.PSD[dissIndex:] > 0]
        #if len(growthFilter) == 0 or np.amax(growthFilter) < 0:
        if len(growthFilter) == 0:
            return currDT
        else:
            if np.amax(np.abs(growthFilter)) == 0:
                return currDT
            else:
                return np.amin([currDT, (self.PSDbounds[1] - self.PSDbounds[0]) / (2 * np.amax(np.abs(growthFilter)))])

    def adjustSizeClassesEuler(self, checkDissolution = False):
        '''
        Adds a size class if last class in PBM is filled
        Changes length of size classes based off number of allowed bins
        '''
        change = False
        newIndices = None
        if self.PSD[-1] > 1:
            #print('adding bins')
            newIndices = self.bins
            self.addSizeClasses(int(self.originalBins/4))
            change = True

        if self._adaptiveBinSize:
            if self.bins > self.maxBins:
                #print('reducing bins')
                self.changeSizeClasses(self.PSDbounds[0], self.PSDbounds[-1], self.minBins)
                change = True
                newIndices = None
            elif checkDissolution and self.PSDbounds[-1] > 10*self.PSDbounds[0]:
                if any(self.PSD > 1) and np.amax(self.PSDsize[self.PSD > 1]) < self.PSDsize[int(self.minBins/2)]:
                    #print('splitting bins')
                    self.changeSizeClasses(self.PSDbounds[0], np.amax(self.PSDbounds[1:][self.PSD > 1]), self.maxBins)
                    change = True
                    newIndices = None
        return change, newIndices
        
    def Normalize(self):
        '''
        Normalizes the PSD to 1
        '''
        total = self.WeightedMoment(0, self.PSDbounds[1:] - self.PSDbounds[:-1])
        self.PSD /= total
            
    def NormalizeToMoment(self, order = 0):
        '''
        Normalizes the PSD with so that the moment of specified order will be 1
        
        Parameters
        ----------
        order : int (optional)
            Order of moment that PSD will be normalized to (defaults to 0)
            Using zeroth order will normalize the PSD so the sum of PSD will be 1
        '''
        total = self.Moment(order)
        self.PSD /= total

    def Nucleate(self, amount, radius):
        '''
        Adds nucleated particles to PSD given radius and amount of particles

        Parameters
        ----------
        amount : float
            Amount of nucleated particles
        radius : float
            Radius of nucleated particles
        '''
        if amount < 1:
            return False
            
        change = False

        #Find size class for nucleated particles
        nRad = np.argmax(self.PSDbounds > radius) - 1

        #If radius is larger than length scale of PBM, adjust PBM such that radius is towards the beginning
        if nRad == -1 and radius > 0:
            #print('adding nucleated bins')
            self.changeSizeClasses(self.PSDbounds[0], 5 * radius, self.originalBins)
            nRad = np.argmax(self.PSDbounds > radius)
            change = True
        self.PSD[nRad] += amount
        return change
        
    def UpdateEuler(self, dt, flux):
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
        '''
        netFlux = np.zeros(self.bins + 1)

        #Array of 0 (flux <= 0), 1 (flux > 0)
        fluxSign = np.sign(flux)
        fluxSign[fluxSign == -1] = 0

        #If flux is negative (from class n to n-1), then take from size class n
        #If flux is positive, then take from size class n-1
        netFlux[:-1] += dt * flux[:-1] * self.PSD * (1-fluxSign[:-1]) / (self.PSDbounds[1:] - self.PSDbounds[:-1])
        netFlux[1:] += dt * flux[1:] * self.PSD * fluxSign[1:] / (self.PSDbounds[1:] - self.PSDbounds[:-1])

        #Brute force stability so PSD is never negative
        #If the time step is determined from getDTEuler, this should be unnecessary
        netFlux[1:-1][netFlux[1:-1] < -self.PSD[1:]] = -self.PSD[1:][netFlux[1:-1] < -self.PSD[1:]]
        netFlux[1:-1][netFlux[1:-1] > self.PSD[:-1]] = self.PSD[:-1][netFlux[1:-1] > self.PSD[:-1]]

        self._fv = netFlux
        
        self.PSD += (netFlux[:-1] - netFlux[1:])
        
        #Adjust size classes and return True if the size classes had changed
        change, newIndices = self.adjustSizeClassesEuler(all(flux<0))
            
        #Set negative frequencies to 0
        self.PSD[self.PSD < 1] = 0

        return change, newIndices

    def UpdateLagrange(self, dt, flux):
        '''
        Updates bounds of size classes with given growth rate
        Fluxes of particles between size classes is d(Gn)/dx,
        however, keeping the number of particles in each size class the same,
        the bounds of the size classes can be updated by r_i = v_i * dt

        Parameters
        ----------
        dt : float
            Time increment
        flux : array
            Growth rate of each particle size class
            Array size must be (bins + 1) since this operates on bounds of size classes
        '''
        self._prevPSDbounds = copy.copy(self.PSDbounds)
        self.PSDbounds += flux * dt
        self.PSDsize = 0.5 * (self.PSDbounds[1:] + self.PSDbounds[:-1])
        
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
        return self.Moment(0)
    
    def FirstMoment(self):
        '''
        Length weighted moment
        '''
        return self.Moment(1)
        
    def SecondMoment(self):
        '''
        Area weighted moment
        '''
        return self.Moment(2)
        
    def ThirdMoment(self):
        '''
        Volume weighted moment
        '''
        return self.Moment(3)
        
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
        if hasattr(scale, '__len__'):
            scale = np.interp(self.PSDsize, self.PSDbounds, scale)
        else:
            scale = scale * np.ones(self.PSDsize)

        if fill:
            axes.fill_between(self.PSDsize * scale, self.PSD, np.zeros(len(self.PSD)), *args, **kwargs)
        else:
            axes.plot(self.PSDsize * scale, self.PSD, *args, **kwargs)
        self.setAxes(axes, scale, logX, logY) 

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
        if hasattr(scale, '__len__'):
            scale = np.interp(self.PSDsize, self.PSDbounds, scale)
        else:
            scale = scale * np.ones(self.PSDsize)

        if fill:
            axes.fill_between(self.PSDsize * scale, self.PSD, np.zeros(len(self.PSD)), *args, **kwargs)
        else:
            axes.plot(self.PSDsize * scale, self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1]), *args, **kwargs)
        
        #Set x-limits
        if logX:
            if self.min == 0:
                axes.set_xlim([self.PSDbounds[1]*scale[0], self.max*scale[-1]])
            else:
                axes.set_xlim([self.min*scale[0], self.max*scale[-1]])
            axes.set_xscale('log')
        else:
            axes.set_xlim([self.min*scale[0], self.max*scale[-1]]) 

        #Set y-limits
        if logY:
            axes.set_ylim([1e-1, np.amax([1.1 * np.amax(self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1])), 1])])
            axes.set_yscale('log')
        else:
            axes.set_ylim([0, np.amax([1.1 * np.amax(self.PSD / (self.PSDbounds[1:] - self.PSDbounds[:-1])), 1])])

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
        x = np.linspace(self.min, self.max, 1000)   
        y = kernel(x) * self.ZeroMoment() * (self.PSDbounds[1] - self.PSDbounds[0])

        if hasattr(scale, '__len__'):
            scale = np.interp(x, self.PSDbounds, scale)
        else:
            scale = scale * np.ones(x)
        
        if fill:
            axes.fill_between(x * scale, y, np.zeros(len(y)), *args, **kwargs)
        else:
            axes.plot(x * scale, y, *args, **kwargs)
        self.setAxes(axes, scale, logX, logY) 
            
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

        if hasattr(scale, '__len__'):
            scale = np.interp(xCoord, self.PSDbounds, scale)
        else:
            scale = scale * np.ones(xCoord)

        if outline != 'no outline':
            axes.plot(xCoord * scale, yCoord, *args, **kwargs)
            if fill:
                axes.fill_between(xCoord * scale, yCoord, np.zeros(len(yCoord)), alpha=0.3, *args, **kwargs)
        else:
            axes.fill_between(xCoord * scale, yCoord, np.zeros(len(yCoord)), *args, **kwargs)
        self.setAxes(axes, scale, logX, logY)

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
        if hasattr(scale, '__len__'):
            scale = np.interp(self.PSDsize, self.PSDbounds, scale)
        else:
            scale = scale * np.ones(self.PSDsize)

        axes.plot(self.PSDsize * scale, self.CumulativeMoment(order) / self.Moment(order), *args, **kwargs)
        self.setAxes(axes, scale, logX, False) 
        axes.set_ylim([0, 1])
        
    def setAxes(self, axes, scale = 1, logX = False, logY = False): 
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
                axes.set_xlim([self.PSDbounds[1]*scale[0], self.max*scale[-1]])
            else:
                axes.set_xlim([self.min*scale[0], self.max*scale[-1]])
            axes.set_xscale('log')
        else:
            axes.set_xlim([self.min*scale[0], self.max*scale[-1]])

        #Don't set y limits if the PSD is empty
        if any(self.PSD > 0): 
            if logY:
                axes.set_ylim([1e-1, np.amax([1.1 * np.amax(self.PSD), 1])])
                axes.set_yscale('log')
            else:
                axes.set_ylim([0, np.amax([1.1 * np.amax(self.PSD), 1])])
        
        axes.set_ylabel('Frequency')
            
                    
        
        