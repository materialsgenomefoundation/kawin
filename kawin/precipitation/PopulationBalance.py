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
        
        self._record = False
        self._recordedBins = None
        self._recordedPSD = None
        self._recordedTime = None

    def reset(self, resetBounds = True):
        '''
        Resets the PSD to 0 and resets bin size and number of bins to original values
        This will remove any size classes that were added since initialization
        '''
        if resetBounds:
            self.min = self.originalMin
            self.max = self.originalMax
            self.bins = self.originalBins
        self.PSDbounds = np.linspace(self.min, self.max, self.bins+1)
        self.PSDsize = 0.5 * (self.PSDbounds[:-1] + self.PSDbounds[1:])
            
        self.PSD = np.zeros(self.bins)

        #Hidden variable for use in KWNEuler when adaptive time stepping is enabled
        #This allows for PSD to revert to its previous value if a time constraint is not met
        self._prevPSD = np.zeros(self.bins)
        self._prevPSDbounds = np.zeros(self.bins+1)

        #Temporary storage for net flux
        #This is used to correct the fluxes once the time step is known
        self._netFlux = None

    def enableRecording(self):
        '''
        Enables recording of particle size distribution per iteration

        The initial data in the recorded bin is t = 0, N_i = 0

        The size of the recorded particle size distribution will be (n x max bins)
            Where n in the number of iterations
            max bins is the maximum number of bins, if the current number is smaller, the rest of the array will be 0
        '''
        self._record = True
        self._recordedBins = np.zeros((1, self.maxBins + 1))
        self._recordedPSD = np.zeros((1, self.maxBins))
        self._recordedTime = np.zeros(1)

    def resetRecordedData(self):
        '''
        If recording, then reset the recorded bins to the original size (starting with t = 0, N_i = 0)
        If not recording, then clear the recorded data
        '''
        if self._record:
            self._recordedBins = np.zeros((1, self.maxBins + 1))
            self._recordedPSD = np.zeros((1, self.maxBins))
            self._recordedTime = np.zeros(1)
        else:
            self._recordedBins = None
            self._recordedPSD = None
            self._recordedTime = None

    def disableRecording(self):
        '''
        Disables recording

        We won't clear the recorded bins here in case the user still wants to grab recorded data
        '''
        self._record = False

    def setRecording(self, record = True):
        '''
        Wrapper around enable and disable recording
        '''
        if record:
            self.enableRecording()
        else:
            self.disableRecording()

    def removeRecordedData(self):
        '''
        Removes recorded data
        '''
        self._recordedBins = None
        self._recordedPSD = None
        self._recordedTime = None

    def record(self, time):
        '''
        Adds current PSD data to recorded arrays

        TODO: Make sure this works when adaptive bins is False
        '''
        if self._record:
            maxBins = self.maxBins if self._adaptiveBinSize else self.bins
            self._recordedBins = np.pad(self._recordedBins, ((0, 1), (0, maxBins+1 - self._recordedBins.shape[1])))
            self._recordedPSD = np.pad(self._recordedPSD, ((0, 1), (0, maxBins - self._recordedPSD.shape[1])))
            self._recordedTime = np.pad(self._recordedTime, (0,1))
            self._recordedBins[-1][:self.PSDbounds.shape[0]] = self.PSDbounds
            self._recordedPSD[-1][:self.PSD.shape[0]] = self.PSD
            self._recordedTime[-1] = time

    def saveRecordedPSD(self, filename, compressed = True):
        '''
        Saves recorded data into npz format

        Note: If recording is disabled, then this function will do nothing since 
              there is nothing to save anyways

        Parameters
        ----------
        filename : str
            File name to save to
        compressed : bool (optional)
            Whether to save as in compressed format (defaults to True)
        '''
        if self._record:
            if compressed:
                np.savez_compressed(filename, time = self._recordedTime, bins = self._recordedBins, PSD = self._recordedPSD)
            else:
                np.savez(filename, time = self._recordedTime, bins = self._recordedBins, PSD = self._recordedPSD)

    def loadRecordedPSD(self, filename):
        '''
        Loads recorded PSD
        '''
        data = np.load(filename)
        self._record = True
        self._recordedTime = data['time']
        self._recordedBins = data['bins']
        self._recordedPSD = data['PSD']

    def _grabPSDfromIndex(self, index):
        '''
        Returns PSD bounds, PSD bins and PSD from recorded data based off index

        Since the number of bins is likely less than the max, we want to grab only the non-zero indices
        TODO: two concerns
            1) this may remove the last 1 bins (this may be okay since we add new bins once the
                list bins has at least 1 particle), so the last bin would be 0 anyways
        '''
        nonzero = len(np.nonzero(self._recordedBins[index])[0])
        if nonzero == 0:
            PSDbounds = np.linspace(self.originalMin, self.originalMax, self.originalBins+1)
            PSDsize = 0.5 * (PSDbounds[1:] + PSDbounds[:-1])
            PSD = np.zeros(self.originalBins)
        else:
            PSDbounds = self._recordedBins[index,:nonzero]
            PSD = self._recordedPSD[index,:nonzero-1]
            PSDsize = 0.5 * (PSDbounds[1:] + PSDbounds[:-1])
        bins = len(PSD)
        minBound, maxBound = np.amin(PSDbounds), np.amax(PSDbounds)
        return PSDbounds, PSD, PSDsize, bins, minBound, maxBound

    def setPSDtoRecordedTime(self, time):
        '''
        Sets particle size distribution to specific time if recorded

        Parameter
        ---------
        time : float
            Time to load PSD from, will load to nearest time available
        '''
        if self._record:
            if time <= self._recordedTime[0]:
                print('Input time is lower than smallest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[0]))
                self.PSDbounds, self.PSD, self.PSDsize, self.bins, self.min, self.max = self._grabPSDfromIndex(0)
            elif time >= self._recordedTime[-1]:
                print('Input time is larger than longest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[-1]))
                self.PSDbounds, self.PSD, self.PSDsize, self.bins, self.min, self.max = self._grabPSDfromIndex(-1)
            else:
                #Upper and lower PSD
                #Note: horrible naming convention here
                #    Upper PSD refers to the PSD just after time
                #    Lower PSD refers to the PSD just before time
                #This does NOT refer to the PSD with the larger or smaller number of bins
                uind = np.argmax(self._recordedTime > time)
                lind = uind - 1

                utime, ltime = self._recordedTime[uind], self._recordedTime[lind]
                uPSDbounds, uPSD, uPSDsize, ubins, umin, umax = self._grabPSDfromIndex(uind)
                lPSDbounds, lPSD, lPSDsize, lbins, lmin, lmax = self._grabPSDfromIndex(lind)

                #Interpolate from lower PSD to upper PSD using bounds of larger PSD
                #This will account for all possible cases if the PSD size classes change
                #This is done by pretending we're calling changeSizeClasses
                #    Where we resize the PSD with the smaller number of bins to have the same bins as the larger PSD
                #    And correct for the possible change in number density
                if ubins >= lbins:
                    #Resize lower PSD to upper PSD
                    oldV = np.sum(lPSD * lPSDsize**3)
                    distDen = lPSD / (lPSDbounds[1:] - lPSDbounds[:-1])
                    rOld = 0.5 * (lPSDbounds[1:] + lPSDbounds[:-1])
                    lPSD = np.interp(uPSDsize, rOld, distDen, left=0, right=0) * (uPSDbounds[1:] - uPSDbounds[:-1])
                    newV = np.sum(lPSD * uPSDsize**3)
                    if newV != 0:
                        lPSD *= oldV / newV
                    else:
                        lPSD = np.zeros(ubins)
                    
                else:
                    #Resize upper PSD to lower PSD
                    oldV = np.sum(uPSD * uPSDsize**3)
                    distDen = uPSD / (uPSDbounds[1:] - uPSDbounds[:-1])
                    rOld = 0.5 * (uPSDbounds[1:] + uPSDbounds[:-1])
                    uPSD = np.interp(lPSDsize, rOld, distDen, left=0, right=0) * (lPSDbounds[1:] - lPSDbounds[:-1])
                    uPSDbounds = lPSDbounds
                    newV = np.sum(uPSD * lPSDsize**3)
                    if newV != 0:
                        uPSD *= oldV / newV
                    else:
                        uPSD = np.zeros(lbins)

                #Now that the bin sizes are the same, we can just interpolate the PSD
                self.PSDbounds = uPSDbounds
                self.PSDsize = 0.5 * (self.PSDbounds[1:] + self.PSDbounds[:-1])
                self.PSD = (uPSD - lPSD) * (time - ltime) / (utime - ltime) + lPSD
                self.bins = len(self.PSDsize)
                self.min, self.max = np.amin(self.PSDbounds), np.amax(self.PSDbounds)

    def setAdaptiveBinSize(self, adaptive):
        '''
        For Euler implementation, sets whether to change the bin size when 
        the number of filled bins > maxBins or < minBins

        If False, the bins will still be if nucleated particles are greater than the max bin size
        and bins will still be added when the last bins starts to fill (but this will not change the bin size)
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

        NOTE: this appears to be unused
            (this was used in the previous KWNEuler implementation when the PSD could change within an iteration)
            (now it changes between iterations, so we don't need to revert back if something goes wrong)
        '''
        self.PSD = copy.copy(self._prevPSD)
        self.PSDbounds = copy.copy(self._prevPSDbounds)
        self.PSDsize = 0.5 * (self.PSDbounds[1:] + self.PSDbounds[:-1])
        self.bins = len(self.PSD)
        self.min, self.max = self.PSDbounds[0], self.PSDbounds[-1]

    def changeSizeClasses(self, cMin, cMax, bins = None, resetPSD = False):
        '''
        Changes the size classes and resets the PSD

        This is done by linear interpolation of the previous bins and PSD
        And interpolating to the new bins and PSD
        Due to differences in bin size (thus resolution of the PSD), the number density
            could be a little different. To correct for this, we get the 3rd moment of the
            previous PSD and the new PSD, and correct the new PSD to have the same 3rd moment
        
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
        Adds an additional number of size classes to end of distribution

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

    def adjustSizeClassesEuler(self, checkDissolution = False):
        '''
        1) adds some bins to the end of the PSD if the last bin has at least 1 precipitate
            Number of bins is 1/4 of the original number of bins
        2) If adaptive bin size is enabled, then two checks
            2a) if number of bins > max bins, then resize to have the number of bins be the minimum
            2b) if checking dissolution and number of filled bins < 1/2 min bins,
                then resize to last filled bin with the number of bins being the maximum

        Parameters
        ----------
        checkDissolution : bool
            Whether to check if the PSD is getting smaller and resize accordingly

        Returns
        -------
        change : bool
            When the number of bins changed
        newIndices : int or None
            The number of bins added to the PSD
            If the size of the bins changed, then this is None to indicate that resizing occured
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

    def getDissolutionIndex(self, maxDissolution, minIndex = 0):
        '''
        Finds indices when the volume fraction of particles below this index is 
        within the maximum amount (fraction-wise) that the PSD is allowed to dissolve

        So find R_max where int(0, R_max, R^3 * dr) < maxDissolution * int(0, infinity, R^3 * dr)
        The index is the correspoinding index to R_max

        Parameters
        ----------
        maxDissolution : float
            Max fraction allowed to dissolve
        minIndex : int
            Minimum index which below, all particles are allowed to dissolve
            Upper limit on dissolution index

        Returns
        -------
        max of [dissolution index, minIndex]
        '''
        dissFrac = maxDissolution * self.ThirdMoment()
        dissIndex = np.argmax(self.CumulativeMoment(3) > dissFrac) - 1
        if dissIndex < 0:
            dissIndex = 0
        return np.amax([np.argmax(self.CumulativeMoment(3) > dissFrac), minIndex])
        

    def getDTEuler(self, currDT, growth, dissolutionIndex, maxBinRatio = 0.4):
        '''
        Calculates time interval for Euler implementation
            dt < dR / (2 * growth rate)
        This ensures that at most, only half of particles in one size class can go to another

        Also finds dt such that the max delta in growth rate is 0.4 dR
            We could use 0.5 dR which is the upper limit
                (for a given bin, the max change in density would remove all particles, with 0.5 getting smaller and 0.5 getting bigger)
            But 0.4 dR should be slightly more stable

        TODO: allow variable ratio - this will make it more flexible for testing different time step constraints

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
        maxBinRatio : float (optional)
            Max ratio of particles in bin allowed to move to a nearby bin
            Default is 0.4
        '''
        self.maxRatio = maxBinRatio
        growthFilter = growth[dissolutionIndex:-1][self.PSD[dissolutionIndex:] > 0]

        if len(growthFilter) == 0:
            return currDT
        else:
            if np.amax(np.abs(growthFilter)) == 0:
                return currDT
            else:
                return self.maxRatio * (self.PSDbounds[1] - self.PSDbounds[0]) / np.amax(np.abs(growthFilter))
    
    def getdXdtEuler(self, flux, nucRate, nucRadius, psd):
        '''
        dn_i/dt = d(G*n)/dr + nucRate

        d(G*n)/dr is calculated from two conditions
        For positive growth rates - d(G*n)/dr|_i = n_i * flux_i / dr
        For negative growth rates - d(G*n)/dr|_i = n_(i-1) * flux_i / dr
        TODO : check that the two equations above represent the implementation

        Parameters
        ----------
        flux : numpy array (bins+1)
            Growth rate of particles in m/s
        nucRate : float
            Nucleation rate in #/m^3/s
        nucRadius : float
            Nucleation radius in m
        psd : numpy array (bins)
            Particle size distribution with number density #/m3

        Returns
        -------
        dXdt (bins) - corresponds to dn_i/dt
        '''
        self._netFlux = np.zeros(self.bins+1)
        fluxSign = np.sign(flux)
        fluxSign[fluxSign == -1] = 0
        dR = self.PSDbounds[1:] - self.PSDbounds[:-1]
        self._netFlux[:-1] += flux[:-1] * psd * (1-fluxSign[:-1]) / dR
        self._netFlux[1:] += flux[1:] * psd * fluxSign[1:] / dR

        dXdt = (self._netFlux[:-1] - self._netFlux[1:])

        #Find size class for nucleated particles
        nRad = np.argmax(self.PSDbounds > nucRadius) - 1
        dXdt[nRad] += nucRate

        return dXdt
    
    def correctdXdtEuler(self, dt, flux, nucRate, nucRadius, psd):
        '''
        Given dt, correct the net flux so PSD will not be negative
            Essentially, the total number of particles leaving a bin should be less than or equal to the number of particles in the bin

        For any term in netFlux where netFlux_i*dt that is larger than n_i
            We correct netFlux such that netFlux_i*dt = n_i

        Parameters
        ----------
        dt : float
            time step
        flux : numpy array (bins+1)
            Growth rate of particles in m/s
        nucRate : float
            Nucleation rate in #/m^3/s
        nucRadius : float
            Nucleation radius in m
        psd : numpy array (bins)
            Particle size distribution with number density #/m3

        Returns
        -------
        dXdt (bins) - corresponds to dn_i/dt corrected to avoid negative bins
        '''
        indBelow = self._netFlux[1:-1]*dt < -psd[1:]
        self._netFlux[1:-1][indBelow] = -psd[1:][indBelow] / dt
        indAbove = self._netFlux[1:-1]*dt > psd[:-1]
        self._netFlux[1:-1][indAbove] = psd[:-1][indAbove] / dt

        dXdt = (self._netFlux[:-1] - self._netFlux[1:])

        #Find size class for nucleated particles
        nRad = np.argmax(self.PSDbounds > nucRadius) - 1
        dXdt[nRad] += nucRate

        return dXdt
    
    def UpdatePBMEuler(self, time, newN):
        '''
        Updates PBM with new values

        Parameters
        ----------
        time : float
            New time
        newN : numpy array
            New number density
        '''
        self.PSD = newN
        self.PSD[self.PSD < 1] = 0
        self.record(time)

    def MomentFromN(self, N, order):
        '''
        Given arbtrary PSD, return moment

        Parameters
        ----------
        N : numpy array
            PSD / number density
        order : float
            Moment order
        '''
        return np.sum(N * self.PSDsize**order)
    
    def CumulativeMomentFromN(self, N, order):
        '''
        Given arbtrary PSD, return cumulative moment (from 0 to max)

        Parameters
        ----------
        N : numpy array
            PSD / number density
        order : float
            Moment order
        '''
        return np.cumsum(N * self.PSDsize**order)
    
    def WeightedMomentFromN(self, N, order, weights):
        '''
        Given arbtrary PSD, return weighted moment

        Parameters
        ----------
        N : numpy array
            PSD / number density
        order : float
            Moment order
        weights : numpy array
            Weights for each bin
        '''
        return np.sum(N * self.PSDsize**order * weights)
    
    def CumulativeWeightedMomentFromN(self, N, order, weights):
        '''
        Given arbtrary PSD, return cumulative weighted moment (from 0 to max)

        Parameters
        ----------
        N : numpy array
            PSD / number density
        order : float
            Moment order
        weights : numpy array
            Weights for each bin
        '''
        return np.cumsum(self.PSD * self.PSDsize**order * weights)
    
    def ZeroMomentFromN(self, N):
        '''
        Sum of N
        '''
        return self.MomentFromN(N, 0)
    
    def FirstMomentFromN(self, N):
        '''
        Length weighted moment of N
        '''
        return self.MomentFromN(N, 1)
        
    def SecondMomentFromN(self, N):
        '''
        Area weighted moment of N
        '''
        return self.MomentFromN(N, 2)
        
    def ThirdMomentFromN(self, N):
        '''
        Volume weighted moment of N
        '''
        return self.MomentFromN(N, 3)
        
    def Moment(self, order):
        '''
        Moment of specified order

        Parameters
        ----------
        order : int
            Order of moment
        '''
        return self.MomentFromN(self.PSD, order)

    def CumulativeMoment(self, order):
        '''
        Cumulative distribution using moment of specified order

        Parameters
        ----------
        order : int
            Order of moment
        '''
        return self.CumulativeMomentFromN(self.PSD, order)
        
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
        return self.WeightedMomentFromN(self.PSD, order, weights)

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
        return self.CumulativeWeightedMomentFromN(self.PSD, order, weights)

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
            scale = scale * np.ones(len(self.PSDsize))

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
            scale = scale * np.ones(len(self.PSDsize))

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
            scale = scale * np.ones(len(x))
        
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
            scale = scale * np.ones(len(xCoord))

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
            scale = scale * np.ones(len(self.PSDsize))

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
            
                    
        
        