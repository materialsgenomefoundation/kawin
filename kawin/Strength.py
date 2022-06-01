import numpy as np

class StrengthModel:
    def __init__(self):
        self.functionName = 'strengthFactor'

        #Solid solution strengthening parameters
        self.ssweights = {}
        self.ssexp = 1

        #Base strength
        self.sigma0 = 0

        #Precipitate strength
        self.M = None      #Taylor factor
        self.b = None       #Burgers vector
        self.k = None       #Constant (~0.5)
        self.G = None       #Shear modulus
        self.rtrans = None      #Transition radius

    def ssStrength(self, model):
        '''
        Solid solution strength model
        \sigma_ss = \sum{k_i * c_i^n}

        Parameters
        ----------
        model : KWNEuler object
            Model to take composition from

        Returns
        -------
        strength : array of floats
            Solid solution strength contribution over time
        '''
        if len(model.xComp.shape) == 1:
            return self.ssweights[model.elements[0]] * model.xComp**self.ssexp
        else:
            return np.sum([self.ssweights[model.elements[i]]*model.xComp[:,i]**self.ssexp for i in range(len(model.elements))], axis=0)

    def Fterm(self, r):
        '''
        Term for precipitate strength model
        F_i = r_i / r_trans if r_i < r_trans else 1

        Parameters
        ----------
        r : array of floats
            Precipitate radii
        
        Returns
        -------
        weights : array of floats
        '''
        weights = np.zeros(len(r))
        weights[r <= self.rtrans] = r[r <= self.rtrans] / self.rtrans
        weights[r > self.rtrans] = 1
        return weights

    def insertStrength(self, model):
        '''
        Inserts Fterm into the KWNmodel to be solved for

        Parameters
        ----------
        model : KWNEuler object
        '''
        model.addPSDOutput(self.functionName, self.Fterm, 0, normalize='none')

    def _totalAvgR(self, model):
        totalN = np.sum(model.precipitateDensity, axis=0)
        totalN[totalN == 0] = 1
        totalR = np.sum(model.precipitateDensity * model.avgR, axis=0) / totalN
        return totalR

    def _precStrength(self, r, fv, fterm, N):
        '''
        Private function of precipitate strength contribution

        Parameters
        r : float or array
            Average radius
        fv : float or array
            Volume fraction of precipitates
        fterm : float or array
            StrengthModel.Fterm results corresponding to r
        N : float or array
            Precipitate density
            Note: if using an average radius to calculate Fterm rather than a particle size distribution,
            then N should be 1
        '''
        l = self._avgD(r, fv)
        prefactor = self.M / self.b * 2 * self.k * self.G * self.b**2

        return prefactor * (fterm/N)**(3/2) / l

    def precStrength(self, model):
        '''
        Precipitate strength model
        \sigma_pp = M/b * 2*k*G*b^2 F^(3/2) / l
        Where M is Taylor factor, b is burgers vector, G is shear modulus, 
        F is Fterm and l is average precipitate distance

        Parameters
        ----------
        model : KWNEuler object

        Returns
        -------
        strength : array of floats
            Preciptate strength contribution over time
        '''
        psdfuncs = [p['name'] for p in model.PSDfunctions]
        return self._precStrength(self._totalAvgR(model), np.sum(model.betaFrac, axis=0), np.sum(model.PSDoutputs[:,:,psdfuncs.index(self.functionName)], axis=0), np.sum(model.precipitateDensity, axis=0))

    def _avgD(self, r, fv):
        '''
        Private function for average distance

        Parameters
        ----------
        r : float or array
            Average radius
        fv : float or array
            volute fraction of precipitates
        '''
        return r * np.sqrt(2*np.pi / (3*fv))
    
    def avgDist(self, model):
        '''
        Average distance between precipitates

        l = r * \sqrt{2\pi / 3f_v}

        Parameters
        ----------
        model : KWNEuler object

        Returns
        -------
        distance : array of floats
            Average distance vs time
        '''
        return self._avgD(self._totalAvgR(model), np.sum(model.betaFrac, axis=0))

    def estimateRtrans(self, Gm, Gp, b):
        '''
        Estimates transition radius

        Stress around a precipitate = G_m * b / 2R
        Max stress on a precipitate before shearing = G_p / 30
        Transition radius is found when the two equations are equal

        Parameters
        ----------
        Gm : float
            Shear modulus of matrix phase
        Gp : float
            Shear modulus of precipitate phase
        b : float
            Magnitude of Burgers vector
        '''
        self.rtrans = 15 * (Gm/Gp) * b

    def getStrengthUnits(self, strengthUnits = 'Pa'):
        yscale = 1
        ylabel = 'Strength (Pa)'
        if strengthUnits.lower() == 'kpa':
            yscale = 1e3
            ylabel = 'Strength (kPa)'
        elif strengthUnits.lower() == 'mpa':
            yscale = 1e6
            ylabel = 'Strength (MPa)'
        elif strengthUnits.lower() == 'gpa':
            yscale = 1e9
            ylabel = 'Strength (GPa)'
        return yscale, ylabel

    def plotStrengthOverR(self, ax, rBounds, fv, strengthUnits = 'Pa', *args, **kwargs):
        '''
        Plots precipitate strength contribution as a function of radius

        Parameters
        ----------
        ax : Axis
        rBounds : tuple
            Lower and upper bounds for radius to plot
        fv : Volume fraction
        strengthUnits : str
            Units for strength, options are 'Pa', 'kPa', 'MPa' or 'GPa'
        '''
        yscale, ylabel = self.getStrengthUnits(strengthUnits)
        r = np.linspace(rBounds[0], rBounds[1], 100)
        fterm = self.Fterm(r)
        strength = self._precStrength(r, fv, fterm, 1)
        ax.plot(r, strength / yscale)
        ax.set_xlabel('Radius (m)')
        ax.set_xlim(rBounds)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    def plotStrength(self, ax, model, bounds = None, timeUnits = 's', strengthUnits = 'Pa', *args, **kwargs):
        '''
        Plots strength over time

        Parameters
        ----------
        ax : Axis
        model : KWNEuler object
        bounds : tuple or None
            Bounds on time axis (if None, the bounds will automatically be set)
        timeUnits : str
            Units of time to plot in, options are 's' for seconds, 'm' for minutes or 'h' for hours
        strengthUnits : str
            Units for strength, options are 'Pa', 'kPa', 'MPa' or 'GPa'
        '''
        timeScale, timeLabel, bounds = model.getTimeAxis(timeUnits, bounds)
        yscale, ylabel = self.getStrengthUnits(strengthUnits)

        ssstrength = self.ssStrength(model)
        precstrength = self.precStrength(model)
        ax.plot(model.time*timeScale, (self.sigma0 + ssstrength + precstrength) / yscale, *args, **kwargs)
        #ax.plot(model.time*timeScale, (ssstrength) / yscale, *args, **kwargs)
        #ax.plot(model.time*timeScale, (ssstrength + precstrength) / yscale, *args, **kwargs)
        ax.set_xlabel(timeLabel)
        ax.set_xlim(bounds)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_xscale('log')

    