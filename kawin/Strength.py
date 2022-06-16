import numpy as np

class StrengthModel:
    def __init__(self):
        self.functionName = 'strengthFactor'

        #Solid solution strengthening parameters
        self.ssweights = {}
        self.ssexp = 1

        #Base strength
        self.sigma0 = None

        #Precipitate strength factors
        self.coherencyEffect = False        #Coherency effect
        self.modulusEffect = False          #Modulus effect
        self.APBEffect = False              #Anti-phase boundary effect
        self.SFEffect = False               #Stacking fault energy effect
        self.IFEffect = False               #Interfacial energy effect
        self.orowanEffect = False           #Non-shearable (Orowan) mechanism

        #Parameters for precipitate strength
        self.nu = None              #Poisson ratio
        self.b = None               #Burgers vector
        self.bp = None              #Burgers vector for precipitate
        self.theta = None           #Angle between dislocation and burgers vector (0 for screw, pi/2 for edge)
        self.G = None               #Shear modulus of matrix
        self.Gp = None              #Shear modulus of precipitate
        self.ri = None              #Dislocation radius
        self.delta = None           #Lattice misfit
        self.eps = None             #Strain from lattice misfit
        self.J = 1                  #Correction factor for strong coherency effect
        self.w1 = 0.0722            #Factor for modulus effect (0.0175 - 0.0722)
        self.w2 = 0.81              #Factor for modulus effect (0.81 +/- 0.09)
        self.yAPB = None            #Anti-phase boundary energy
        self.s = None               #Number of dislocations per group for weak APB effect
        self.beta = 1               #Factor for trailing dislocation for weak APB effect
        self.V = 2.8                #Factor for strong APB effect
        self.ySFM = None            #Stacking fault energy for matrix
        self.ySFP = None            #Stacking fault energy for precipitate
        self.gamma = None           #Interfacial energy of precipitate

    def epsMisfit(self):
        '''
        Strain from lattice misfit
        '''
        self.eps = (1/3) * (1+self.nu) / (1-self.nu) * self.delta

    def T(self, theta, r0):
        '''
        Dislocation line tension
        '''
        return self.G*self.b**2 / (4*np.pi) * (1 + self.nu - 3*self.nu*np.sin(theta)**2) / (1 - self.nu) * np.log(r0 / self.ri)

    @property
    def Tedge(self):
        '''
        Edge dislocation line tension assuming ln(r0/ri) = 2pi
        '''
        return self.G * self.b**2 / 4

    @property
    def Tscrew(self):
        '''
        Screw dislocation line tension assuming ln(r0/ri) = 2pi
        '''
        return self.G * self.b**2

    @property
    def Tgeneral(self):
        '''
        Approximation of line tension, combining edge and screw dislocations
        '''
        return 0.5 * self.G * self.b**2

    def coherencyWeakEdge(self, r, Ls, r0):
        '''
        Coherency effect for edge dislocation on weak and shearable particles
        '''
        return np.sqrt((592 / 35) * self.G**3 * self.b * self.eps**3 * r**3 / (Ls**2 * self.T(np.pi/2, r0)))

    def coherencyWeakScrew(self, r, Ls, r0):
        '''
        Coherency effect for screw dislocation on weak and shearable particles
        '''
        return np.sqrt((9/5) * self.G**3 * self.b * self.eps**3 * r**3 / (Ls**2 * self.T(0, r0)))

    def coherencyStrongEdge(self, r, Ls, r0):
        '''
        Coherency effect for edge dislocation on strong and shearable particles
        '''
        return np.sqrt(2) * np.power(3, 3/8) * self.J / Ls * np.power((self.T(np.pi/2, r0)**3 * self.G * self.eps * r) / self.b**3, 1/4)

    def coherencyStrongScrew(self, r, Ls, r0):
        '''
        Coherency effect for screw dislocation on strong and shearable particles
        '''
        return 2 * self.J / Ls * np.power((self.T(0, r0)**3 * self.G * self.eps * r) / self.b**3, 1/4)

    def modulusWeakEdge(self, r, Ls, r0):
        '''
        Modulus effect for edge dislocation on weak and shearable particles
        '''
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(np.pi/2, r0)), 3/2)

    def modulusWeakScrew(self, r, Ls, r0):
        '''
        Modulus effect for screw dislocation on weak and shearable particles
        '''
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(0, r0)), 3/2)

    def modulusStrong(self, r, Ls):
        '''
        Modulus effect for edge or screw dislocation on strong and shearable particles
        '''
        return self.J * self.w1 * np.abs(self.Gp - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (self.b * Ls)

    def APBweakedge(self, r, Ls, r0):
        '''
        Anti-phase boundary effect for edge dislocation on weak and shearable particles
        '''
        xi = 16 * self.yAPB * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(2 * self.yAPB * r / (2*self.T(np.pi/2, r0)), 3/2) - self.beta * xi)

    def APBweakscrew(self, r, Ls, r0):
        '''
        Anti-phase boundary effect for screw dislocation on weak and shearable particles
        '''
        xi = 16 * self.yAPB * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(0, r0) / (self.b * Ls) * np.power(2 * self.yAPB * r / (2*self.T(0, r0)), 3/2) - self.beta * xi)

    def APBstrongedge(self, r, Ls, r0):
        '''
        Anti-phase boundary effect for edge dislocation on strong and shearable particles
        Equation from paper gives np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2, r0)) - 1), but their plots say otherwise
        '''
        return (2 * self.V * self.T(np.pi/2, r0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2, r0)))

    def APBstrongscrew(self, r, Ls, r0):
        '''
        Anti-phase boundary effect for screw dislocation on strong and shearable particles
        '''
        return (2 * self.V * self.T(0, r0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(0, r0)) - 1)

    def SFEweakWideEdge(self, r, Ls, r0):
        '''
        Stacking fault energy effect for edge dislocation on weak and shearable particles for a wide stacking fault
        '''
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP) / self.T(np.pi/2, r0), 3/2)

    def SFEweakWideScrew(self, r, Ls, r0):
        '''
        Stacking fault energy effect for screw dislocation on weak and shearable particles for a wide stacking fault
        '''
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP) / self.T(0, r0), 3/2)

    def SFEstrongWide(self, r, Ls):
        '''
        Stacking fault energy effect for edge or screw dislocation on strong and shearable particles for a wide stacking fault
        '''
        return self.J * 2 * r * (self.ySFM - self.ySFP) / (self.b * Ls)

    def K(self, theta):
        return self.G * self.bp**2 * (2 - self.nu) / (8 * np.pi * (1 - self.nu)) * (1 - 2 * self.nu * np.cos(2 * theta) / (2 - self.nu))

    @property
    def SFEWeffEdge(self):
        '''
        Effective stacking fault width for edge dislocations
        '''
        return 2 * self.K(np.pi/2) / (self.ySFM + self.ySFP)

    @property
    def SFEWeffScrew(self):
        '''
        Effective stacking fault width for screw dislocations
        '''
        return 2 * self.K(0) / (self.ySFM + self.ySFP)

    def SFEweakNarrowEdge(self, r, Ls, r0):
        '''
        Stacking fault energy effect for edge dislocation on weak and shearable particles for a narrow stacking fault
        '''
        return (2 * self.T(np.pi/2, r0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffEdge * r - self.SFEWeffEdge**2 / 4) / self.T(np.pi/2, r0), 3/2)

    def SFEweakNarrowScrew(self, r, Ls, r0):
        '''
        Stacking fault energy effect for screw dislocation on weak and shearable particles for a narrow stacking fault
        '''
        return (2 * self.T(0, r0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffScrew * r - self.SFEWeffScrew**2 / 4) / self.T(0, r0), 3/2)

    def SFEstrongNarrowEdge(self, r, Ls, r0):
        '''
        Stacking fault energy effect for edge dislocation on strong and shearable particles for a narrow stacking fault
        '''
        return self.J * 2 * (self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffEdge * r - self.SFEWeffEdge**2 / 4) / (self.b * Ls)

    def SFEstrongNarrowScrew(self, r, Ls, r0):
        '''
        Stacking fault energy effect for screw dislocation on strong and shearable particles for a narrow stacking fault
        '''
        return self.J * 2 * (self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffScrew * r - self.SFEWeffScrew**2 / 4) / (self.b * Ls)

    def interfacialWeakEdge(self, r, Ls, r0):
        '''
        Interfacial energy effect for edge dislocation on weak and shearable particles
        '''
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(self.gamma * self.b / self.T(np.pi/2, r0), 3/2)

    def interfacialWeakScrew(self, r, Ls, r0):
        '''
        Interfacial energy effect for screw dislocation on weak and shearable particles
        '''
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(self.gamma * self.b / self.T(0, r0), 3/2)

    def interfacialStrong(self, r, Ls, r0):
        '''
        Interfacial energy effect for edge or screw dislocation on strong and shearable particles
        '''
        return self.J * 2 * self.gamma / Ls

    def orowan(self, r, Ls):
        '''
        Orowan strengthening for non-shearable particles
        '''
        return self.J * self.G * self.b / (2 * np.pi * np.sqrt(1 - self.nu) * Ls) * np.log(2 * r / self.ri)

    def checkPrecipitateStrengthParameters(self):
        '''
        Checks if all parameters for precipitate strength contributions are supplied
        Return True if all parameters are not None
            Parameters: Taylor factor (M), burgers vector (b), constant (k), shear modulus (G) and transition radius (rtrans)
        '''
        return self.M is not None and self.b is not None and self.k is not None and self.G is not None and self.rtrans is not None

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

    def plotStrength(self, ax, model, plotContributions = False, bounds = None, timeUnits = 's', strengthUnits = 'Pa', *args, **kwargs):
        '''
        Plots strength over time

        Parameters
        ----------
        ax : Axis
        model : KWNEuler object
        plotContributions : boolean
            Whether to plot each contribution of the strength model (defaults to False)
        bounds : tuple or None
            Bounds on time axis (if None, the bounds will automatically be set)
        timeUnits : str
            Units of time to plot in, options are 's' for seconds, 'm' for minutes or 'h' for hours
        strengthUnits : str
            Units for strength, options are 'Pa', 'kPa', 'MPa' or 'GPa'
        '''
        timeScale, timeLabel, bounds = model.getTimeAxis(timeUnits, bounds)
        yscale, ylabel = self.getStrengthUnits(strengthUnits)

        ssstrength = self.ssStrength(model) if len(self.ssweights) > 0 else np.zeros(len(model.time))
        sigma0 = self.sigma0*np.ones(len(model)) if self.sigma0 is not None else np.zeros(len(model.time))
        precstrength = self.precStrength(model) if self.checkPrecipitateStrengthParameters() else np.zeros(len(model.time))

        ax.plot(model.time*timeScale, (sigma0 + ssstrength + precstrength) / yscale, *args, **kwargs)

        if plotContributions:
            ax.plot(model.time*timeScale, sigma0 / yscale, *args, **kwargs)
            ax.plot(model.time*timeScale, ssstrength / yscale, *args, **kwargs)
            ax.plot(model.time*timeScale, precstrength / yscale, *args, **kwargs)
            ax.legend(['Total Strength', r'$\sigma_0$', 'SS Strength', 'Precipitate Strength'])

        ax.set_xlabel(timeLabel)
        ax.set_xlim(bounds)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_xscale('log')

    