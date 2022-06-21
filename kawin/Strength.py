import numpy as np

class StrengthModel:
    def __init__(self):
        self.rssName = 'rss'
        self.LsName = 'Ls'

        #Solid solution strengthening parameters
        self.ssweights = {}
        self.ssexp = 1

        #Base strength
        self.sigma0 = 0

        self.M = 2.24

        #Precipitate strength factors
        self.coherencyEffect = False        #Coherency effect
        self.modulusEffect = False          #Modulus effect
        self.APBEffect = False              #Anti-phase boundary effect
        self.SFEffect = False               #Stacking fault energy effect
        self.IFEffect = False               #Interfacial energy effect
        self.orowanEffect = False           #Non-shearable (Orowan) mechanism

        #Parameters for dislocation line tension and general shear strength
        self.G, self.b, self.nu, self.ri, self.r0, self.J = None, None, None, None, None, 1

        #Parameters for coherency effect
        self.eps = None

        #Parameters for modulus effect
        self.Gp, self.w1, self.w2 = None, 0.0722, 0.81

        #Parameters for anti-phase boundary effect
        self.yAPB, self.s, self.beta, self.V = None, 2, 1, 2.8

        #Parameters for stacking fault energy effect
        self.ySFM, self.ySFP, self.bp = None, None, None

        #Parameters for interfacial effect
        self.gamma = None

        self.edgeDis = True
        self._setFunctions()

    def setBaseStrength(self, sigma0):
        '''
        Sets base strength of matrix

        Parameters
        ----------
        sigma0 : float
            Base strength (Pa)
        '''
        self.sigma0 = sigma0

    def setSolidSolutionStrength(self, weights, exp = 1):
        '''
        Parameters for solid solution strengthening

        Parameters
        ----------
        weights : dictionary
            Dictionary mapping element (str) to weight (float). Weights are in units of (Pa)
        exp : float
            Exponential factor for weights
        '''
        self.ssweights = weights
        self.ssexp = exp

    def setDislocationParameters(self, G, b, nu, ri = None, r0 = None, J = 1):
        '''
        Parameters for dislocation line tension, used for all precipitate strength models

        Parameters
        ----------
        G : float
            Shear modulus of matrix (Pa)
        b : float
            Burgers vector (meters)
        nu : float
            Poisson ratio
        ri : float
            Dislocation core radius (meters)
            If None, ri will be set to Burgers vector
        r0 : float
            Closest distance between parallel dislocations
            For shearable precipitates, r0 is average distance between particles on slip plane
            For non-shearable precipitates, r0 is average particle diameter on slip plane
            If None, r0 will be set such that ln(r0/ri) = 2*pi
        J : float
            Correction coefficient for mean particle distance between 0 to 1
            Default at 1 based off current particle distance equations
        '''
        self.G = G
        self.b = b
        if self.bp is None:
            self.bp = b
        self.nu = nu
        self.ri = b if ri is None else ri
        self.r0 = self.ri * np.exp(2*np.pi) if r0 is None else r0
        self.J = J

    def setDislocationType(self, disType = 'edge'):
        '''
        Set type of dislocation

        Parameters
        ----------
        disType : str
            Type of dislocation ('edge' or 'screw')
            Default to 'edge'
        '''
        if disType.lower() == 'screw':
            self.edgeDis = False
        self._setFunctions()

    def _setFunctions(self):
        '''
        Sets strength contribution functions based off edge or screw dislocation
        '''
        if self.edgeDis:
            self.coherencyWeakFunction = self.coherencyWeakEdge
            self.coherencyStrongFunction = self.coherencyStrongEdge
            self.modulusWeakFunction = self.modulusWeakEdge
            self.APBweakFunction = self.APBweakEdge
            self.APBstrongFunction = self.APBstrongEdge
            self.SFEweakNarrowFunction = self.SFEweakNarrowEdge
            self.SFEweakWideFunction = self.SFEweakWideEdge
            self.SFEstrongNarrowFunction = self.SFEstrongNarrowEdge
            self.interfacialweakFunction = self.interfacialWeakEdge
        else:
            self.coherencyWeakFunction = self.coherencyWeakScrew
            self.coherencyStrongFunction = self.coherencyStrongScrew
            self.modulusWeakFunction = self.modulusWeakScrew
            self.APBweakFunction = self.APBweakScrew
            self.APBstrongFunction = self.APBstrongScrew
            self.SFEweakNarrowFunction = self.SFEweakNarrowScrew
            self.SFEweakWideFunction = self.SFEweakWideScrew
            self.SFEstrongNarrowFunction = self.SFEstrongNarrowScrew
            self.interfacialweakFunction = self.interfacialWeakScrew

        self.modulusStrongFunction = self.modulusStrong
        self.SFEstrongWidefunction = self.SFEstrongWide
        self.interfacialstrongfunction = self.interfacialStrong
        self.orowanfunction = self.orowan

    def setCoherencyParameters(self, eps):
        '''
        Parameters for coherency effect

        Parameters
        ----------
        eps : float
            Lattice misfit strain
        '''
        self.coherencyEffect = True
        self.eps = eps

    def setModulusParameters(self, Gp, w1 = 0.0722, w2 = 0.81):
        '''
        Parameters for modulus effect

        Parameters
        ----------
        Gp : float
            Shear modulus of precipitate
        w1 : float
            First factor for Nembach model taking value between 0.0175 and 0.0722
            Default at 0.0722
        w2 : float
            Second factor for Nembach model taking value of 0.81 +/- 0.09
            Default at 0.81
        '''
        self.modulusEffect = True
        self.Gp = Gp
        self.w1 = w1
        self.w2 = w2

    def setAPBParameters(self, yAPB, s = 2, beta = 1, V = 2.8):
        '''
        Parameters for anti-phase boundary effect for ordered precipitates in a disordered matrix

        Parameters
        ----------
        yAPB : float
            Anti-phase boundary energy
        s : int
            Number of leading + trailing dislocations to repair anti-phase boundary
            Default at 2
        beta : float
            Factor representating dislocation shape (between 0 and 1)
            Default at 1
        V : float
            Correction factor accounting for extra dislocations and uncertainties
            Default at 2.8
        '''
        self.APBEffect = True
        self.yAPB = yAPB
        self.s = s
        self.beta = beta
        self.V = V

    def setSFEParameters(self, ySFM, ySFP, bp = None):
        '''
        Parameters for stacking fault energy effect

        Parameters
        ----------
        ySFM : float
            Stacking fault energy of matrix
        ySFP : float
            Stacking fault energy of precipitate
        bp : float
            Burgers vector in precipitate
            If None, will be set to burgers vector in matrix
        '''
        self.SFEffect = True
        self.ySFM = ySFM
        self.ySFP = ySFP
        self.bp = self.b if bp is None else bp

    def setInterfacialParameters(self, gamma):
        '''
        Parameters for interfacial effect

        Parameters
        ----------
        gamma : float
            Interfacial energy of matrix/precipitate surface
        '''
        self.IFEffect = True
        self.gamma = gamma

    def epsMisfit(self):
        '''
        Strain from lattice misfit
        '''
        self.eps = (1/3) * (1+self.nu) / (1-self.nu) * self.delta

    def T(self, theta):
        '''
        Dislocation line tension
        '''
        return self.G*self.b**2 / (4*np.pi) * (1 + self.nu - 3*self.nu*np.sin(theta)**2) / (1 - self.nu) * np.log(self.r0 / self.ri)

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

    def coherencyWeakEdge(self, r, Ls):
        '''
        Coherency effect for edge dislocation on weak and shearable particles
        '''
        return np.sqrt((592 / 35) * self.G**3 * self.b * self.eps**3 * r**3 / (Ls**2 * self.T(np.pi/2)))

    def coherencyWeakScrew(self, r, Ls):
        '''
        Coherency effect for screw dislocation on weak and shearable particles
        '''
        return np.sqrt((9/5) * self.G**3 * self.b * self.eps**3 * r**3 / (Ls**2 * self.T(0)))

    def coherencyStrongEdge(self, r, Ls):
        '''
        Coherency effect for edge dislocation on strong and shearable particles
        '''
        return np.sqrt(2) * np.power(3, 3/8) * self.J / Ls * np.power((self.T(np.pi/2)**3 * self.G * self.eps * r) / self.b**3, 1/4)

    def coherencyStrongScrew(self, r, Ls):
        '''
        Coherency effect for screw dislocation on strong and shearable particles
        '''
        return 2 * self.J / Ls * np.power((self.T(0)**3 * self.G * self.eps * r) / self.b**3, 1/4)

    def modulusWeakEdge(self, r, Ls):
        '''
        Modulus effect for edge dislocation on weak and shearable particles
        '''
        return 2 * self.T(np.pi/2) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(np.pi/2)), 3/2)

    def modulusWeakScrew(self, r, Ls):
        '''
        Modulus effect for screw dislocation on weak and shearable particles
        '''
        return 2 * self.T(0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(0)), 3/2)

    def modulusStrong(self, r, Ls):
        '''
        Modulus effect for edge or screw dislocation on strong and shearable particles
        '''
        return self.J * self.w1 * np.abs(self.Gp - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (self.b * Ls)

    def APBweakEdge(self, r, Ls):
        '''
        Anti-phase boundary effect for edge dislocation on weak and shearable particles
        '''
        xi = 16 * self.yAPB * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(np.pi/2) / (self.b * Ls) * np.power(2 * self.yAPB * r / (2*self.T(np.pi/2)), 3/2) - self.beta * xi)

    def APBweakScrew(self, r, Ls):
        '''
        Anti-phase boundary effect for screw dislocation on weak and shearable particles
        '''
        xi = 16 * self.yAPB * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(0) / (self.b * Ls) * np.power(2 * self.yAPB * r / (2*self.T(0)), 3/2) - self.beta * xi)

    def APBstrongEdge(self, r, Ls):
        '''
        Anti-phase boundary effect for edge dislocation on strong and shearable particles
        Equation from paper gives np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2)) - 1), but their plots say otherwise
        '''
        return (2 * self.V * self.T(np.pi/2)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2)) - 1)

    def APBstrongScrew(self, r, Ls):
        '''
        Anti-phase boundary effect for screw dislocation on strong and shearable particles
        Equation from paper gives np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(0)) - 1), but their plots say otherwise
        '''
        return (2 * self.V * self.T(0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(0)) - 1)

    def SFEweakWideEdge(self, r, Ls):
        '''
        Stacking fault energy effect for edge dislocation on weak and shearable particles for a wide stacking fault
        '''
        return 2 * self.T(np.pi/2) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP) / self.T(np.pi/2), 3/2)

    def SFEweakWideScrew(self, r, Ls):
        '''
        Stacking fault energy effect for screw dislocation on weak and shearable particles for a wide stacking fault
        '''
        return 2 * self.T(0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP) / self.T(0), 3/2)

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

    def SFEweakNarrowEdge(self, r, Ls):
        '''
        Stacking fault energy effect for edge dislocation on weak and shearable particles for a narrow stacking fault
        '''
        return (2 * self.T(np.pi/2)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffEdge * r - self.SFEWeffEdge**2 / 4) / self.T(np.pi/2), 3/2)

    def SFEweakNarrowScrew(self, r, Ls):
        '''
        Stacking fault energy effect for screw dislocation on weak and shearable particles for a narrow stacking fault
        '''
        return (2 * self.T(0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffScrew * r - self.SFEWeffScrew**2 / 4) / self.T(0), 3/2)

    def SFEstrongNarrowEdge(self, r, Ls):
        '''
        Stacking fault energy effect for edge dislocation on strong and shearable particles for a narrow stacking fault
        '''
        return self.J * 2 * (self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffEdge * r - self.SFEWeffEdge**2 / 4) / (self.b * Ls)

    def SFEstrongNarrowScrew(self, r, Ls):
        '''
        Stacking fault energy effect for screw dislocation on strong and shearable particles for a narrow stacking fault
        '''
        return self.J * 2 * (self.ySFM - self.ySFP) * np.sqrt(self.SFEWeffScrew * r - self.SFEWeffScrew**2 / 4) / (self.b * Ls)

    def interfacialWeakEdge(self, r, Ls):
        '''
        Interfacial energy effect for edge dislocation on weak and shearable particles
        '''
        return 2 * self.T(np.pi/2) / (self.b * Ls) * np.power(self.gamma * self.b / self.T(np.pi/2), 3/2)

    def interfacialWeakScrew(self, r, Ls):
        '''
        Interfacial energy effect for screw dislocation on weak and shearable particles
        '''
        return 2 * self.T(0) / (self.b * Ls) * np.power(self.gamma * self.b / self.T(0), 3/2)

    def interfacialStrong(self, r, Ls):
        '''
        Interfacial energy effect for edge or screw dislocation on strong and shearable particles
        '''
        return self.J * 2 * self.gamma / Ls

    def orowan(self, r, Ls):
        '''
        Orowan strengthening for non-shearable particles
        '''
        return self.J * self.G * self.b / (2 * np.pi * np.sqrt(1 - self.nu) * Ls) * np.log(2 * r / self.ri)

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

    def rssterm(self, model):
        r1 = np.sum([model.PBM[p].PSD * model.PBM[p].PSDsize for p in range(len(model.phases))])
        r2 = np.sum([model.PBM[p].PSD * model.PBM[p].PSDsize**2 for p in range(len(model.phases))])
        if r1 == 0:
            rss = 0
        else:
            rss = np.sqrt(2/3) * r2 / r1
        return rss

    def Lsterm(self, model):
        r1 = np.sum([model.PBM[p].PSD * model.PBM[p].PSDsize for p in range(len(model.phases))])
        r2 = np.sum([model.PBM[p].PSD * model.PBM[p].PSDsize**2 for p in range(len(model.phases))])
        if r1 == 0:
            Ls = 0
        else:
            rss = np.sqrt(2/3) * r2 / r1
            Ls = np.sqrt(np.log(3) / (2*np.pi*r1) + (2*rss)**2) - 2*rss
        return Ls

    def insertStrength(self, model):
        '''
        Inserts Fterm into the KWNmodel to be solved for

        Parameters
        ----------
        model : KWNEuler object
        '''
        model.addAdditionalOutput(self.rssName, self.rssterm)
        model.addAdditionalOutput(self.LsName, self.Lsterm)

    def precStrength(self, model):
        funcs = [p['name'] for p in model.additionalFunctions]
        rss = model.additionalOutputs[0,:,funcs.index('rss')]
        rss = np.pi/4 * model.avgR[0,:]
        Ls = model.additionalOutputs[0,:,funcs.index('Ls')]
        self.r0 = Ls
        return self._precStrength(rss, Ls)

    def _precStrength(self, rss, Ls):
        weakContributions = []
        strongContributions = []
        if self.coherencyEffect:
            weakContributions.append(self.coherencyWeakFunction(rss, Ls))
            strongContributions.append(self.coherencyStrongFunction(rss, Ls))
        if self.modulusEffect:
            weakContributions.append(self.modulusWeakFunction(rss, Ls))
            strongContributions.append(self.modulusStrongFunction(rss, Ls))
        if self.APBEffect:
            weakContributions.append(self.APBweakFunction(rss, Ls))
            strongContributions.append(self.APBstrongFunction(rss, Ls))
        if self.SFEffect:
            #sfw = np.amin([self.SFEweakNarrowFunction(rss, Ls), self.SFEweakWideFunction(rss, Ls)], axis=0)
            #weakContributions.append(sfw)
            #sfs = np.amin([self.SFEstrongNarrowFunction(rss, Ls), self.SFEstrongWidefunction(rss, Ls)], axis=0)
            #strongContributions.append(sfs)
            weakContributions.append(self.SFEweakNarrowFunction(rss, Ls))
            strongContributions.append(self.SFEstrongNarrowFunction(rss, Ls))
        if self.IFEffect:
            weakContributions.append(self.interfacialweakFunction(rss, Ls))
            strongContributions.append(self.interfacialstrongfunction(rss, Ls))
        tausumweak = np.power(np.sum(np.power(weakContributions, 1.8), axis=0), 1/1.8)
        tausumstrong = np.power(np.sum(np.power(strongContributions, 1.8), axis=0), 1/1.8)
        tauowo = self.orowan(rss, Ls)
        taumin = np.amin(np.array([tausumweak, tausumstrong, tauowo]), axis=0)
        return self.M * taumin

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

    def plotStrengthOverR(self, ax, r, Ls, strengthUnits = 'Pa', *args, **kwargs):
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
        strength = self._precStrength(r, Ls)
        ax.plot(r, strength / yscale)
        ax.set_xlabel('Radius (m)')
        ax.set_xlim([np.amin(r), np.amax(r)])
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
        sigma0 = self.sigma0*np.ones(len(model.time)) if self.sigma0 is not None else np.zeros(len(model.time))
        precstrength = self.precStrength(model)

        total = np.power(np.power(sigma0, 1.8) + np.power(ssstrength+precstrength, 1.8), 1/1.8)

        ax.plot(model.time*timeScale, total / yscale, *args, **kwargs)

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

    