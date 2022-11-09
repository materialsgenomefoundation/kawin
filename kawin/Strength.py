import numpy as np

class StrengthModel:
    '''
    Defines strength model

    6 contributions are accounted for
    For dislocation cutting, contributions are coherency, modulus, anti-phase boundary, stacking fault energy and interfacial energy
    For dislocation bowing, contribution is orowan

    Contributions can be added for all phases or for a single phase
    '''
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
        self.coherencyEffect = {'all': False}        #Coherency effect
        self.modulusEffect = {'all': False}          #Modulus effect
        self.APBEffect = {'all': False}              #Anti-phase boundary effect
        self.SFEffect = {'all': False}               #Stacking fault energy effect
        self.IFEffect = {'all': False}               #Interfacial energy effect
        self.orowanEffect = {'all': False}           #Non-shearable (Orowan) mechanism

        #Parameters for dislocation line tension, general shear strength and orowan strengthening
        #These parameters are for the matrix phase
        self.G, self.b, self.nu, self.ri, self.theta, self.psi = None, None, None, None, 90*np.pi/180, 120*np.pi/180

        #eps, GP, yAPB, ySFP and gamma are specific to a phase
        #Parameters for coherency effect
        self.eps = {}

        #Parameters for modulus effect
        self.Gp, self.w1, self.w2 = {}, 0.0722, 0.81

        #Parameters for anti-phase boundary effect
        self.yAPB, self.s, self.beta, self.V = {}, 2, 1, 2.8

        #Parameters for stacking fault energy effect
        self.ySFM, self.ySFP, self.bp = None, {}, {}

        #Parameters for interfacial effect
        self.gamma = {}

        #Model types for line tension and J
        self.T = self.Tcomplex
        self.J = self.Jsimple

        #Single phase superposition functions exponent
        self.singlePhaseExp = 1.8

        #Multi-phase superposition functions exponents
        self.multiphaseSameExp = 1.8
        self.multiphaseMixedExp = 1.4

    def _getStrengthFunctions(self):
        '''
        Internal function that creates arrays for dislocation cutting mechanisms

        wfuncs, sfuncs - list of functions for each contribution for weak and strong effects
        contributions - each contribution has a dictionary of str : boolean to say whether a phase has that contribution
        labels - labels for plotting
        '''
        wfuncs = [self.coherencyWeak, self.modulusWeak, self.APBweak, self.SFEweak, self.interfacialWeak]
        sfuncs = [self.coherencyStrong, self.modulusStrong, self.APBstrong, self.SFEstrong, self.interfacialStrong]
        contributions = [self.coherencyEffect, self.modulusEffect, self.APBEffect, self.SFEffect, self.IFEffect]
        labels = ['Coherency', 'Modulus', 'APB', 'SFE', 'Interfacial']
        return wfuncs, sfuncs, contributions, labels

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

    def setTaylorFactor(self, M):
        self.M = M

    def setDislocationParameters(self, G, b, nu = 1/3, ri = None, theta = 90, psi = 120):
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
        ri : float (optional)
            Dislocation core radius (meters)
            If None, ri will be set to Burgers vector
        r0 : float (optional)
            Closest distance between parallel dislocations
            For shearable precipitates, r0 is average distance between particles on slip plane
            For non-shearable precipitates, r0 is average particle diameter on slip plane
            If None, r0 will be set such that ln(r0/ri) = 2*pi
        theta : float (optional)
            Dislocation characteristic, 90 for edge, 0 for screw (default is 90)
        psi : float (optional)
            Dislocation bending angle (default is 120)
        '''
        self.G = G
        self.b = b
        self.nu = nu
        self.ri = b if ri is None else ri
        self.theta = theta * np.pi/180
        self.psi = psi * np.pi/180

    def setCoherencyParameters(self, eps, phase = 'all'):
        '''
        Parameters for coherency effect

        Parameters
        ----------
        eps : float
            Lattice misfit strain
        phase : str (optional)
            Defaults to 'all'
            If 'all', contribution and parameters will be applied to all phases
            If name of a specific phase (must be one that is defined in the PrecipitateModel)
                contribution and parameters will only be applied to that phase
        '''
        self.coherencyEffect[phase] = True
        self.eps[phase] = eps

    def setModulusParameters(self, Gp, w1 = 0.05, w2 = 0.85, phase = 'all'):
        '''
        Parameters for modulus effect

        Parameters
        ----------
        Gp : float
            Shear modulus of precipitate
        w1 : float (optional)
            First factor for Nembach model taking value between 0.0175 and 0.0722
            Default at 0.05
        w2 : float (optional)
            Second factor for Nembach model taking value of 0.81 +/- 0.09
            Default at 0.85
        phase : str (optional)
            Defaults to 'all'
            If 'all', contribution and parameters will be applied to all phases
            If name of a specific phase (must be one that is defined in the PrecipitateModel)
                contribution and parameters will only be applied to that phase
        '''
        self.modulusEffect[phase] = True
        self.Gp[phase] = Gp
        self.w1 = w1
        self.w2 = w2

    def setAPBParameters(self, yAPB, s = 2, beta = 1, V = 2.8, phase = 'all'):
        '''
        Parameters for anti-phase boundary effect for ordered precipitates in a disordered matrix

        Parameters
        ----------
        yAPB : float
            Anti-phase boundary energy
        s : int (optional)
            Number of leading + trailing dislocations to repair anti-phase boundary
            Default at 2
        beta : float (optional)
            Factor representating dislocation shape (between 0 and 1)
            Default at 1
        V : float (optional)
            Correction factor accounting for extra dislocations and uncertainties
            Default at 2.8
        phase : str (optional)
            Defaults to 'all'
            If 'all', contribution and parameters will be applied to all phases
            If name of a specific phase (must be one that is defined in the PrecipitateModel)
                contribution and parameters will only be applied to that phase
        '''
        self.APBEffect[phase] = True
        self.yAPB[phase] = yAPB
        self.s = s
        self.beta = beta
        self.V = V

    def setSFEParameters(self, ySFM, ySFP, bp = None, phase = 'all'):
        '''
        Parameters for stacking fault energy effect

        Parameters
        ----------
        ySFM : float
            Stacking fault energy of matrix
        ySFP : float
            Stacking fault energy of precipitate
        bp : float (optional)
            Burgers vector in precipitate
            If None, will be set to burgers vector in matrix
        phase : str (optional)
            Defaults to 'all'
            If 'all', contribution and parameters will be applied to all phases
            If name of a specific phase (must be one that is defined in the PrecipitateModel)
                contribution and parameters will only be applied to that phase
        '''
        self.SFEffect[phase] = True
        self.ySFM = ySFM
        self.ySFP[phase] = ySFP
        self.bp[phase] = self.b if bp is None else bp

    def setInterfacialParameters(self, gamma, phase = 'all'):
        '''
        Parameters for interfacial effect

        Parameters
        ----------
        gamma : float
            Interfacial energy of matrix/precipitate surface
        phase : str (optional)
            Defaults to 'all'
            If 'all', contribution and parameters will be applied to all phases
            If name of a specific phase (must be one that is defined in the PrecipitateModel)
                contribution and parameters will only be applied to that phase
        '''
        self.IFEffect[phase] = True
        self.gamma[phase] = gamma

    def setTmodel(self, modelType = 'complex'):
        '''
        Set model for line tension

        Parameters
        ----------
        modelType : str
            'complex' for line tension based off dislocation character
            'simple' for simple line tension model (T = G * b^2 / 2)
        '''
        if modelType == 'simple':
            self.T = self.Tsimple
        else:
            self.T = self.Tcomplex

    def setJfactor(self, modelType = 'complex'):
        '''
        Set model for J

        Parameters
        ----------
        modelType : str
            'complex' for J based off dislocation character
            'simple' for J = 1
        '''
        if modelType == 'simple':
            self.J = self.Jsimple
        else:
            self.J = self.Jcomplex

    def epsMisfit(self, delta, phase='all'):
        '''
        Strain from lattice misfit
        '''
        self.eps['all'] = (1/3) * (1+self.nu) / (1-self.nu) * delta

    def Tcomplex(self, theta, r0):
        '''
        Dislocation line tension
        '''
        return self.G*self.b**2 / (4*np.pi) * (1 + self.nu - 3*self.nu*np.sin(theta)**2) / (1 - self.nu) * np.log(r0 / self.ri)

    def Tsimple(self, theta, r0):
        '''
        Simple model for dislocation tension
        '''
        return 0.5 * self.G * self.b**2

    @property
    def Jcomplex(self):
        return (1 - self.nu * np.cos(np.pi/2 - self.theta)**2) / np.sqrt(1 - self.nu)

    @property
    def Jsimple(self):
        return 1

    def coherencyWeak(self, r, Ls, r0, phase='all'):
        '''
        Coherency effect for mixed dislocation on weak and shearable particles
        '''
        return (1.3416*np.cos(self.theta)**2 + 4.1127*np.sin(self.theta)**2) / Ls * np.sqrt(self.G**3 * self.eps[phase]**3 * r**3 * self.b / self.T(self.theta, r0))

    def coherencyStrong(self, r, Ls, r0, phase='all'):
        '''
        Coherency effect for mixed dislocation on strong and shearable particles
        '''
        return (2*np.cos(self.theta)**2 + 2.1352*np.sin(self.theta)**2) / Ls * np.power(self.T(self.theta, r0)**3 * self.G * self.eps[phase] * r / self.b**3, 1/4)

    def Fmod(self, r, phase='all'):
        '''
        Term for modulus effect
        '''
        return self.w1 * np.abs(self.G - self.Gp[phase]) * self.b**2 * np.power(r / self.b, self.w2)

    def modulusWeak(self, r, Ls, r0, phase='all'):
        '''
        Modulus effect for mixed dislocation on weak and shearable particles
        '''
        return 2 * self.T(self.theta, r0) / (self.b * Ls) * np.power(self.Fmod(r, phase) / (2 * self.T(self.theta, r0)), 3/2)

    def modulusStrong(self, r, Ls, r0, phase='all'):
        '''
        Modulus effect for edge or screw dislocation on strong and shearable particles
        '''
        return self.J * self.Fmod(r, phase) / (self.b * Ls)

    def APBweak(self, r, Ls, r0, phase='all'):
        '''
        Anti-phase boundary effect for mixed dislocation on weak and shearable particles
        '''
        return 2 / (self.s * self.b * Ls) * (2 * self.T(self.theta, r0) * np.power(r * self.yAPB[phase] / self.T(self.theta, r0), 3/2) - 16 * self.beta * self.yAPB[phase] * r**2 / (3 * np.pi * Ls))

    def APBstrong(self, r, Ls, r0, phase='all'):
        '''
        Anti-phase boundary effect for mixed dislocation on strong and shearable particles
        '''
        return 0.69 / (self.b * Ls) * np.sqrt(8 * self.V * self.T(self.theta, r0) * r * self.yAPB[phase] / 3)

    def K(self, theta, phase='all'):
        return self.G * self.bp[phase]**2 * (2 - self.nu - 2 * self.nu * np.cos(2 * theta)) / (8 * np.pi * (1 - self.nu))

    def SFEWeff(self, theta, phase='all'):
        '''
        Effective stacking fault width for mixed dislocations
        '''
        return 2* self.K(theta, phase) / (self.ySFM + self.ySFP[phase])

    def SFEFterm(self, r, phase='all'):
        '''
        Stacking fault term
        '''
        return 2 * (self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(self.theta, phase) * r - self.SFEWeff(self.theta, phase)**2 / 4)

    def interfacialWeak(self, r, Ls, r0, phase='all'):
        '''
        Interfacial energy effect for mixed dislocation on weak and shearable particles
        '''
        return 2 * self.T(self.theta, r0) / (self.b * Ls) * np.power(2 * self.gamma[phase] * self.b / (2 * self.T(self.theta, r0)), 3/2)

    def interfacialStrong(self, r, Ls, r0, phase='all'):
        '''
        Interfacial energy effect for mixed dislocations on strong and shearable particles
        '''
        return 2 * self.gamma[phase] / Ls

    def SFEweak(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for mixed dislocations on weak and shearable particles
        '''
        return 2 * self.T(self.theta, r0) / (self.b * Ls) * np.power(self.SFEFterm(r, phase) / (2 * self.T(self.theta, r0)), 3/2)

    def SFEstrong(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for mixed dislocations on strong and shearable particles
        '''
        return self.SFEFterm(r, phase) / (self.b * Ls)

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

    def rssterm(self, model, p, i):
        r1 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize)
        r2 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize**2)
        if r1 == 0:
            rss = 0
        else:
            rss = np.sqrt(2/3) * r2 / r1
        return rss

    def Lsterm(self, model, p, i):
        r1 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize)
        r2 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize**2)
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

    def getParticleSpacing(self, model, phase = None):
        index = model.phaseIndex(phase)
        rss = model.getAdditionalOutput(self.rssName)[index]
        Ls = model.getAdditionalOutput(self.LsName)[index]
        return rss, Ls

    def precStrength(self, model):
        rss = model.getAdditionalOutput(self.rssName)
        Ls = model.getAdditionalOutput(self.LsName)

        ps = []
        totalCompare = np.zeros(len(rss[0]))
        for i in range(len(model.phases)):
            r0Strong = Ls[i]
            r0Weak = Ls[i] / np.sqrt(np.cos(self.psi / 2))
            weakContributions, strongContributions, orowan, _ = self.getStrengthContributions(rss[i], Ls[i], r0Weak, r0Strong, model.phases[i])
            strength, compare = self.combineStrengthContributions(weakContributions, strongContributions, orowan, returnComparison=True)
            compare[~np.isfinite(strength)] = 0
            strength[~np.isfinite(strength)] = 0
            ps.append(strength)
            totalCompare += np.array(compare, dtype='int')
        ps = np.array(ps)
        totalStrength = np.zeros(len(ps[0]))
        indices = (totalCompare == 0) | (totalCompare == len(model.phases))
        totalStrength[indices] = np.power(np.sum(np.power(ps[:,indices], self.multiphaseSameExp), axis=0), 1/self.multiphaseSameExp)
        totalStrength[~indices] = np.power(np.sum(np.power(ps[:,~indices], self.multiphaseMixedExp), axis=0), 1/self.multiphaseMixedExp)
        return totalStrength

    def getStrengthContributions(self, rss, Ls, r0Weak, r0Strong, phase = 'all'):
        weakContributions = []
        strongContributions = []
        contributionsList = []
        wfuncs, sfuncs, contributions, ylabel = self._getStrengthFunctions()
        for i in range(len(wfuncs)):
            if contributions[i]['all'] or (phase in contributions[i] and contributions[i][phase]):
                with np.errstate(divide='ignore', invalid='ignore'):
                    if (phase in contributions[i] and contributions[i][phase]):
                        weakContributions.append(wfuncs[i](rss, Ls, r0Weak, phase))
                        strongContributions.append(sfuncs[i](rss, Ls, r0Strong, phase))
                    else:
                        weakContributions.append(wfuncs[i](rss, Ls, r0Weak, 'all'))
                        strongContributions.append(sfuncs[i](rss, Ls, r0Strong, 'all'))
                    contributionsList.append(ylabel[i])
        weakContributions = np.array(weakContributions)
        weakContributions[(weakContributions < 0)] = 0
        strongContributions = np.array(strongContributions)
        strongContributions[(strongContributions < 0)] = 0
        tauowo = self.orowan(rss, Ls)

        return weakContributions, strongContributions, tauowo, contributionsList

    def combineStrengthContributions(self, weakContributions, strongContributions, orowan, returnComparison = False):
        tausumweak = np.power(np.sum(np.power(weakContributions, self.singlePhaseExp), axis=0), 1/self.singlePhaseExp)
        tausumstrong = np.power(np.sum(np.power(strongContributions, self.singlePhaseExp), axis=0), 1/self.singlePhaseExp)
        taumin = np.amin(np.array([tausumweak, tausumstrong, orowan]), axis=0)
        if returnComparison:
            return self.M * taumin, (tausumweak > tausumstrong) & (tausumweak > orowan)
        else:
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

    def plotPrecipitateStrengthOverR(self, ax, r, Ls, phase=None, strengthUnits = 'MPa', plotContributions = False, *args, **kwargs):
        '''
        Plots precipitate strength contribution as a function of radius

        Parameters
        ----------
        ax : Axis
        r : list
            Equivalent radius
        Ls : list
            Surface to surface particle distance
        strengthUnits : str
            Units for strength, options are 'Pa', 'kPa', 'MPa' or 'GPa'
        plotContributions : bool
            Whether to plot all contributions
        '''
        if phase is None:
            phase = 'all'

        self.plotPrecipitateStrengthOverX(ax, r, r, Ls, phase, strengthUnits, plotContributions, *args, **kwargs)

    def plotPrecipitateStrengthOverTime(self, ax, model, phase = None, bounds = None, timeUnits = 's', strengthUnits = 'MPa', plotContributions = False, *args, **kwargs):
        '''
        Plots precipitate strength contribution as a function of time

        Parameters
        ----------
        ax : Axis
        r : list
            Equivalent radius
        Ls : list
            Surface to surface particle distance
        strengthUnits : str
            Units for strength, options are 'Pa', 'kPa', 'MPa' or 'GPa'
        plotContributions : bool
            Whether to plot all contributions
        '''
        r, Ls = self.getParticleSpacing(model, phase)

        timeScale, timeLabel, bounds = model.getTimeAxis(timeUnits, bounds)

        self.plotPrecipitateStrengthOverX(ax, model.time*timeScale, r, Ls, phase, strengthUnits, plotContributions, *args, **kwargs)
        if plotContributions:
            row, col = [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]
            for i in range(len(row)):
                ax[row[i], col[i]].set_xlabel(timeLabel)
                ax[row[i], col[i]].set_xlim([model.time[0]*timeScale, model.time[-1]*timeScale])
                ax[row[i], col[i]].set_xscale('log')
        else:
            ax.set_xlabel(timeLabel)
            ax.set_xlim([model.time[0]*timeScale, model.time[-1]*timeScale])
            ax.set_xscale('log')

    def plotPrecipitateStrengthOverX(self, ax, x, r, Ls, phase = None, strengthUnits = 'MPa', plotContributions = False, *args, **kwargs):
        '''
        Plots precipitate strength contribution as a function of x

        Parameters
        ----------
        ax : Axis
        x : list
            X coordinates to plot against
        r : list
            Equivalent radius, must correspond to x
        Ls : list
            Surface to surface particle distance, must correspond to x
        strengthUnits : str
            Units for strength, options are 'Pa', 'kPa', 'MPa' or 'GPa'
        plotContributions : bool
            Whether to plot all contributions
        '''
        yscale, ylabel = self.getStrengthUnits(strengthUnits)
        Leff = Ls / np.sqrt(np.cos(self.psi / 2))
        if plotContributions:
            _, _, _, ylabel = self._getStrengthFunctions()
            row, col = [0, 0, 1, 1, 2], [0, 1, 0, 1, 0]
            weak, strong, oro, contributionList = self.getStrengthContributions(r, Ls, Leff, Ls, phase)
            for i in range(len(row)):
                if ylabel[i] in contributionList:
                    index = contributionList.index(ylabel[i])
                    ax[row[i], col[i]].plot(x, self.M * weak[index] / yscale, x, self.M * strong[index] / yscale, *args, **kwargs)
                    ax[row[i], col[i]].legend(['Weak', 'Strong'])
                    ax[row[i], col[i]].set_ylim(bottom=0)
                else:
                    ax[row[i], col[i]].plot(x, np.zeros(len(x)), *args, **kwargs)
                    ax[row[i], col[i]].set_ylim([-1, 1])
                ax[row[i], col[i]].set_xlabel('Radius (m)')
                ax[row[i], col[i]].set_ylabel(r'$\tau_{' + ylabel[i] + '}$ (' + strengthUnits + ')')

            wtot = np.power(np.sum(np.power(weak, self.singlePhaseExp), axis=0), 1/self.singlePhaseExp)
            stot = np.power(np.sum(np.power(strong, self.singlePhaseExp), axis=0), 1/self.singlePhaseExp)
            smin = np.amin([wtot, stot, oro], axis=0)
            ax[2,1].plot(x, self.M * wtot/yscale, x, self.M * stot/yscale, x, self.M * oro/yscale, x, self.M * smin/yscale, *args, **kwargs)
            ax[2,1].set_ylim(bottom=0)
            ax[2,1].set_ylabel(r'$\tau$ (' + strengthUnits + ')')
            ax[2,1].set_xlabel('Radius (m)')
            ax[2,1].legend(['Weak', 'Strong', 'Orowan', 'Minimum'])

        else:
            weak, strong, oro, contributionList = self.getStrengthContributions(r, Ls, Leff, Ls, phase)
            strength = self.combineStrengthContributions(weak, strong, oro)
            ax.plot(x, strength / yscale, *args, **kwargs)
            ax.set_ylabel('Yield ' + ylabel)
            ax.set_ylim(bottom=0)


    def plotStrength(self, ax, model, plotContributions = False, bounds = None, timeUnits = 's', strengthUnits = 'MPa', *args, **kwargs):
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


    '''
    Comparison functions

    Functions used for the strength model are for mixed dislocations from a MatCalc presentation

    The following functions are from Ahmadi et al and are for specific cases regarding dislocation behavior
    '''
    def coherencyWeakEdge(self, r, Ls, r0, phase='all'):
        return np.sqrt((592 / 35) * self.G**3 * self.b * self.eps[phase]**3 * r**3 / (Ls**2 * self.T(np.pi/2, r0)))

    def coherencyWeakScrew(self, r, Ls, r0, phase='all'):
        return np.sqrt((9/5) * self.G**3 * self.b * self.eps[phase]**3 * r**3 / (Ls**2 * self.T(0, r0)))

    def coherencyStrongEdge(self, r, Ls, r0, phase='all'):
        return np.sqrt(2) * np.power(3, 3/8) * self.J / Ls * np.power((self.T(np.pi/2, r0)**3 * self.G * self.eps[phase] * r) / self.b**3, 1/4)

    def coherencyStrongScrew(self, r, Ls, r0, phase='all'):
        return 2 * self.J / Ls * np.power((self.T(0, r0)**3 * self.G * self.eps[phase] * r) / self.b**3, 1/4)

    def modulusWeakEdge(self, r, Ls, r0, phase='all'):
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp[phase] - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(np.pi/2, r0)), 3/2)

    def modulusWeakScrew(self, r, Ls, r0, phase='all'):
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp[phase] - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(0, r0)), 3/2)

    def APBweakEdge(self, r, Ls, r0, phase='all'):
        xi = 16 * self.yAPB[phase] * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(2 * self.yAPB[phase] * r / (2*self.T(np.pi/2, r0)), 3/2) - self.beta * xi)

    def APBweakScrew(self, r, Ls, r0, phase='all'):
        xi = 16 * self.yAPB[phase] * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(0, r0) / (self.b * Ls) * np.power(2 * self.yAPB[phase] * r / (2*self.T(0, r0)), 3/2) - self.beta * xi)

    def APBstrongEdge(self, r, Ls, r0, phase='all'):
        #Equation from paper gives np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2)) - 1), but their plots say otherwise
        return (2 * self.V * self.T(np.pi/2, r0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB[phase] * r) / (self.V * self.T(np.pi/2, r0)))

    def APBstrongScrew(self, r, Ls, r0, phase='all'):
        #Equation from paper gives np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(0)) - 1), but their plots say otherwise
        return (2 * self.V * self.T(0, r0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB[phase] * r) / (self.V * self.T(0, r0)))

    def SFEweakWideEdge(self, r, Ls, r0, phase='all'):
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP[phase]) / self.T(np.pi/2, r0), 3/2)

    def SFEweakWideScrew(self, r, Ls, r0, phase='all'):
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP[phase]) / self.T(0, r0), 3/2)

    def SFEstrongWide(self, r, Ls, r0, phase='all'):
        return self.J * 2 * r * (self.ySFM - self.ySFP[phase]) / (self.b * Ls)

    def SFEweakNarrowEdge(self, r, Ls, r0, phase='all'):
        return (2 * self.T(np.pi/2, r0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(np.pi/2, phase) * r - self.SFEWeff(np.pi/2, phase)**2 / 4) / self.T(np.pi/2, r0), 3/2)

    def SFEweakNarrowScrew(self, r, Ls, r0, phase='all'):
        return (2 * self.T(0, r0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(0, phase) * r - self.SFEWeff(0, phase)**2 / 4) / self.T(0, r0), 3/2)

    def SFEstrongNarrowEdge(self, r, Ls, r0, phase='all'):
        return self.J * 2 * (self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(np.pi/2, phase) * r - self.SFEWeff(np.pi/2, phase)**2 / 4) / (self.b * Ls)

    def SFEstrongNarrowScrew(self, r, Ls, r0, phase='all'):
        return self.J * 2 * (self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(0, phase) * r - self.SFEWeff(0, phase)**2 / 4) / (self.b * Ls)

    def interfacialWeakEdge(self, r, Ls, r0, phase='all'):
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(self.gamma[phase] * self.b / self.T(np.pi/2, r0), 3/2)

    def interfacialWeakScrew(self, r, Ls, r0, phase='all'):
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(self.gamma[phase] * self.b / self.T(0, r0), 3/2)

    def interfacialStrongOld(self, r, Ls, r0, phase='all'):
        return self.J * 2 * self.gamma[phase] / Ls

    