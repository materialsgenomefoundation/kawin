import numpy as np
from kawin.precipitation.Plot import getTimeAxis

class StrengthModel:
    '''
    Defines strength model

    Following implementation described in
    M.R. Ahmadi, E. Povoden-Karadeni, K.I. Oksuz, A. Falahati and E. Kozeschnik
        Computational Materials Science 91 (2014) 173-186

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

        #Single phase superposition exponent
        self.singlePhaseExp = 1.8

        #Multi-phase superposition exponents
        self.multiphaseSameExp = 1.8
        self.multiphaseMixedExp = 1.4

        #Superposition exponent for total strength
        self.totalStrengthExp = 1.8

        #Strength terms
        self.rss = None
        self.ls = None
        self.solidStrength = None

    def save(self, filename, compressed = True):
        '''
        Saves strength model data

        Note, this only saves solid solution strength, the rss and ls terms
        Parameters should be free so user can load model and evaluate different parameters
        '''
        if compressed:
            np.savez_compressed(filename, ssStrength=self.solidStrength, rss = self.rss, ls = self.ls)
        else:
            np.savez(filename, ssStrength=self.solidStrength, rss = self.rss, ls = self.ls)

    def load(self, filename):
        data = np.load(filename)
        self.solidStrength = data['ssStrength']
        self.rss = data['rss']
        self.ls = data['ls']

    def _getStrengthFunctions(self, selectedContributions = None):
        '''
        Internal function that creates arrays for dislocation cutting mechanisms

        wfuncs, sfuncs - list of functions for each contribution for weak and strong effects
        contributions - each contribution has a dictionary of str : boolean to say whether a phase has that contribution
        labels - labels for plotting

        Parameters
        ----------
        selectedContributions : None or List[str]
            If None, will return weak/strong functions and labels for all contributions
            If List[str], will return weak/strong functions and labels for only the contributions defined in list
                Options are: Coherency, Modulus, APB, SFE and/or Interfacial
        
        Returns
        -------
        wfuncs - List of functions for weak contributions
        sfuncs - List of functions for strong contributions
        contributions - List of {phase str:boolean} for whether the contribution is enabled
        labels - List of labels for plotting
        '''
        wfuncs = [self.coherencyWeak, self.modulusWeak, self.APBweak, self.SFEweak, self.interfacialWeak]
        sfuncs = [self.coherencyStrong, self.modulusStrong, self.APBstrong, self.SFEstrong, self.interfacialStrong]
        contributions = [self.coherencyEffect, self.modulusEffect, self.APBEffect, self.SFEffect, self.IFEffect]
        labels = ['Coherency', 'Modulus', 'APB', 'SFE', 'Interfacial']
        if selectedContributions is None:
            return wfuncs, sfuncs, contributions, labels
        else:
            wfuncsSub, sfuncsSub, contributionsSub, labelsSub = [], [], [], []
            lowerLabels = [l.lower() for l in labels]
            for c in selectedContributions:
                if c.lower() in lowerLabels:
                    index = lowerLabels.index(c.lower())
                    wfuncsSub.append(wfuncs[index])
                    sfuncsSub.append(sfuncs[index])
                    contributionsSub.append(contributions[index])
                    labelsSub.append(labels[index])
            return wfuncsSub, sfuncsSub, contributionsSub, labelsSub 
        

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

    def setTaylorFactor(self, M = 2.24):
        '''
        Taylor factor for converting critical resolved shear stress to yield strength

        Default is 2.24
        '''
        self.M = M

    def setStrengthSuperpositionExponent(self, singlePhaseExp = 1.8, multiPhaseSameExp = 1.8, multiPhaseMixedExp = 1.4, totalExp = 1.8):
        '''
        Sets exponent (n) for superposition function when combining strength contributions

        sigma^n = sigma_1^n + sigma_2^n + sigma_3^n + ...

        Parameters
        ----------
        singlePhaseExp : float (optional)
            Exponent for adding strength contributions for a single precipitate phase
            Default = 1.8
        multiPhaseSameExp : float (optional)
            Exponent for adding strength contributions for multiple precipitate phases when the 
            strengthening mechanisms are the same
            Default = 1.8
        multiPhaseMixedExp : float (optional)
            Exponent for adding strength contributions for multiple precipitate phases when the
            strengthening mechanisms differ
            Default = 1.4
        totalExp : float (optional)
            Exponent for adding strength mechanisms for total strength
            This includes base strength, solid solution strengthening and precipitation hardening
            Default = 1.8
        '''
        self.singlePhaseExp = singlePhaseExp
        self.multiphaseSameExp = multiPhaseSameExp
        self.multiphaseMixedExp = multiPhaseMixedExp
        self.totalStrengthExp = totalExp

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
        '''
        Correction constant for mean distance between randomly arranged particles accounting for dislocation characteristic
        '''
        return (1 - self.nu * np.cos(np.pi/2 - self.theta)**2) / np.sqrt(1 - self.nu)

    @property
    def Jsimple(self):
        '''
        Same as Jcomplex but assumes edge dislocation
        '''
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

    def ssStrength(self, model, n):
        '''
        Solid solution strength model
        sigma_ss = sum(k_i * c_i^n)

        Parameters
        ----------
        model : KWNEuler object
            Model to take composition from

        Returns
        -------
        strength : array of floats
            Solid solution strength contribution over time
        '''
        val = 0
        for i in range(len(model.elements)):
            if model.elements[i] in self.ssweights:
                val += self.ssweights[model.elements[i]]*model.pData.composition[n,i]**self.ssexp
        return val

    def rssterm(self, model, p):
        '''
        Mean projected radius of particles

        r1 = first ordered moment of particle size distribution
        r2 = second ordered moment of particle size distribution
        rss = sqrt(2/3) * r2 / r1

        Parameters
        ----------
        model : PrecipitateModel
        p : int
            Phase index
        i : int
            Iteration of the model
        '''
        r1 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize)
        r2 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize**2)
        if r1 == 0:
            rss = 0
        else:
            rss = np.sqrt(2/3) * r2 / r1
        return rss

    def Lsterm(self, model, p):
        '''
        Mean surface to surface distance between particles

        r1 = first ordered moment of particle size distribution
        r2 = second ordered moment of particle size distribution
        rss = sqrt(2/3) * r2 / r1
        ls = sqrt(ln(3)/(2*pi*r1) + (2*rss)^2) - 2*rss

        Parameters
        ----------
        model : PrecipitateModel
        p : int
            Phase index
        i : int
            Iteration of the model
        '''
        r1 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize)
        r2 = np.sum(model.PBM[p].PSD * model.PBM[p].PSDsize**2)
        if r1 == 0:
            Ls = 0
        else:
            rss = np.sqrt(2/3) * r2 / r1
            Ls = np.sqrt(np.log(3) / (2*np.pi*r1) + (2*rss)**2) - 2*rss
        return Ls
    
    def updateCoupledModel(self, model):
        '''
        Computes rss, ls and solid solution strengthening terms
        from current state of the PrecipitateModel

        Parameters
        ----------
        model : PrecpitateModel
        '''
        if self.rss is None:
            self.rss = np.zeros((1, len(model.phases)))
            self.ls = np.zeros((1, len(model.phases)))
            self.solidStrength = np.zeros(1)
            self.solidStrength[0] = self.ssStrength(model, 0)

        self.rss = np.append(self.rss, [[self.rssterm(model, p) for p in range(len(model.phases))]], axis=0)
        self.ls = np.append(self.ls, [[self.Lsterm(model, p) for p in range(len(model.phases))]], axis=0)
        self.solidStrength = np.append(self.solidStrength, [self.ssStrength(model, model.pData.n)], axis=0)

    def precStrength(self, model):
        '''
        Gets strength as a function of time

        Parameters
        ----------
        model : PrecipitateModel
        '''
        rss = self.rss
        Ls = self.ls

        ps = []
        totalCompare = np.zeros(len(rss[:,0]))
        for i in range(len(model.phases)):
            weakContributions, strongContributions, orowan, _ = self.getStrengthContributions(rss[:,i], Ls[:,i], model.phases[i])
            strength, compare, _ = self.combineStrengthContributions(weakContributions, strongContributions, orowan, returnComparison=True)
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

    def getStrengthContributions(self, rss, Ls, phase = 'all', selectedContributions=None):
        '''
        Gets strength contributions from a model

        Parameters
        ----------
        rss : array
            Mean projected radius
        Ls : array
            Mean surface to surface particle spacing
        phase : str (optional)
            Phase name
            Defaults to 'all'
        selectedContributions : None or List[str]
            If None, will return weak/strong functions and labels for all contributions
            If List[str], will return weak/strong functions and labels for only the contributions defined in list
                Options are: Coherency, Modulus, APB, SFE and/or Interfacial
        '''
        r0Weak = Ls / np.sqrt(np.cos(self.psi / 2))
        r0Strong = Ls
        weakContributions = []
        strongContributions = []
        contributionsList = []
        wfuncs, sfuncs, contributions, ylabel = self._getStrengthFunctions(selectedContributions)
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
        weakContributions[(weakContributions < 0) | ~np.isfinite(weakContributions)] = 0
        strongContributions = np.array(strongContributions)
        strongContributions[(strongContributions < 0) | ~np.isfinite(strongContributions)] = 0
        tauowo = np.array(self.orowan(rss, Ls))
        tauowo[~np.isfinite(tauowo)] = 0
        return weakContributions, strongContributions, tauowo, contributionsList
    
    def combineStrengthContributions(self, weakContributions, strongContributions, orowan, returnComparison = False):
        '''
        Combines weak, strong and orowan contributions

        Parameters
        ----------
        weakContributions : 2D array
            List of arrays for weak contribution mechanisms
        strongContributions : 2D array
            List of arrays for strong contribution mechanisms
        orowan : array
            Orowan strengthening contribution
        returnComparison : boolean (optional)
            If True, returns additional array of boolean where
                True - weak contribution is higher than strong and orowan contributions
                False - weak contribution is lower than strong and orowan contributions
        '''
        tausumweak = np.zeros(orowan.shape) if len(weakContributions) == 0 else np.array(np.power(np.sum(np.power(weakContributions, self.singlePhaseExp), axis=0), 1/self.singlePhaseExp))
        tausumstrong = np.zeros(orowan.shape) if len(strongContributions) == 0 else np.array(np.power(np.sum(np.power(strongContributions, self.singlePhaseExp), axis=0), 1/self.singlePhaseExp))
        tausumweak[~np.isfinite(tausumweak)] = 0
        tausumstrong[~np.isfinite(tausumstrong)] = 0
        orowan[~np.isfinite(orowan)] = 0
        taumin = np.amin(np.array([tausumweak, tausumstrong, orowan]), axis=0)
        if returnComparison:
            return self.M * taumin, (tausumweak > tausumstrong) & (tausumweak > orowan), (self.M * tausumweak, self.M * tausumstrong, self.M * orowan)
        else:
            return self.M * taumin

    def totalStrength(self, ssStrength, precStrength):
        '''
        Combined base strength, solid solution strength and precipitate strength

        Parameters
        ----------
        ssStrength : array
            Solid solution strengthening
        precStrength : array
            Precipitate strengthening
        '''
        sigma0 = self.sigma0*np.ones(len(ssStrength))
        return np.power(np.sum(np.power([sigma0, ssStrength, precStrength], self.totalStrengthExp), axis=0), 1/self.totalStrengthExp)

    def getStrengthUnits(self, strengthUnits = 'Pa'):
        '''
        Internal function to return scale and label for the y-axis based off units of strength
        '''
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

    def plotPrecipitateStrengthOverR(self, ax, r, Ls, phase=None, strengthUnits = 'MPa', contribution = None, *args, **kwargs):
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
        contribution : None or str
            If None, will plot overall strength
            If str, will plot selected contribution or all contributions
                Options are: Coherency, Modulus, APB, SFE or Interfacial
        '''
        if phase is None:
            phase = 'all'

        self.plotPrecipitateStrengthOverX(ax, r, r, Ls, phase, strengthUnits, contribution, *args, **kwargs)
        ax.set_xlabel('Radius (m)')

    def plotPrecipitateStrengthOverTime(self, ax, model, phase = None, bounds = None, timeUnits = 's', strengthUnits = 'MPa', contribution = None, *args, **kwargs):
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
        contribution : None or str
            If None, will plot overall strength
            If str, will plot selected contribution or all contributions
                Options are: Coherency, Modulus, APB, SFE or Interfacial
        '''
        timeScale, timeLabel, bounds = getTimeAxis(model.pData.time, timeUnits, bounds)

        self.plotPrecipitateStrengthOverX(ax, model.pData.time*timeScale, self.rss, self.ls, phase, strengthUnits, contribution, *args, **kwargs)
        ax.set_xlabel(timeLabel)
        ax.set_xscale('log')
        ax.set_xlim(bounds)

    def plotPrecipitateStrengthOverX(self, ax, x, r, Ls, phase = None, strengthUnits = 'MPa', contribution = None, *args, **kwargs):
        '''
        Plots precipitate strength contribution as a function of x

        TODO: make this a bit more generalized where you can set the contribution you want to plot
            This should also remove the restriction that axes subplot must be 3x2

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
        contribution : None or str
            If None, will plot overall strength
            If str, will plot selected contribution or all contributions
                Options are: Coherency, Modulus, APB, SFE, Interfacial, Orowan or All
        '''
        yscale, ylabel = self.getStrengthUnits(strengthUnits)
        if contribution is not None:
            if contribution.lower() == 'orowan':
                tauowo = np.array(self.orowan(r, Ls))
                tauowo[~np.isfinite(tauowo)] = 0
                ax.plot(x, self.M * tauowo / yscale, *args, **kwargs)
                ax.set_ylabel(r'$\tau_{orowan}$ (' + strengthUnits + ')')
                ax.set_ylim(bottom=0)
                ax.set_xlim([x[0], x[-1]])

            elif contribution.lower() != 'all':
                _, _, _, ylabel = self._getStrengthFunctions([contribution])
                weak, strong, oro, contributionList = self.getStrengthContributions(r, Ls, phase, [contribution])
                if ylabel[0] in contributionList:
                    ax.plot(x, self.M * weak[0] / yscale, x, self.M * strong[0] / yscale, *args, **kwargs)
                    ax.set_ylim(bottom=0)
                    ax.legend(['Weak', 'Strong'])
                else:
                    ax.plot(x, np.zeros(len(x)), *args, **kwargs)
                    ax.set_ylim([-1, 1])
                ax.set_ylabel(r'$\tau_{' + ylabel[0] + '}$ (' + strengthUnits + ')')
                ax.set_xlim([x[0], x[-1]])

            else:
                _, _, _, ylabel = self._getStrengthFunctions()
                weak, strong, oro, contributionList = self.getStrengthContributions(r, Ls, phase)
                strength, _, summedContributions = self.combineStrengthContributions(weak, strong, oro, returnComparison=True)
                wtot, stot, oro = summedContributions

                ax.plot(x, wtot/yscale, x, stot/yscale, x, oro/yscale, x, strength/yscale, *args, **kwargs)
                ax.set_ylim(bottom=0)
                ax.set_ylabel(r'$\tau$ (' + strengthUnits + ')')
                ax.legend(['Weak', 'Strong', 'Orowan', 'Minimum'])
                ax.set_xlim([x[0], x[-1]])
            

        else:
            weak, strong, oro, contributionList = self.getStrengthContributions(r, Ls, phase)
            strength = self.combineStrengthContributions(weak, strong, oro)
            ax.plot(x, strength / yscale, *args, **kwargs)
            ax.set_ylabel('Yield ' + ylabel)
            ax.set_ylim(bottom=0)
            ax.set_xlim([x[0], x[-1]])

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
        timeScale, timeLabel, bounds = getTimeAxis(model.pData.time, timeUnits, bounds)
        yscale, ylabel = self.getStrengthUnits(strengthUnits)

        sigma0 = self.sigma0 * np.ones(len(model.pData.time))
        #ssStrength = self.ssStrength(model) if len(self.ssweights) > 0 else np.zeros(len(model.time))
        ssStrength = self.solidStrength
        precStrength = self.precStrength(model)

        total = self.totalStrength(ssStrength, precStrength)

        ax.plot(model.pData.time*timeScale, total / yscale, *args, **kwargs)

        if plotContributions:
            ax.plot(model.pData.time*timeScale, sigma0 / yscale, *args, **kwargs)
            ax.plot(model.pData.time*timeScale, ssStrength / yscale, *args, **kwargs)
            ax.plot(model.pData.time*timeScale, precStrength / yscale, *args, **kwargs)
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
        '''
        Coherency effect for edge dislocation on weak and shearable particles
        '''
        return np.sqrt((592 / 35) * self.G**3 * self.b * self.eps[phase]**3 * r**3 / (Ls**2 * self.T(np.pi/2, r0)))

    def coherencyWeakScrew(self, r, Ls, r0, phase='all'):
        '''
        Coherency effect for screw dislocation on weak and shearable particles
        '''
        return np.sqrt((9/5) * self.G**3 * self.b * self.eps[phase]**3 * r**3 / (Ls**2 * self.T(0, r0)))

    def coherencyStrongEdge(self, r, Ls, r0, phase='all'):
        '''
        Coherency effect for edge dislocation on strong and shearable particles
        '''
        return np.sqrt(2) * np.power(3, 3/8) * self.J / Ls * np.power((self.T(np.pi/2, r0)**3 * self.G * self.eps[phase] * r) / self.b**3, 1/4)

    def coherencyStrongScrew(self, r, Ls, r0, phase='all'):
        '''
        Coherency effect for screw dislocation on strong and shearable particles
        '''
        return 2 * self.J / Ls * np.power((self.T(0, r0)**3 * self.G * self.eps[phase] * r) / self.b**3, 1/4)

    def modulusWeakEdge(self, r, Ls, r0, phase='all'):
        '''
        Modulus effect for edge dislocation on weak and shearable particles
        '''
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp[phase] - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(np.pi/2, r0)), 3/2)

    def modulusWeakScrew(self, r, Ls, r0, phase='all'):
        '''
        Modulus effect for screw dislocation on weak and shearable particles
        '''
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(self.w1 * np.abs(self.Gp[phase] - self.G) * self.b**2 * np.power(r/self.b, self.w2) / (2*self.T(0, r0)), 3/2)

    def APBweakEdge(self, r, Ls, r0, phase='all'):
        '''
        Anti-phase boundary effect for edge dislocation on weak and shearable particles
        '''
        xi = 16 * self.yAPB[phase] * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(2 * self.yAPB[phase] * r / (2*self.T(np.pi/2, r0)), 3/2) - self.beta * xi)

    def APBweakScrew(self, r, Ls, r0, phase='all'):
        '''
        Anti-phase boundary effect for screw dislocation on weak and shearable particles
        '''
        xi = 16 * self.yAPB[phase] * r**2 / (3 * np.pi * self.b * Ls**2)
        return 2/self.s * (2 * self.T(0, r0) / (self.b * Ls) * np.power(2 * self.yAPB[phase] * r / (2*self.T(0, r0)), 3/2) - self.beta * xi)

    def APBstrongEdge(self, r, Ls, r0, phase='all'):
        '''
        Anti-phase boundary effect for edge dislocation on strong and shearable particles

        NOTE: Equation from paper would give np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2)) - 1), but this equation does not follow the plots they have
        '''
        return (2 * self.V * self.T(np.pi/2, r0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB[phase] * r) / (self.V * self.T(np.pi/2, r0)))

    def APBstrongScrew(self, r, Ls, r0, phase='all'):
        '''
        Anti-phase boundary effect for screw dislocation on strong and shearable particles

        NOTE: Equation from paper would give np.sqrt((np.pi * self.yAPB * r) / (self.V * self.T(np.pi/2)) - 1), but this equation does not follow the plots they have
        '''
        return (2 * self.V * self.T(0, r0)) / (np.pi * self.b * Ls) * np.sqrt((np.pi * self.yAPB[phase] * r) / (self.V * self.T(0, r0)))

    def SFEweakWideEdge(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for edge dislocation on weak and shearable particles for wide stacking faults
        '''
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP[phase]) / self.T(np.pi/2, r0), 3/2)

    def SFEweakWideScrew(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for screw dislocation on weak and shearable particles for wide stacking faults
        '''
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(r * (self.ySFM - self.ySFP[phase]) / self.T(0, r0), 3/2)

    def SFEstrongWide(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for edge dislocation on strong and shearable particles for wide stacking faults
        '''
        return self.J * 2 * r * (self.ySFM - self.ySFP[phase]) / (self.b * Ls)

    def SFEweakNarrowEdge(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for edge dislocation on weak and shearable particles for narrow stacking faults
        '''
        return (2 * self.T(np.pi/2, r0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(np.pi/2, phase) * r - self.SFEWeff(np.pi/2, phase)**2 / 4) / self.T(np.pi/2, r0), 3/2)

    def SFEweakNarrowScrew(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for screw dislocation on weak and shearable particles for narrow stacking faults
        '''
        return (2 * self.T(0, r0)) / (self.b * Ls) * np.power((self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(0, phase) * r - self.SFEWeff(0, phase)**2 / 4) / self.T(0, r0), 3/2)

    def SFEstrongNarrowEdge(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for edge dislocation on strong and shearable particles for narrow stacking faults
        '''
        return self.J * 2 * (self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(np.pi/2, phase) * r - self.SFEWeff(np.pi/2, phase)**2 / 4) / (self.b * Ls)

    def SFEstrongNarrowScrew(self, r, Ls, r0, phase='all'):
        '''
        Stacking fault energy effect for screw dislocation on strong and shearable particles for narrow stacking faults
        '''
        return self.J * 2 * (self.ySFM - self.ySFP[phase]) * np.sqrt(self.SFEWeff(0, phase) * r - self.SFEWeff(0, phase)**2 / 4) / (self.b * Ls)

    def interfacialWeakEdge(self, r, Ls, r0, phase='all'):
        '''
        Interfacial energy effect for edge dislocation on weak and shearable particles
        '''
        return 2 * self.T(np.pi/2, r0) / (self.b * Ls) * np.power(self.gamma[phase] * self.b / self.T(np.pi/2, r0), 3/2)

    def interfacialWeakScrew(self, r, Ls, r0, phase='all'):
        '''
        Interfacial energy effect for screw dislocation on weak and shearable particles
        '''
        return 2 * self.T(0, r0) / (self.b * Ls) * np.power(self.gamma[phase] * self.b / self.T(0, r0), 3/2)

    def interfacialStrongOld(self, r, Ls, r0, phase='all'):
        '''
        Interfacial energy effect strong and shearable particles (independent of dislocation type)
        '''
        return self.J * 2 * self.gamma[phase] / Ls

    