'''
Defines strength model for precipitation hardening

Following implementation described in
M.R. Ahmadi, E. Povoden-Karadeni, K.I. Oksuz, A. Falahati and E. Kozeschnik
    Computational Materials Science 91 (2014) 173-186
and Alloy yield strength modeling with MatCalc, P. Warczok, MatCalc, presentation

The paper and presentation give the same equations, except that the presentation combines
the edge and screw coherency contributions in one equation to give a mixed dislocation character contribution

6 contributions are accounted for
For dislocation cutting, contributions are coherency, modulus, anti-phase boundary, stacking fault energy and interfacial energy
For dislocation bowing, contribution is orowan

Contributions can be added for all phases or for a single phase
'''

from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from kawin.precipitation import PrecipitateModel, PrecipitateParameters
from kawin.precipitation.Plot import _get_time_axis, _get_axis, _adjust_kwargs

def ignore_numpy_warnings(func):
    '''
    The strength contributions may be undefined at rss=0 or ls=0, so this
    is just an easy way to ignore any divide by 0 warnings

    TODO: a better way would be to know the function limits for rss=0 and ls=0,
          then apply the function limits at those values instead of computing, but
          ignoring the warnings here is a lot easier for now
    '''
    def wrapper(*args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            return func(*args, **kwargs)
    return wrapper

class DislocationParameters:
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
    def __init__(self, G, b, nu = 1/3, ri = None, theta = 90, psi = 120):
        self.G = G
        self.b = b
        self.nu = nu
        self.ri = b if ri is None else ri
        self.theta = theta * np.pi/180
        self.psi = psi * np.pi/180

    def tension(self, r0, theta = None):
        theta = self.theta if theta is None else theta
        return self.G*self.b**2 / (4*np.pi) * (1 + self.nu - 3*self.nu*np.sin(theta)**2) / (1 - self.nu) * np.log(r0 / self.ri)
    
ShearingStrength = namedtuple('ShearingStrength', ['weak', 'strong'])

class StrengthContributionBase(ABC):
    name = 'STRENGTH_BASE'

    def __init__(self, phase = 'all'):
        self.phase = phase

    def r0Weak(self, Ls, dislocations: DislocationParameters):
        return Ls / np.sqrt(np.cos(dislocations.psi / 2))
    
    def r0Strong(self, Ls, dislocations: DislocationParameters):
        return Ls

    @abstractmethod
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        raise NotImplementedError()
    
class OrowanContribution(StrengthContributionBase):
    '''
    Orowan strengthening contribution
    '''
    name = 'OROWAN'

    @ignore_numpy_warnings
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        G = dislocations.G
        b = dislocations.b
        nu = dislocations.nu
        ri = dislocations.ri
        return G*b / (2*np.pi*np.sqrt(1 - nu)*Ls) * np.log(2*r/ri)
    
class CoherencyContribution(StrengthContributionBase):
    '''
    Parameters for coherency effect

    Parameters
    ----------
    eps : float
        Lattice misfit strain
    '''
    name = 'COHERENCY'

    def __init__(self, eps, phase='all'):
        super().__init__(phase)
        self.eps = eps

    @staticmethod
    def latticeMisfit(delta, dislocations: DislocationParameters):
        '''
        Strain (eps) from lattice misfit (delta)
        '''
        nu = dislocations.nu
        return (1/3)*(1+nu)/(1-nu)*delta

    @ignore_numpy_warnings
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        theta = dislocations.theta
        G = dislocations.G
        b = dislocations.b
        tensionWeak = dislocations.tension(self.r0Weak(Ls, dislocations))
        tensionStrong = dislocations.tension(self.r0Strong(Ls, dislocations))

        weak = (1.3416*np.cos(theta)**2 + 4.1127*np.sin(theta)**2)/Ls * np.sqrt(b*(G*self.eps*r)**3/tensionWeak)
        strong = (2*np.cos(theta)**2 + 2.1352*np.sin(theta)**2)/Ls * np.power(G*self.eps*r*(tensionStrong/b)**3, 1/4)
        return ShearingStrength(weak=weak, strong=strong)

class ModulusContribution(StrengthContributionBase):
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
    '''
    name = 'MODULUS'

    def __init__(self, Gp, w1=0.05, w2=0.85, phase='all'):
        super().__init__(phase)
        self.Gp = Gp
        self.w1 = w1
        self.w2 = w2

    def f(self, r, G, b):
        '''
        Term for modulus effect
        '''
        return self.w1*np.abs(G-self.Gp)*np.power(r/b, self.w2)*b**2

    @ignore_numpy_warnings
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        '''
        Modulus effect for mixed dislocation on weak and shearable particles
        '''
        G = dislocations.G
        b = dislocations.b
        tensionWeak = dislocations.tension(self.r0Weak(Ls, dislocations))
        weak = 2*tensionWeak/(b*Ls) * np.power(self.f(r, G, b)/(2*tensionWeak), 3/2)
        strong = self.f(r, G, b)/(b*Ls)
        return ShearingStrength(weak=weak, strong=strong)
    
class APBContribution(StrengthContributionBase):
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
    '''
    name = 'APB'

    def __init__(self, yAPB, s=2, beta=1, V=2.8, phase='all'):
        super().__init__(phase)
        self.yAPB = yAPB
        self.s = s
        self.beta = beta
        self.V = V

    @ignore_numpy_warnings
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        b = dislocations.b
        tensionWeak = dislocations.tension(self.r0Weak(Ls, dislocations))
        tensionStrong = dislocations.tension(self.r0Strong(Ls, dislocations))
        weak = 2/(self.s*b*Ls) * (2*tensionWeak*np.power(r*self.yAPB/tensionWeak, 3/2) - 16*self.beta*self.yAPB*r**2/(3*np.pi*Ls))
        strong = 0.69/(b*Ls) * np.sqrt(8*self.V*tensionStrong*r*self.yAPB/3)
        return ShearingStrength(weak=weak, strong=strong)
    
class SFEContribution(StrengthContributionBase):
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
    '''
    name = 'SFE'

    def __init__(self, ySFM, ySFP, bp = None, phase='all'):
        super().__init__(phase)
        self.ySFM = ySFM
        self.ySFP = ySFP
        self.bp = bp

    def K(self, dislocations: DislocationParameters):
        G = dislocations.G
        bp = dislocations.b if self.bp is None else self.bp
        nu = dislocations.nu
        theta = dislocations.theta
        return G*bp**2 * (2 - nu - 2*nu*np.cos(2*theta))/(8*np.pi*(1 - nu))

    def Weff(self, dislocations: DislocationParameters):
        '''
        Effective stacking fault width for mixed dislocations
        '''
        return 2*self.K(dislocations)/(self.ySFM + self.ySFP)

    def f(self, r, dislocations: DislocationParameters):
        '''
        Stacking fault term
        '''
        return 2*(self.ySFM - self.ySFP)*np.sqrt(self.Weff(dislocations)*r - self.Weff(dislocations)**2/4)

    @ignore_numpy_warnings
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        b = dislocations.b
        tensionWeak = dislocations.tension(self.r0Weak(Ls, dislocations))
        weak = 2*tensionWeak/(b*Ls)*np.power(self.f(r, dislocations)/(2*tensionWeak), 3/2)
        strong = self.f(r, dislocations)/(b*Ls)
        return ShearingStrength(weak=weak, strong=strong)

class InterfacialContribution(StrengthContributionBase):
    '''
    Parameters for interfacial effect

    Parameters
    ----------
    gamma : float
        Interfacial energy of matrix/precipitate surface
    '''
    name = 'INTERFACIAL'

    def __init__(self, gamma, phase='all'):
        super().__init__(phase)
        self.gamma = gamma

    @ignore_numpy_warnings
    def computeCRSS(self, r, Ls, dislocations: DislocationParameters):
        b = dislocations.b
        tensionWeak = dislocations.tension(self.r0Weak(Ls, dislocations))
        weak = 2*tensionWeak/(b*Ls)*np.power(self.gamma*b/tensionWeak, 3/2)
        strong = 2*self.gamma/Ls
        return ShearingStrength(weak=weak, strong=strong)
    
class SolidSolutionStrength:
    '''
    Model for solid solution strengthening

    Parameters
    ----------
    weights : dictionary
        Dictionary mapping element (str) to weight (float). Weights are in units of (Pa)
    exp : float
        Exponential factor for weights
    '''
    def __init__(self, weights = {}, exp = {}):
        self.weights = weights
        self.exp = exp
    
    def compute(self, composition, elements):
        composition = np.atleast_2d(composition)
        val = np.zeros(len(composition))
        for i in range(len(elements)):
            e = elements[i]
            val += np.power(self.weights.get(e,0)*composition[:,i], self.exp.get(e,1))
        return np.squeeze(val)
    
def computeCRSS(rss, Ls, contributions: list[StrengthContributionBase], dislocations: DislocationParameters, phase):
    '''
    Computes critical resolved shear stress from precipitate contributions over rss, Ls
    Weak and strong values will be mapped to the specific contribution
    Orowan is a single contribution that's added by default
    '''
    rss = np.atleast_1d(rss)
    Ls = np.atleast_1d(Ls)

    phaseContributions = {}
    for c in contributions:
        if isinstance(c, OrowanContribution):
            continue
        # This will override any existing contribution to the phase specific one
        if c.phase == phase:
            phaseContributions[c.name] = c
        elif c.phase == 'all':
            # If we already added a phase-specific contribution, don't add it again
            if c.name not in contributions:
                phaseContributions[c.name] = c

    # weak and strong values will be a mapping from contribution name -> CRSS
    weakValues = {}
    strongValues = {}
    for name, c in phaseContributions.items():
        strength = c.computeCRSS(rss, Ls, dislocations)
        # If strength is a ShearingStrength, then split between strong and weak contributions
        if isinstance(strength, ShearingStrength):
            weak, strong = strength.weak, strength.strong
        else:
            weak, strong = strength, strength

        weak[(weak < 0) | ~np.isfinite(weak)] = 0
        strong[(strong < 0) | ~np.isfinite(strong)] = 0
        weakValues[name] = np.squeeze(weak)
        strongValues[name] = np.squeeze(strong)

    orowan = np.squeeze(OrowanContribution().computeCRSS(rss, Ls, dislocations))
    return weakValues, strongValues, orowan
    
def combineCRSS(weak, strong, owo, exp = 1.8, returnContributions = False):
    '''
    Sums the different critical resolved shear stress contributions together
    '''
    # sums contribution by (s1^exp + s2^exp + ...)^(1/exp)
    # returns 0 if no contribution exist
    def sumArray(contribution):
        if len(contribution) == 0:
            conSum = np.zeros(owo.shape)
        else:
            con = np.array(list(contribution.values()))
            conSum = np.power(np.sum(np.power(con, exp), axis=0), 1/exp)
        return conSum
    
    # ensures that array is 1d and all negative/undefined values are 0
    def processArray(contribution):
        contribution = np.atleast_1d(contribution)
        contribution[(contribution < 0) | ~np.isfinite(contribution)] = 0
        return contribution
    
    owo = processArray(owo)
    weakSum = processArray(sumArray(weak))
    strongSum = processArray(sumArray(strong))

    taumin = np.amin(np.array([weakSum, strongSum, owo]), axis=0)
    if returnContributions:
        return np.squeeze(taumin), np.squeeze(weakSum), np.squeeze(strongSum), np.squeeze(owo)
    else:
        return np.squeeze(taumin)
    
class StrengthModel:
    '''
    Strength model for coupling with precipitate model

    Parameters
    ----------
    phases: list[PrecipitateParameters|str]
        List of phases to model
        If coupling with a precipitate model, must be the same phases
    contributions: StrengthContributionBase | list[StrengthContributionBase]
    dislocations: DislocationParameters
    ssModel: SolidSolutionStrength (optional)
        If None, then solid solution strength model will output 0
    sigma0: float (optional)
        Base strength of all
        Default is 0
    '''
    def __init__(self, phases: list[PrecipitateParameters | str], 
                 contributions: StrengthContributionBase | list[StrengthContributionBase], 
                 dislocations: DislocationParameters,
                 ssModel: SolidSolutionStrength = None, 
                 sigma0: float = 0):
        if isinstance(phases, PrecipitateParameters) or isinstance(phases, str):
            phases = [phases]
        if isinstance(phases[0], PrecipitateParameters):
            phases = [p.phase for p in phases]
        self.phases = phases

        #Taylor factor for converting critical resolved shear stress to yield strength
        self.M = 2.24

        #Single phase superposition exponent
        self.singlePhaseExp = 1.8

        #Multi-phase superposition exponents
        self.multiphaseSameExp = 1.8
        self.multiphaseMixedExp = 1.4

        #Superposition exponent for total strength
        self.totalStrengthExp = 1.8

        #Strength terms
        self.rss = np.zeros((1, len(self.phases)))
        self.Ls = np.zeros((1, len(self.phases)))

        self.dislocations = dislocations
        if isinstance(contributions, StrengthContributionBase):
            contributions = [contributions]
        self.contributions = contributions
        self.ssModel = ssModel
        self.sigma0 = sigma0

    def save(self, filename):
        '''
        Saves strength model data

        Note, this only saves solid solution strength, the rss and ls terms
        Parameters should be free so user can load model and evaluate different parameters
        '''
        np.savez_compressed(filename, rss = self.rss, Ls = self.Ls)

    def load(self, filename):
        data = np.load(filename)
        self.rss = data['rss']
        self.Ls = data['Ls']
    
    def updateCoupledModel(self, model: PrecipitateModel):
        '''
        Computes rss and Ls from current state of the PrecipitateModel

        rss - mean projected radius of particles
        Ls - mean surface to surface distance between particles

        r1 = first ordered moment of particle size distribution
        r2 = second ordered moment of particle size distribution
        rss = sqrt(2/3) * r2 / r1
        ls = sqrt(ln(3)/(2*pi*r1) + (2*rss)^2) - 2*rss

        Parameters
        ----------
        model : PrecpitateModel
        '''
        rss = np.zeros(len(model.phases))
        Ls = np.zeros(len(model.phases))
        for p in range(len(model.phases)):
            r1 = model.PBM[p].firstMoment()
            r2 = model.PBM[p].secondMoment()
            if r1 > 0:
                rss[p] = np.sqrt(2/3) * r2 / r1
                Ls[p] = np.sqrt(np.log(3) / (2*np.pi*r1) + (2*rss)**2) - 2*rss

        self.rss = np.append(self.rss, [rss], axis=0)
        self.Ls = np.append(self.Ls, [Ls], axis=0)

    def computePrecipitateStrength(self, model: PrecipitateModel):
        '''
        Computes yield strength from precipitate contributions

        Precipitate contributions compute CRSS, so we multiply by Taylor factor (M) to get YS
        '''
        ps = []
        totalCompare = np.zeros(len(self.rss[:,0]))
        totalStrength = np.zeros(len(self.rss[:,0]))
        for i in range(len(model.phases)):
            weak, strong, owo = computeCRSS(self.rss[:,i], self.Ls[:,i], self.contributions, self.dislocations, model.phases[i])
            strength, weakSum, strongSum, owo = combineCRSS(weak, strong, owo, self.singlePhaseExp, True)
            
            # if weak is larger than strong or orowan contributions, then add to compare
            compare = (weakSum > strongSum) & (weakSum > owo)
            compare[~np.isfinite(strength)] = 0
            strength[~np.isfinite(strength)] = 0
            ps.append(strength)
            totalCompare += np.array(compare, dtype='int')

        ps = np.array(ps)
        # If contributions from each phase are all weak or all (strong or orowan), then we use the multiphaseSameExp (for same strengthening mechanism)
        # else, we have mixed mechanisms for the strength contributions, so use the multiphaseMixedExp when summing the contributions
        indices = (totalCompare == 0) | (totalCompare == len(model.phases))
        totalStrength[indices] = np.power(np.sum(np.power(ps[:,indices], self.multiphaseSameExp), axis=0), 1/self.multiphaseSameExp)
        totalStrength[~indices] = np.power(np.sum(np.power(ps[:,~indices], self.multiphaseMixedExp), axis=0), 1/self.multiphaseMixedExp)
        return self.M*totalStrength
    
    def totalStrength(self, model: PrecipitateModel, returnContributions = False):
        '''
        Compute total strength from precipitate, solid solution and base contributions
        All strength are assumed to be yield strength
        '''
        precStrength = self.computePrecipitateStrength(model)
        if self.ssModel is None:
            ssStrength = np.zeros(len(precStrength))
        else:
            ssStrength = self.ssModel.compute(model.data.composition, model.elements)
        baseStrength = self.sigma0*np.ones(len(precStrength))
        strength = np.power(np.sum(np.power([baseStrength, ssStrength, precStrength], self.totalStrengthExp), axis=0), 1/self.totalStrengthExp)
        strength[(strength < 0) | ~np.isfinite(strength)] = 0
        if returnContributions:
            return strength, precStrength, ssStrength, baseStrength
        else:
            return strength

def _get_strength_units(units = 'Pa'):
    yscale = 1
    formattedUnit = 'Pa'
    if units.lower() == 'kpa':
        yscale = 1e3
        formattedUnit = 'kPa'
    elif units.lower() == 'mpa':
        yscale = 1e6
        formattedUnit = 'MPa'
    elif units.lower() == 'gpa':
        yscale = 1e9
        formattedUnit = 'GPa'
    return yscale, formattedUnit

def _plotContributionOverX(x, r, Ls, contribution: StrengthContributionBase, dislocations: DislocationParameters, strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots single strengthening contribution
    
    Parameters
    ----------
    x: np.array
        x coordinates
    r: np.array
        Mean projected particle radius
    Ls: np.array
        Surface-surface particle distance
    contribution: StrengthContributionBase
    dislocations: DislocationParameters
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    ax = _get_axis(ax)
    yscale, unit = _get_strength_units(strengthUnits)
    strength = contribution.computeCRSS(r, Ls, dislocations)
    if isinstance(strength, ShearingStrength):
        ax.plot(x, strength.weak/yscale, *args, **_adjust_kwargs('Weak', {'label': 'Weak'}, **kwargs))
        ax.plot(x, strength.strong/yscale, *args, **_adjust_kwargs('Strong', {'label': 'Strong'}, **kwargs))
        ax.legend()
    else:
        ax.plot(x, strength/yscale, *args, **kwargs)

    label = contribution.name.lower()
    ax.set_ylabel(r'$\tau_{' + label + '}$ (' + unit + ')')
    ax.set_ylim(bottom=0)
    ax.set_xlim([x[0], x[-1]])
    return ax

def plotContribution(r, Ls, contribution: StrengthContributionBase, dislocations: DislocationParameters, strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots single strengthening contribution
    This plots the critical resolved shear stress vs. mean projected particle radius

    Parameters
    ----------
    r: np.array
        Mean projected particle radius
    Ls: np.array
        Surface-surface particle distance
    contribution: StrengthContributionBase
    dislocations: DislocationParameters
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    ax = _plotContributionOverX(r, r, Ls, contribution, dislocations, strengthUnits=strengthUnits, ax=ax, *args, **kwargs)
    ax.set_xlabel('Radius (m)')
    return ax

def plotContributionOverTime(model: PrecipitateModel, strengthModel: StrengthModel, contribution: StrengthContributionBase, timeUnits='s', strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots single strengthening contribution from precipitate evolution in a precipitation model
    This plots the critical resolved shear stress vs. time

    Parameters
    ----------
    model: PrecipitateModel
    strengthModel: StrengthModel
    contribution: StrengthContributionBase
    timeUnits: str (optional)
        s, min, or h
        Default is s
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    timeScale, timeLabel, bounds = _get_time_axis(model.data.time, timeUnits=timeUnits)
    ax = _plotContributionOverX(timeScale*model.data.time, strengthModel.rss, strengthModel.Ls, contribution, strengthModel.dislocations, strengthUnits=strengthUnits, ax=ax, *args, **kwargs)
    ax.set_xlabel(timeLabel)
    ax.set_xlim(bounds)
    ax.set_xscale('log')
    return ax

def _plotPrecipitateStrengthOverX(x, r, Ls, strengthModel: StrengthModel, phase, plotContributions=False, strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots total strengthening contribution of single precipitate

    Parameters
    ----------
    x: np.array
        x coordinates
    r: np.array
        Mean projected particle radius
    Ls: np.array
        Surface-surface particle distance
    strengthModel: StrengthModel
    phase: str (optional)
        If None, then first phase of strength model is used
    plotContributions: bool (optional)
        If True, will plot the weak, strong, orowan and minimum contribution
        If False, will only plot the minimum contribution
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    ax = _get_axis(ax)
    yscale, unit = _get_strength_units(strengthUnits)
    weak, strong, owo = computeCRSS(r, Ls, strengthModel.contributions, strengthModel.dislocations, phase)
    if plotContributions:
        strength, weak, strong, owo = combineCRSS(weak, strong, owo, strengthModel.singlePhaseExp, True)
        ax.plot(x, weak/yscale, *args, **_adjust_kwargs('Weak', {'label': 'Weak'}, **kwargs))
        ax.plot(x, strong/yscale, *args, **_adjust_kwargs('Strong', {'label': 'Strong'}, **kwargs))
        ax.plot(x, owo/yscale, *args, **_adjust_kwargs('Orowan', {'label': 'Orowan'}, **kwargs))
        ax.plot(x, strength/yscale, *args, **_adjust_kwargs('Minimum', {'label': 'Minimum'}, **kwargs))
        ax.legend()
    else:
        strength = combineCRSS(weak, strong, owo, strengthModel.singlePhaseExp, False)
        ax.plot(x, strength/yscale, *args, **kwargs)

    ax.set_ylabel(r'$\tau$ (' + unit + ')')
    ax.set_ylim(bottom=0)
    ax.set_xlim([x[0], x[-1]])
    return ax

def plotPrecipitateStrength(r, Ls, strengthModel: StrengthModel, phase=None, plotContributions=False, strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots total strengthening contribution of single precipitate
    This plots the critical resolved shear stress vs. mean projected particle radius

    Parameters
    ----------
    r: np.array
        Mean projected particle radius
    Ls: np.array
        Surface-surface particle distance
    strengthModel: StrengthModel
    phase: str (optional)
        If None, then first phase of strength model is used
    plotContributions: bool (optional)
        If True, will plot the weak, strong, orowan and minimum contribution
        If False, will only plot the minimum contribution
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    if phase is None:
        phase = strengthModel.phases[0]
    ax = _plotPrecipitateStrengthOverX(r, r, Ls, strengthModel, phase, plotContributions=plotContributions, strengthUnits=strengthUnits, ax=ax, *args, **kwargs)
    ax.set_xlabel('Radius (m)')
    return ax

def plotPrecipitateStrengthOverTime(model: PrecipitateModel, strengthModel: StrengthModel, phase=None, plotContributions=False, timeUnits='s', strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots total strengthening contribution of single precipitate
    This plots the critical resolved shear stress vs. time

    Parameters
    ----------
    model: PrecipitateModel
    strengthModel: StrengthModel
    phase: str (optional)
        If None, then first phase of strength model is used
    plotContributions: bool (optional)
        If True, will plot the weak, strong, orowan and minimum contribution
        If False, will only plot the minimum contribution
    timeUnits: str (optional)
        s, min, or h
        Default is s
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    if phase is None:
        phase = strengthModel.phases[0]
    timeScale, timeLabel, bounds = _get_time_axis(model.data.time, timeUnits=timeUnits)
    ax = _plotPrecipitateStrengthOverX(timeScale*model.data.time, strengthModel.rss, strengthModel.Ls, strengthModel, phase, plotContributions=plotContributions, strengthUnits=strengthUnits, ax=ax, *args, **kwargs)
    ax.set_xlabel(timeLabel)
    ax.set_xlim(bounds)
    ax.set_xscale('log')
    return ax

def plotAlloyStrength(model: PrecipitateModel, strengthModel: StrengthModel, plotContributions=False, timeUnits='s', strengthUnits='MPa', ax=None, *args, **kwargs):
    '''
    Plots total predicted strength of an alloy during precipitate evolution
    This plots the yield strength vs. time

    Parameters
    ----------
    model: PrecipitateModel
    strengthModel: StrengthModel
    plotContributions: bool (optional)
        If True, will plot the base, solid solution, precipitate and total contribution
        If False, will only plot the total contribution
    timeUnits: str (optional)
        s, min, or h
        Default is s
    strengthUnits: str (optional)
        Pa, kPa, MPa or GPa
        Default is MPa
    ax: matplotlib axis
    '''
    ax = _get_axis(ax)
    timeScale, timeLabel, bounds = _get_time_axis(model.data.time, timeUnits=timeUnits)
    yscale, unit = _get_strength_units(strengthUnits)

    if plotContributions:
        strength, prec, ss, base = strengthModel.totalStrength(model, True)
        ax.plot(timeScale*model.data.time, prec/yscale, *args, **_adjust_kwargs('Precipitate', {'label': 'Precipitate'}, **kwargs))
        ax.plot(timeScale*model.data.time, ss/yscale, *args, **_adjust_kwargs('Solid Solution', {'label': 'Solid Solution'}, **kwargs))
        ax.plot(timeScale*model.data.time, base/yscale, *args, **_adjust_kwargs('Base Strength', {'label': 'Base Strength'}, **kwargs))
        ax.plot(timeScale*model.data.time, strength/yscale, *args, **_adjust_kwargs('Total', {'label': 'Total'}, **kwargs))
        ax.legend()
    else:
        strength = strengthModel.totalStrength(model, False)
        ax.plot(timeScale*model.data.time, strength/yscale, *args, **kwargs)

    ax.set_ylabel(r'Strength (' + unit + ')')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(timeLabel)
    ax.set_xlim(bounds)
    ax.set_xscale('log')
    return ax