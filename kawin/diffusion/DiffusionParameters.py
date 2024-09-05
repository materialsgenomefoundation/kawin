import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable
import numpy as np

from kawin.thermo import GeneralThermodynamics
from kawin.thermo.Mobility import mobility_from_composition_set, x_to_u_frac, interstitials

MobilityData = namedtuple('MobilityData',
                         ['mobility', 'phases', 'phase_fractions', 'chemical_potentials'])

def wienerUpper(mobility, phase_fracs, *args, **kwargs):
    '''
    Upper wiener bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(phase_fracs[:,np.newaxis], modified_mob), axis=0)
    return avg_mob

def wienerLower(mobility, phase_fracs, *args, **kwargs):
    '''
    Lower wiener bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    avg_mob = 1/np.sum(np.multiply(phase_fracs[:,np.newaxis], 1/(modified_mob)), axis=0)
    return avg_mob

def labyrinth(mobility, phase_fracs, *args, **kwargs):
    '''
    Labyrinth mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    labyrinth_factor = kwargs.get('labyrinth_factor', 1)
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(np.power(phase_fracs[:,np.newaxis], labyrinth_factor), modified_mob), axis=0)
    return avg_mob

def _hashinShtrikmanGeneral(mobility, phase_fracs, extreme_mob):
    '''
    General hashin shtrikman bounds

    Ak = f^phi / (1 / (gamma^phi - gamma^alpha) + 1/(3*gamma^alpha))
    gamma^* = gamma^alpha + Ak / (1 - Ak / (3*gamma^alpha))

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)
    extreme_mob : list of length (e,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    Ak = phase_fracs[:,np.newaxis] * (mobility - extreme_mob) * (3*extreme_mob) / (2*extreme_mob + mobility)
    Ak = np.sum(Ak, axis=0)
    avg_mob = extreme_mob + Ak / (1 - Ak / (3*extreme_mob))
    return avg_mob

def hashinShtrikmanUpper(mobility, phase_fracs, *args, **kwargs):
    '''
    Upper hashin shtrikman bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    max_mob = np.amax(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashinShtrikmanGeneral(modified_mob, phase_fracs, max_mob)

def hashinShtrikmanLower(mobility, phase_fracs, *args, **kwargs):
    '''
    Lower hashin shtrikman bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    min_mob = np.amin(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashinShtrikmanGeneral(modified_mob, phase_fracs, min_mob)

def _postProcessDoNothing(therm, mobility, phase_fracs, *args, **kwargs):
    return mobility, phase_fracs

def _postProcessPredefinedMatrixPhase(therm, mobility, phase_fracs, *args, **kwargs):
    '''
    User will supply a predefined "alpha" phase, which the mobility is taken
    from for all undefined mobility

    Note: this assumes the user will know that "alpha" is continuously stable
    across the diffusion couple
    '''
    alpha_phase = args[0]
    alpha_idx = therm.phases.index(alpha_phase)
    alpha_mob = mobility[alpha_idx]
    for i in range(mobility.shape[1]):
        mobility[:,i][mobility[:,i] == -1] = alpha_mob[i]
    return mobility, phase_fracs

def _postProcessMajorityPhase(therm, mobility, phase_fracs, *args, **kwargs):
    '''
    Takes the majority phase and applies the mobility for all other phases
    with undefined mobility
    '''
    max_idx = np.argmax(phase_fracs)
    for i in range(mobility.shape[1]):
        mobility[:,i][mobility[:,i] == -1] = mobility[max_idx,i]
    return mobility, phase_fracs

def _postProcessExcludePhases(therm, mobility, phase_fracs, *args, **kwargs):
    '''
    For all excluded phases, the mobility and phase fraction will be set to 0
    This assumes that user knows the excluded phases to be minor or that the
    mobility is unknown
    '''
    excluded_phases = args[0]
    phase_idxs = [therm.phases.index(p) for p in excluded_phases]
    for p in phase_idxs:
        phase_fracs[p] = 0
    return mobility, phase_fracs

class HashTable:
    '''
    Implements a hash table that stores mobility, phases, phase fractions and
    chemical potentials for a given (composition, temperature) pair
    '''
    def __init__(self):
        self._cache = True
        self.cachedData = {}
        self.setHashSensitivity(4)

    def enableCaching(self, is_caching):
        self._cache = is_caching

    def clearCache(self):
        self.cachedData = {}

    def setHashSensitivity(self, s):
        '''
        Sets sensitivity of the hash table by significant digits

        For example, if a composition set is (0.5693, 0.2937) and s = 3, then
        the hash will be stored as (0.569, 0.294)

        Lower s values will give faster simulation times at the expense of accuracy

        Parameters
        ----------
        s : int
            Number of significant digits to keep for the hash table
        '''
        self.hash_sensitivity = np.power(10, int(s))

    def _hashingFunction(self, x, T):
        '''
        Gets hash value for a (compostion, temperature) pair

        Parameters
        ----------
        x : float, list[float]
        T : float
        '''
        return hash(tuple((np.concatenate((x, [T]))*self.hash_sensitivity).astype(np.int32)))

    def retrieveFromHashTable(self, x, T):
        '''
        Attempts to retrieve a stored value from the hash table

        Parameters
        ----------
        x : float, [float]
        T : float

        Returns
        -------
        Value or None (if no hash table for cached value does not exist)
        '''
        if self._cache is None:
            return None
        else:
            hash_value = self._hashingFunction(x, T)
            return self.cachedData.get(hash_value, None)
        
    def addToHashTable(self, x, T, value):
        '''
        Attempts to add a value to the hash table
        If no hash table, then this will not do anything

        Parameters
        ----------
        x : float, list[float]
        T : float
        '''
        if self._cache is not None:
            hash_value = self._hashingFunction(x, T)
            self.cachedData[hash_value] = value

class BoundaryConditions:
    '''
    Stores information about the boundary conditions
    Boundary conditions are stored as (type, value) where type can be
        FLUX_BC - Neumann condition
        COMPOSITION_BC = Dirichlet condition
    '''
    FLUX_BC = 0
    COMPOSITION_BC = 1

    LEFT = 3
    RIGHT = 4

    def __init__(self, elements):
        self.elements = elements
        self.leftBCtype, self.rightBCtype = self.FLUX_BC*np.ones(len(elements)), self.FLUX_BC*np.ones(len(elements))
        self.leftBC, self.rightBC = np.zeros(len(elements)), np.zeros(len(elements))

    def setBoundaryCondition(self, side, bc_type, value, element):
        '''
        Sets left boundary condition

        Parameters
        ----------
        side : BoundaryCondition.LEFT or BoundaryCondition.RIGHT
        bc_type : BoundaryCondition.FLUX_BC or BoundaryCondition.COMPOSITION_BC
        value : float
        element : str
        '''
        if element not in self.elements:
            print(f'Warning: {element} not in {self.elements}')
            return
        index = self.elements.index(element)
        if side == self.LEFT:
            self.leftBCtype[index] = bc_type
            self.leftBC[index] = value
        elif side == self.RIGHT:
            self.rightBCtype[index] = bc_type
            self.rightBC[index] = value

    def applyBoundaryConditionsToInitialProfile(self, x, z):
        for e in range(len(self.elements)):
            if self.leftBCtype[e] == self.COMPOSITION_BC:
                x[e,0] = self.leftBC[e]
            if self.rightBCtype[e] == self.COMPOSITION_BC:
                x[e,-1] = self.rightBC[e]

    def applyBoundaryConditionsToFluxes(self, fluxes):
        for e in range(len(self.elements)):
            fluxes[e,0] = self.leftBC[e] if self.leftBCtype[e] == self.FLUX_BC else fluxes[e,1]
            fluxes[e,-1] = self.rightBC[e] if self.rightBCtype[e] == self.FLUX_BC else fluxes[e,-2]

class CompositionProfile:
    '''
    Stores a series of information for how to build a composition profile
    for each element

    When building the profile, this will go through the list of steps
    that the user inputted
    '''
    LINEAR = 0
    STEP = 1
    SINGLE = 2
    BOUNDED = 3
    FUNCTION = 4
    PROFILE = 5
    
    def __init__(self, elements):
        self.elements = elements
        self.compositionSteps = [[] for _ in self.elements]

    def _getElementIndex(self, element):
        if element not in self.elements:
            print(f'Warning: {element} not in {self.elements}')
            return None
        return self.elements.index(element)

    def addCompositionBuildStep(self, element, profile_type, *args, **kwargs):
        index = self._getElementIndex(element)
        if index is not None:
            self.compositionSteps[index].append((profile_type, args, kwargs))

    def clearCompositionBuildSteps(self, element = None):
        if element is None:
            self.compositionSteps = [[] for _ in self.elements]

        index = self._getElementIndex(element)
        if index is not None:
            self.compositionSteps[index].clear()

    def addLinearCompositionStep(self, element, left_value, right_value):
        self.addCompositionBuildStep(element, self.LINEAR, left_value=left_value, right_value=right_value)

    def addStepCompositionStep(self, element, left_value, right_value, z_value):
        self.addCompositionBuildStep(element, self.STEP, left_value=left_value, right_value=right_value, z_value=z_value)

    def addSingleCompositionStep(self, element, value, z_value):
        self.addCompositionBuildStep(element, self.SINGLE, value=value, z_value=z_value)

    def addBoundedCompositionStep(self, element, value, left_z, right_z):
        self.addCompositionBuildStep(element, self.BOUNDED, value=value, left_z=left_z, right_z=right_z)

    def addFunctionCompositionStep(self, element, function):
        self.addCompositionBuildStep(element, self.FUNCTION, function=function)

    def addProfileCompositionStep(self, element, x_list, z_list):
        self.addCompositionBuildStep(element, self.PROFILE, x_list=x_list, z_list=z_list)

    def _setLinearComposition(self, element_idx, x, z, left_value, right_value):
        x[element_idx,:] = np.linspace(left_value, right_value, len(z))

    def _setStepComposition(self, element_idx, x, z, left_value, right_value, z_value):
        left_indices = z <= z_value
        x[element_idx,left_indices] = left_value
        x[element_idx,~left_indices] = right_value

    def _setSingleComposition(self, element_idx, x, z, value, z_value):
        z_idx = np.argmin(np.abs(z - z_value))
        x[element_idx,z_idx] = value

    def _setBoundedComposition(self, element_idx, x, z, value, left_z, right_z):
        z_indices = (z >= left_z) & (z <= right_z)
        x[element_idx,z_indices] = value

    def _setFunctionComposition(self, element_idx, x, z, function):
        x[element_idx,:] = function(z)

    def _setProfileComposition(self, element_idx, x, z, x_list, z_list):
        x[element_idx,:] = np.interp(z, z_list, x_list, x_list[0], x_list[-1])

    def buildProfile(self, x, z):
        build_functions = {
            self.LINEAR: self._setLinearComposition,
            self.STEP: self._setStepComposition,
            self.SINGLE: self._setSingleComposition,
            self.BOUNDED: self._setBoundedComposition,
            self.FUNCTION: self._setFunctionComposition,
            self.PROFILE: self._setProfileComposition
        }

        for i in range(len(self.elements)):
            for step_info in self.compositionSteps[i]:
                build_functions[step_info[0]](i, x, z, *step_info[1], **step_info[2])

class TemperatureParameters:
    def __init__(self):
        self.Tparameters = None
        self.Tfunction = None

    def setIsothermalTemperature(self, T):
        self.Tparameters = T
        self.Tfunction = lambda z, t: self.Tparameters*np.ones(len(z))

    def setTemperatureArray(self, times, temperatures):
        self.Tparameters = (times, temperatures)
        self.Tfunction = lambda z, t: np.interp(t/3600, self.Tparameters[0], self.Tparameters[1], self.Tparameters[1][0], self.Tparameters[1][-1]) * np.ones(len(z))

    def setTemperatureFunction(self, func):
        self.Tparameters = func
        self.Tfunction = lambda z, t: self.Tparameters(z, t)

    def __call__(self, z, t):
        return self.Tfunction(z, t)

class HomogenizationParameters:
    def __init__(self):
        self.setHomogenizationFunction('wiener upper')
        self.labyrinthFactor: float = 1
        self.setPostProcessFunction(None)
        self.eps = 0.05

    def setPostProcessFunction(self, functionName, functionArgs = None):
        '''
        Returns post process function
        '''
        self.postProcessParameters = [functionArgs]
        if functionName is None:
            self.postProcessFunction = _postProcessDoNothing
        else:
            if functionName == 'predefined':
                self.postProcessFunction = _postProcessPredefinedMatrixPhase
            elif functionName == 'majority':
                self.postProcessFunction = _postProcessMajorityPhase
            elif functionName == 'exclude':
                self.postProcessFunction = _postProcessExcludePhases
            else:
                raise "Error: post process function should be \'predefined\', \'majority\' or \'exclude\'"
            
    def setHomogenizationFunction(self, function):
        '''
        Sets averaging function to use for mobility

        Default mobility value should be that a phase of unknown mobility will be ignored for average mobility calcs

        Parameters
        ----------
        function : str
            Options - 'upper wiener', 'lower wiener', 'upper hashin-shtrikman', 'lower hashin-strikman', 'labyrinth'
        '''
        if 'upper' in function and 'wiener' in function:
            self.homogenizationFunction = wienerUpper
        elif 'lower' in function and 'wiener' in function:
            self.homogenizationFunction = wienerLower
        elif 'upper' in function and 'hashin' in function:
            self.homogenizationFunction = hashinShtrikmanUpper
        elif 'lower' in function and 'hashin' in function:
            self.homogenizationFunction = hashinShtrikmanLower
        elif 'lab' in function:
            self.homogenizationFunction = labyrinth

    def setLabyrinthFactor(self, n):
        '''
        Labyrinth factor

        Parameters
        ----------
        n : int
            Either 1 or 2
            Note: n = 1 will the same as the weiner upper bounds
        '''
        self.labyrinthFactor = np.clip(n, 1, 2)

class DiffusionParameters:
    def __init__(self, elements, 
                 temperatureParameters = None, 
                 boundaryCondition = None,
                 compositionProfile = None,
                 hashTable = None,
                 homogenizationParameters = None,
                 minComposition = 1e-8,
                 maxCompositionChange = 0.002):
        self.temperature = TemperatureParameters() if temperatureParameters is None else temperatureParameters
        self.boundaryConditions = BoundaryConditions(elements) if boundaryCondition is None else boundaryCondition
        self.compositionProfile = CompositionProfile(elements) if compositionProfile is None else compositionProfile
        self.hashTable = HashTable() if hashTable is None else hashTable
        self.homogenizationParameters = HomogenizationParameters() if homogenizationParameters is None else homogenizationParameters

        self.minComposition = minComposition
        self.maxCompositionChange = maxCompositionChange

def _computeSingleMobility(therm: GeneralThermodynamics, x, T, unsortIndices, diffusionParameters: DiffusionParameters) -> MobilityData:
    '''
    Gets mobility data for x and T

    If mobility does not exist for a phase or a phase is unstable, then the mobility is set to -1

    Parameters
    ----------
    therm : GeneralThermodynamics object
    x : float, list[float]
        For binary, it must be a float
        For ternary, it must be list[float]
    T : float
        Temperature(s)
    homogenization_parameters: HomogenizationParameters
        
    Returns
    -------
    MobilityData
    '''
    mobility_data = diffusionParameters.hashTable.retrieveFromHashTable(x, T)
    if mobility_data is None:
        # Compute equilibrium
        try:
            wks = therm.getEq(x, T, 0, therm.phases)
            chemical_potentials = np.squeeze(wks.eq.MU)[unsortIndices]
            comp_sets = wks.get_composition_sets()
        except Exception as e:
            print(f'Error at {x}, {T}')
            raise e
        
        # Compute mobility and phase fractions
        mob = -1*np.ones((len(comp_sets), len(therm.elements)-1))
        phases, phase_fracs = zip(*[(cs.phase_record.phase_name, cs.NP) for cs in comp_sets])
        phases = np.array(phases)
        phase_fracs = np.array(phase_fracs, dtype=np.float64)
        for p, cs in enumerate(comp_sets):
            if therm.mobCallables.get(phases[p], None) is not None:
                mob[p,:] = mobility_from_composition_set(cs, therm.mobCallables[phases[p]], therm.mobility_correction)[unsortIndices]
                mob[p,:] *= x_to_u_frac(np.array(cs.X, dtype=np.float64)[unsortIndices], therm.elements[:-1], interstitials)

        mobility_data = MobilityData(mobility = mob, phases = phases, phase_fractions = phase_fracs, chemical_potentials=chemical_potentials)
        diffusionParameters.hashTable.addToHashTable(x, T, mobility_data)

    return mobility_data

def computeMobility(therm : GeneralThermodynamics, x, T, diffusionParameters : DiffusionParameters):
    # x should always be 2d, T should always be 1d
    x = np.atleast_2d(np.squeeze(x))
    if therm._isBinary:
        x = x.T
    T = np.atleast_1d(T)

    # we want to make sure that the mobility is sorted to be the same order as listed in therm
    sortIndices = np.argsort(therm.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    mob = []
    phases = []
    phase_fracs = []
    chemical_potentials = []
    for i in range(len(x)):
        mobility_data = _computeSingleMobility(therm, x[i], T[i], unsortIndices, diffusionParameters)
        mob.append(mobility_data.mobility)
        phases.append(mobility_data.phases)
        phase_fracs.append(mobility_data.phase_fractions)
        chemical_potentials.append(mobility_data.chemical_potentials)
    return MobilityData(mobility=mob, phases=phases, phase_fractions=phase_fracs, chemical_potentials=chemical_potentials)

def computeHomogenizationFunction(therm : GeneralThermodynamics, x, T, diffusionParameters : DiffusionParameters):
    '''
    Compute homogenization function (defined by HomogenizationParameters) for list of x,T

    Parameters
    ----------
    therm : GeneralThermodynamics object
    x : float, list[float], list[list[float]]
        Compositions
        For binary, it must be either a float or list[float] (shape N) for multiple compositions
        For ternary, it must be either list[float] (shape e) or list[list[float]] (shape N x e) for multiple compositions
    T : float, list[float]
        Temperature(s)
        First dimensions must correspond to first dimension of x
    diffusion_parameters: DiffusionParameters
        
    Returns
    -------
    average mobility array - (N, e)
    chemical potential array - (N, e)

    Where N is size of (x,T), and e is number of elements
    '''
    # x should always be 2d, T should always be 1d
    x = np.atleast_2d(np.squeeze(x))
    if therm._isBinary:
        x = x.T
    T = np.atleast_1d(T)

    # we want to make sure that the mobility is sorted to be the same order as listed in therm
    sortIndices = np.argsort(therm.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    avg_mob = np.zeros((x.shape[0], len(therm.elements)-1))
    chemical_potentials = np.zeros((x.shape[0], len(therm.elements)-1))
    for i in range(len(x)):
        mobility_data = _computeSingleMobility(therm, x[i], T[i], unsortIndices, diffusionParameters)
        mob = mobility_data.mobility
        phase_fracs = mobility_data.phase_fractions
        chemical_potentials[i,:] = mobility_data.chemical_potentials

        mob, phase_fracs = diffusionParameters.homogenizationParameters.postProcessFunction(therm, mob, phase_fracs, *diffusionParameters.homogenizationParameters.postProcessParameters)
        avg_mob[i] = diffusionParameters.homogenizationParameters.homogenizationFunction(mob, phase_fracs, labyrinth_factor = diffusionParameters.homogenizationParameters.labyrinthFactor)

    return np.squeeze(avg_mob), np.squeeze(chemical_potentials)

