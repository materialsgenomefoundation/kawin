import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Union
import warnings

import numpy as np

from kawin.thermo.utils import _process_xT_arrays
from kawin.thermo import GeneralThermodynamics
from kawin.thermo.Mobility import mobility_from_composition_set, x_to_u_frac, interstitials

MobilityData = namedtuple('MobilityData',
                         ['mobility', 'phases', 'phase_fractions', 'chemical_potentials'])

def wienerUpper(mobility: np.array, phase_fracs: np.array, *args, **kwargs) -> np.array:
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

def wienerLower(mobility: np.array, phase_fracs: np.array, *args, **kwargs) -> np.array:
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

def labyrinth(mobility: np.array, phase_fracs: np.array, *args, **kwargs) -> np.array:
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

def _hashinShtrikmanGeneral(mobility: np.array, phase_fracs: np.array, extreme_mob: np.array) -> np.array:
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

def hashinShtrikmanUpper(mobility: np.array, phase_fracs: np.array, *args, **kwargs) -> np.array:
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

def hashinShtrikmanLower(mobility: np.array, phase_fracs: np.array, *args, **kwargs) -> np.array:
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

def _postProcessDoNothing(therm: GeneralThermodynamics, mobility: np.array, phase_fracs: np.array, *args, **kwargs):
    return mobility, phase_fracs

def _postProcessPredefinedMatrixPhase(therm: GeneralThermodynamics, mobility: np.array, phase_fracs: np.array, *args, **kwargs):
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

def _postProcessMajorityPhase(therm: GeneralThermodynamics, mobility: np.array, phase_fracs: np.array, *args, **kwargs):
    '''
    Takes the majority phase and applies the mobility for all other phases
    with undefined mobility
    '''
    max_idx = np.argmax(phase_fracs)
    for i in range(mobility.shape[1]):
        mobility[:,i][mobility[:,i] == -1] = mobility[max_idx,i]
    return mobility, phase_fracs

def _postProcessExcludePhases(therm: GeneralThermodynamics, mobility: np.array, phase_fracs: np.array, *args, **kwargs):
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

def _validateDictionary(inputDict: dict[str, float], elements: list[str]):
    stepElements = list(inputDict.keys())
    extraElements = list(set(stepElements) - set(elements))
    missingElements = list(set(elements) - set(stepElements))

    if len(missingElements) > 0:
        raise ValueError(f"Composition profile needs to be defined for {elements}. Profile steps for only {stepElements} are found.")
    if len(extraElements) > 0:
        warnings.warn(f"Composition profile is defined for {stepElements} but only {elements} will be used.")

class HashTable:
    '''
    Implements a hash table that stores mobility, phases, phase fractions and
    chemical potentials for a given (composition, temperature) pair
    '''
    def __init__(self):
        self._cache = True
        self.cachedData = {}
        self.setHashSensitivity(4)

    def enableCaching(self, is_caching: bool):
        self._cache = is_caching

    def clearCache(self):
        self.cachedData = {}

    def setHashSensitivity(self, s: int):
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

    def _hashingFunction(self, x: np.array, T: np.array):
        '''
        Gets hash value for a (compostion, temperature) pair

        Parameters
        ----------
        x : float, list[float]
        T : float
        '''
        return hash(tuple((np.concatenate((x, [T]))*self.hash_sensitivity).astype(np.int32)))

    def retrieveFromHashTable(self, x: np.array, T: np.array):
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
        
    def addToHashTable(self, x: np.array, T: np.array, value: any):
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

    def __init__(self):
        self.leftBCtype, self.rightBCtype = {}, {}
        self.leftBC, self.rightBC = {}, {}

    def setBoundaryCondition(self, side: int, bcType: int, value: float, element: str):
        '''
        Sets left boundary condition

        Parameters
        ----------
        side : BoundaryCondition.LEFT or BoundaryCondition.RIGHT
        bc_type : BoundaryCondition.FLUX_BC or BoundaryCondition.COMPOSITION_BC
        value : float
        element : str
        '''
        if side == self.LEFT:
            self.leftBCtype[element] = bcType
            self.leftBC[element] = value
        elif side == self.RIGHT:
            self.rightBCtype[element] = bcType
            self.rightBC[element] = value

    def _setupBoundary(self, element: str, boundaryType: dict[str, int], boundaryValue: dict[str, int]):
        if element not in boundaryType:
            boundaryType[element] = self.FLUX_BC
        if element not in boundaryValue:
            boundaryValue[element] = 0

    def setupDefaults(self, elements: list[str]):
        for e in elements:
            self._setupBoundary(e, self.leftBCtype, self.leftBC)
            self._setupBoundary(e, self.rightBCtype, self.rightBC)

    def applyBoundaryConditionsToInitialProfile(self, elements: list[str], x: np.array, z: np.array):
        for i, e in enumerate(elements):
            if self.leftBCtype[e] == self.COMPOSITION_BC:
                x[i,0] = self.leftBC[e]
            if self.rightBCtype[e] == self.COMPOSITION_BC:
                x[i,-1] = self.rightBC[e]

    def applyBoundaryConditionsToFluxes(self, elements: list[str], fluxes: np.array):
        for i, e in enumerate(elements):
            fluxes[i,0] = self.leftBC[e] if self.leftBCtype[e] == self.FLUX_BC else fluxes[i,1]
            fluxes[i,-1] = self.rightBC[e] if self.rightBCtype[e] == self.FLUX_BC else fluxes[i,-2]

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
    
    def __init__(self):
        self.compositionSteps = {}

    def addCompositionBuildStep(self, element: str, profile_type: int, *args, **kwargs):
        if element in self.compositionSteps:
            self.compositionSteps[element].append((profile_type, args, kwargs))
        else:
            self.compositionSteps[element] = [(profile_type, args, kwargs)]

    def clearCompositionBuildSteps(self, element: str = None):
        if element is None:
            self.compositionSteps = {}
        else:
            if element in self.compositionSteps:
                self.compositionSteps.pop(element)

    def addLinearCompositionStep(self, element: str, left_value: float, right_value: float):
        self.addCompositionBuildStep(element, self.LINEAR, left_value=left_value, right_value=right_value)

    def addStepCompositionStep(self, element: str, left_value: float, right_value: float, z_value: float):
        self.addCompositionBuildStep(element, self.STEP, left_value=left_value, right_value=right_value, z_value=z_value)

    def addSingleCompositionStep(self, element: str, value: float, z_value: float):
        self.addCompositionBuildStep(element, self.SINGLE, value=value, z_value=z_value)

    def addBoundedCompositionStep(self, element: str, value: float, left_z: float, right_z: float):
        self.addCompositionBuildStep(element, self.BOUNDED, value=value, left_z=left_z, right_z=right_z)

    def addFunctionCompositionStep(self, element: str, function: Callable):
        self.addCompositionBuildStep(element, self.FUNCTION, function=function)

    def addProfileCompositionStep(self, element: str, x_list: list[float], z_list: list[float]):
        self.addCompositionBuildStep(element, self.PROFILE, x_list=x_list, z_list=z_list)

    def _setLinearComposition(self, element_idx: int, x: np.array, z: np.array, left_value: float, right_value: float):
        x[element_idx,:] = np.linspace(left_value, right_value, len(z))

    def _setStepComposition(self, element_idx: int, x: np.array, z: np.array, left_value: float, right_value: float, z_value: float):
        left_indices = z <= z_value
        x[element_idx,left_indices] = left_value
        x[element_idx,~left_indices] = right_value

    def _setSingleComposition(self, element_idx: int, x: np.array, z: np.array, value: float, z_value: float):
        z_idx = np.argmin(np.abs(z - z_value))
        x[element_idx,z_idx] = value

    def _setBoundedComposition(self, element_idx: int, x: np.array, z: np.array, value: float, left_z: float, right_z: float):
        z_indices = (z >= left_z) & (z <= right_z)
        x[element_idx,z_indices] = value

    def _setFunctionComposition(self, element_idx: int, x: np.array, z: np.array, function: Callable):
        x[element_idx,:] = function(z)

    def _setProfileComposition(self, element_idx: int, x: np.array, z: np.array, x_list: list[float], z_list: list[float]):
        x[element_idx,:] = np.interp(z, z_list, x_list, x_list[0], x_list[-1])

    def buildProfile(self, elements: list[str], x: np.array, z: np.array):
        _validateDictionary(self.compositionSteps, elements)

        build_functions = {
            self.LINEAR: self._setLinearComposition,
            self.STEP: self._setStepComposition,
            self.SINGLE: self._setSingleComposition,
            self.BOUNDED: self._setBoundedComposition,
            self.FUNCTION: self._setFunctionComposition,
            self.PROFILE: self._setProfileComposition
        }

        for i in range(len(elements)):
            if elements[i] in self.compositionSteps:
                for step_info in self.compositionSteps[elements[i]]:
                    build_functions[step_info[0]](i, x, z, *step_info[1], **step_info[2])

class TemperatureParameters:
    def __init__(self, *args):
        if len(args) == 2:
            self.setTemperatureArray(*args)
        elif len(args) == 1:
            if callable(args[0]):
                self.setTemperatureFunction(args[0])
            else:
                self.setIsothermalTemperature(args[0])
        else:
            self.Tparameters = None
            self.Tfunction = None

    def setIsothermalTemperature(self, T: float):
        self.Tparameters = T
        self.Tfunction = lambda z, t: self.Tparameters*np.ones(len(z))

    def setTemperatureArray(self, times: float, temperatures: list[float]):
        self.Tparameters = (times, temperatures)
        self.Tfunction = lambda z, t: np.interp(t/3600, self.Tparameters[0], self.Tparameters[1], self.Tparameters[1][0], self.Tparameters[1][-1]) * np.ones(len(z))

    def setTemperatureFunction(self, func: Callable):
        self.Tparameters = func
        self.Tfunction = lambda z, t: self.Tparameters(z, t)

    def __call__(self, z: float, t: float):
        return self.Tfunction(z, t)

class HomogenizationParameters:
    WIENER_UPPER = 0
    WIENER_LOWER = 1
    HASHIN_UPPER = 2
    HASHIN_LOWER = 3
    LABYRINTH = 4

    NO_POST = 5
    PREDEFINED = 6
    MAJORITY = 7
    EXCLUDE = 8

    def __init__(self, 
                 homogenizationFunction: Callable = None, 
                 labyrinthFactor: float = 1, 
                 eps: float = 0.05,
                 postProcessFunction: Callable = None,
                 postProcessArgs = None):
        if homogenizationFunction is None:
            homogenizationFunction = self.WIENER_UPPER
        self.setHomogenizationFunction(homogenizationFunction)

        self.labyrinthFactor = labyrinthFactor

        if postProcessFunction is None:
            postProcessFunction = self.NO_POST
        self.setPostProcessFunction(postProcessFunction, postProcessArgs)
        self.eps = eps

    def setPostProcessFunction(self, functionName: Union[str, int], functionArgs = None):
        '''
        Returns post process function
        '''
        self.postProcessParameters = [functionArgs]
        if isinstance(functionName, str):
            self._setPostProcessFunctionByStr(functionName)
        else:
            self._setPostProcessFunctionByID(functionName)

    def _setPostProcessFunctionByStr(self, functionName: str):
        keywords_map = {
            'none': self.NO_POST,
            'predefined': self.PREDEFINED,
            'majority': self.MAJORITY,
            'exclude': self.EXCLUDE
        }
        if functionName in keywords_map:
            self._setHomogenizationFunctionByID(keywords_map[functionName])
        
        str_options = ', '.join(list(keywords_map.keys()))
        raise Exception(f'Error: post process function by str should be {str_options}')

    def _setPostProcessFunctionByID(self, functionName: int):
        if functionName == self.NO_POST:
            self.postProcessFunction = _postProcessDoNothing
        elif functionName == self.PREDEFINED:
            self.postProcessFunction = _postProcessPredefinedMatrixPhase
        elif functionName == self.MAJORITY:
            self.postProcessFunction = _postProcessMajorityPhase
        elif functionName == self.EXCLUDE:
            self.postProcessFunction = _postProcessExcludePhases
        else:
            func_types = ['NO_POST', 'PREDEFINED', 'MAJORITY', 'EXCLUDE']
            int_options = ', '.join([f'HomogenizationParameters.{t}' for t in func_types])
            raise Exception(f'Error: post process function by ID should be {int_options}')
            
    def setHomogenizationFunction(self, function: Union[str, int]):
        '''
        Sets averaging function to use for mobility

        Default mobility value should be that a phase of unknown mobility will be ignored for average mobility calcs

        Parameters
        ----------
        function : str
            Options - 'upper wiener', 'lower wiener', 'upper hashin-shtrikman', 'lower hashin-strikman', 'labyrinth'
        '''
        if isinstance(function, str):
            self._setHomogenizationFunctionByStr(function)
        else:
            self._setHomogenizationFunctionByID(function)

    def _setHomogenizationFunctionByStr(self, function: str):
        keywords_map = {
            self.WIENER_UPPER: ['wiener', 'upper'],
            self.WIENER_LOWER: ['wiener', 'lower'],
            self.HASHIN_UPPER: ['hashin', 'upper'],
            self.HASHIN_LOWER: ['hashin', 'lower'],
            self.LABYRINTH: ['lab'],
        }
        for func_id, keywords in keywords_map.items():
            if all([kw in function for kw in keywords]):
                self._setHomogenizationFunctionByID(func_id)
                return
        
        func_types = ['wiener upper', 'wiener lower', 'hashin upper', 'hashin lower', 'labyrinth']
        str_options = ', '.join(func_types)
        raise Exception(f'Error: homogenization function by str should be {str_options}')

    def _setHomogenizationFunctionByID(self, function: int):
        if function == self.WIENER_UPPER:
            self.homogenizationFunction = wienerUpper
        elif function == self.WIENER_LOWER:
            self.homogenizationFunction = wienerLower
        elif function == self.HASHIN_UPPER:
            self.homogenizationFunction = hashinShtrikmanUpper
        elif function == self.HASHIN_LOWER:
            self.homogenizationFunction = hashinShtrikmanLower
        elif function == self.LABYRINTH:
            self.homogenizationFunction = labyrinth
        else:
            func_types = ['WIENER_UPPER', 'WIENER_LOWER', 'HASHIN_UPPER', 'HASHIN_LOWER', 'LABYRINTH']
            int_options = ', '.join([f'HomogenizationParameters.{t}' for t in func_types])
            raise Exception(f'Error: homogenization function by ID should be {int_options}')

    def setLabyrinthFactor(self, n: float):
        '''
        Labyrinth factor

        Parameters
        ----------
        n : int
            Either 1 or 2
            Note: n = 1 will the same as the weiner upper bounds
        '''
        self.labyrinthFactor = np.clip(n, 1, 2)

def _computeSingleMobility(therm: GeneralThermodynamics, x: np.array, T: np.array, unsortIndices: np.array, hashTable : HashTable = None) -> MobilityData:
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
    mobility_data = None
    if hashTable is not None:
        mobility_data = hashTable.retrieveFromHashTable(x, T)

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
        
        if hashTable is not None:
            hashTable.addToHashTable(x, T, mobility_data)

    return mobility_data

def computeMobility(therm : GeneralThermodynamics, x, T, hashTable : HashTable = None):
    x, T = _process_xT_arrays(x, T, therm.numElements == 2)

    # we want to make sure that the mobility is sorted to be the same order as listed in therm
    sortIndices = np.argsort(therm.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    mob = []
    phases = []
    phase_fracs = []
    chemical_potentials = []
    for i in range(len(x)):
        mobility_data = _computeSingleMobility(therm, x[i], T[i], unsortIndices, hashTable)
        mob.append(mobility_data.mobility)
        phases.append(mobility_data.phases)
        phase_fracs.append(mobility_data.phase_fractions)
        chemical_potentials.append(mobility_data.chemical_potentials)
    return MobilityData(mobility=mob, phases=phases, phase_fractions=phase_fracs, chemical_potentials=chemical_potentials)

def computeHomogenizationFunction(therm : GeneralThermodynamics, x, T, homogenizationParameters : HomogenizationParameters, hashTable : HashTable = None):
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
    x, T = _process_xT_arrays(x, T, therm.numElements == 2)

    # we want to make sure that the mobility is sorted to be the same order as listed in therm
    sortIndices = np.argsort(therm.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    avg_mob = np.zeros((x.shape[0], len(therm.elements)-1))
    chemical_potentials = np.zeros((x.shape[0], len(therm.elements)-1))
    for i in range(len(x)):
        mobility_data = _computeSingleMobility(therm, x[i], T[i], unsortIndices, hashTable)
        mob = mobility_data.mobility
        phase_fracs = mobility_data.phase_fractions
        chemical_potentials[i,:] = mobility_data.chemical_potentials

        mob, phase_fracs = homogenizationParameters.postProcessFunction(therm, mob, phase_fracs, *homogenizationParameters.postProcessParameters)
        avg_mob[i] = homogenizationParameters.homogenizationFunction(mob, phase_fracs, labyrinth_factor = homogenizationParameters.labyrinthFactor)

    return np.squeeze(avg_mob), np.squeeze(chemical_potentials)

class DiffusionConstraints:
    def __init__(self):
        self.reset()

    def reset(self):
        # Minimum composition that a node can take
        self.minComposition = 1e-8

        # Factor to ensure numerical stability in the single phase diffusion model
        # In theory, from von Neumann stability analysis for the heat equation, this can be 0.5
        # But I will use 0.4 to be safe
        self.vonNeumannThreshold = 0.4

        # Max composition change a node can see per iteration in the homogenization model
        # There is not an analagous method for von Neumann stability as we have for the single
        # phase diffusion model, so this is a naive approach to numerical stability
        self.maxCompositionChange = 0.002

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