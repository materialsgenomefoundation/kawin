from typing import Callable, Union

import numpy as np

from kawin.thermo.utils import _process_xT_arrays
from kawin.thermo import GeneralThermodynamics
from kawin.diffusion.DiffusionParameters import HashTable, _computeSingleMobility, MobilityData

def wienerUpper(mobility: np.array, phaseFracs: np.array, *args, **kwargs) -> np.array:
    '''
    Upper wiener bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phaseFracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(phaseFracs[:,np.newaxis], modified_mob), axis=0)
    return avg_mob

def wienerLower(mobility: np.array, phaseFracs: np.array, *args, **kwargs) -> np.array:
    '''
    Lower wiener bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phaseFracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    avg_mob = 1/np.sum(np.multiply(phaseFracs[:,np.newaxis], 1/(modified_mob)), axis=0)
    return avg_mob

def labyrinth(mobility: np.array, phaseFracs: np.array, *args, **kwargs) -> np.array:
    '''
    Labyrinth mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phaseFracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    labyrinth_factor = kwargs.get('labyrinth_factor', 1)
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(np.power(phaseFracs[:,np.newaxis], labyrinth_factor), modified_mob), axis=0)
    return avg_mob

def _hashinShtrikmanGeneral(mobility: np.array, phaseFracs: np.array, extreme_mob: np.array) -> np.array:
    '''
    General hashin shtrikman bounds

    Ak = f^phi / (1 / (gamma^phi - gamma^alpha) + 1/(3*gamma^alpha))
    gamma^* = gamma^alpha + Ak / (1 - Ak / (3*gamma^alpha))

    Parameters
    ----------
    mobility : list of length (p, e)
    phaseFracs : list of length (p,)
    extreme_mob : list of length (e,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    Ak = phaseFracs[:,np.newaxis] * (mobility - extreme_mob) * (3*extreme_mob) / (2*extreme_mob + mobility)
    Ak = np.sum(Ak, axis=0)
    avg_mob = extreme_mob + Ak / (1 - Ak / (3*extreme_mob))
    return avg_mob

def hashinShtrikmanUpper(mobility: np.array, phaseFracs: np.array, *args, **kwargs) -> np.array:
    '''
    Upper hashin shtrikman bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phaseFracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    max_mob = np.amax(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashinShtrikmanGeneral(modified_mob, phaseFracs, max_mob)

def hashinShtrikmanLower(mobility: np.array, phaseFracs: np.array, *args, **kwargs) -> np.array:
    '''
    Lower hashin shtrikman bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phaseFracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    min_mob = np.amin(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashinShtrikmanGeneral(modified_mob, phaseFracs, min_mob)

def _postProcessDoNothing(mobData: MobilityData, *args, **kwargs):
    return np.array(mobData.mobility), np.array(mobData.phase_fractions)

def _postProcessPredefinedMatrixPhase(mobData: MobilityData, *args, **kwargs):
    '''
    User will supply a predefined "alpha" phase, which the mobility is taken
    from for all undefined mobility

    Note: this assumes the user will know that "alpha" is continuously stable
    across the diffusion couple
    '''
    alpha_phase = args[0]
    mobility, phases, phase_fractions = np.array(mobData.mobility), np.array(mobData.phases), np.array(mobData.phase_fractions)
    alpha_idx = np.squeeze(np.where(phases == alpha_phase)[0])
    if len(alpha_idx.shape) > 0:
        raise ValueError(f'{alpha_phase} must be stable as a single phase.')
    for i in range(mobility.shape[0]):
        if mobility[i][0] == -1:
            mobility[i] = mobility[alpha_idx]
    return mobility, phase_fractions

def _postProcessMajorityPhase(mobData: MobilityData, *args, **kwargs):
    '''
    Takes the majority phase and applies the mobility for all other phases
    with undefined mobility
    '''
    mobility, phases, phase_fractions = np.array(mobData.mobility), np.array(mobData.phases), np.array(mobData.phase_fractions)
    max_idx = np.argmax(phase_fractions)
    for i in range(mobility.shape[0]):
        if mobility[i][0] == -1:
            mobility[i] = mobility[max_idx]
    return mobility, phase_fractions

def _postProcessExcludePhases(mobData: MobilityData, *args, **kwargs):
    '''
    For all excluded phases, the mobility and phase fraction will be set to 0
    This assumes that user knows the excluded phases to be minor or that the
    mobility is unknown
    '''
    excluded_phases = args[0]
    mobility, phases, phase_fractions = np.array(mobData.mobility), np.array(mobData.phases), np.array(mobData.phase_fractions)
    for i in range(mobility.shape[0]):
        if phases[i] in excluded_phases:
            mobility[i] = 0
            phase_fractions[i] = 0
    return mobility, phase_fractions

class HomogenizationParameters:
    '''
    Defines homogenization and pre-process functions for the homogenization model

    Parameters
    ----------
    homogenizationFunction: Union[str, int]
        - 'wiener upper' or HomogenizationParameters.WIENER_UPPER
        - 'wiener lower' or HomogenizationParameters.WIENER_LOWER
        - 'hashin upper' or HomogenizationParameters.HASHIN_UPPER
        - 'hashin lower' or HomogenizationParameters.HASHIN_LOWER
        - 'lab' or HomogenizationParameters.LABYRINTH
    labyrinthFactor : float
        If labyrinth function is used
        Must be between 1 and 2
    eps : float
        Factor for the ideal entropy contribution
    postProcessFunction: Union[str, int]
        Function that can modify the phase mobility based off certain conditions
        - 'none' or HomogenizationParameters.NO_POST
            - performs no preprocessing on phase mobilities
        - 'predefined' or HomogenizationParameters.PREDEFINED
            - phases with no mobility models will take on the same mobility value as the predefined phase
        - 'majority' or HomogenizationParameters.MAJORITY
            - phases with no mobility will take on the same mobility of the phase with the largest phase fraction
        - 'exclude' or HomogenizationParameters.EXCLUDE
            - phases in the exclude list will take on a mobility of 0
    postProcessArgs: Any
        If 'majority' post process function is used, postProcessArgs is str corresponding to the pre-defined phase
        If 'exlude' post process function is used, postProcessArgs is list[str] corresponding to list of phases to exlude
    '''
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
                 homogenizationFunction: Union[str,int] = 'wiener upper', 
                 labyrinthFactor: float = 1, 
                 eps: float = 0.05,
                 postProcessFunction: Union[str, int] = 'none',
                 postProcessArgs = None):
        self.setHomogenizationFunction(homogenizationFunction)

        self.labyrinthFactor = labyrinthFactor

        self.setPostProcessFunction(postProcessFunction, postProcessArgs)
        self.eps = eps

    def setPostProcessFunction(self, functionName: Union[str, int], functionArgs = None):
        '''
        Sets post process function by str or int

        Parameters
        ----------
        functionName : Union[str, int]
            Key for post process function ('none', 'predefined', 'majority', 'exclude')
        functionArgs : Any
            Additional function arguments (particularly for 'predefined' or 'exclude')
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
            self._setPostProcessFunctionByID(keywords_map[functionName])
            return
        
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
            Options - 'upper wiener', 'lower wiener', 'upper hashin', 'lower hashin', 'lab'
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
        chemical_potentials[i,:] = mobility_data.chemical_potentials

        mob, phase_fracs = homogenizationParameters.postProcessFunction(mobility_data, *homogenizationParameters.postProcessParameters)
        avg_mob[i] = homogenizationParameters.homogenizationFunction(mob, phase_fracs, labyrinth_factor = homogenizationParameters.labyrinthFactor)

    return np.squeeze(avg_mob), np.squeeze(chemical_potentials)