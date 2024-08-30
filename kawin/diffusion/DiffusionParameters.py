import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable
import numpy as np

from kawin.thermo import GeneralThermodynamics
from kawin.thermo.Mobility import mobility_from_composition_set, x_to_u_frac, interstitials

MobilityData = namedtuple('MobilityData',
                                 ['mobility', 'phases', 'phase_fractions', 'chemical_potentials'])

class HashTable:
    '''
    Implements a hash table that stores mobility, phases, phase fractions and
    chemical potentials for a given (composition, temperature) pair
    '''
    def __init__(self):
        self._cache = True
        self.hash_table = {}
        self.setHashSensitivity(4)

    def enableCaching(self, isCaching):
        self._cache = isCaching

    def clearCache(self):
        self.hash_table = {}

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

    def hashing_function(self, x, T):
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
            hash_value = self.hashing_function(x, T)
            return self.hash_table.get(hash_value, None)
        
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
            hash_value = self.hashing_function(x, T)
            self.hash_table[hash_value] = value

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
        self.left_BC_type, self.right_BC_type = self.FLUX_BC*np.ones(len(elements)), self.FLUX_BC*np.ones(len(elements))
        self.left_BC, self.right_BC = np.zeros(len(elements)), np.zeros(len(elements))

    def set_boundary_condition(self, side, bc_type, value, element):
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
            self.left_BC_type[index] = bc_type
            self.left_BC[index] = value
        elif side == self.RIGHT:
            self.right_BC_type[index] = bc_type
            self.right_BC[index] = value

    def apply_boundary_conditions_to_initial_profile(self, x, z):
        for e in range(len(self.elements)):
            if self.left_BC_type[e] == self.COMPOSITION_BC:
                x[e,0] = self.left_BC[e]
            if self.right_BC_type[e] == self.COMPOSITION_BC:
                x[e,-1] = self.right_BC[e]

    def apply_boundary_conditions_to_fluxes(self, fluxes):
        for e in range(len(self.elements)):
            fluxes[e,0] = self.left_BC[e] if self.left_BC_type[e] == self.FLUX_BC else fluxes[e,1]
            fluxes[e,-1] = self.right_BC[e] if self.right_BC_type[e] == self.FLUX_BC else fluxes[e,-2]

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
        self.composition_steps = [[] for _ in self.elements]

    def _get_element_index(self, element):
        if element not in self.elements:
            print(f'Warning: {element} not in {self.elements}')
            return None
        return self.elements.index(element)

    def add_composition_build_step(self, element, profile_type, *args, **kwargs):
        index = self._get_element_index(element)
        if index is not None:
            self.composition_steps[index].append((profile_type, args, kwargs))

    def clear_composition_build_steps(self, element = None):
        if element is None:
            self.composition_steps = [[] for _ in self.elements]

        index = self._get_element_index(element)
        if index is not None:
            self.composition_steps[index].clear()

    def add_linear_composition_step(self, element, left_value, right_value):
        self.add_composition_build_step(element, self.LINEAR, left_value=left_value, right_value=right_value)

    def add_step_composition_step(self, element, left_value, right_value, z_value):
        self.add_composition_build_step(element, self.STEP, left_value=left_value, right_value=right_value, z_value=z_value)

    def add_single_composition_step(self, element, value, z_value):
        self.add_composition_build_step(element, self.SINGLE, value=value, z_value=z_value)

    def add_bounded_composition_step(self, element, value, left_z, right_z):
        self.add_composition_build_step(element, self.BOUNDED, value=value, left_z=left_z, right_z=right_z)

    def add_function_composition_step(self, element, function):
        self.add_composition_build_step(element, self.FUNCTION, function=function)

    def add_profile_composition_step(self, element, x_list, z_list):
        self.add_composition_build_step(element, self.PROFILE, x_list=x_list, z_list=z_list)

    def _set_linear_composition(self, element_idx, x, z, left_value, right_value):
        x[element_idx,:] = np.linspace(left_value, right_value, len(z))

    def _set_step_composition(self, element_idx, x, z, left_value, right_value, z_value):
        left_indices = z <= z_value
        x[element_idx,left_indices] = left_value
        x[element_idx,~left_indices] = right_value

    def _set_single_composition(self, element_idx, x, z, value, z_value):
        z_idx = np.argmin(np.abs(z - z_value))
        x[element_idx,z_idx] = value

    def _set_bounded_composition(self, element_idx, x, z, value, left_z, right_z):
        z_indices = (z >= left_z) & (z <= right_z)
        x[element_idx,z_indices] = value

    def _set_function_composition(self, element_idx, x, z, function):
        x[element_idx,:] = function(z)

    def _set_profile_composition(self, element_idx, x, z, x_list, z_list):
        x[element_idx,:] = np.interp(z, z_list, x_list, x_list[0], x_list[-1])

    def build_profile(self, x, z):
        build_functions = {
            self.LINEAR: self._set_linear_composition,
            self.STEP: self._set_step_composition,
            self.SINGLE: self._set_single_composition,
            self.BOUNDED: self._set_bounded_composition,
            self.FUNCTION: self._set_function_composition,
            self.PROFILE: self._set_profile_composition
        }

        for i in range(len(self.elements)):
            for step_info in self.composition_steps[i]:
                build_functions[step_info[0]](i, x, z, *step_info[1], **step_info[2])

class TemperatureParameters:
    def __init__(self):
        self.T_parameters = None
        self.T_function = None

    def set_isothermal_temperature(self, T):
        self.T_parameters = T
        self.T_function = lambda z, t: self.T_parameters*np.ones(len(z))

    def set_temperature_array(self, times, temperatures):
        self.T_parameters = (times, temperatures)
        self.T_function = lambda z, t: np.interp(t/3600, self.T_parameters[0], self.T_parameters[1], self.T_parameters[1][0], self.T_parameters[1][-1]) * np.ones(len(z))

    def set_temperature_function(self, func):
        self.T_parameters = func
        self.T_function = lambda z, t: self.T_parameters(z, t)

    def __call__(self, z, t):
        return self.T_function(z, t)

def wiener_upper(mobility, phase_fracs, *args, **kwargs):
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

def wiener_lower(mobility, phase_fracs, *args, **kwargs):
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

def _hashin_shtrikman_general(mobility, phase_fracs, extreme_mob):
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

def hashin_shtrikman_upper(mobility, phase_fracs, *args, **kwargs):
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
    return _hashin_shtrikman_general(modified_mob, phase_fracs, max_mob)

def hashin_shtrikman_lower(mobility, phase_fracs, *args, **kwargs):
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
    return _hashin_shtrikman_general(modified_mob, phase_fracs, min_mob)

def _post_process_do_nothing(therm, mobility, phase_fracs, *args, **kwargs):
    return mobility, phase_fracs

def _post_process_predefined_matrix_phase(therm, mobility, phase_fracs, *args, **kwargs):
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

def _post_process_majority_phase(therm, mobility, phase_fracs, *args, **kwargs):
    '''
    Takes the majority phase and applies the mobility for all other phases
    with undefined mobility
    '''
    max_idx = np.argmax(phase_fracs)
    for i in range(mobility.shape[1]):
        mobility[:,i][mobility[:,i] == -1] = mobility[max_idx,i]
    return mobility, phase_fracs

def _post_process_exclude_phases(therm, mobility, phase_fracs, *args, **kwargs):
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

class DiffusionParameters:
    FLUX_BC = 0
    COMPOSITION_BC = 1

    def __init__(self, elements):
        self.min_composition = 1e-8
        self.max_composition_change = 0.002

        self.temperature = TemperatureParameters()
        self.boundary_conditions = BoundaryConditions(elements)
        self.composition_profile = CompositionProfile(elements)
        self.hash_table = HashTable()

        self.homogenization_function: Callable = wiener_upper
        self.labyrinth_factor: float = 1
        self.setPostProcessFunction(None)
        self.eps = 0.05

    def setPostProcessFunction(self, function_name, function_args = None):
        '''
        Returns post process function
        '''
        self.post_process_parameters = [function_args]
        if function_name is None:
            self.post_process_function = _post_process_do_nothing
        else:
            if function_name == 'predefined':
                self.post_process_function = _post_process_predefined_matrix_phase
            elif function_name == 'majority':
                self.post_process_function = _post_process_majority_phase
            elif function_name == 'exclude':
                self.post_process_function = _post_process_exclude_phases
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
            self.homogenization_function = wiener_upper
        elif 'lower' in function and 'wiener' in function:
            self.homogenization_function = wiener_lower
        elif 'upper' in function and 'hashin' in function:
            self.homogenization_function = hashin_shtrikman_upper
        elif 'lower' in function and 'hashin' in function:
            self.homogenization_function = hashin_shtrikman_lower
        elif 'lab' in function:
            self.homogenization_function = labyrinth

    def setLabyrinthFactor(self, n):
        '''
        Labyrinth factor

        Parameters
        ----------
        n : int
            Either 1 or 2
            Note: n = 1 will the same as the weiner upper bounds
        '''
        self.labyrinth_factor = np.clip(n, 1, 2)

def _compute_single_mobility(therm: GeneralThermodynamics, x, T, unsortIndices, diffusion_parameters: DiffusionParameters) -> MobilityData:
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
    mobility_data = diffusion_parameters.hash_table.retrieveFromHashTable(x, T)
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
        diffusion_parameters.hash_table.addToHashTable(x, T, mobility_data)

    return mobility_data

def compute_mobility(therm : GeneralThermodynamics, x, T, diffusion_parameters : DiffusionParameters):
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
        mobility_data = _compute_single_mobility(therm, x[i], T[i], unsortIndices, diffusion_parameters)
        mob.append(mobility_data.mobility)
        phases.append(mobility_data.phases)
        phase_fracs.append(mobility_data.phase_fractions)
        chemical_potentials.append(mobility_data.chemical_potentials)
    return MobilityData(mobility=mob, phases=phases, phase_fractions=phase_fracs, chemical_potentials=chemical_potentials)

def compute_homogenization_function(therm : GeneralThermodynamics, x, T, diffusion_parameters : DiffusionParameters):
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
        mobility_data = _compute_single_mobility(therm, x[i], T[i], unsortIndices, diffusion_parameters)
        mob = mobility_data.mobility
        phases = mobility_data.phases
        phase_fracs = mobility_data.phase_fractions
        chemical_potentials[i,:] = mobility_data.chemical_potentials

        mob, phase_fracs = diffusion_parameters.post_process_function(therm, mob, phase_fracs, *diffusion_parameters.post_process_parameters)
        avg_mob[i] = diffusion_parameters.homogenization_function(mob, phase_fracs, labyrinth_factor = diffusion_parameters.labyrinth_factor)

    return np.squeeze(avg_mob), np.squeeze(chemical_potentials)

