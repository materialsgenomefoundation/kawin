from collections import namedtuple
from typing import Callable

import numpy as np

from kawin.thermo.utils import _process_xT_arrays
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

class TemperatureParameters:
    '''
    Parameter for temperature that acts like a callable function

    Parameters can be one of the following:
        T : float
            Isothermal temperature
        times : list[float], temperatures : list[float]
            Array of times and temperatures (temperature becomes temperatures[i] at times[i])
        func : callable
            Function in the form of f(z,t) = T where z is spatial coordinate and t is time
    '''
    def __init__(self, *args):
        if len(args) == 2:
            self.setTemperatureArray(*args)
        elif len(args) == 1:
            if isinstance(args[0], TemperatureParameters):
                self.Tparameters = args[0].Tparameters
                self.Tfunction = args[0].Tfunction
            elif callable(args[0]):
                self.setTemperatureFunction(args[0])
            else:
                self.setIsothermalTemperature(args[0])
        else:
            self.Tparameters = None
            self.Tfunction = None

    def setIsothermalTemperature(self, T: float):
        '''
        Sets isothermal temperature

        Parameters
        ----------
        T : float
        '''
        self.Tparameters = T
        self.Tfunction = lambda z, t: self.Tparameters*np.ones(len(z))

    def setTemperatureArray(self, times: list[float], temperatures: list[float]):
        '''
        Sets array of times/temperatures

        Example:
            time = [0, 1, 2]
            temperature = [100, 200, 300]
            This will set temperature to 100 at t = 0 hours, then at 1 hour, temperature = 200, then after 2 hours, temperature = 300

        Parameters
        ----------
        times : list[float]
        temperatures : list[float]
        '''
        self.Tparameters = (times, temperatures)
        self.Tfunction = lambda z, t: np.interp(t/3600, self.Tparameters[0], self.Tparameters[1], self.Tparameters[1][0], self.Tparameters[1][-1]) * np.ones(len(z))

    def setTemperatureFunction(self, func: Callable):
        '''
        Sets temperature function

        Parameters
        ----------
        func : Callable
            Function is in the form f(z,t) = T, where z is spatial coordinate and t is time
        '''
        self.Tparameters = func
        self.Tfunction = lambda z, t: self.Tparameters(z, t)

    def __call__(self, z: float, t: float):
        return self.Tfunction(z, t)

def _computeSingleMobility(therm: GeneralThermodynamics, x: np.array, T: np.array, unsortIndices: np.array, hashTable : HashTable = None) -> MobilityData:
    '''
    Gets mobility data for x and T (at single composition/temperature)

    If mobility does not exist for a phase or a phase is unstable, then the mobility is set to -1

    Parameters
    ----------
    therm : GeneralThermodynamics object
    x : float, list[float]
        For binary, it must be a float
        For ternary, it must be list[float]
    T : float
        Temperature(s)
        
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
    '''
    Gets mobility data for x and T where x and T is an array

    If mobility does not exist for a phase or a phase is unstable, then the mobility is set to -1

    Parameters
    ----------
    therm : GeneralThermodynamics object
    x : float or array
        For binary, it must be (1,) or (N,)
        For ternary, it must be (1,e) or (N,e)
    T : float or array
        Must be (1,) or (N,)
        
    Returns
    -------
    MobilityData
    '''
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

class DiffusionConstraints:
    '''
    Constraints for numerical stability

    Attributes
    ----------
    minComposition : float
        Minimum composition that a node can take
    vonNeumannThreshold : float
        Factor to ensure numerical stability in the single phase diffusion model
        In theory, from von Neumann stability analysis for the heat equation, this can be 0.5
        But I will use 0.4 to be safe
    maxCompositionChange : float
        Max composition change a node can see per iteration in the homogenization model
        There is not an analagous method for von Neumann stability as we have for the single
        phase diffusion model, so this is a naive approach to numerical stability
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.minComposition = 1e-8
        self.vonNeumannThreshold = 0.4
        self.maxCompositionChange = 0.002