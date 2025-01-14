from collections import namedtuple
from typing import Callable
import warnings

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
        Sets boundary condition for left or right side

        Parameters
        ----------
        side : BoundaryCondition.LEFT or BoundaryCondition.RIGHT
        bc_type : BoundaryCondition.FLUX_BC or BoundaryCondition.COMPOSITION_BC
        value : float
        element : str
        '''
        if isinstance(bcType, str):
            if bcType == 'flux':
                bcType = BoundaryConditions.FLUX_BC
            elif bcType == 'composition':
                bcType = BoundaryConditions.COMPOSITION_BC
            else:
                options = ['BoundaryCondition.FLUX_BC', 'BoundaryCondition.COMPOSITION_BC', 'flux', 'composition']
                raise ValueError(f'Parameter bcType must be the following: {options}')

        if side == self.LEFT or side == 'left':
            self.leftBCtype[element] = bcType
            self.leftBC[element] = value
        elif side == self.RIGHT or side == 'right':
            self.rightBCtype[element] = bcType
            self.rightBC[element] = value
        else:
            options = ['BoundaryCondition.LEFT', 'BoundaryCondition.RIGHT', 'left', 'right']
            raise ValueError(f'Parameter side must be the following: {options}')

    def setLeftBoundaryCondition(self, bcType: int, value: float, element: str):
        '''
        Sets boundary condition for left side

        Parameters
        ----------
        bc_type : BoundaryCondition.FLUX_BC or BoundaryCondition.COMPOSITION_BC
        value : float
        element : str
        '''
        self.setBoundaryCondition(self.LEFT, bcType, value, element)

    def setRightBoundaryCondition(self, bcType: int, value: float, element: str):
        '''
        Sets boundary condition for right side

        Parameters
        ----------
        bc_type : BoundaryCondition.FLUX_BC or BoundaryCondition.COMPOSITION_BC
        value : float
        element : str
        '''
        self.setBoundaryCondition(self.RIGHT, bcType, value, element)

    def _setupBoundary(self, element: str, boundaryType: dict[str, int], boundaryValue: dict[str, int]):
        if element not in boundaryType:
            boundaryType[element] = self.FLUX_BC
        if element not in boundaryValue:
            boundaryValue[element] = 0

    def setupDefaults(self, elements: list[str]):
        '''
        Sets up default boundary conditions for list of elements
        Default is no flux conditions

        Parameters
        ----------
        elements : list[str]
        '''
        for e in elements:
            self._setupBoundary(e, self.leftBCtype, self.leftBC)
            self._setupBoundary(e, self.rightBCtype, self.rightBC)

    def applyBoundaryConditionsToInitialProfile(self, elements: list[str], x: np.array, z: np.array):
        '''
        Applies composition boundary conditions to initial profile
        
        Parameters
        ----------
        elements: list[str]
            List of elements defining profile (e,)
        x : np.array
            Composition profile (e,N)
        z : np.array
            Spatial coordinates (N,)
        '''
        for i, e in enumerate(elements):
            if self.leftBCtype[e] == self.COMPOSITION_BC:
                x[i,0] = self.leftBC[e]
            if self.rightBCtype[e] == self.COMPOSITION_BC:
                x[i,-1] = self.rightBC[e]

    def applyBoundaryConditionsToFluxes(self, elements: list[str], fluxes: np.array):
        '''
        Applies boundary conditions to fluxes

        For composition boundary conditions, fluxes[0] = fluxes[1] and fluxes[-1] = fluxes[-2]
            This allows for the composition at the node to stay constant (i.e. dx/dt = fluxes[1] - fluxes[0] = 0)

        Parameters
        ----------
        elements : list[str]
            List of elements defined profile (e,)
        fluxes : np.array
            Fluxes (e,N+1)
        '''
        for i, e in enumerate(elements):
            fluxes[i,0] = self.leftBC[e] if self.leftBCtype[e] == self.FLUX_BC else fluxes[i,1]
            fluxes[i,-1] = self.rightBC[e] if self.rightBCtype[e] == self.FLUX_BC else fluxes[i,-2]

class CompositionProfile:
    '''
    Stores a series of information for how to build a composition profile
    for each element

    When building the profile, this will go through the list of steps
    in the order that the user inputted
    '''
    LINEAR = 0
    STEP = 1
    SINGLE = 2
    BOUNDED = 3
    FUNCTION = 4
    PROFILE = 5
    
    def __init__(self):
        self.compositionSteps = {}

    def addCompositionBuildStep(self, element: str, profileType: int, *args, **kwargs):
        '''
        Adds a composition build step for an element. It is recommended to use the 
        specific functions such as addLinearCompositionStep, addSingleCompositionStep, etc.

        Parameters
        ----------
        element : str
        profileType : int
            ID for profile type
        *args, **kwargs
            Parameters for the profile function
        '''
        if element in self.compositionSteps:
            self.compositionSteps[element].append((profileType, args, kwargs))
        else:
            self.compositionSteps[element] = [(profileType, args, kwargs)]

    def clearCompositionBuildSteps(self, element: str = None):
        '''
        Removes all composition steps for an element

        Parameters
        ----------
        element : str (optional)
            Element to remove steps from
            If None, steps for all elements will be removed
        '''
        if element is None:
            self.compositionSteps = {}
        else:
            if element in self.compositionSteps:
                self.compositionSteps.pop(element, None)

    def addLinearCompositionStep(self, element: str, leftValue: float, rightValue: float):
        '''
        Add a step to define a linear profile

        Parameters
        ----------
        element : str
        leftValue : float
            Value on left side of mesh
        rightValue : float
            Value on right side of mesh
        '''
        self.addCompositionBuildStep(element, self.LINEAR, leftValue=leftValue, rightValue=rightValue)

    def addStepCompositionStep(self, element: str, leftValue: float, rightValue: float, zValue: float):
        '''
        Add a step to define a step profile

        Parameters
        ----------
        element : str
        leftValue : float
            Value on left side of mesh
        rightValue : float
            Value on right side of mesh
        zValue : float
            Spatial coordinate where compostion switches from leftValue to rightValue
        '''
        self.addCompositionBuildStep(element, self.STEP, leftValue=leftValue, rightValue=rightValue, zValue=zValue)

    def addSingleCompositionStep(self, element: str, value: float, zValue: float):
        '''
        Add a step to define a composition at a single node

        Parameters
        ----------
        element : str
        value : float
            Value of node
        zValue : float
            Spatial coordinate of node, the nearest node to zValue will be used
        '''
        self.addCompositionBuildStep(element, self.SINGLE, value=value, zValue=zValue)

    def addBoundedCompositionStep(self, element: str, value: float, leftZ: float, rightZ: float):
        '''
        Add a step to defined composition between two bounds

        Parameters
        ----------
        element : str
        value : float
            Composition
        leftZ : float
            Spatial coordinate of left side of bound
        rightZ : float
            Spatial coordinate on right side of bound
        '''
        self.addCompositionBuildStep(element, self.BOUNDED, value=value, leftZ=leftZ, rightZ=rightZ)

    def addFunctionCompositionStep(self, element: str, function: Callable):
        '''
        Add a step to define a profile by a function

        Parameters
        ----------
        element : str
        function : Callable
            Function in the form of f(z) = x where z is spatial coordinate and x is composition
        '''
        self.addCompositionBuildStep(element, self.FUNCTION, function=function)

    def addProfileCompositionStep(self, element: str, xList: list[float], zList: list[float]):
        '''
        Add a step to define a profile by data points

        Parameters
        ----------
        element : str
        xList : list[float]
            Data point for composition
        zList : list[float]
            Data points for spatial coordinates, must correspond to xList
        '''
        self.addCompositionBuildStep(element, self.PROFILE, xList=xList, zList=zList)

    def _setLinearComposition(self, elementIdx: int, x: np.array, z: np.array, leftValue: float, rightValue: float):
        x[elementIdx,:] = np.linspace(leftValue, rightValue, len(z))

    def _setStepComposition(self, elementIdx: int, x: np.array, z: np.array, leftValue: float, rightValue: float, zValue: float):
        leftIndices = z <= zValue
        x[elementIdx,leftIndices] = leftValue
        x[elementIdx,~leftIndices] = rightValue

    def _setSingleComposition(self, elementIdx: int, x: np.array, z: np.array, value: float, zValue: float):
        z_idx = np.argmin(np.abs(z - zValue))
        x[elementIdx,z_idx] = value

    def _setBoundedComposition(self, elementIdx: int, x: np.array, z: np.array, value: float, leftZ: float, rightZ: float):
        z_indices = (z >= leftZ) & (z <= rightZ)
        x[elementIdx,z_indices] = value

    def _setFunctionComposition(self, elementIdx: int, x: np.array, z: np.array, function: Callable):
        x[elementIdx,:] = function(z)

    def _setProfileComposition(self, elementIdx: int, x: np.array, z: np.array, xList: list[float], zList: list[float]):
        x[elementIdx,:] = np.interp(z, zList, xList, xList[0], xList[-1])

    def _validateDictionary(self, elements: list[str]):
        '''
        Validates that all input elements are defined in the composition profile
        '''
        stepElements = list(self.compositionSteps.keys())
        extraElements = list(set(stepElements) - set(elements))
        missingElements = list(set(elements) - set(stepElements))

        if len(missingElements) > 0:
            raise ValueError(f"Composition profile needs to be defined for {elements}. Profile steps for only {stepElements} are found.")
        if len(extraElements) > 0:
            warnings.warn(f"Composition profile is defined for {stepElements} but only {elements} will be used.")

    def buildProfile(self, elements: list[str], x: np.array, z: np.array):
        '''
        Builds a composition profile for list of elements

        Parameters
        ----------
        elements : list[str]
            List of elements to build profile
            Shape of (e,)
        x : np.array
            Composition profile to fill
            Shape of (e,N)
        z : np.array
            Spatial coordinates
            Shape of (N,)
        '''
        self._validateDictionary(elements)

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
            if callable(args[0]):
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