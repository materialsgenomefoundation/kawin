import itertools
from collections import namedtuple
from typing import Union

from symengine import Symbol, Basic, S
from tinydb import Query

from pycalphad import Database, variables as v
from pycalphad.io.tdb import _molmass
from espei.utils import database_symbols_to_fit

from kawin.thermo.Mobility import MobilityModel

DerivativePair = namedtuple('DerivativePair', ['functions', 'symbols'])
DatasetPair = namedtuple('SystemPair', ['site_fractions', 'values'])
FittingResult = namedtuple('FittingResult', ['template', 'parameters', 'aicc'])

def _upsert_parameters(dbf: Database, parameter_type: str, phase: str, constituent_array, parameter_order: int, diffusing_species: str, function: Basic):
    species_dict = {s.name: s for s in dbf.species}
    tuple_const = tuple(tuple(species_dict.get(s.upper(), v.Species(s)) for s in xs) for xs in constituent_array)
    query = Query()
    search_query = (query.parameter_type == parameter_type) &\
                    (query.phase_name == phase) & \
                    (query.constituent_array == tuple_const) & \
                    (query.diffusing_species == diffusing_species) & \
                    (query.parameter_order == parameter_order)
    param = dbf.search(search_query)
    if len(param) == 0:
        dbf.add_parameter(parameter_type, phase, constituent_array, 
                          parameter_order, function, 
                          diffusing_species=diffusing_species)
    else:
        dbf._parameters.upsert({'parameter': param[0]['parameter'] + function}, search_query)

class MobilityTemplate:
    '''
    Creates a template of a mobility model for a single diffusing species

    Attributes
    ----------
    dbf: Database
    phase: str
    sublattices: list[int]
    constituents: list
    diffusing_species: v.Species
    num_params: int
        Number of parameters added to the database
    elements: list[str]

    Parameters
    ----------
    phase: str
    diffusing_species: str | v.Species
    sublattices: list[int]
    constituents: list[list[str]]
    '''
    def __init__(self, phase: str, diffusing_species: Union[str, v.Species], sublattices: list[int], constituents: list, add_endmembers: bool = True):
        self.dbf = Database()
        self.phase = phase
        self.sublattices = sublattices
        self.constituents = constituents
        self.diffusing_species = v.Species(diffusing_species)
        self.num_params = 0
        self._model = None

        unique_species = list(set(itertools.chain(*constituents)))
        for s in unique_species:
            sp = v.Species(s)
            self.dbf.species.add(sp)
            self.dbf.elements.update(list(sp.constituents.keys()))

        # Add reference state, this is to be compatible for phase records, but values aren't
        # necessary since we intend to add the parameters to the thermodynamic database in the end  
        for el in self.dbf.elements:
            self.dbf.refstates[el] = {'phase': phase, 'mass': _molmass.get(el, 0.0), 'H298': 0, 'S298': 0}

        self.dbf.add_phase(phase, {}, sublattices)
        self.dbf.add_phase_constituents(phase, constituents)

        if add_endmembers:
            self.add_endmember_parameters()

    @property
    def elements(self) -> list[str]:
        return list(self.dbf.elements)

    def _create_constant(self) -> Basic:
        '''
        Creates a constant parameter symbol as VV00XX
        '''
        parameter_symbol = Symbol(_vname(self.num_params))
        self.num_params += 1
        return parameter_symbol
    
    def _create_T_term(self) -> Basic:
        '''
        Creates a T dependent parameter symbol as VV00XX*T
        '''
        parameter_symbol = Symbol(_vname(self.num_params))*v.T
        self.num_params += 1
        return parameter_symbol

    def add_endmember_parameters(self):
        '''
        Creates A+B*T terms for all endmembers of phase
        '''
        for prod in itertools.product(*self.constituents):
            constituent_array = [[species] for species in prod]
            self.dbf.add_parameter('MQ', self.phase, constituent_array,
                                   0, self._create_constant() + self._create_T_term(), 
                                   diffusing_species=self.diffusing_species)
            
        # Not really needed since this function is called upon initialization, but to be safe since we modified the database here
        self._model = None

    def add_term(self, constituent_array: list, parameter_order: Union[int, list[int]], parameter: Union[callable, Basic], parameter_type: str = 'MQ'):
        '''
        Adds a term to a parameter in the database

        If parameter already exists (same symbol, constituent_array and parameter order), then the parameter is
        added on to the existing value rather than overwriting it

        Parameters
        ----------
        constituent_array: list[list[str]]
        param_order: int | list[int]
            If list, will add parameters for all specified orders
        parameter: callable | Any
            If callable, should be a function that generates a parameter symbol
            If not callable, should be compatible with a symengine expression
        symbol: str (Optional)
            Defaults to 'MQ'
            Parameter symbol to add in database
            By default, we always use 'MQ', but this adds the option to create the 'MF' terms
            if they the prefactor and activation energies need to be separate terms (i.e. for 
            magnetic contributions which are yet unsupported)
        '''
        if not isinstance(parameter_order, list):
            parameter_order = [parameter_order]

        # If param_func is not callable, then convert to a self-generating function
        if callable(parameter):
            param_func = parameter
        else:
            param_func = lambda: parameter

        for p in parameter_order:
            _upsert_parameters(self.dbf, parameter_type, self.phase, constituent_array, p, self.diffusing_species, param_func())
        
        # Since we updated the database, clear the mobility model if it was made at some point
        self._model = None

    def add_prefactor(self, constituent_array: list, parameter_order: Union[int, list[int]], parameter_type: str = 'MQ'):
        '''
        Adds a diffusion prefactor term (in form of VV00XX*T)
            While the prefactor is defined as M0 in 
                M = M0/RT * exp(Q/RT),
            the prefactor is treated as part of the exponential term as
                M = 1/RT * exp((MQ+MF)/RT) where MQ+MF = A+B*T
                Here, A refers to the activation energy and B refers to the prefactor

        Parameters
        ----------
        constituent_array: list[list[str]]
        param_order: int | list[int]
            If list, then terms will be added to all specified orders
        symbols: str (Optional)
            Defaults to 'MQ'
        '''
        self.add_term(constituent_array, parameter_order, self._create_T_term, parameter_type=parameter_type)
        
    def add_activation_energy(self, constituent_array: list, parameter_order: Union[int, list[int]], parameter_type: str='MQ'):
        '''
        Adds a activation energy term (in form of VV00XX)

        Parameters
        ----------
        constituent_array: list[list[str]]
        param_order: int | list[int]
            If list, then terms will be added to all specified orders
        symbols: str (Optional)
            Defaults to 'MQ'
        '''
        self.add_term(constituent_array, parameter_order, self._create_constant, parameter_type=parameter_type)

    def copy_parameter(self, constituent_array: list, new_constituent_array: list, parameter_order: int, parameter_type: str = 'MQ'):
        '''
        Copies parameter function to new set of constituent array
        This is for cases to avoid wildcard usage when a value might be constant across composition space
        '''
        species_dict = {s.name: s for s in self.dbf.species}
        tuple_const = tuple(tuple(species_dict.get(s.upper(), v.Species(s)) for s in xs) for xs in constituent_array)
        query = Query()
        search_query = (query.parameter_type == parameter_type) &\
                        (query.phase_name == self.phase) & \
                        (query.constituent_array == tuple_const) & \
                        (query.diffusing_species == self.diffusing_species) & \
                        (query.parameter_order == parameter_order)
        param = self.dbf.search(search_query)
        if len(param) > 0:
            p = param[0]
            self.dbf.add_parameter(p['parameter_type'], p['phase_name'], new_constituent_array,
                                   p['parameter_order'], p['parameter'], 
                                   diffusing_species=p['diffusing_species'])

    @property
    def parameters(self) -> list:
        '''
        Returns all database parameters for mobility model
        '''
        query = Query()
        search_query = (query.phase_name == self.phase) & (query.diffusing_species == self.diffusing_species)
        return self.dbf._parameters.search(search_query)

    @property
    def model(self) -> MobilityModel:
        '''
        Returns mobility model built from database parameters
        '''
        if self._model is None:
            self._model = MobilityModel(self.dbf, self.elements, self.phase)
        return self._model
    
    # mobility_function, MQ, pre_factor and activation_energy are wrappers to
    # retrieve the underlying functions in the MobilityModel directly
    @property
    def mobility_function(self) -> Basic:
        '''
        Mobility function: M = 1/RT * exp(-(MQ+MF)/RT)
        '''
        return getattr(self.model, f'MOB_{self.diffusing_species.name}')
    
    @property
    def MQ(self) -> Basic:
        '''
        Exp part of mobility function: MQ+MF
        '''
        return getattr(self.model, f'MQ_{self.diffusing_species}')

    @property
    def prefactor(self) -> Basic:
        '''
        Natural log of pre-diffusion factor
        This refers to all the T-dependent terms in MQ+MF
        '''
        return getattr(self.model, f'lnM0_{self.diffusing_species.name}')
    
    @property
    def activation_energy(self) -> Basic:
        '''
        Activation energy of pre-diffusion factor
        This refers to all the constant terms in MQ+MF
        '''
        return getattr(self.model, f'MQa_{self.diffusing_species.name}')
    
    def get_derivatives(self, function: Basic) -> DerivativePair:
        '''
        Creates derivative and corresponding variables for function (would be either lnM0 or Q)
        '''
        # NOTE: this assumes that all free variables of interest are in VV00XX format
        # If the user doesn't add custom parameters directly, then this assumption should hold
        symbol_list = _get_variable_terms(function)
        diffs, symbols = [], []
        for s in symbol_list:
            diff = function.diff(s)
            if diff != S.Zero:
                diffs.append(diff)
                symbols.append(s)
        return DerivativePair(functions=diffs, symbols=symbols)
    
    def add_to_database(self, dbf: Database, parameter_values: dict[Union[Symbol, str], float]):
        '''
        Adds parameters to an existing database

        Parameters
        ----------
        dbf: Database
            database to add parameters to
        parameter_values: dict[Symbol|str, float]
            Values for each VV00XX parameter in the mobility model
            Any parameter that isn't specified is assumed to be 0 and won't be added to database
        '''
        last_var_index = find_last_variable(dbf)
        parameter_values = {Symbol(key) if isinstance(key, str) else key: value for key,value in parameter_values.items()}

        symbol_replace_map = {param_name: _vname(i+last_var_index) for i, param_name in enumerate(parameter_values)}
        unused_symbols = {s: 0 for s in (set(self.MQ.free_symbols) - set(symbol_replace_map.keys())) if str(s).startswith('VV')}

        for p in self.parameters:
            new_func = p['parameter'].subs(unused_symbols)
            new_func = new_func.xreplace(symbol_replace_map)
            constituent_array = tuple(tuple(s.name for s in xs) for xs in p['constituent_array'])
            _upsert_parameters(dbf, p['parameter_type'], p['phase_name'], constituent_array,
                            p['parameter_order'], p['diffusing_species'], new_func)

        for s in parameter_values:
            dbf.symbols[symbol_replace_map[s]] = parameter_values[s]

def get_used_database_symbols(dbf, elements, diffusingSpecies, phases = None, includeSub = False):
    '''
    Given the database, grab all symbols that pertain only to the given elements and phases

    example:
        PARAMETER(BCC_A2&C,AL:0) 298.15 VV0001; 6000.0 N !
        PARAMETER(BCC_A2&C,CR:0) 298.15 VV0002; 6000.0 N !
        PARAMETER(FCC_A1&C,AL:0) 298.15 VV0003; 6000.0 N !
        PARAMETER(FCC_A1&C,CR:0) 298.15 VV0004; 6000.0 N !

        get_used_database_symbols(db, ['AL'], ['C'], ['BCC_A2'], True) -> VV0001
        get_used_database_symbols(db, ['AL', 'CR'], ['C'], ['FCC_A1'], True) -> [VV0003, VV0004]
        get_used_database_symbols(db, ['CR'], ['C'], ['BCC_A2', 'FCC_A1'], True) -> [VV0002, VV0004]

    Parameters
    ----------
    dbf : Database
    elements : list[str]
        Symbols will be taken from parameters that have all elements in the constituents
    diffusingSpecies : str
        Name of diffusing species
    phases : list[str] (optional)
        Default = None
        Symbols will be taken from parameters attributed to the phases
        If None, all phases will be considered
    includeSub : bool (optional)
        Default = False
        If True, then symbols will also be taken from parameters where constituents is a subset of elements
    '''
    if not isinstance(dbf, Database):
        dbf = Database(dbf)
    elements = sorted([e.upper() for e in elements])
    numElements = len(elements)
    elSet = frozenset(elements + ['VA'])
    usedSyms = frozenset()
    for p in dbf._parameters:
        if phases is not None:
            if p['phase_name'] not in phases:
                continue
        parameterSpecies = frozenset([s.name for c in p['constituent_array'] for s in c])
        #Add symbols under 2 conditions
        #   1. parameterSpecies is a subset of elSet
        #   2. freeSub is False and length of parameter species (excluding VA) is equal to number of elements
        pIsSubset = len((parameterSpecies-set(['VA','*'])) - elSet) == 0
        shouldBeFree = includeSub or len(parameterSpecies - set(['VA', '*'])) == numElements
        isDiffusingSpecies = p['diffusing_species'].name == diffusingSpecies
        if pIsSubset and shouldBeFree and isDiffusingSpecies:
            usedSyms = usedSyms.union(frozenset([s.name for s in p['parameter'].free_symbols]))
        #if len(parameterSpecies - elSet) == 0:
        #    usedSyms = usedSyms.union(frozenset([s.name for s in p['parameter'].free_symbols]))
    allSyms = database_symbols_to_fit(dbf)
    usedSyms = sorted(list(usedSyms.intersection(frozenset(allSyms))))
    return usedSyms

def _vname(index: int) -> str:
    '''
    Converts index to VV00XX format
    '''
    return 'VV{:04d}'.format(index)

def find_last_variable(database: Database) -> int:
    '''
    Searches database to find the last symbol noted as VV00XX
    '''
    index = 0
    while _vname(index) in database.symbols:
        index += 1
    return index

def _get_variable_terms(function: Basic) -> list[Symbol]:
    return sorted([s for s in function.free_symbols if str(s).startswith('VV')], key=str)

def _eval_symengine_expr(function: Basic, symbol_map: dict[Symbol, float], ignore_missing = True) -> float:
    default_symbols = {}
    if ignore_missing:
        default_symbols = {s: 0 for s in function.free_symbols}
    default_symbols.update(symbol_map)
    return float(function.subs(default_symbols).n(73, real=True))