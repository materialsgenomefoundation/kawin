from symengine import Symbol, Basic, S
from tinydb import Query

from pycalphad import Database, variables as v
from espei.utils import database_symbols_to_fit

def get_used_database_symbols(dbf, elements, diffusing_species, phases = None, include_subsystems = False):
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
    diffusing_species : str
        Name of diffusing species
    phases : list[str] (optional)
        Default = None
        Symbols will be taken from parameters attributed to the phases
        If None, all phases will be considered
    include_subsystems : bool (optional)
        Default = False
        If True, then symbols will also be taken from parameters where constituents is a subset of elements
    '''
    if not isinstance(dbf, Database):
        dbf = Database(dbf)
    elements = sorted([e.upper() for e in elements])
    num_elements = len(elements)
    element_set = frozenset(elements + ['VA'])
    used_syms = frozenset()
    for p in dbf._parameters:
        if phases is not None:
            if p['phase_name'] not in phases:
                continue
        parameter_species = frozenset([s.name for c in p['constituent_array'] for s in c])
        #Add symbols under 2 conditions
        #   1. parameterSpecies is a subset of elSet
        #   2. freeSub is False and length of parameter species (excluding VA) is equal to number of elements
        p_is_subset = len((parameter_species-set(['VA','*'])) - element_set) == 0
        should_be_free = include_subsystems or len(parameter_species - set(['VA', '*'])) == num_elements
        is_diffusing_species = p['diffusing_species'].name == diffusing_species
        if p_is_subset and should_be_free and is_diffusing_species:
            used_syms = used_syms.union(frozenset([s.name for s in p['parameter'].free_symbols]))
    all_syms = database_symbols_to_fit(dbf)
    used_syms = sorted(list(used_syms.intersection(frozenset(all_syms))))
    return used_syms

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