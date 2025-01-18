import numpy as np
from kawin.thermo import GeneralThermodynamics
from pycalphad import Database, variables as v
from espei.utils import database_symbols_to_fit

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

class MobilityTerm:
    '''
    Utility class to hold data for a mobility term

    Attributes
    ----------
    constituent_array : list[list[str]]
        Constituent array
    order : int
        Redlich-kister polynomial order
    '''
    def __init__(self, constituent_array : list[list[v.SiteFraction]], order : int = 0):
        self.constituent_array = tuple(constituent_array)
        self.order = order
        self.expr = 0

    def generate_multiplier(self, site_fractions):
        '''
        Given site fraction values, return result for xA*xB*(xA-xB)**n
        
        Notes
            This will account for multiple sublattices, but mixing on both sublattices will
            assume to have the same order polynomial
            Tertiary contributions are not included yet
        '''
        val = 1
        for clist in self.constituent_array:
            clist = np.atleast_1d(clist)
            for c in clist:
                #The wildcards will be treated as the sum for each component on the sublattice
                #However, since the sum is 1, we can ignore it here
                if c.species != v.Species('*'):
                    val *= site_fractions.get(c,0)
            if len(clist) == 2:
                ordered = sorted(clist)
                val *= (site_fractions.get(ordered[1],0) - site_fractions.get(ordered[0],0))**self.order
        return val
    
    def create_constituent_array_list(self):
        '''
        Create constituent array list compatible with Database.add_parameter
        '''
        return [[c.species.name for c in np.atleast_1d(clist)] for clist in self.constituent_array]
    
    def __eq__(self, other):
        return self.constituent_array == other.constituent_array and self.order == other.order

def add_mobility_model(database : Database, phase : str, diffusing_species : str, mobility_terms : list[MobilityTerm], symbols : dict[str,float] = {}):
    '''
    Add mobility parameters to database

    Parameters
    ----------
    database : Pycalphad database object
    phase : str
    diffusing_species : str
    mobility_terms : list[MobilityParameter]
    '''
    for mp in mobility_terms:
        database.add_parameter('MQ', phase, mp.create_constituent_array_list(), mp.order, mp.expr, diffusing_species=diffusing_species)
    for s,val in symbols.items():
        database.symbols[s] = val

def _vname(index):
    '''
    Converts index to VV00XX format
    '''
    return 'VV{:04d}'.format(index)

def find_last_variable(database):
    '''
    Searches database to find the last symbol noted as VV00XX
    '''
    index = 0
    while _vname(index) in database.symbols:
        index += 1
    return index