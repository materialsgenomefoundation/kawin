import numpy as np
from kawin.thermo import GeneralThermodynamics
from pycalphad import Database, variables as v
from espei.utils import database_symbols_to_fit

def get_used_database_symbols(dbname, elements, refElement, phases = None, freeSub = False):
    '''
    Given the database, grab all symbols that pertain only to the given elements and phases

    If freeSub, then parameters for all subsystems (unaries, binaries, ...) will be added
    If not, then only grab parameters for the specific number of elements
        Ex. If fitting ternary A-B-C
            If freeSub, grab parameters from A, B, C, A-B, A-C, B-C and A-B-C
            Else, grab parameters only from A-B-C

    ex:
        PARAMETER(BCC_A2,AL:0) 298.15 VV0001; 6000.0 N !
        PARAMETER(BCC_A2,CR:0) 298.15 VV0002; 6000.0 N !
        PARAMETER(FCC_A1,AL:0) 298.15 VV0003; 6000.0 N !
        PARAMETER(FCC_A1,CR:0) 298.15 VV0004; 6000.0 N !

    Inputting ['AL'] and ['BCC_A2'] will output VV0001
    Inputting ['AL', 'CR'] and ['FCC_A1'] will output VV0003 and VV0004
    Inputting ['CR'] and ['BCC_A2', 'FCC_A1'] will output VV0002 and VV0004
    '''
    db = Database(dbname)
    elements = sorted([e.upper() for e in elements])
    numElements = len(elements)
    elSet = frozenset(elements + ['VA'])
    usedSyms = frozenset()
    for p in db._parameters:
        if phases is not None:
            if p['phase_name'] not in phases:
                continue
        parameterSpecies = frozenset([s.name for c in p['constituent_array'] for s in c])
        #Add symbols under 2 conditions
        #   1. parameterSpecies is a subset of elSet
        #   2. freeSub is False and length of parameter species (excluding VA) is equal to number of elements
        pIsSubset = len((parameterSpecies-set(['VA','*'])) - elSet) == 0
        shouldBeFree = freeSub or len(parameterSpecies - set(['VA', '*'])) == numElements
        isDiffusingSpecies = p['diffusing_species'].name == refElement
        if pIsSubset and shouldBeFree and isDiffusingSpecies:
            usedSyms = usedSyms.union(frozenset([s.name for s in p['parameter'].free_symbols]))
        #if len(parameterSpecies - elSet) == 0:
        #    usedSyms = usedSyms.union(frozenset([s.name for s in p['parameter'].free_symbols]))
    allSyms = database_symbols_to_fit(db)
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