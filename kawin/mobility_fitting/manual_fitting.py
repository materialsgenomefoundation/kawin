import numpy as np
from pycalphad import Database, Model, variables as v
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import unpack_condition
from symengine import Piecewise, And
from tinydb import where
from kawin.thermo.LocalEquilibrium import local_equilibrium
from espei.datasets import load_datasets, recursive_glob

class EquilibriumSiteFractionGenerator:
    def __init__(self, database : Database, phase : str):
        self.db = database
        self.phase = phase
        self.models = {}
        self.phase_records = {}
        self.constituents = {}

        self.full_constituents = [c for cons in self.db.phases[self.phase].constituents for c in sorted(list(cons))]
        self._conditions_override = {v.N: 1, v.GE: 0}

    def set_override_condition(self, variable, value):
        self._conditions_override[variable] = value

    def remove_override_condition(self, variable):
        self._conditions_override.pop(variable)

    def _generate_comps_key(self, components):
        comps = sorted(components)
        return frozenset(comps), comps

    def _generate_phase_records(self, components):
        active_comps, comps = self._generate_comps_key(components)
        if active_comps not in self.models:
            self.models[active_comps] = {self.phase: Model(self.db, comps, self.phase)}
            self.phase_records[active_comps] = build_phase_records(self.db, comps, [self.phase], {v.T, v.P, v.N, v.GE}, self.models[active_comps])
            self.constituents[active_comps] = [c for cons in self.models[active_comps][self.phase].constituents for c in sorted(list(cons))]

    def __call__(self, components, conditions : dict[v.StateVariable: float]) -> dict[v.Species: float]:
        active_comps, comps = self._generate_comps_key(components)
        self._generate_phase_records(components)

        for oc in self._conditions_override:
            conditions[oc] = self._conditions_override[oc]
        results, comp_sets = local_equilibrium(self.db, comps, [self.phase], conditions, self.models[active_comps], self.phase_records[active_comps])
        sfg = {c:val for c,val in zip(self.constituents[active_comps], comp_sets[0].dof[4:])}
        for c in self.full_constituents:
            sfg[c] = sfg.get(c,0)
        return sfg
    
class SiteFractionGenerator:
    def create_site_fractions(self, composition):
        return NotImplementedError()

    def __call__(self, components, conditions):
        comps_no_va = list(set(components) - set(['VA']))
        composition = {c:1 for c in comps_no_va}
        for key,val in conditions.items():
            if type(key) == v.MoleFraction:
                for c in composition:
                    composition[c] -= val
                composition[key.name[2:]] = val

        return self.create_site_fractions(composition)

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
        self.expr = None

    def generate_multiplier(self, site_fractions):
        val = 1
        for clist in self.constituent_array:
            clist = np.atleast_1d(clist)
            for c in clist:
                val *= site_fractions.get(c,0)
            if len(clist) == 2:
                ordered = sorted(clist)
                val *= (site_fractions.get(ordered[1],0) - site_fractions.get(ordered[0],0))**self.order
        return val
    
    def create_constituent_array_list(self):
        return [[c.species.name for c in np.atleast_1d(clist)] for clist in self.constituent_array]
    
    def __eq__(self, other):
        return self.constituent_array == other.constituent_array and self.order == other.order

def add_mobility_model(database : Database, phase : str, diffusing_species : str, mobility_terms : list[MobilityTerm]):
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

def least_squares_fit(A, b, p=1):
    '''
    Given site fractions and function to generate Redlich-kister terms,
    compute RK coefficients and AICC criteria
    '''
    A
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    k = len(A[0])
    b_pred = np.matmul(A, x)
    L = np.log(np.sum((b_pred - b)**2) / len(A))
    aicc = np.array(2*p*k + len(A)*L + (2*(k*p)**2 + 2*k*p) / (len(A) - k*p - 1))
    return x, aicc

def find_last_variable(database):
    index = 0
    vname = 'VV{:04d}'.format(index)
    while vname in database.symbols:
        index += 1
        vname = 'VV{:04d}'.format(index)
    return vname, index


def fit(datasets, database, components, phase, diffusing_species, mobility_test_models, site_fraction_generator, p = 1):
    '''
    Fit mobility models to datasets

    Parameters
    ----------
    datasets : list[dict]
        Espei datasets
    species : list[v.Species]
        List of species in mobility_test_models
    components : list[str]
    phase : str
    diffusing_species : str
    mobility_test_models : list[list[MobilityTerm]]
    site_fraction_generator : function
        Takes in components and list of conditions and returns dictionary {v.SiteFraction : float}
    '''
    if type(datasets) == str:
        datasets = load_datasets(sorted(recursive_glob(datasets, '*.json')))

    components = list(set(components).union(set(['VA'])))

    fitted_mobility_model = []

    q_query = (
        (where('components').test(lambda x: set(x).issubset(components))) & 
        (where('phases').test(lambda x: len(x) == 1 and x[0] == phase)) &
        (where('output').test(lambda x : 'TRACER_Q' in x and x.endswith(diffusing_species)))
    )
    d0_query = (
        (where('components').test(lambda x: set(x).issubset(components))) & 
        (where('phases').test(lambda x: len(x) == 1 and x[0] == phase)) &
        (where('output').test(lambda x : 'TRACER_D0' in x and x.endswith(diffusing_species)))
    )
    q_transform = lambda x : x
    d0_transform = lambda x : np.log(x)
    data_types = {
        'Q': (q_query, q_transform, -1),
        'D0': (d0_query, d0_transform, 8.314*v.T)
    }
    
    for data_key, data_val in data_types.items():
        query, transform, multiplier = data_val
        data = datasets.search(query)

        site_fractions = []
        Y = []
        for d in data:
            conds_grid = []
            conds_key = []
            for c in d['conditions']:
                conds_grid.append(np.atleast_1d(d['conditions'][c]))
                conds_key.append(v.X(c[2:]) if c.startswith('X_') else getattr(v, c))
            conds_grid = np.meshgrid(*conds_grid)
            y_sub = transform(np.array(d['values']).flatten())

            conds_list = {key:val.flatten() for key,val in zip(conds_key, conds_grid)}

            if 'solver' in d:
                for sub_conf, sub_lat in zip(d['solver']['sublattice_configurations'], d['solver']['sublattice_occupancies']):
                    sub_index = 0
                    sf = {}
                    for species, occs in zip(sub_conf, sub_lat):
                        species, occs = np.atleast_1d(species), np.atleast_1d(occs)
                        for s, o in zip(species, occs):
                            y = v.SiteFraction(phase, sub_index, s)
                            sf[y] = o
                        sub_index += 1
                    site_fractions.append(sf)

            else:
                for i in range(len(Y)):
                    sf = site_fraction_generator(d['components'], {key:val[i] for key,val in conds_list.items()})
                    site_fractions.append(sf)

            Y = np.concatenate((Y, y_sub))

        fitted_models = []
        aiccs = []
        for mob_model in mobility_test_models:
            X = [[mi.generate_multiplier(sf) for mi in mob_model] for sf in site_fractions]
            X = np.array(X)
            terms, aicc = least_squares_fit(X, Y, p)
            fitted_models.append(terms*multiplier)
            aiccs.append(aicc)

        index = np.argmin(aiccs)
        best_model = mobility_test_models[index]
        best_fit = fitted_models[index]

        for term, coef in zip(best_model, best_fit):
            if term not in fitted_mobility_model:
                fitted_mobility_model.append(MobilityTerm(term.constituent_array, term.order))
            
            combined_term = fitted_mobility_model[fitted_mobility_model.index(term)]
            if combined_term.expr is None:
                combined_term.expr = 0
            combined_term.expr += coef

    for f in fitted_mobility_model:
        f.expr = Piecewise((f.expr, And(0 <= v.T, v.T < 10000)), (0, True))
        print(f.constituent_array, f.expr)

    return fitted_mobility_model




            




