import itertools
from typing import Union
from abc import abstractmethod, ABC
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from symengine import Symbol, Basic, S
from tinydb import Query

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.io.tdb import _molmass

from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.thermo.Mobility import MobilityModel
from kawin.mobility_fitting.utils import _vname, _get_variable_terms, _eval_symengine_expr, find_last_variable

DerivativePair = namedtuple('DerivativePair', ['functions', 'symbols'])

def _upsert_parameters(dbf: Database, parameter_type: str, phase: str, constituent_array, parameter_order: int, diffusing_species: str, function: Basic, add_if_exists: bool = True):
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
        # If parameter already exists, we can either add the function to the present paramter or overwrite it
        if add_if_exists:
            dbf._parameters.upsert({'parameter': param[0]['parameter'] + function}, search_query)
        else:
            dbf._parameters.upsert({'parameter': param[0]['parameter']}, search_query)

def transform_activation_energy(Q: float) -> Basic:
    '''Transform activation energy to A term in MQ=A+B*T'''
    return -Q

def transform_prefactor(D0: float) -> Basic:
    '''Transforms diffusion prefactor to B term in MQ=A+B*T (note, this returns B, not B*T)'''
    return v.R*np.log(D0)

class SiteFractionGenerator(ABC):
    '''
    Object to compute site fractions from conditions
    '''
    @abstractmethod
    def generate_site_fractions(self, phase: str, components: list[str], conditions : dict[v.StateVariable, float]) -> dict[v.Species, float]:
        '''
        Generate site fractions from conditions

        Parameters
        ----------
        phase: str
        components: list[str]
        conditions: dict[v.StateVariable, float]

        Returns
        -------
        {v.SiteFraction: float}
        '''
        raise NotImplementedError()

class EquilibriumSiteFractionGenerator(SiteFractionGenerator):
    '''
    Grabs site fraction values from local equilibrium calculations

    Parameters
    ----------
    database: Database
    '''
    def __init__(self, database : Database, override_conditions = {}):
        if isinstance(database, str):
            database = Database(database)
        self.dbf = database
        self.models = {}
        self.phase_records = {}

        self._conditions_override = {v.N: 1, v.GE: 0}
        for key, value in override_conditions:
            self.set_override_condition(key, value)

    def set_override_condition(self, variable, value):
        self._conditions_override[variable] = value

    def remove_override_condition(self, variable):
        self._conditions_override.pop(variable)

    def _generate_comps_key(self, components):
        comps = sorted(components)
        return frozenset(comps), comps

    def _generate_phase_records(self, phase, components):
        # Caches models, phase_records and constituents based off active components and phase
        active_comps, comps = self._generate_comps_key(components)
        if active_comps not in self.models:
            self.models[active_comps] = {}
            self.phase_records[active_comps] = {}
        
        if phase not in self.models[active_comps]:
            self.models[active_comps][phase] = {phase: Model(self.dbf, comps, phase)}
            self.phase_records[active_comps][phase] = PhaseRecordFactory(self.dbf, comps, {v.T, v.P, v.N, v.GE}, self.models[active_comps][phase])
        
        return self.models[active_comps][phase], self.phase_records[active_comps][phase]

    def generate_site_fractions(self, phase: str, components: list[str], conditions : dict[v.StateVariable: float]) -> dict[v.Species: float]:
        # Get phase records from active components
        active_comps, comps = self._generate_comps_key(components)
        models, prf = self._generate_phase_records(phase, components)

        # Override any conditions
        for oc in self._conditions_override:
            conditions[oc] = self._conditions_override[oc]

        # Compute local equilibrium (to avoid miscibility gaps)
        # Store site fractions
        #    The first 4 items of CompositionSet.dof refers to v.GE, v.N, v.P and v.T
        #    The order of the site fractions in CompositionSet.dof should correspond to the order in the pycalphad model
        results, comp_sets = local_equilibrium(self.dbf, comps, [phase], conditions, models, prf)
        sfg = {c:val for c,val in zip(prf[phase].variables, comp_sets[0].dof[4:])}
        return sfg

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
            _upsert_parameters(self.dbf, parameter_type, self.phase, constituent_array, p, self.diffusing_species, param_func(), add_if_exists=True)
        
        # Since we updated the database, clear the mobility model if it was made at some point
        self._model = None

    def add_prefactor(self, constituent_array: list, parameter_order: Union[int, list[int]], parameter_type: str = 'MQ', parameter = None):
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
        parameter: float (Optional)
            If not supplied (default), them a VV00XX*T term will be added
            If supplied, this will assumed to be the prefactor and will be transformed into B*T term accordingly
        '''
        if parameter is None:
            parameter = self._create_T_term
        else:
            parameter = transform_prefactor(parameter)*v.T
        self.add_term(constituent_array, parameter_order, parameter, parameter_type=parameter_type)
        
    def add_activation_energy(self, constituent_array: list, parameter_order: Union[int, list[int]], parameter_type: str='MQ', parameter = None):
        '''
        Adds a activation energy term (in form of VV00XX)

        Parameters
        ----------
        constituent_array: list[list[str]]
        param_order: int | list[int]
            If list, then terms will be added to all specified orders
        symbols: str (Optional)
            Defaults to 'MQ'
        parameter: float (Optional)
            If not supplied (default), them a VV00XX term will be added
            If supplied, this will assumed to be the activation energy and will be transformed into A term accordingly
        '''
        if parameter is None:
            parameter = self._create_constant
        else:
            parameter = transform_activation_energy(parameter)
        self.add_term(constituent_array, parameter_order, parameter, parameter_type=parameter_type)

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
    
    def evaluate(self, function: Basic, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float]):
        '''
        Evaluates model template along single variable

        Parameters
        ----------
        template: MobilityTemplate
        function: symengine.Basic
            Function to evaluate
        parameter_values: dict[Symbol, float]
            Fitted parameters
            Parameters in function that are not defined here will assume to be 0
        site_fraction_generator: SiteFractionGenerator
        conditions: dict[v.StateVariable, float]
            Conditions to evaluate function, 1 variable must be a list

        Returns
        -------
        xs: values of evaluated dependent condition
        ys: values of evaluated function
        dependent_var: dependent condition
        '''
        dependent_var = [key for key, val in conditions.items() if len(np.atleast_1d(val)) > 1]
        # Single point condition, then return evaluated function
        if len(dependent_var) == 0:
            site_fractions = site_fraction_generator.generate_site_fractions(self.phase, self.elements, conditions)
            y = _eval_symengine_expr(function, {**site_fractions, **parameter_values, **conditions})
            return y
        # If 1 dependent variable, then return coordinates of dependent variable (x), evaluated function (y), and variable
        elif len(dependent_var) == 1:
            dependent_vals = conditions[dependent_var[0]]
            xs = np.linspace(dependent_vals[0], dependent_vals[1], int((dependent_vals[1] - dependent_vals[0])/dependent_vals[2]))
            ys = np.zeros(len(xs))
            for i,x in enumerate(xs):
                new_conds = {v.N: 1, v.P: 101325, v.GE: 0, v.T: 298.15}
                new_conds.update({**conditions})
                new_conds[dependent_var[0]] = x
                site_fractions = site_fraction_generator.generate_site_fractions(self.phase, self.elements, new_conds)
                ys[i] = _eval_symengine_expr(function, {**site_fractions, **parameter_values, **new_conds})

            return xs, ys, dependent_var[0]
        else:
            raise ValueError(f"Number of free conditions must be 1, but is {len(dependent_var)}")
    
    def add_to_database(self, dbf: Database, parameter_values: dict[Union[Symbol, str], float], add_if_exists=True):
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
                            p['parameter_order'], p['diffusing_species'], new_func, add_if_exists=add_if_exists)

        for s in parameter_values:
            dbf.symbols[symbol_replace_map[s]] = parameter_values[s]

def _plot_model(template: MobilityTemplate, function: Basic, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float], ax=None, scale=1, transform = None, *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    xs, ys, dep_var = template.evaluate(function, parameter_values, site_fraction_generator, conditions)
    if transform is not None:
        ys = transform(ys)
    ax.plot(xs, scale*ys, *args, **kwargs)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_xlabel(dep_var)
    return ax

def plot_prefactor(template: MobilityTemplate, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float], ax=None, scale=1, *args, **kwargs):
    '''
    Plots mobility prefactor using input parameters
    '''
    ax = _plot_model(template, template.prefactor, parameter_values, site_fraction_generator, conditions, transform=lambda x: np.exp(x), ax=ax, scale=scale, *args, **kwargs)
    ax.set_ylabel(r'Diffusion Prefactor ($m^2/s$)')
    ax.set_yscale('log')
    return ax

def plot_activation_energy(template: MobilityTemplate, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float], ax=None, scale=1, *args, **kwargs):
    '''
    Plots mobility activation energy using input parameters
    '''
    ax = _plot_model(template, template.activation_energy, parameter_values, site_fraction_generator, conditions, ax=ax, scale=scale, *args, **kwargs)
    ax.set_ylabel('Activation Energy (J/mol)')
    return ax