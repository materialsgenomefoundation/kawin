import itertools
from typing import Union
from abc import abstractmethod, ABC

import numpy as np
import matplotlib.pyplot as plt
from symengine import Symbol, Basic
from tinydb import TinyDB, where

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from espei.datasets import load_datasets, recursive_glob

from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.mobility_fitting.utils import DatasetPair, FittingResult, MobilityTemplate, _eval_symengine_expr
    
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
    
def grab_datasets(datasets: Union[TinyDB, str], data_type: str, phase: str, components: list[str], diffusing_species: str) -> tuple[list, list]:
    '''
    Grabs all datasets for diffusing species of data type

    Parameters
    ----------
    datasets: TinyDB | str
        If str, must be a folder to load datasets from
    data_type: str
        Must be Q or D0
    phase: str
    components: list[str]
    diffusing_species: str
    '''
    if data_type != 'Q' and data_type != 'D0':
        raise ValueError('Data type must be Q or D0')
    
    if type(datasets) == str:
        datasets = load_datasets(sorted(recursive_glob(datasets, '*.json')))

    components = list(set(components).union(set(['VA'])))

    query = (
        (where('components').test(lambda x: set(x).issubset(components))) & 
        (where('phases').test(lambda x: len(x) == 1 and x[0] == phase)) &
        (where('output').test(lambda x : f'TRACER_{data_type}' in x and x.endswith(diffusing_species)))
    )
    return datasets.search(query)

def generate_site_fractions(data: list, site_fraction_generator: SiteFractionGenerator = None) -> DatasetPair:
    '''
    Collect site fractions and output values from datasets

    data: list
        List of datasets to grab conditions from
    site_fraction_generator: SiteFractionGenerator (optional)
        Object that converts conditions to site fraction values
        If all datasets are non-equilibrium data, then this is not needed
    '''
    site_fractions = []
    Y = []
    for d in data:
        conds_grid = []
        conds_key = []
        for c in d['conditions']:
            conds_grid.append(np.atleast_1d(d['conditions'][c]))
            conds_key.append(v.X(c[2:]) if c.startswith('X_') else getattr(v, c))
        conds_grid = np.meshgrid(*conds_grid)
        y_sub = np.array(d['values']).flatten()

        conds_list = {key:val.flatten() for key,val in zip(conds_key, conds_grid)}

        # If non-equilibrium data, we could grab the site fractions directly
        if 'solver' in d:
            for sub_conf, sub_lat in zip(d['solver']['sublattice_configurations'], d['solver']['sublattice_occupancies']):
                sub_index = 0
                sf = {}
                for species, occs in zip(sub_conf, sub_lat):
                    species, occs = np.atleast_1d(species), np.atleast_1d(occs)
                    for s, o in zip(species, occs):
                        y = v.SiteFraction(d['phases'][0], sub_index, s)
                        sf[y] = o
                    sub_index += 1
                site_fractions.append(sf)
        # If equilibrium data, we need a site_fraction_generator function to
        # convert composition to site fractions
        else:
            for i in range(len(y_sub)):
                sf = site_fraction_generator.generate_site_fractions(d['phases'][0], d['components'], {key:val[i] for key,val in conds_list.items()})
                site_fractions.append(sf)

        Y = np.concatenate((Y, y_sub))
    
    return DatasetPair(site_fractions=site_fractions, values=Y)

def least_squares_fit(A: np.ndarray, b: np.ndarray, p: int = 1) -> tuple[np.ndarray, float]:
    '''
    Given site fractions and function to generate Redlich-kister terms,
    compute RK coefficients and AICC criteria

    Parameters
    ----------
    A: np.ndarray (M,N)
    b: np.ndarray (M,1)
    p: int
        Penalty factor

    Returns
    -------
    x: np.ndarray
    aicc: float
    '''
    # Fit coefficients using least squares regression
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # AICC criteria
    k = len(A[0])
    n = len(A)
    b_pred = np.matmul(A, x)
    rss = np.sum((b_pred-b)**2)
    pk = p*k
    aic = 2*pk + n*np.log(rss/n)
    if pk >= n-1:
        correction = (2*pk**2 + 2*pk) / (-n + pk + 3)
    else:
        correction = (2*pk**2 + 2*pk) / (n - pk - 1)
    aicc = aic + correction
    return x, aicc

def _fit_model(datasets: Union[TinyDB, str], data_type: str, template: MobilityTemplate, function: Basic, site_fraction_generator: SiteFractionGenerator = None, p = 1, transform = None):
    '''
    Fits activation energy or prefactor model
    This is really more of a helper function for fit_prefactor and fit_activation_energy

    Parameters
    ----------
    datasets: Union[TinyDB, str]
    data_type: str
        Either 'D0' or 'Q'
    template: MobilityTemplate
    function: Basic
        either MobilityTemplate.prefactor or MobilityTemplate.activation_energy
    site_fraction_generate: SiteFractionGenerator
    p: int
        AICC penalty factor
    transform: callable (optional)
        Transformation on dataset values to be compatible with the mobility template function
        For prefactor, this should be f(x) = ln(x)
        For activation energy, this is not needed (i.e. f(x) = x)
    '''
    data = grab_datasets(datasets, data_type, template.phase, template.elements, template.diffusing_species.name)
    derivatives = template.get_derivatives(function)
    sf_data = generate_site_fractions(data, site_fraction_generator)

    A = np.zeros((len(sf_data.values), len(derivatives.symbols)))
    b = np.array(sf_data.values)
    if transform is not None:
        b = transform(b)

    for i,j in itertools.product(range(len(sf_data.site_fractions)), range(len(derivatives.functions))):
        A[i,j] = _eval_symengine_expr(derivatives.functions[j], sf_data.site_fractions[i])

    x, aicc = least_squares_fit(A, b, p)
    return {s:xi for s,xi in zip(derivatives.symbols, x)}, aicc

def fit_prefactor(datasets: Union[TinyDB, str], template: MobilityTemplate, site_fraction_generator: SiteFractionGenerator = None, p = 1):
    '''
    Fits prefactor model

    Parameters
    ----------
    datasets: Union[TinyDB, str]
    template: MobilityTemplate
    site_fraction_generate: SiteFractionGenerator
    p: int
        AICC penalty factor

    Returns
    -------
    symbols: {Symbol: float}
    aicc: int
    '''
    return _fit_model(datasets, 'D0', template, template.prefactor, site_fraction_generator, p, lambda x: np.log(x))

def fit_activation_energy(datasets: Union[TinyDB, str], template: MobilityTemplate, site_fraction_generator: SiteFractionGenerator = None, p = 1):
    '''
    Fits activation energy model

    Parameters
    ----------
    datasets: Union[TinyDB, str]
    template: MobilityTemplate
    site_fraction_generate: SiteFractionGenerator
    p: int
        AICC penalty factor

    Returns
    -------
    symbols: {Symbol: float}
    aicc: int
    '''
    return _fit_model(datasets, 'Q', template, template.activation_energy, site_fraction_generator, p)

def select_best_model(datasets: Union[TinyDB, str], templates: list[MobilityTemplate], fit_function: callable, site_fraction_generator: SiteFractionGenerator = None, p = 1, return_all_models = False):
    '''
    Fits multiple templates and selects the best one based off the AICC criteria

    Parameters
    ----------
    datasets: Union[TinyDB, str]
    templates: list[MobliityTemplate]
        List of mobility templates to fit and compare
    fit_function: callable
        Function that takes in (datasets, MobilityTemplate, SiteFractionGenerate, penalty factor) and returns fitted parameters and AICC
        Options are fit_prefactor and fit_activation_energy
    site_fraction_generator: SiteFractionGenerator
    p: int
        AICC penalty factor
    return_all_models: bool
        If true, this will also return a list of FittingResults for each template

    Returns
    -------
    FittingResults for best model
    list[FittingResults] of all models if return_all_models = True
    '''
    params_list = []
    aicc_list = []
    for template in templates:
        params, aicc = fit_function(datasets, template, site_fraction_generator, p=p)
        params_list.append(params)
        aicc_list.append(aicc)

    best_index = np.argmin(aicc_list)

    if return_all_models:
        all_fits = [FittingResult(template=t, parameters=p, aicc=a) for t,p,a in zip(templates, params_list, aicc_list)]
        return all_fits[best_index], all_fits
    else:
        return FittingResult(template=templates[best_index], parameters=params_list[best_index], aicc=aicc_list[best_index])

def evaluate_model(template: MobilityTemplate, function: Basic, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float]):
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
    if len(dependent_var) != 1:
        raise ValueError(f"Number of free conditions must be 1, but is {len(dependent_var)}")
    
    dependent_vals = conditions[dependent_var[0]]
    xs = np.linspace(dependent_vals[0], dependent_vals[1], int((dependent_vals[1] - dependent_vals[0])/dependent_vals[2]))
    ys = np.zeros(len(xs))
    for i,x in enumerate(xs):
        new_conds = {v.N: 1, v.P: 101325, v.GE: 0, v.T: 298.15}
        new_conds.update({**conditions})
        new_conds[dependent_var[0]] = x
        site_fractions = site_fraction_generator.generate_site_fractions(template.phase, template.elements, new_conds)
        ys[i] = _eval_symengine_expr(function, {**site_fractions, **parameter_values, **new_conds})

    return xs, ys, dependent_var[0]

def plot_prefactor(template: MobilityTemplate, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float], ax=None, *args, **kwargs):
    '''
    Plots mobility prefactor using input parameters
    '''
    if ax is None:
        fig, ax = plt.subplots()

    xs, ys, dep_var = evaluate_model(template, template.prefactor, parameter_values, site_fraction_generator, conditions)
    ax.plot(xs, np.exp(ys), *args, **kwargs)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_xlabel(dep_var)
    ax.set_ylabel(r'Diffusion Prefactor ($cm^2/s$)')
    ax.set_yscale('log')
    return ax

def plot_activation_energy(template: MobilityTemplate, parameter_values: dict[Symbol, float], site_fraction_generator: SiteFractionGenerator, conditions: dict[v.StateVariable, float], ax=None, *args, **kwargs):
    '''
    Plots mobility activation energy using input parameters
    '''
    if ax is None:
        fig, ax = plt.subplots()

    xs, ys, dep_var = evaluate_model(template, template.activation_energy, parameter_values, site_fraction_generator, conditions)
    ax.plot(xs, ys, *args, **kwargs)
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_xlabel(dep_var)
    ax.set_ylabel('Activation Energy (J/mol)')
    return ax
