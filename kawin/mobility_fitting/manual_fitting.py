import itertools
from typing import Union
from collections import namedtuple

import numpy as np
from symengine import Basic
from tinydb import TinyDB, where

from pycalphad import variables as v
from espei.datasets import load_datasets, recursive_glob

from kawin.mobility_fitting.template import MobilityTemplate, SiteFractionGenerator
from kawin.mobility_fitting.utils import _eval_symengine_expr

DatasetPair = namedtuple('SystemPair', ['site_fractions', 'values'])
FittingResult = namedtuple('FittingResult', ['template', 'parameters', 'aicc'])
    
def grab_datasets(datasets: Union[TinyDB, str], data_type: str, phase: str, components: list[str], diffusing_species: str, include_disabled: bool = False) -> tuple[list, list]:
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
        datasets = load_datasets(sorted(recursive_glob(datasets, '*.json')), include_disabled)

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

def _fit_model(data: list, template: MobilityTemplate, function: Basic, site_fraction_generator: SiteFractionGenerator = None, p = 1, transform = None):
    '''
    Fits activation energy or prefactor model
    This is really more of a helper function for fit_prefactor and fit_activation_energy

    Parameters
    ----------
    datasets: list
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

def fit_prefactor(data: list, template: MobilityTemplate, site_fraction_generator: SiteFractionGenerator = None, p = 1):
    '''
    Fits prefactor model

    Parameters
    ----------
    data: list
    template: MobilityTemplate
    site_fraction_generate: SiteFractionGenerator
    p: int
        AICC penalty factor

    Returns
    -------
    symbols: {Symbol: float}
    aicc: int
    '''
    return _fit_model(data, template, template.prefactor, site_fraction_generator, p, lambda x: np.log(x))

def fit_activation_energy(data: list, template: MobilityTemplate, site_fraction_generator: SiteFractionGenerator = None, p = 1):
    '''
    Fits activation energy model

    Parameters
    ----------
    data: list
    template: MobilityTemplate
    site_fraction_generate: SiteFractionGenerator
    p: int
        AICC penalty factor

    Returns
    -------
    symbols: {Symbol: float}
    aicc: int
    '''
    return _fit_model(data, template, template.activation_energy, site_fraction_generator, p)

def select_best_model(data: list, templates: list[MobilityTemplate], fit_function: callable, site_fraction_generator: SiteFractionGenerator = None, p = 1, return_all_models = False):
    '''
    Fits multiple templates and selects the best one based off the AICC criteria

    Parameters
    ----------
    data: list
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
        params, aicc = fit_function(data, template, site_fraction_generator, p=p)
        params_list.append(params)
        aicc_list.append(aicc)

    best_index = np.argmin(aicc_list)

    if return_all_models:
        all_fits = [FittingResult(template=t, parameters=p, aicc=a) for t,p,a in zip(templates, params_list, aicc_list)]
        return all_fits[best_index], all_fits
    else:
        return FittingResult(template=templates[best_index], parameters=params_list[best_index], aicc=aicc_list[best_index])
