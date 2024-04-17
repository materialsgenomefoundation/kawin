"""
Calculate error due to thermochemical quantities: heat capacity, entropy, enthalpy.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, NewType, Optional, Tuple, Union, Sequence

from scipy.stats import norm
import numpy as np
import numpy.typing as npt
from tinydb import where

from pycalphad import Database, variables as v
from pycalphad.core.utils import unpack_components, filter_phases, unpack_condition, extract_parameters

from espei.datasets import Dataset
from espei.core_utils import ravel_conditions, get_prop_data, filter_temperatures
from espei.phase_models import PhaseModelSpecification
from espei.sublattice_tools import canonicalize, recursive_tuplify, tuplify
from espei.typing import SymbolName
from espei.utils import database_symbols_to_fit, PickleableTinyDB
from espei.error_functions.residual_base import ResidualFunction, residual_function_registry
from espei.error_functions.non_equilibrium_thermochemical_error import FixedConfigurationCalculationData, filter_sublattice_configurations, calculate_points_array

import kawin.mobility_fitting.error_functions.cached_mobility as cmob
from kawin.thermo.Mobility import MobilityModel
from kawin.mobility_fitting.error_functions.utils import get_output_base_name, get_base_names, build_model, get_base_std

import tinydb

_log = logging.getLogger(__name__)

class NonEquilibriumMobilityData:
    def __init__(self, dbf, data, parameters = None, data_weight_dict = None):
        self.parameters = parameters if parameters is not None else {}
        self.parameter_keys = [p for p in self.parameters]
        self.data_weight_dict = data_weight_dict if data_weight_dict is not None else {}
        self.output = get_output_base_name(data['output'])

        self.comps = data['components']
        self.elements = sorted(list(set([v.Species(c).name for c in self.comps])))
        self.non_va_elements = sorted(list(set(self.elements) - set(['VA'])))
        self.phases = data['phases']
        self.refComp = data.get('ref_el', None)
        if self.refComp is None and 'TRACER' in data['output']:
            self.refComp = [data['output'][len(self.output)+1:]]

        self.depComps = data.get('dependent_el', [])
        self.diffusing_species = sorted(list(set(self.refComp) | set(self.depComps) | set(self.elements) - set(['VA'])))
        self.models, self.phase_records, self.mob_models, self.mob_callables = build_model(dbf, self.elements, self.phases[0], parameters, self.diffusing_species)
        self.mobility_correction = {c:1 for c in self.non_va_elements}

        self._processConditions(data)

        dataset_weights = np.array(data.get('weight', 1.0)) * np.ones(len(self.values))
        property_std_deviation = get_base_std()
        self.weights = (property_std_deviation.get(self.output, 1.0)/data_weight_dict.get(self.output, 1.0)/dataset_weights).flatten()

    def _processConditions(self, data):
        T = data['conditions']['T']
        P = data['conditions']['P']
        values = np.array(data['values'])
        self.values = values.flatten()
        self.P, self.T = ravel_conditions(values, P, T)
        
        sub_conf = data['solver']['sublattice_configurations']
        sub_occ = data['solver']['sublattice_occupancies']
        phase_cons = [[c.name for c in s] for s in self.models[self.phases[0]].constituents]
        single_points = np.array([calculate_points_array(phase_cons, conf, occ) for conf, occ in zip(sub_conf, sub_occ)])
        self.points = np.tile(single_points, (values.shape[0]*values.shape[1], 1))
        
def get_mob_data(dbf: Database, comps: Sequence[str], phases: Sequence[str], datasets: PickleableTinyDB, parameters: Dict[str, float], data_weight_dict: Optional[Dict[str, float]] = None):
    '''
    Return the ZPF data used in the calculation of ZPF error

    Parameters
    ----------
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data
    parameters : dict
        Dictionary mapping symbols to optimize to their initial values
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.

    Returns
    -------
    list
        List of data dictionaries with keys ``weight``, ``phase_regions`` and ``dataset_references``.
    '''
    base_names = get_base_names()
    #Don't support interdiffusivity for now
    #I suppose it is possible to support it by computing composition from
    # the site fractions and computing it similar to equilibrium mobility error,
    # but I feel like you may as well set the dataset to be an equilibrium data at
    # that point
    #Does interdiffusivity even make sense to have a value at a specific sublattice configuration?
    base_names = list(set(base_names) - set(['INTER_DIFF']))
    desired_data = datasets.search((tinydb.where('output').test(lambda x: get_output_base_name(x) in base_names)) &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)) &
                                   (tinydb.where('solver').exists()))
    
    mod_data = []
    for data in desired_data:
        mod_data.append(NonEquilibriumMobilityData(dbf, data, parameters, data_weight_dict))   
    return mod_data 

def calc_mob_differences(data : NonEquilibriumMobilityData, parameters : np.ndarray):
    diffs, wts = [], []
    paramDict = {data.parameter_keys[i] : parameters[i] for i in range(len(data.parameter_keys))}
        
    #Update phase record parameters
    param_keys, param_values = extract_parameters(paramDict)
    for p in data.phases:
        data.phase_records[p].parameters[:] = np.asarray(param_values, dtype=np.float_)

    state_variables = sorted([v.T, v.P, v.N, v.GE])

    for i in range(len(data.values)):
        T = data.T[i]
        P = data.P[i]
        value = data.values[i]
        point = data.points[i]

        dof = np.zeros(4 + len(data.mob_models[data.phases[0]].site_fractions))
        dof[state_variables.index(v.T)] = T
        dof[state_variables.index(v.P)] = P
        dof[state_variables.index(v.N)] = 1
        dof[4:] = point

        refIndex = data.diffusing_species.index(data.refComp[0])

        if data.output == 'TRACER_DIFF':
            calculated_values = cmob.mobility_from_composition_set_symbolic_quick(dof, data.diffusing_species, data.mob_models[data.phases[0]], paramDict)
            r = calculated_values[refIndex]
        elif data.output == 'TRACER_Q':
            calculated_values = cmob.activation_energy_from_composition_set_quick(dof, data.diffusing_species, data.mob_models[data.phases[0]], paramDict)
            r = calculated_values[refIndex]
        elif data.output == 'TRACER_D0':
            calculated_values = cmob.prefactor_from_composition_set_quick(dof, data.diffusing_species, data.mob_models[data.phases[0]], paramDict)
            r = calculated_values[refIndex]

        if data.output == 'TRACER_DIFF':
            r = np.sign(r) * np.log10(np.abs(r))
            value = np.sign(value) * np.log10(np.abs(value))
        elif data.output == 'TRACER_D0':
            r = np.sign(r) * np.log10(np.exp(r))
            value = np.sign(value) * np.log10(np.abs(value))

        diffs.append(r - value)
        wts.append(data.weights[i])

    return diffs, wts

def calculate_mob_probability(mob_data : Sequence[NonEquilibriumMobilityData], parameters : np.ndarray) -> float:
    differences = []
    weights = []
    for data in mob_data:
        diffs, wts = calc_mob_differences(data, parameters)
        if np.any(np.isinf(diffs) | np.isnan(diffs)):
            return -np.inf
        differences.append(diffs)
        weights.append(wts)
    
    differences = np.concatenate(differences, axis=0)
    weights = np.concatenate(weights, axis=0)
    probs = norm(loc=0.0, scale=weights).logpdf(differences)
    return np.sum(probs)

class NonEquilibriumMobilityResidual(ResidualFunction):
    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: Union[PhaseModelSpecification, None],
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        additional_mcmc_args: Optional[Dict] = {},
        ):
        super().__init__(database, datasets, phase_models, symbols_to_fit)
        if weight is not None:
            weight = {name: weight.get(name, 1.0) for name in get_base_names()}
        else:
            weight = {name: 1.0 for name in get_base_names()}
        if phase_models is not None:
            comps = sorted(phase_models.components)
        else:
            comps = sorted(database.elements)
        phases = sorted(filter_phases(database, unpack_components(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        # okay if parameters are initialized to zero, we only need the symbol names
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))

        self.mob_data = get_mob_data(database, comps, phases, datasets, parameters, weight)
        
    def get_likelihood(self, parameters) -> float:
        likelihood = calculate_mob_probability(self.mob_data, parameters)
        return likelihood


residual_function_registry.register(NonEquilibriumMobilityResidual)

