"""
Calculate error due to interdiffusivity, tracer diffusivity, activation energy and diffusion prefactor
based off equilibrium
"""
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Union, Sequence

from itertools import product
import numpy as np
from scipy.stats import norm
import tinydb

from pycalphad import Database, variables as v
from pycalphad.core.utils import filter_phases, unpack_components, unpack_condition, extract_parameters

from espei.phase_models import PhaseModelSpecification
from espei.typing import SymbolName
from espei.utils import PickleableTinyDB, database_symbols_to_fit
from espei.error_functions.residual_base import ResidualFunction, residual_function_registry

from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.thermo.FreeEnergyHessian import partialdMudX
from kawin.thermo.Mobility import interstitials

import kawin.mobility_fitting.error_functions.cached_mobility as cmob
from kawin.mobility_fitting.error_functions.utils import get_output_base_name, get_base_names, build_model, get_base_std

_log = logging.getLogger(__name__)

class EquilibriumMobilityData:
    """
    Stores internal values for each dataset needed to compute output
    """
    def __init__(self, dbf, data, parameters = None, data_weight_dict = None):
        self.parameters = parameters if parameters is not None else {}
        self.parameter_keys = [p for p in self.parameters]
        self.data_weight_dict = data_weight_dict if data_weight_dict is not None else {}
        self.output = get_output_base_name(data['output'])
        self.reference = data['reference']

        pot_conds = OrderedDict([(getattr(v, key), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if not key.startswith('X_')])
        comp_conds = OrderedDict([(v.X(key[2:]), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if key.startswith('X_')])
        rav_comp_conds = [OrderedDict(zip(comp_conds.keys(), pt_comps)) for pt_comps in zip(*comp_conds.values())]

        self.comps = data['components']
        self.elements = sorted(list(set([v.Species(c).name for c in self.comps])))
        self.non_va_elements = sorted(list(set(self.elements) - set(['VA'])))
        self.phases = data['phases']
        self.refComp = data.get('ref_el', None)
        if self.refComp is None and 'TRACER' in data['output']:
            self.refComp = [data['output'][len(self.output)+1:]]
        self.depComps = data.get('dependent_el', [])

        #Set diffusing species - this is necessary for tracer data of a component at the limit of X->0
        #  in which case, the dataset will not include the component when building the model
        self.diffusing_species = sorted(list(set(self.refComp) | set(self.depComps) | set(self.elements) - set(['VA'])))
        self.models, self.phase_records, self.mob_models, self.mob_callables = build_model(dbf, self.elements, self.phases[0], parameters, self.diffusing_species)
        self.mobility_correction = {c:1 for c in self.non_va_elements}

        self.conditions = {}
        for c in data['conditions']:
            condList = unpack_condition(data['conditions'][c])
            if 'X_' not in c:
                self.conditions[getattr(v,c)] = condList
        self.conditions['composition'] = rav_comp_conds
        
        self.values = np.array(data['values'])
        self._compute_cached_equilibrium(dbf)

        total_num_calculations = len(rav_comp_conds)*np.prod([len(vals) for vals in pot_conds.values()])
        dataset_weights = np.array(data.get('weight', 1.0)) * np.ones(total_num_calculations)
        property_std_deviation = get_base_std()
        self.weights = (property_std_deviation.get(self.output, 1.0)/data_weight_dict.get(self.output, 1.0)/dataset_weights).flatten()

    def _compute_cached_equilibrium(self, dbf):
        """
        Computes local equilibrium at each condition and stores results
        This is to avoid computing equilibrium every time the log probability is calculated

        NOTE: this is only valid if the thermodynamic parameters are fixed
        TODO: add option to disable this in case the user wants to fit both thermodynamic and mobility parameters (why?)

        Cached data includes: non-vacant elements, temperature, internal degrees of freedom, composition and chemical potential gradient
        """
        self.cache = {}

        keys, values = zip(*self.conditions.items())
        allConds = [dict(zip(keys,p)) for p in product(*values)]

        for i in range(len(allConds)):
            inds = np.array([self.conditions[k].index(v) for k,v in allConds[i].items()], dtype=np.int32)

            currConds = {v.N: 1, v.GE: 0, v.P: allConds[i][v.P], v.T: allConds[i][v.T]}
            for c in allConds[i]['composition']:
                currConds[c] = allConds[i]['composition'][c]

            #Grab data from global cache
            result, cs = local_equilibrium(dbf, self.elements, self.phases, currConds, self.models, self.phase_records)
            cs_el = list(cs[0].phase_record.nonvacant_elements)
            cs_T = cs[0].dof[cs[0].phase_record.state_variables.index(v.T)]
            cs_dof = np.array(cs[0].dof)
            cs_X = np.array(cs[0].X)
            cs_dmudx = partialdMudX(result.chemical_potentials, cs[0])
            self.cache[tuple(inds)] = (cs_el, cs_T, cs_dof, cs_X, cs_dmudx)

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
    desired_data = datasets.search((tinydb.where('output').test(lambda x: get_output_base_name(x) in base_names)) &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)) &
                                   (~tinydb.where('solver').exists()))
    
    mod_data = []
    for data in desired_data:
        mod_data.append(EquilibriumMobilityData(dbf, data, parameters, data_weight_dict))   
    return mod_data 

def calc_mob_differences(data : EquilibriumMobilityData, parameters : np.ndarray):
    diffs, wts = [], []
    paramDict = {data.parameter_keys[i] : parameters[i] for i in range(len(data.parameter_keys))}
        
    #Update phase record parameters
    param_keys, param_values = extract_parameters(paramDict)
    for p in data.phases:
        data.phase_records[p].parameters[:] = np.asarray(param_values, dtype=np.float_)

    keys, values = zip(*data.conditions.items())
    allConds = [dict(zip(keys,p)) for p in product(*values)]
    for i in range(len(allConds)):
        inds = np.array([data.conditions[k].index(v) for k,v in allConds[i].items()], dtype=np.int32)
        value = data.values[tuple(inds)]
        cs_data = data.cache[tuple(inds)]

        #Compute mobility from cached equilibrium data
        cs_el, cs_T, cs_dof, cs_X, cs_dmudx = cs_data
        mob_from_CS = cmob.mobility_from_composition_set_quick(cs_dof, cs_el, data.mob_callables[data.phases[0]], data.mobility_correction, parameters = paramDict)

        refIndex = data.non_va_elements.index(data.refComp[0])
        
        #NOTE: for interdiffusivity, tracer diffusivity and prefactor, we multiply by the sign of the value
        #      This is for cases if the computed and desired values are of different signs
        #      and taking the log of the absolute value will remove this possible difference
        #
        #      1) In reality, tracer diffusivity and prefactor should always be positive, so
        #         this may not be necessary
        #      2) If the desired interdiffusivity correctly accounts for potential miscibility gaps
        #         that exists in the database, then the sign of the calculated and desired interdiffusivity
        #         should be the same
        if data.output == 'INTER_DIFF':
            mobMatrix = cmob.mobility_matrix_quick(cs_dof, cs_X, cs_el, mob_from_CS, data.phase_records[data.phases[0]])
            cd, _ = cmob.chemical_diffusivity_quick(cs_dmudx, mobMatrix)

            depComp1 = data.non_va_elements.index(data.depComps[0])
            depComp2 = data.non_va_elements.index(data.depComps[1])
            if data.depComps[1] in interstitials:
                D = cd[depComp1, depComp2]
            else:
                D = cd[depComp1, depComp2] - cd[depComp1, refIndex]
            diffs.append((np.sign(D)*np.log10(np.abs(D)) - np.sign(value)*np.log10(np.abs(value))))

        elif data.output == 'TRACER_DIFF':
            tracer_diff = cmob.tracer_diffusivity_quick(cs_T, mob_from_CS)
            D = tracer_diff[refIndex]
            diffs.append((np.sign(D)*np.log10(np.abs(D)) - np.sign(value)*np.log10(np.abs(value))))

        elif data.output == 'TRACER_D0':
            D0 = cmob.prefactor_from_composition_set_quick(cs_dof, cs_el, data.mob_models[data.phases[0]], parameters = paramDict)
            D = np.exp(D0[refIndex])
            diffs.append((np.sign(D)*np.log10(np.abs(D)) - np.sign(value)*np.log10(np.abs(value))))

        elif data.output == 'TRACER_Q':
            Qs = cmob.activation_energy_from_composition_set_quick(cs_dof, cs_el, data.mob_models[data.phases[0]], parameters = paramDict)
            Q = Qs[refIndex]
            diffs.append(Q - value)

        wts.append(data.weights[i])

    _log.debug('Output: %s differences: %s, weights: %s, reference: %s', data.output, diffs, wts, data.reference)

    return diffs, wts

def calculate_mob_probability(mob_data : Sequence[EquilibriumMobilityData], parameters : np.ndarray) -> float:
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

class EquilibriumMobilityResidual(ResidualFunction):
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

    def get_likelihood(self, parameters):
        likelihood = calculate_mob_probability(self.mob_data, parameters)
        return likelihood

residual_function_registry.register(EquilibriumMobilityResidual)
