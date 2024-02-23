from collections import OrderedDict
from typing import Sequence, Dict, Union, List, Optional

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.utils import filter_phases, unpack_components, unpack_condition
from scipy.stats import norm
import tinydb

from espei.phase_models import PhaseModelSpecification
from espei.typing import SymbolName
from espei.utils import PickleableTinyDB, database_symbols_to_fit
from espei.error_functions.residual_base import ResidualFunction, residual_function_registry

from kawin.thermo import GeneralThermodynamics
from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.thermo.FreeEnergyHessian import partialdMudX
from itertools import product

import kawin.mobility_fitting.cached_mobility as cmob

def getCachedData(cache, conds, data):
    strConds = {c.__str__(): conds[c] for c in conds}
    hashStr = str(strConds)
    if hashStr in cache:
        return cache[hashStr]
    else:
        result, cs = local_equilibrium(data.thermo.db, data.thermo.elements, data.thermo.phases, conds, data.thermo.models, data.thermo.phase_records)
        cs_el = list(cs[0].phase_record.nonvacant_elements)
        cs_T = cs[0].dof[cs[0].phase_record.state_variables.index(v.T)]
        cs_dof = np.array(cs[0].dof)
        cs_X = np.array(cs[0].X)
        cs_dmudx = partialdMudX(result.chemical_potentials, cs[0])
        cache[hashStr] = (cs_el, cs_T, cs_dof, cs_X, cs_dmudx)
        return cache[hashStr]

class InterdiffusivityData:
    def __init__(self, dbf, data, parameters = None, data_weight_dict = None):
        self.dbf = dbf
        self.parameters = parameters if parameters is not None else {}
        self.parameter_keys = [p for p in self.parameters]
        self.data_weight_dict = data_weight_dict if data_weight_dict is not None else {}
        self.output = data['output']
        self.property_std_deviation = {
            'D': 0.1  # decade
        }

        pot_conds = OrderedDict([(getattr(v, key), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if not key.startswith('X_')])
        comp_conds = OrderedDict([(v.X(key[2:]), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if key.startswith('X_')])
        rav_comp_conds = [OrderedDict(zip(comp_conds.keys(), pt_comps)) for pt_comps in zip(*comp_conds.values())]

        comps = data['components']
        phases = data['phases']
        self.refComp = data.get('ref_el')
        self.depComps = data.get('dependent_el', [])
        self.thermo = GeneralThermodynamics(dbf, comps, phases, parameters=parameters)
        self.conditions = {}
        for c in data['conditions']:
            condList = unpack_condition(data['conditions'][c])
            if 'X_' not in c:
                self.conditions[getattr(v,c)] = condList
        self.conditions['composition'] = rav_comp_conds
        
        self.values = np.array(data['values'])

        total_num_calculations = len(rav_comp_conds)*np.prod([len(vals) for vals in pot_conds.values()])
        dataset_weights = np.array(data.get('weight', 1.0)) * np.ones(total_num_calculations)
        self.weights = (self.property_std_deviation.get('D', 1.0)/data_weight_dict.get('D', 1.0)/dataset_weights).flatten()

def get_diff_data(dbf: Database, comps: Sequence[str], phases: Sequence[str], datasets: PickleableTinyDB, parameters: Dict[str, float], data_weight_dict: Optional[Dict[str, float]] = None):
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
    desired_data = datasets.search((tinydb.where('output').test(lambda x: 'INTER_DIFF' in x)) &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))
    
    diff_data = []
    for data in desired_data:
        diff_data.append(InterdiffusivityData(dbf, data, parameters, data_weight_dict))   
    return diff_data 

def calc_diff_differences(data : InterdiffusivityData, parameters : np.ndarray, thermoCache : Dict):
    diffs, wts = [], []
    paramDict = {}
    for i in range(len(data.parameter_keys)):
        paramDict[data.parameter_keys[i]] = parameters[i]
    data.thermo.updateParameters(paramDict)

    nonVaElements = sorted([e for e in data.thermo.elements if e != 'VA'])

    keys, values = zip(*data.conditions.items())
    allConds = [dict(zip(keys,p)) for p in product(*values)]
    for i in range(len(allConds)):
        inds = np.array([data.conditions[k].index(v) for k,v in allConds[i].items()], dtype=np.int32)
        dexp = data.values[tuple(inds)]

        currConds = {v.N: 1, v.GE: 0, v.P: allConds[i][v.P], v.T: allConds[i][v.T]}
        for c in allConds[i]['composition']:
            currConds[c] = allConds[i]['composition'][c]

        #Grab data from global cache
        cs_data = getCachedData(thermoCache, currConds, data)
        cs_el, cs_T, cs_dof, cs_X, cs_dmudx = cs_data
        mob_from_CS = cmob.mobility_from_composition_set_quick(cs_dof, cs_el, data.thermo.mobCallables[data.thermo.phases[0]], data.thermo.mobility_correction, parameters = data.thermo._parameters)
        
        #eq_result = local_equilibrium(data.thermo.db, data.thermo.elements, data.thermo.phases, currConds, data.thermo.models, data.thermo.phase_records)
        #result, cs = eq_result

        if data.refComp is None:
            refComp = data.output.split('_')[-1]
            refIndex = nonVaElements.index(refComp)
        else:
            refIndex = nonVaElements.index(data.refComp[0])
        
        if 'INTER_DIFF' in data.output:
            #cd, _ = chemical_diffusivity(result.chemical_potentials, cs[0], data.thermo.mobCallables[data.thermo.phases[0]], data.thermo.mobility_correction, parameters = data.thermo._parameters)

            mobMatrix = cmob.mobility_matrix_quick(cs_dof, cs_X, cs_el, mob_from_CS, data.thermo.phase_records[data.thermo.phases[0]])
            cd, _ = cmob.chemical_diffusivity_quick(cs_dmudx, mobMatrix)

            depComp1 = nonVaElements.index(data.depComps[0])
            depComp2 = nonVaElements.index(data.depComps[1])
            D = cd[depComp1, depComp2] - cd[depComp1, refIndex]
            diffs.append((np.sign(D)*np.log10(np.abs(D)) - np.sign(dexp)*np.log10(np.abs(dexp))))
            wts.append(data.weights[i])

    return diffs, wts

def calculate_diff_probability(diff_data : Sequence[InterdiffusivityData], parameters : np.ndarray, thermoCache : Dict) -> float:
    differences = []
    weights = []
    for data in diff_data:
        diffs, wts = calc_diff_differences(data, parameters, thermoCache)
        if np.any(np.isinf(diffs) | np.isnan(diffs)):
            return -np.inf
        differences.append(diffs)
        weights.append(wts)
    
    differences = np.concatenate(differences, axis=0)
    weights = np.concatenate(weights, axis=0)
    probs = norm(loc=0.0, scale=weights).logpdf(differences)
    return np.sum(probs)

class InterdiffusivityResidual(ResidualFunction):
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
            self.weight = weight.get("ZPF", 1.0)
        else:
            self.weight = 1.0
        if phase_models is not None:
            comps = sorted(phase_models.components)
            model_dict = phase_models.get_model_dict()
        else:
            comps = sorted(database.elements)
            model_dict = dict()
        phases = sorted(filter_phases(database, unpack_components(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        # okay if parameters are initialized to zero, we only need the symbol names
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
        self.diff_data = get_diff_data(database, comps, phases, datasets, parameters, model_dict)

        self.thermoCache = {}

    def get_likelihood(self, parameters):
        likelihood = calculate_diff_probability(self.diff_data, parameters, self.thermoCache)
        return likelihood

residual_function_registry.register(InterdiffusivityResidual)
