import json

from tinydb import where

import pycalphad
from pycalphad import Database
from espei.datasets import load_datasets, recursive_glob
from espei.paramselect import generate_parameters
from espei.parameter_selection.fitting_descriptions import ModelFittingDescription

from kawin.mobility_fitting.fitting_steps import StepQ, StepD0, StepTracerDiffusivity
from kawin.thermo.Mobility import MobilityModel

def add_mobility_keywords(elements):
    '''
    Adds MQ_el keywords to the pycalphad tdb keyword list
    
    Custom fitting in espei doesn't support diffusing species yet so as a workaround,
    we could fit to terms of MQ_el where el is the diffusing species, then rewrite
    the database to replace MQ_el terms with MQ(Phase&el,...)

    In order to do this, pycalphad will need to be able to recognize that MQ_el is a valid
    keyword in the tdb file

    Parameters
    ----------
    elements : [str]
        List of diffusing species to fit mobility models to

    Returns
    -------
    aliases : {str: str}
        Maps tdb keyword (MQ_el) to element (el)
    '''
    aliases = []
    for e in elements:
        aliases.append('MQ_{}'.format(e))
        pycalphad.io.tdb_keywords.TDB_PARAM_TYPES.append(aliases[-1])
    return aliases

def rewrite_mobility_terms(dbf, mob_aliases):
    '''
    Rewrites the database to replace MQ_el with MQ(Phase&el,...)

    Parameters
    ----------
    db_name : str
        Name of thermodynamic database
    mob_aliases : {str: str}
        Maps tdb keyword (MQ_el) to element (el)
    '''
    for p in dbf._parameters:
        if p['parameter_type'] in mob_aliases:
            ptype = p['parameter_type']
            index = ptype.index('_')
            diff_species = ptype[index+1:]
            cons_array = tuple(tuple(s.name for s in xs) for xs in p['constituent_array'])
            dbf.add_parameter(
                'MQ', p['phase_name'], cons_array, 
                p['parameter_order'], p['parameter'], 
                ref=p['reference'], diffusing_species = diff_species)

    for m in mob_aliases:
        dbf._parameters.remove(where('parameter_type') == m)

def setup_mobility_fitting_description(diffusing_species, fit_to_tracer = False):
    '''
    Defines custom classes for diffusion prefactor, activation energy and tracer diffusivity
    and add to the global mobility_fitting_description

    Parameters
    ----------
    diffusing_species : [str]
        List of elements to fit mobility models to
    fit_to_tracer : bool
        If False, will fit to datasets of activation energy and pre-factor terms
        If True, will fit to tracer diffusivity directly
            Not recommended due to overfitting. It is better to convert the
            tracer diffusivity to activation energy and pre-factor terms and fit
            to those values instead
    '''
    steps = []
    for e in diffusing_species:
        if fit_to_tracer:
            Dstar = type(f'Tracer_{e}', (StepTracerDiffusivity,), {'parameter_name': f'MQ_{e}', 'data_types_read': f'TRACER_DIFF_{e}'})
            steps.append(Dstar)
        else:
            D0 = type(f'D0_{e}', (StepD0,), {'parameter_name': f'MQ_{e}', 'data_types_read': f'TRACER_D0_{e}'})
            Q = type(f'Q_{e}', (StepQ,), {'parameter_name': f'MQ_{e}', 'data_types_read': f'TRACER_Q_{e}'})
            steps += [D0, Q]
    return ModelFittingDescription(steps, model=MobilityModel)

def generate_mobility(phase_models, datasets, diffusing_species = None, ridge_alpha=None, aicc_penalty_factor=None, dbf=None, fit_to_tracer=False):
    '''
    Wrapper around run_espei to do the following:
        1. Find components to fit mobility to
        2. Add 'MQ_el' keywords into pycalphad tdb_keyword list
        3. Set up mobility fitting description
        4. Run espei
        5. Rewrite generated database to replace 'MQ_el' with 'MQ(ph&el)'

    Parameters
    ----------
    input_settings : dict
        Input dictionary used for an espei assessment
    '''
    # If phase models is a file, then load it in 
    if isinstance(phase_models, str):
        with open(phase_models, 'r') as f:
            phase_models = json.load(f)

    # If datasets is a folder, then load it in
    if isinstance(datasets, str):
        datasets = load_datasets(sorted(recursive_glob(datasets, '*.json')))

    # Get non-VA components
    components = sorted(list(set(phase_models['components']) - set(['VA'])))

    # If no diffusing species, then assume same as components
    if diffusing_species is None:
        diffusing_species = components

    # Get mobility fitting description
    mobility_fitting_description = setup_mobility_fitting_description(diffusing_species, fit_to_tracer=fit_to_tracer)

    # Generate dbf
    mob_dbf = generate_parameters(phase_models, datasets, 'SGTE91', 'linear', 
                                  ridge_alpha=ridge_alpha, aicc_penalty_factor=aicc_penalty_factor,
                                  dbf=dbf, fitting_description=mobility_fitting_description)
    
    aliases = add_mobility_keywords(diffusing_species)
    rewrite_mobility_terms(mob_dbf, aliases)
    return mob_dbf

    