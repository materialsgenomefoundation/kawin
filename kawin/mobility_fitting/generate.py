import json
import copy

from tinydb import where

import pycalphad
from pycalphad import Database
from espei.espei_script import run_espei
from espei.parameter_selection.fitting_descriptions import ModelFittingDescription

from kawin.mobility_fitting.fitting_steps import StepQ, StepD0, StepTracerDiffusivity
from kawin.thermo.Mobility import MobilityModel

fitting_description = None

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

def rewrite_mobility_terms(db_name, mob_aliases, tdb_write_kwargs = {}):
    '''
    Rewrites the database to replace MQ_el with MQ(Phase&el,...)

    Parameters
    ----------
    db_name : str
        Name of thermodynamic database
    mob_aliases : {str: str}
        Maps tdb keyword (MQ_el) to element (el)
    '''
    db = Database(db_name)
    for p in db._parameters:
        if p['parameter_type'] in mob_aliases:
            ptype = p['parameter_type']
            index = ptype.index('_')
            diff_species = ptype[index+1:]
            cons_array = tuple(tuple(s.name for s in xs) for xs in p['constituent_array'])
            db.add_parameter(
                'MQ', p['phase_name'], cons_array, 
                p['parameter_order'], p['parameter'], 
                ref=p['reference'], diffusing_species = diff_species)

    for m in mob_aliases:
        db._parameters.remove(where('parameter_type') == m)

    tdb_write_kwargs['if_exists'] = tdb_write_kwargs.get('if_exists', 'overwrite')
    db.to_file(db_name, **tdb_write_kwargs)

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
        
    global fitting_description
    fitting_description = ModelFittingDescription(steps, model=MobilityModel)

def fit_mobility(input_settings, diffusing_species = None, fit_to_tracer=False, tdb_write_kwargs = {}):
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
    #Get phase models to get components in system to fit to
    phase_model_file = input_settings['system']['phase_models']
    with open(phase_model_file, 'r') as f:
        phase_models = json.load(f)
    
    #Get non-VA components
    components = list(set(phase_models['components']) - set(['VA']))

    # If no diffusing species, then assume same as components
    if diffusing_species is None:
        diffusing_species = components

    aliases = add_mobility_keywords(diffusing_species)
    setup_mobility_fitting_description(diffusing_species, fit_to_tracer=fit_to_tracer)

    out_db = input_settings['output']['output_db']
    run_espei(input_settings)
    rewrite_mobility_terms(out_db, aliases, tdb_write_kwargs=tdb_write_kwargs)