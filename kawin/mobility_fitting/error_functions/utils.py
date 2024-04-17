from pycalphad import Model, Database, calculate, equilibrium, variables as v
from pycalphad.codegen.callables import build_callables, build_phase_records
from pycalphad.core.utils import extract_parameters
from kawin.thermo.Mobility import MobilityModel

def get_output_base_name(output_name):
    base_names = get_base_names()
    for bn in base_names:
        if bn in output_name:
            return bn
        
def get_base_names():
    return ['INTER_DIFF', 'TRACER_DIFF', 'TRACER_D0', 'TRACER_Q']

def get_base_std():
    return {
        'INTER_DIFF': 0.1,  # decade
        'TRACER_D0': 0.1,   # decade
        'TRACER_DIFF': 0.1, # decade
        'TRACER_Q': 10000,  # J/mol
        }

def build_model(db, elements, phase, parameters, diffusing_species):
    param_keys, _ = extract_parameters(parameters)
    model = {phase: Model(db, elements, phase, parameters=param_keys)}
    model[phase].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)

    phase_record = build_phase_records(db, elements, [phase], model[phase].state_variables, 
                                       model, build_gradients=True, build_hessians=True, 
                                       parameters=parameters)
    
    mob_model = {phase: MobilityModel(db, elements, phase, parameters=param_keys)}
    mob_model[phase].set_diffusing_species(db, diffusing_species)
    mob_callable = {phase: {}}
    diffusing_species = sorted(list(set(diffusing_species) - set(['VA'])))
    for c in diffusing_species:
        bcp = build_callables(db, elements, [phase], mob_model, parameter_symbols=parameters, output=f'MOB_{c}', build_gradients=False, build_hessians=False, additional_statevars=[v.T, v.P, v.N, v.GE])
        mob_callable[phase][c] = bcp[f'MOB_{c}']['callables'][phase]

    return model, phase_record, mob_model, mob_callable