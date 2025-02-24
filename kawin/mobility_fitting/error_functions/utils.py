from pycalphad import Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
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

    #Include v.GE to be compatible with kawin
    #TODO: this may not be necessary since v.GE is only added if the model
    #      is built in the GeneralThermodynamics module
    model[phase].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)
    prf = PhaseRecordFactory(db, elements, model[phase].state_variables, model, parameters=parameters)
    
    mob_model = {phase: MobilityModel(db, elements, phase, parameters=param_keys)}
    mob_model[phase].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)
    mob_model[phase].set_diffusing_species(db, diffusing_species)
    mob_prf = PhaseRecordFactory(db, elements, mob_model[phase].state_variables, mob_model, parameters=parameters)

    return model, prf, mob_model, mob_prf