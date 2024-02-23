import numpy as np
from pycalphad.core.utils import extract_parameters
from pycalphad import variables as v

interstitials = ['C', 'N', 'O', 'H', 'B']

# ------------------------------------------------------------------
# For ESPEI assessments
# Quick compute functions using serializable composition set data
# The CompositionSet object itself doesn't appear serializable
#   So this will take in data such as dof or elements, which are serializable
# If assessing only mobility parameters, then the thermodynamic factor will be constant for each data point
#   Thus, we can cache all the thermodynamic factors and use them to compute mobility without having to compute equilibrium each time
# ------------------------------------------------------------------
def mobility_from_composition_set_quick(dof, elements, mobility_callables = None, mobility_correction = None, parameters = {}):
    if mobility_callables is None:
        raise ValueError('mobility_callables is required')

    #Set mobility correction if not set
    if mobility_correction is None:
        mobility_correction = {A: 1 for A in elements}
    else:
        for A in elements:
            if A not in mobility_correction:
                mobility_correction[A] = 1

    #return np.array([mobility_correction[elements[A]] * mobility_callables[elements[A]](composition_set.dof) for A in range(len(elements))])
    param_keys, param_values = extract_parameters(parameters)
    if len(param_values) > 0:
        callableInput = np.concatenate((dof, param_values[0]), dtype=np.float_)
    else:
        callableInput = dof
    return np.array([mobility_correction[elements[A]] * mobility_callables[elements[A]](callableInput) for A in range(len(elements))])

def compute_symbolic_quick(dof, elements, mobility_model, symbol, parameters = {}):
    param_keys, param_values = extract_parameters(parameters)
    
    symbols = sorted([v.T, v.P, v.N, v.GE]) + mobility_model.site_fractions
    var_dict = {s:val for s,val in zip(symbols, np.array(dof))}
    if len(param_keys) > 0:
        for p,val in zip(param_keys, param_values[0]):
            var_dict[p] = val

    return np.array([getattr(mobility_model, f'{symbol}_{elements[A]}').subs(var_dict).n(53, real=True) for A in range(len(elements))])

def mobility_from_composition_set_symbolic_quick(dof, elements, mobility_model, parameters = {}):
    return compute_symbolic_quick(dof, elements, mobility_model, 'MQ', parameters)

def activation_energy_from_composition_set_quick(dof, elements, mobility_model, parameters = {}):
    return compute_symbolic_quick(dof, elements, mobility_model, 'MQa', parameters)

def prefactor_from_composition_set_quick(dof, elements, mobility_model, parameters = {}):
    return compute_symbolic_quick(dof, elements, mobility_model, 'lnM0', parameters)
    
def tracer_diffusivity_quick(T, mob_from_CS):
    R = 8.314
    return R * T * mob_from_CS

def mobility_matrix_quick(dof, composition, elements, mob_from_CS, phase_record):
    X = composition
    Usum = np.sum([X[A] for A in range(len(elements)) if elements[A] not in interstitials])
    U = X / Usum

    mob = np.array([U[A] * mob_from_CS[A] for A in range(len(elements))])

    #Find vacancy site fractions for multiplying with interstitials when making the mobility matrix
    #If vacancies are not found on the same sublattice, we'll defualt to 1 so there's at least some mobility and not 0
    #       A mobility of 0 would be quite unrealistic
    #       In addition, as we're working with interstitals, the vacancies are going to be close to 1, so this assumption wouldn't hurt
    vaTerms = {}            #Maps sublattice index to site fraction index for vacancies
    interstitialTerms = {}  #Maps interstitial to sublattice index
    index = len(phase_record.state_variables)
    for i in range(len(phase_record.variables)):
        if phase_record.variables[i].species.name == 'VA':
            vaTerms[phase_record.variables[i].sublattice_index] = dof[index+i]
        if phase_record.variables[i].species.name in interstitials:
            interstitialTerms[phase_record.variables[i].species.name] = phase_record.variables[i].sublattice_index

    mobMatrix = np.zeros((len(elements), len(elements)))
    for a in range(len(elements)):
        if elements[a] in interstitials:
            mobMatrix[a, a] = vaTerms.get(interstitialTerms[elements[a]], 1) * mob[a]
        else:
            for b in range(len(elements)):
                if elements[b] not in interstitials:
                    if a == b:
                        mobMatrix[a, b] = (1 - U[a]) * mob[b]
                    else:
                        mobMatrix[a, b] = -U[a] * mob[b]
    mobMatrix *= Usum

    #for a in range(len(elements)):
    #    for b in range(len(elements)):
    #        if a == b:
    #            mobMatrix[a, b] = (1 - U[a]) * mob[b]
    #        else:
    #           mobMatrix[a, b] = -U[a] * mob[b]

    return mobMatrix

def chemical_diffusivity_quick(dmudx, mobMatrix, returnHessian = False):
    Dkj = np.matmul(mobMatrix, dmudx)
    
    if returnHessian:
        return Dkj, dmudx
    else:
        return Dkj, None

def interdiffusivity_quick(dmudx, mobMatrix, elements, refElement, mobility_callables = None, mobility_correction = None, returnHessian = False, parameters = {}):
    Dkj, hessian = chemical_diffusivity_quick(dmudx, mobMatrix, returnHessian)

    refIndex = 0
    for a in range(len(elements)):
        if elements[a] == refElement:
            refIndex = a
            break

    Dnkj = np.zeros((len(elements) - 1, len(elements) - 1))
    c = 0
    d = 0
    for a in range(len(elements)):
        if a != refIndex:
            for b in range(len(elements)):
                if b != refIndex:
                    if elements[b] in interstitials:
                        Dnkj[c, d] = Dkj[a, b]
                    else:
                        Dnkj[c, d] = Dkj[a, b] - Dkj[a, refIndex]
                    d += 1
            c += 1
            d = 0

    return Dnkj, hessian