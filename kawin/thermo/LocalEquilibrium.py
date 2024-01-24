from pycalphad.core.solver import Solver
from pycalphad.core.composition_set import CompositionSet
from pycalphad.codegen.callables import build_phase_records
from pycalphad import calculate, variables as v
import numpy as np

def local_equilibrium(dbf, comps, phases, conds, models, phase_records, composition_sets=None):
    '''
    Local equilibrium calculation

    Chemical potential in a miscibility gap will be constant
    This method allows the user to get the free energy at the specified composition
    ignoring possible miscibility gaps

    Parameters
    ----------
    dbf : Database
    comps : list
        List of elements to consider
    phases : list
        List of phases to consider
    conds : dict
        Dictionary of conditions (v.N needs to be included)

    Returns
    -------
    Dataset containing free energy and chemical potential
    '''
    # Broadcasting conditions not supported
    cur_conds = {str(k): float(v) for k, v in conds.items()}
    if 'GE' in cur_conds:
        state_variables = np.array([cur_conds['GE'], cur_conds['N'], cur_conds['P'], cur_conds['T']], dtype=np.float64)
    else:
        state_variables = np.array([0, cur_conds['N'], cur_conds['P'], cur_conds['T']], dtype=np.float64)
    if composition_sets is None:
        # Note: filter_phases() not called, so all specified phases must be valid
        composition_sets = []

        # Choose a naive starting point for each phase
        # only one composition set per phase is chosen
        # here, we just choose the point with the minimum Gibbs energy
        # mass balance does not have to be preserved at the starting point
        for phase in phases:
            # arbitrary guess
            phase_amt = 1./len(phases)
            calc_p = calculate(dbf, comps, phase, T=cur_conds['T'], P=cur_conds['P'], N=cur_conds['N'], GE=cur_conds['GE'],
                               pdens=10, model=models, phase_records=phase_records)
            idx_p = np.argmin(calc_p.GM.values.squeeze())
            compset = CompositionSet(phase_records[phase])
            site_fractions = np.array(calc_p.Y.isel(points=idx_p).values.squeeze())
            compset.update(site_fractions, phase_amt, state_variables)
            composition_sets.append(compset)
    else:
        #Update state variables in composition sets if supplied
        #pycalphad doesn't seem to update the temperature if it changes
        for cs in composition_sets:
            cs.dof[:state_variables.shape[0]] = state_variables


    # Calculate a local equilibrium for the specified phases
    solver = Solver()
    #print('initial', composition_sets)
    result = solver.solve(composition_sets, cur_conds)
    return result, composition_sets
