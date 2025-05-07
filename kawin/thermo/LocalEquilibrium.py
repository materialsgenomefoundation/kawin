from pycalphad.core.solver import Solver
from pycalphad.core.composition_set import CompositionSet
from pycalphad import calculate, variables as v
import numpy as np

def local_equilibrium(dbf, comps, phases, conds, models, phase_records, composition_sets=None, pDens=10):
    '''
    Local equilibrium calculation

    Chemical potential in a miscibility gap will be constant
    This method allows the user to get the free energy at the specified composition
    ignoring possible miscibility gaps

    Parameters
    ----------
    dbf: Database
    comps: list[str]
        List of elements to consider
    phases: list[str]
        List of phases to consider
    conds: dict[v.StateVariable, float]
        Dictionary of conditions (v.N needs to be included)
    models: dict[str, Model]
    phase_records : PhaseRecordFactory
    composition_sets: list[CompositionSet] (optional)
        If None, then composition sets will be determined through sampling
    pDens: int (optional)
        Sampling density when composition sets are not supplied

    Returns
    -------
    Dataset containing free energy and chemical potential
    '''
    # Broadcasting conditions not supported
    cur_conds = {str(k): float(v) for k, v in conds.items()}
    cur_conds = {k: v for k, v in conds.items()}

    # State variables are in alphabetical order
    if v.GE in cur_conds:
        state_variables = np.array([cur_conds[v.GE], cur_conds[v.N], cur_conds[v.P], cur_conds[v.T]], dtype=np.float64)
    else:
        state_variables = np.array([0, cur_conds[v.N], cur_conds[v.P], cur_conds[v.T]], dtype=np.float64)
    if composition_sets is None:
        # Note: filter_phases() not called, so all specified phases must be valid
        composition_sets = []

        # Special case for single phase where we can set a local phase condition for better sampling
        if len(phases) == 1:
            local_phase_conds = {v.X(phases[0], var.species): conds[var] for var in conds if isinstance(var, v.X)}
            calc_p = calculate(dbf, comps, phases[0], T=cur_conds[v.T], P=cur_conds[v.P], N=cur_conds[v.N], GE=cur_conds[v.GE],
                               pdens=pDens, model=models, phase_records=phase_records, conditions=local_phase_conds)
            idx_p = np.argmin(calc_p.GM.values.squeeze())
            compset = CompositionSet(phase_records[phases[0]])
            site_fractions = np.array(calc_p.Y.isel(points=idx_p).values.squeeze())
            compset.update(site_fractions, 1.0, state_variables)
            composition_sets.append(compset)
        else:
            # Choose a naive starting point for each phase
            # only one composition set per phase is chosen
            # here, we just choose the point with the minimum Gibbs energy
            # mass balance does not have to be preserved at the starting point
            for phase in phases:
                # arbitrary guess
                phase_amt = 1./len(phases)
                calc_p = calculate(dbf, comps, phase, T=cur_conds[v.T], P=cur_conds[v.P], N=cur_conds[v.N], GE=cur_conds[v.GE],
                                pdens=pDens, model=models, phase_records=phase_records)
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
