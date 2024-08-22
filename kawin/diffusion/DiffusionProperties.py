import copy

import numpy as np

from kawin.thermo import GeneralThermodynamics
from kawin.thermo.Mobility import mobility_from_composition_set, x_to_u_frac, interstitials

def compute_mobility(therm : GeneralThermodynamics, x, T, hash_table = None, hashing_function = None):
    '''
    Gets mobility of all phases at x and T

    If mobility does not exist for a phase or a phase is unstable, then the mobility is set to -1

    Parameters
    ----------
    therm : GeneralThermodynamics object
    x : float, list[float], list[list[float]]
        Compositions
        For binary, it must be either a float or list[float] (shape N) for multiple compositions
        For ternary, it must be either list[float] (shape e) or list[list[float]] (shape N x e) for multiple compositions
    T : float, list[float]
        Temperature(s)
        First dimensions must correspond to first dimension of x
    hash_table: dict (optional)
        Dictionary to cache mobility outputs to
    hashing_function : function (optional)
        Function that takes in (x,T) and returns a hash for the hash table
        Must be supplied if the hash_table is supplied
        
    Returns
    -------
    mobility array - (N, p, e)
    phase fraction array - (N, p)
    chemical potential array - (N, e)

    Where N is size of (x,T), p is number of phases and e is number of elements
    '''
    x = np.atleast_2d(np.squeeze(x))
    if therm._isBinary:
        x = x.T
    T = np.atleast_1d(T)

    sortIndices = np.argsort(therm.elements[:-1])
    unsortIndices = np.argsort(sortIndices)

    mob = -1 * np.ones((x.shape[0], len(therm.phases), len(therm.elements)-1))
    phase_fracs = np.zeros((x.shape[0], len(therm.phases)))
    chemical_potentials = np.zeros((x.shape[0], len(therm.elements)-1))
    for i in range(len(x)):
        cachedTerms = None
        if hash_table is not None:
            hashValue = hashing_function(x[i], T[i])
            cachedTerms = hash_table.get(hashValue, None)

        if cachedTerms is not None:
            mobVals, phaseVals, chemPotVals = cachedTerms
            mob[i,:,:] = mobVals
            phase_fracs[i,:] = phaseVals
            chemical_potentials[i,:] = chemPotVals
        else:
            try:
                wks = therm.getEq(x[i], T[i], 0, therm.phases)
                chemical_potentials[i,:] = np.squeeze(wks.eq.MU)[unsortIndices]
                comp_sets = wks.get_composition_sets()
            except Exception as e:
                print(f'Error at {x[i]}, {T[i]}')
                raise e

            for p in range(len(therm.phases)):
                cs = [cs for cs in comp_sets if cs.phase_record.phase_name == therm.phases[p]]
                if len(cs) == 0:
                    continue
                cs = cs[0]
                phase_fracs[i,p] = cs.NP
                
                if therm.mobCallables.get(therm.phases[p], None) is not None:
                    mob[i,p,:] = mobility_from_composition_set(cs, therm.mobCallables[therm.phases[p]], therm.mobility_correction)[unsortIndices]
                    mob[i,p,:] *= x_to_u_frac(np.array(cs.X, dtype=np.float64)[unsortIndices], therm.elements[:-1], interstitials)

            if hash_table is not None:
                hashValue = hashing_function(x[i], T[i])
                hash_table[hashValue] = (copy.copy(mob[i,:,:]), copy.copy(phase_fracs[i,:]), copy.copy(chemical_potentials[i,:]))

    return mob, phase_fracs, chemical_potentials

def wiener_upper(mob_array, phase_fracs, *args, **kwargs):
    '''
    Upper wiener bounds for average mobility

    Returns
    -------
    (N, e) mobility array - N is number of nodes, e is number of elements
    '''
    modified_mob = np.where(mob_array != -1, mob_array, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(phase_fracs[:,:,np.newaxis], modified_mob), axis=1)
    return avg_mob

def wiener_lower(mob_array, phase_fracs, *args, **kwargs):
    '''
    Lower wiener bounds for average mobility

    Returns
    -------
    (N, e) mobility array - N is number of nodes, e is number of elements
    '''
    modified_mob = np.where(mob_array != -1, mob_array, np.finfo(np.float64).max)
    avg_mob = 1/np.sum(np.multiply(phase_fracs[:,:,np.newaxis], 1/(modified_mob)), axis=1)
    return avg_mob

def labyrinth(mob_array, phase_fracs, *args, **kwargs):
    '''
    Labyrinth mobility

    Returns
    -------
    (N, e) mobility array - N is number of nodes, e is number of elements
    '''
    labyrinth_factor = kwargs.get('labyrinth_factor', 1)
    modified_mob = np.where(mob_array != -1, mob_array, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(np.power(phase_fracs[:,:,np.newaxis], labyrinth_factor), modified_mob), axis=1)
    return avg_mob

def _hashin_shtrikman_general(mob_array, phase_fracs, extreme_mob, extreme_arg):
    '''
    General hashin shtrikman bounds

    Returns
    -------
    (N, e) mobility array - N is number of nodes, e is number of elements
    '''
    Ak = 3*extreme_mob*(mob_array - extreme_mob) / (2*extreme_mob + mob_array)
    Ak *= phase_fracs[:,:,np.newaxis]
    Ak[:,extreme_arg,:] = 0
    Ak = np.sum(Ak, axis=1)
    avg_mob = extreme_mob[:,0,:] + Ak / (1 - Ak / (3*extreme_mob[:,0,:]))
    return avg_mob

def hashin_shtrikman_upper(mob_array, phase_fracs, *args, **kwargs):
    '''
    Upper hashin shtrikman bounds for average mobility

    Returns
    -------
    (N, e) mobility array - N is number of nodes, e is number of elements
    '''
    modified_mob = np.where(mob_array != -1, mob_array, np.finfo(np.float64).tiny)
    max_mob = np.amax(modified_mob, axis=1)[:,np.newaxis,:]
    max_arg = np.argmax(modified_mob, axis=1)
    return _hashin_shtrikman_general(modified_mob, phase_fracs, max_mob, max_arg)

def hashin_shtrikman_lower(mob_array, phase_fracs, *args, **kwargs):
    '''
    Lower hashin shtrikman bounds for average mobility

    Returns
    -------
    (N, e) mobility array - N is number of nodes, e is number of elements
    '''
    modified_mob = np.where(mob_array != -1, mob_array, np.finfo(np.float64).max)
    min_mob = np.amin(modified_mob, axis=1)[:,np.newaxis,:]
    min_arg = np.argmin(modified_mob, axis=1)
    return _hashin_shtrikman_general(modified_mob, phase_fracs, min_mob, min_arg)
