import numpy as np
from pycalphad import Database, variables as v
from symengine import Piecewise, And, Symbol, log
from kawin.mobility_fitting.utils import _vname, find_last_variable, MobilityTerm

class SpeciesData:
    def __init__(self, name, atomic_number, diameter, Tm, melting_phase = None):
        self.name = name
        self.atomic_number = atomic_number
        self.diameter = diameter
        self.Tm = Tm
        self.melting_phase = melting_phase
        self.K0 = self._getK0(melting_phase)

    def _getK0(self, melting_phase):
        if melting_phase is None:
            return 0
        
        phases = {'BCC': 1, 'HCP': 2, 'FCC': 3}
        for p in phases:
            if p in melting_phase:
                return phases[p]
        return 0
    
def _get_liquid_phase_name(database):
    for ph in database.phases:
        if ph.upper() == 'LIQUID':
            return ph
        elif ph.upper() == 'LIQ':
            return ph
        elif ph.upper() == 'L':
            return ph
    return 'LIQUID'

def _getK0(melting_phase):
    phases = {'BCC': 1, 'HCP': 2, 'FCC': 3}
    for p in phases:
        if p in melting_phase:
            return phases[p]
    return 0

def compute_liquid_mobility_liu(database, species):
    '''
    Species data will include {diameter, atomic_number, T_m, phase}

    Units:
        diameter: Angstroms
        T_m: K
    '''
    R = 8.314
    
    liq_name = _get_liquid_phase_name(database)
    mobility_models = {}
    for solute in species:
        models = []
        for solvent in species:
            d_solv, A_solv, Tm_solv, phase_solv = (species[solvent][t] for t in ['diameter', 'atomic_number', 'T_m', 'phase'])
            K0_solv = _getK0(phase_solv)
            d_solu = species[solute]['diameter']

            Q = -0.17 * R * Tm_solv * (16 + K0_solv)
            D0 = (8.95 - 0.0734 * A_solv) * 1e-8
            D0 *= d_solv / d_solu

            const_array = [[v.SiteFraction(liq_name, 0, solvent)]]
            mob_term = MobilityTerm(const_array, 0)
            mob_term.expr = Piecewise((Q + R*np.log(D0)*v.T, And(1.0 < v.T, v.T < 10000)), (0, True))
            models.append(mob_term)

        mobility_models[solute] = models

    return mobility_models

def compute_liquid_mobility_su(database, species):
    '''
    Species data will include {atomic_mass, density, T_m, CTE (optional)}
    If CTE is included, then density/molar volume will be assumed to be at room temperature
        And will be adjusted to the melting point

    Units:
        atomic_mass: g/mol
        density: g/cm3
        T_m: K
        CTE: m/m
    '''
    vIndex = find_last_variable(database)
    C1 = 1.8e-8
    C2 = 2.34
    R = 8.314
    k = 1.38e-23
    R0 = 0.644e-10

    liq_name = _get_liquid_phase_name(database)
    mobility_models = {}
    for solute in species:
        models = []
        for solvent in species:
            M_solv, rho_solv, Tm_solv = (species[solvent][t] for t in ['atomic_mass', 'density', 'T_m'])
            cte_solv = species[solvent].get('CTE', 0)

            M_solu, rho_solu, Tm_solu = (species[solute][t] for t in ['atomic_mass', 'density', 'T_m'])
            cte_solu = species[solute].get('CTE', 0)

            rho_solv *= (1 + cte_solv * (Tm_solv - 298.15))**3
            rho_solu *= (1 + cte_solu * (Tm_solv - 298.15))**3
            V_solv = M_solv / rho_solv * 1e-6       #m3/mol
            V_solu = M_solu / rho_solu * 1e-6       #m3/mol

            Q = -C2 * Tm_solv * R
            D1 = C1 * np.power(M_solv/1000, 1/2) / np.power(V_solv, 2/3)
            D2 = R0 * np.power(M_solu / rho_solu, 1/3)
            D3 = k / (4*np.pi)
            D0 = D3/(D1*D2) * v.T**(1/2) / (1 - 0.112 * (v.T / Tm_solv)**(1/2))

            const_array = [[v.SiteFraction(liq_name, 0, solvent)]]
            mob_term = MobilityTerm(const_array, 0)
            mob_term.expr = Piecewise((Q + R*log(D0)*v.T, And(1.0 < v.T, v.T < 10000)), (0, True))
            models.append(mob_term)

        mobility_models[solute] = models

    return mobility_models



    



