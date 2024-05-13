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

    From Y. Liu et al, "A predictive equation for solute diffusivity in liquid metals" Scripta Materialia 55 (2006), 367

    Q = 0.17*R*Tm*(16+K0) J/mol
        K0 is integer corresponding to solid phase at melting (BCC = 1, HCP = 2, FCC = 3)
        If the phase is neither of those 3, then we assume K0 = 0

    For self diffusion
        D0_BB = (8.95 - 0.0734*A) * 1e-8 m2/s

    For solute diffusion
        D0_AB = r_B / r_A * D0_BB

    Then M_AB = D0_AB * exp(-Q/RT)
    
    In a calphad database, this becomes MQ = -Q + R*T*ln(D0_AB)
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

    From X. Su et al, "A new equation for temperature dependent solute impurity diffusivity in liquid metals" Journal of Phase Equilibria and Diffusion 31 (2010) 333

    Diffusivity starts with Sutherlan-Einstein equation - D = k*T / 4*pi*mu*r

    Dynamic viscosity (mu)
        mu_B = C1 * M_B^1/2 * T^1/2 / V_B^2/3 * exp(C2 * TM_B / T) J*s/m3
        C1 = 1.8e-8 (J/K/mol^1/3)^1/2
        C2 = 2.34

    Radius in liquid
        r_A = R0 * (M_A/rho_A)^1/3 * (1 - 0.122 * (T/TM)^1/2) m
            R0 = 0.644e-10
        Note: In Su et al, TM is suggested to be melting point of solute (TM_A), but in
        P. Protopapas et al, J. Chem. Phys. 59 (1973) 15 (where this equation comes from), 
        TM is noted to be the melting point of the alloy, in which the solvent (TM_B) is used
        Here, we used TM_B for the melting point since it seems to be the trend of smaller radius
        leading to higher diffusivity

    D_AB = k*T / 4*pi*mu_B*r_A

    To put in a Calphad database, it's helpful to define equivalents for D0_AB and Q
        Q = -C2 * TM_B * R (coming from the viscosity equation)
        D0_AB = D0 * T^1/2 / (1 - 0.112 * (T/TM_B)^1/2)
            D0 = (k / 4*pi) * (V_B^2/3 / C1*M_B^1/2) * (rho_A^1/3 / M_A^1/3*R0)
        Then M = D_AB = D0_AB * exp(-Q/RT)
        And MQ = -Q + R*T*ln(D0_AB)
    
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



    



