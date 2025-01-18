import numpy as np
from symengine import Piecewise, And, Symbol, log

from pycalphad import Database, variables as v

from kawin.Constants import GAS_CONSTANT, BOLTZMANN_CONSTANT
from kawin.mobility_fitting.utils import _vname, find_last_variable, MobilityTerm

def _getK0(melting_phase):
    phases = {'BCC': 1, 'HCP': 2, 'FCC': 3}
    for p in phases:
        if p in melting_phase:
            return phases[p]
    return 0

class LiuSpecies:
    def __init__(self, name, diameter, atomic_number, Tm, melting_phase = None):
        self.name = name
        self.diameter = diameter
        self.atomic_number = atomic_number
        self.Tm = Tm
        self.melting_phase = melting_phase
        self.K0 = _getK0(melting_phase)

class SuSpecies:
    def __init__(self, name, atomic_mass, density, Tm, cte = 0):
        self.name = name
        self.atomic_mass = atomic_mass
        self.density = density
        self.Tm = Tm
        self.cte = cte
    
def _get_liquid_phase_name(database):
    for ph in database.phases:
        if ph.upper() == 'LIQUID':
            return ph
        elif ph.upper() == 'LIQ':
            return ph
        elif ph.upper() == 'L':
            return ph
    return 'LIQUID'

def compute_liquid_mobility_liu(database: Database, species: list[LiuSpecies]):
    '''
    Species data will include {diameter, atomic_number, T_m, phase}

    Units:
        diameter: Angstroms
        T_m: K

    From Y. Liu et al, "A predictive equation for solute diffusivity in liquid metals" 
    Scripta Materialia 55 (2006), 367. doi:10.1016/j.scriptamat.2006.04.019

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
    liq_name = _get_liquid_phase_name(database)
    mobility_models = {}
    for solute in species:
        models = []
        for solvent in species:
            Q = -0.17 * GAS_CONSTANT * solvent.Tm * (16 + solvent.K0)
            D0 = (8.95 - 0.0734 * solvent.atomic_number) * 1e-8
            D0 *= solvent.diameter / solute.diameter

            const_array = [[v.SiteFraction(liq_name, 0, solvent.name)]]
            mob_term = MobilityTerm(const_array, 0)
            mob_term.expr = Piecewise((Q + GAS_CONSTANT*np.log(D0)*v.T, And(1.0 < v.T, v.T < 10000)), (0, True))
            models.append(mob_term)

        mobility_models[solute.name] = models

    return mobility_models

def compute_liquid_mobility_su(database: Database, species: list[SuSpecies]):
    '''
    Species data will include {atomic_mass, density, T_m, CTE (optional)}
    If CTE is included, then density/molar volume will be assumed to be at room temperature
        And will be adjusted to the melting point

    Units:
        atomic_mass: g/mol
        density: g/cm3
        T_m: K
        CTE: m/m

    From X. Su et al, "A new equation for temperature dependent solute impurity diffusivity in liquid metals" 
    Journal of Phase Equilibria and Diffusion 31 (2010) 333. doi:10.1007/s11669-010-9726-4

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
    C1 = 1.8e-8
    C2 = 2.34
    R0 = 0.644e-10

    liq_name = _get_liquid_phase_name(database)
    mobility_models = {}
    for solute in species:
        models = []
        for solvent in species:
            rho_solv = solvent.density
            rho_solu = solute.density

            # If CTE is non-zero, adjust density to that of the solvent melting point
            rho_solv *= (1 + solvent.cte * (solvent.Tm - 298.15))**3
            rho_solu *= (1 + solute.cte * (solvent.Tm - 298.15))**3
            V_solv = solvent.atomic_mass / rho_solv * 1e-6       #m3/mol
            V_solu = solute.atomic_mass / rho_solu * 1e-6       #m3/mol

            Q = -C2 * solvent.Tm * GAS_CONSTANT
            D1 = C1 * np.power(solvent.atomic_mass/1000, 1/2) / np.power(V_solv, 2/3)
            D2 = R0 * np.power(solute.atomic_mass / rho_solu, 1/3)
            D3 = BOLTZMANN_CONSTANT / (4*np.pi)
            D0 = D3/(D1*D2) * v.T**(1/2) / (1 - 0.112 * (v.T / solvent.Tm)**(1/2))

            const_array = [[v.SiteFraction(liq_name, 0, solvent.name)]]
            mob_term = MobilityTerm(const_array, 0)
            mob_term.expr = Piecewise((Q + GAS_CONSTANT*log(D0)*v.T, And(1.0 < v.T, v.T < 10000)), (0, True))
            models.append(mob_term)

        mobility_models[solute.name] = models

    return mobility_models

