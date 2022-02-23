from tinydb import where
import numpy as np
from pycalphad import Model, variables as v
from symengine import exp, Symbol
from kawin.FreeEnergyHessian import partialdMudX, dMudX

setattr(v, 'GE', v.StateVariable('GE'))

class MobilityModel(Model):
    '''
    Handles mobility and diffusivity data from .tdb files

    Parameters
    ----------
    dbe : Database
    comps : list
        List of elements to consider
    phase_name : str
    parameters : dict or list (optional)
        Dictionary of parameters to be kept symbolic

    Attributes
    ----------
    mobility_variables : dict
        Dictionary of symbols in mobility functions for each element
    diffusivity_variables : dict
        Dictionary of symbols in diffusivity functions for each element
    '''
    def __init__(self, dbe, comps, phase_name, parameters=None):
        super().__init__(dbe, comps, phase_name, parameters)

        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}
        for name, value in self.mobility.items():
            self.mobility[name] = self.symbol_replace(value, symbols)

        for name, value in self.diffusivity.items():
            self.diffusivity[name] = self.symbol_replace(value, symbols)

        self.mob_site_fractions = {c: sorted([x for x in self.mobility_variables[c] if isinstance(x, v.SiteFraction)], key=str) for c in self.mobility}
        self.diff_site_fractions = {c: sorted([x for x in self.diffusivity_variables[c] if isinstance(x, v.SiteFraction)], key=str) for c in self.diffusivity}
        self.mob_state_variables = {c: sorted([x for x in self.mobility_variables[c] if not isinstance(x, v.SiteFraction)], key=str) for c in self.mobility}
        self.diff_state_variables = {c: sorted([x for x in self.diffusivity_variables[c] if not isinstance(x, v.SiteFraction)], key=str) for c in self.diffusivity}

    @property
    def mobility_variables(self):
        return {c: sorted([x for x in self.mobility[c].free_symbols if isinstance(x, v.StateVariable)], key=str) for c in self.mobility}

    @property
    def diffusivity_variables(self):
        return {c: sorted([x for x in self.diffusivity[c].free_symbols if isinstance(x, v.StateVariable)], key=str) for c in self.diffusivity}

    def build_phase(self, dbe):
        '''
        Builds free energy and mobility/diffusivity models as abstract syntax tree

        Parameters
        ----------
        dbe : Database
        '''
        super().build_phase(dbe)
        self.mobility, self.diffusivity = self.build_mobility(dbe)

    def _mobility_validity(self, constituent_array):
        '''
        Returns true if constituent_array contains only active species of current model
        Ignores vacancies - if a sublattice is only vacancies, then it should be ignored
            For now, phases where this occurs will have the parameters listed for n-1 sublattices
        '''
        for param_sublattice, model_sublattice in zip(constituent_array, self.constituents):
            if not (set(param_sublattice).issubset(model_sublattice) or (param_sublattice[0] == v.Species('*'))):
                return False
        return True
        
    def build_mobility(self, dbe):
        '''
        Builds mobility/diffusivity models as abstract syntax tree

        Parameters
        ----------
        dbe : Database
        '''
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search

        self.mob_models = {}
        mob = {}
        diff = {}

        param_names = ['MF', 'MQ', 'DF', 'DQ']

        for c in self.components:
            if c.name != 'VA':
                for name in param_names:
                    param_query = (
                        (where('phase_name') == phase.name) & \
                        (where('parameter_type') == name) & \
                        (where('constituent_array').test(self._mobility_validity)) & \
                        (where('diffusing_species') == c)
                    )
                    rk = self.redlich_kister_sum(phase, param_search, param_query)

                    if name not in self.mob_models:
                        self.mob_models[name] = {}
                    self.mob_models[name][c.name] = rk
 
                mob[c.name] = (1 / (v.R * v.T)) * exp((self.mob_models['MF'][c.name] + self.mob_models['MQ'][c.name]) / (v.R * v.T))
                setattr(self, 'mob_'+str(c.name).upper(), mob[c.name])
                diff[c.name] = exp((self.mob_models['DF'][c.name] + self.mob_models['DQ'][c.name]) / (v.R * v.T))
                setattr(self, 'diff_' + str(c.name).upper(), diff[c.name])

        return mob, diff

def tracer_diffusivity(composition_set, mobility_callables = None, mobility_correction = None):
    '''
    Computers tracer diffusivity for given equilibrium results
    D = MRT

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_callables : dict
        Pre-computed mobility callables for each element
    mobility_correction : dict (optional)
        Factor to multiply mobility by for each given element (defaults to 1)

    Returns
    -------
    Array of floats of diffusivity for each element (alphabetical order)
    '''
    if mobility_callables is None:
        raise ValueError('mobility_callables is required')

    R = 8.314
    T = composition_set.dof[composition_set.phase_record.state_variables.index(v.T)]
    elements = list(composition_set.phase_record.nonvacant_elements)

    #Set mobility correction if not set
    if mobility_correction is None:
        mobility_correction = {A: 1 for A in elements}
    else:
        for A in elements:
            if A not in mobility_correction:
                mobility_correction[A] = 1

    return R * T * np.array([mobility_correction[elements[A]] * mobility_callables[elements[A]](composition_set.dof) for A in range(len(elements))])

def tracer_diffusivity_from_diff(composition_set, diffusivity_callables = None, diffusivity_correction = None):
    '''
    Tracer diffusivity from diffusivity callables

    This will just return the Da as an array

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    diffusivity_callables : dict
        Pre-computed diffusivity callables for each element
    diffusivity_correction : dict (optional)
        Factor to multiply diffusivity by for each given element (defaults to 1)

    Returns
    -------
    Array of floats of diffusivity for each element (alphabetical order)
    '''
    if diffusivity_callables is None:
        raise ValueError('diffusivity_callables is required')

    elements = list(composition_set.phase_record.nonvacant_elements)

    #Set diffusivity correction if not set
    if diffusivity_correction is None:
        diffusivity_correction = {A: 1 for A in elements}
    else:
        for A in elements:
            if A not in diffusivity_correction:
                diffusivity_correction[A] = 1

    return np.array([diffusivity_correction[elements[A]] * diffusivity_callables[elements[A]](composition_set.dof) for A in range(len(elements))])

def mobility_matrix(composition_set, mobility_callables = None, mobility_correction = None):
    '''
    Mobility matrix
    Used to obtain diffusivity when multipled with free energy hessian

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_callables : dict (optional)
        Pre-computed mobility callables for each element
    mobility_correction : dict (optional)
        Factor to multiply mobility by for each given element (defaults to 1)

    Returns
    -------
    Matrix of floats
        Each index along an axis correspond to elements in alphabetical order
    '''
    if mobility_callables is None:
        raise ValueError('mobility_callables is required')

    elements = list(composition_set.phase_record.nonvacant_elements)
    X = composition_set.X

    #Set mobility correction if not set
    if mobility_correction is None:
        mobility_correction = {A: 1 for A in elements}
    else:
        for A in elements:
            if A not in mobility_correction:
                mobility_correction[A] = 1

    mob = np.array([X[A] * mobility_correction[elements[A]] * mobility_callables[elements[A]](composition_set.dof) for A in range(len(elements))])

    mobMatrix = np.zeros((len(elements), len(elements)))
    for a in range(len(elements)):
        for b in range(len(elements)):
            if a == b:
                mobMatrix[a, b] = (1 - X[a]) * mob[b]
            else:
                mobMatrix[a, b] = -X[a] * mob[b]

    return mobMatrix

def chemical_diffusivity(chemical_potentials, composition_set, mobility_callables, mobility_correction = None, returnHessian = False):
    '''
    Chemical diffusivity (D_ab)
        D_ab = mobility matrix * free energy hessian

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_callables : dict
        Pre-computed mobility callables for each element
    mobility_correction : dict (optional)
        Factor to multiply mobility by for each given element (defaults to 1)
    returnHessian : bool (optional)
        Whether to return chemical potential derivative (defaults to False)

    Returns
    -------
    (matrix of floats, free energy hessian)
        Each index along an axis correspond to elements in alphabetical order
        free energy hessian will be None if returnHessian is False
    '''
    dmudx = partialdMudX(chemical_potentials, composition_set)
    #print('dmudx', dmudx)
    mobMatrix = mobility_matrix(composition_set, mobility_callables, mobility_correction)
    #print('mobMatrix', mobMatrix)
    Dkj = np.matmul(mobMatrix, dmudx)
    
    if returnHessian:
        return Dkj, dmudx
    else:
        return Dkj, None

def interdiffusivity(chemical_potentials, composition_set, refElement, mobility_callables = None, mobility_correction = None, returnHessian = False):
    '''
    Interdiffusivity (D^n_ab)

    D^n_ab = D_ab - D_an (for substitutional element)
    D^n_ab = D_ab (for interstitial element)

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    refElement : str
        Reference element n
    mobility_callables : dict (optional)
        Pre-computed mobility callables for each element
    mobility_correction : dict (optional)
        Factor to multiply mobility by for each given element (defaults to 1)
    returnHessian : bool (optional)
        Whether to return chemical potential derivative (defaults to False)

    Returns
    -------
    (matrix of floats, free energy hessian)
        Each index along an axis correspond to elements in 
            alphabetical order excluding reference element
        free energy hessian will be None if returnHessian is False
    '''
    #List of interstitial elements - do not require reference element when calculating interdiffusivity
    interstitials = ['C', 'N', 'O', 'H', 'B']

    Dkj, hessian = chemical_diffusivity(chemical_potentials, composition_set, mobility_callables, mobility_correction, returnHessian)
    #print('Dkj', Dkj)
    elements = list(composition_set.phase_record.nonvacant_elements)

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


def interdiffusivity_from_diff(composition_set, refElement, diffusivity_callables, diffusivity_correction = None):
    '''
    Interdiffusivity (D^n_ab) calculated from diffusivity callables
        This is if the TDB database only has diffusivity data and no mobility data

    D^n_ab = D_ab - D_an (for substitutional element)
    D^n_ab = D_ab (for interstitial element)

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    refElement : str
        Reference element n
    diffusivity_callables : dict
        Pre-computed callables for diffusivity for each element
    diffusivity_correction : dict
        Factor to multiply diffusivity by for each element (defaults to 1)

    Returns
    -------
    matrix of floats
        Each index along an axis correspond to elements in 
            alphabetical order excluding reference element
    '''
    elements = list(composition_set.phase_record.nonvacant_elements)

    if diffusivity_correction is None:
        diffusivity_correction = {elements[A]: 1 for A in elements}
    else:
        for A in elements:
            if A not in diffusivity_correction:
                diffusivity_correction[A] = 1

    Dnkj = np.zeros((len(elements) - 1, len(elements) - 1))
    eleIndex = 0
    for a in range(len(elements) - 1):
        if elements[eleIndex] == refElement:
            eleIndex += 1

        Daa = diffusivity_correction[elements[eleIndex]] * diffusivity_callables[elements[eleIndex]](composition_set.dof)
        Dnkj[a, a] = Daa

        eleIndex += 1

    return Dnkj


def inverseMobility(chemical_potentials, composition_set, refElement, mobility_callables, mobility_correction = None, returnOther = True):
    '''
    Inverse mobility matrix for determining interfacial composition

    M^-1 = (free energy hessian) * Dnkj^-1

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    refElement : str
        Reference element n
    mobility_callables : dict
        Pre-computed mobility callables for each element
    mobility_correction : dict (optional)
        Factor to multiply mobility by for each given element (defaults to 1)
    returnOther : bool (optional)
        Whether to return interdiffusivity and hessian (defaults to False)

    Returns
    -------
    (interdiffusivity, hessian, inverse mobility)
        Interdiffusivity and hessian will be None if returnOther is False
    '''
    Dnkj, _ = interdiffusivity(chemical_potentials, composition_set, refElement, mobility_callables, mobility_correction, False)
    totalH = dMudX(chemical_potentials, composition_set, refElement)
    #print('totalH', totalH)
    if returnOther:
        return Dnkj, totalH, np.matmul(totalH, np.linalg.inv(Dnkj))
    else:
        return None, None, np.matmul(totalH, np.linalg.inv(Dnkj))


def inverseMobility_from_diffusivity(chemical_potentials, composition_set, refElement, diffusivity_callables, diffusivity_correction = None, returnOther = True):
    '''
    Inverse mobility matrix for determining interfacial composition

    M^-1 = (free energy hessian) * Dnkj^-1

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    refElement : str
        Reference element n
    diffusivity_callables : dict
        Pre-computed callables for diffusivity for each element
    diffusivity_correction : dict
        Factor to multiply diffusivity by for each element (defaults to 1)
    returnOther : bool (optional)
        Whether to return interdiffusivity and hessian (defaults to False)

    Returns
    -------
    (interdiffusivity, hessian, inverse mobility)
        Interdiffusivity and hessian will be None if returnOther is False
    '''
    Dnkj = interdiffusivity_from_diff(composition_set, refElement, diffusivity_callables, diffusivity_correction)
    totalH = dMudX(chemical_potentials, composition_set, refElement)

    if returnOther:
        return Dnkj, totalH, np.matmul(totalH, np.linalg.inv(Dnkj))
    else:
        return None, None, np.matmul(totalH, np.linalg.inv(Dnkj))
