from tinydb import where
import numpy as np
from pycalphad import Model, variables as v
from pycalphad.core.utils import wrap_symbol, extract_parameters
from symengine import exp, Symbol, Add
from kawin.thermo.FreeEnergyHessian import partialdMudX, dMudX

setattr(v, 'GE', v.StateVariable('GE'))

#List of interstitial elements
# When calculating interdiffusivity, we do not require reference element
# When calculating the mobility factor, we have an additional vacancy term to multiply by
#As a list here, hopefully this should be editable by a user outside of this module - may have to edit __init__.py
interstitials = ['C', 'N', 'O', 'H', 'B']

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
        self.diffusing_species = None

        super().__init__(dbe, comps, phase_name, parameters)
        self._setup_mobility_models(dbe)

    def set_diffusing_species(self, dbe, diffusing_species):
        '''
        Allows user to override diffusing species

        This will rebuild all the models, which is a bit
        inefficient, but I can't seem to add a parameter to
        the constructor due to the _dispatch_on function in
        pycalphad.model.Model
        '''
        # diffusing_species is list[v.Species] to be similar to components
        self.diffusing_species = [v.Species(s) for s in diffusing_species]

        #We only need to rebuild the mobility models since we're just changing
        #what mobility models we have
        self.mobility, self.diffusivity = self.build_mobility(dbe)
        self._setup_mobility_models(dbe)

    def _setup_mobility_models(self, dbe):
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}
        if self._parameters_arg is not None:
            if isinstance(self._parameters_arg, dict):
                symbols.update([(wrap_symbol(s), val) for s, val in self._parameters_arg.items()])
            else:
                # Lists of symbols that should remain symbolic
                for s in self._parameters_arg:
                    symbols.pop(wrap_symbol(s))

        #Replace symbols with database symbols for mobility and exponential term
        #Also store a copy of the mobility/diffusivity as MOB_A or DIFF_A
        for name, value in self.mobility.items():
            self.mobility[name] = self.symbol_replace(value, symbols).xreplace(v.supported_variables_in_databases)
            setattr(self, 'MOB_'+name, self.mobility[name])
            setattr(self, 'MQ_'+name, self.symbol_replace(getattr(self, 'MQ_'+name), symbols).xreplace(v.supported_variables_in_databases))

            setattr(self, 'lnM0_'+name, getattr(self, 'MQ_'+name).diff(v.T)/v.R)
            setattr(self, 'MQa_'+name, v.T*getattr(self, 'MQ_'+name).diff(v.T) - getattr(self, 'MQ_'+name))

        for name, value in self.diffusivity.items():
            self.diffusivity[name] = self.symbol_replace(value, symbols).xreplace(v.supported_variables_in_databases)
            setattr(self, 'DIFF_'+name, self.diffusivity[name])
            setattr(self, 'DQ_'+name, self.symbol_replace(getattr(self, 'DQ_'+name), symbols).xreplace(v.supported_variables_in_databases))

            setattr(self, 'lnD0_'+name, getattr(self, 'DQ_'+name).diff(v.T)/v.R)
            setattr(self, 'DQa_'+name, v.T*getattr(self, 'DQ_'+name).diff(v.T) - getattr(self, 'DQ_'+name))

        self.mob_site_fractions = {c: sorted([x for x in self.mobility_variables[c] if isinstance(x, v.SiteFraction)], key=str) for c in self.mobility}
        self.diff_site_fractions = {c: sorted([x for x in self.diffusivity_variables[c] if isinstance(x, v.SiteFraction)], key=str) for c in self.diffusivity}
        self.mob_state_variables = {c: sorted([x for x in self.mobility_variables[c] if not isinstance(x, v.SiteFraction)], key=str) for c in self.mobility}
        self.diff_state_variables = {c: sorted([x for x in self.diffusivity_variables[c] if not isinstance(x, v.SiteFraction)], key=str) for c in self.diffusivity}

    @property
    def mobility_variables(self):
        '''
        List of variables in the mobility functions

        Returns
        -------
        dictionary {component name (str) : variables (list)}
        '''
        return {c: sorted([x for x in self.mobility[c].free_symbols if isinstance(x, v.StateVariable)], key=str) for c in self.mobility}

    @property
    def diffusivity_variables(self):
        '''
        List of variables in the diffusivity functions

        Returns
        -------
        dictionary {component name (str) : variables (list)}
        '''
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
        if self.diffusing_species is None:
            self.diffusing_species = self.components

        phase = dbe.phases[self.phase_name]
        param_search = dbe.search

        self.mob_models = {}
        mob = {}
        diff = {}

        param_names = ['MF', 'MQ', 'DF', 'DQ']

        for c in self.diffusing_species:
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

                #Additional parameters search if diffusing species are not included
                #   This is mainly intended to help with parameter fitting
                #   Parameters will be in the format for MOB_A, MOB_B (or the respective keyword)
                #   This will reflect how the models are stored in MobilityModel
                #       Ex. MOB_A is the entire mobility model of A while MQ_A is the redlich kister polynomial used for A mobility
                #Additional parameters will be in tuples of (database keyword, mob_models keyword)
                additional_params = [('MOB', 'MQ'), ('MQ', 'MQ'), ('DIFF', 'DQ'), ('DQ', 'DQ')]
                for p in additional_params:
                    fit_name = p[0] + '_' + c.name
                    param_query = (
                        (where('phase_name') == phase.name) & \
                        (where('parameter_type') == fit_name) & \
                        (where('constituent_array').test(self._mobility_validity))
                    )
                    rk = self.redlich_kister_sum(phase, param_search, param_query)
                    self.mob_models[p[1]][c.name] += rk

        self.checkOrderingContribution(dbe)
        for c in self.diffusing_species:
            if c.name != 'VA':
                #In thermo-calc, the mobility model is defined as exp(sum(MF)/RT) * exp(sum(MQ)/RT) / RT
                #The diffusivity model is defined either as dilute - exp(sum(DF)/RT) * exp(sum(DQ)/RT)
                #                                        or simple - sum(DF) + sum(DQ)
                # We use the dilute assumption here
                #In summary, there's no difference between MF and MQ, or between DF and DQ
                # For papers using Q and theta (pre-exponential term), corrections must be made to theta have it fit the definitions above
                mqsum = self.mob_models['MF'][c.name] + self.mob_models['MQ'][c.name]
                dqsum = self.mob_models['DF'][c.name] + self.mob_models['DQ'][c.name]
                mob[c.name] = (1 / (v.R * v.T)) * exp(mqsum / (v.R * v.T))
                diff[c.name] = exp(dqsum / (v.R * v.T))

                #Also store the exponential term in case we want to grab the activation energy or pre-exp term
                setattr(self, 'MQ_'+str(c.name).upper(), mqsum)
                setattr(self, 'DQ_'+str(c.name).upper(), dqsum)

        return mob, diff
    
    def checkOrderingContribution(self, dbe):
        '''
        Checks if phase is an ordered part of a order-disorder model

        The ordered part of the phase double counts the disordered contribution, so the model is
        G = G_dis + G_ord(y) - G_ord(y=x)

        This is straight up copied from Model.atomic_ordering_energy in pycalphad with
          the minor difference that we replace the symbols in mob and diff
        '''
        phase = dbe.phases[self.phase_name]
        ordered_phase_name = phase.model_hints.get('ordered_phase', None)
        disordered_phase_name = phase.model_hints.get('disordered_phase', None)

        #If not order-disorder model, then return as unchanged
        if phase.name != ordered_phase_name:
            return
        
        ordered_phase = dbe.phases[ordered_phase_name]
        constituents = [sorted(set(c).intersection(self.components)) for c in ordered_phase.constituents]
        disordered_phase = dbe.phases[disordered_phase_name]
        disordered_model = self.__class__(dbe, sorted(self.components), disordered_phase_name)
        
        #If diffusing species are different, then set diffusing species for disordered model
        if set(self.components).symmetric_difference(self.diffusing_species) != 0:
            disordered_model.set_diffusing_species(dbe, self.diffusing_species)

        disordered_subl_constituents = disordered_phase.constituents[0]
        ordered_constituents = ordered_phase.constituents
        substitutional_sublattice_idxs = []
        for idx, subl_constituents in enumerate(ordered_constituents):
            if len(disordered_subl_constituents.symmetric_difference(subl_constituents)) == 0:
                substitutional_sublattice_idxs.append(idx)

        num_substitutional_sublattice_idxs = len(substitutional_sublattice_idxs)
        num_ordered_interstitial_subls = len(ordered_phase.sublattices) - num_substitutional_sublattice_idxs
        num_disordered_interstitial_subls = len(disordered_phase.sublattices) - 1
        if num_ordered_interstitial_subls != num_disordered_interstitial_subls:
            raise ValueError(
                f'Number of interstitial sublattices for the disordered phase '
                f'({num_disordered_interstitial_subls}) and the ordered phase '
                f'({num_ordered_interstitial_subls}) do not match. Got '
                f'substitutional sublattice indices of {substitutional_sublattice_idxs}.'
                )
        
        for c in self.mob_models['MF']:
            ordered_mobQ = Add(self.mob_models['MQ'][c])
            ordered_mobF = Add(self.mob_models['MF'][c])
            ordered_diffQ = Add(self.mob_models['DQ'][c])
            ordered_diffF = Add(self.mob_models['DF'][c])

            # Compute the molefraction_dict, which will map ordered phase site
            # fractions to the quasi mole fractions representing the disordered state
            molefraction_dict = {}
            ordered_sitefracs = [x for x in ordered_mobQ.free_symbols if isinstance(x, v.SiteFraction)]
            for sitefrac in ordered_sitefracs:
                if sitefrac.sublattice_index in substitutional_sublattice_idxs:
                    molefraction_dict[sitefrac] = \
                        self._quasi_mole_fraction(sitefrac.species,
                                                ordered_phase_name,
                                                constituents,
                                                ordered_phase.sublattices,
                                                substitutional_sublattice_idxs,
                                                )

            # Compute the variable_rename_dict, which will map disordered phase site
            # fractions to the quasi mole fractions representing the disordered state
            variable_rename_dict = {}
            disordered_sitefracs = [x for x in disordered_model.energy.free_symbols if isinstance(x, v.SiteFraction)]
            for atom in disordered_sitefracs:
                if atom.sublattice_index == 0:  # only the first sublattice is substitutional
                    variable_rename_dict[atom] = \
                        self._quasi_mole_fraction(atom.species,
                                                ordered_phase_name,
                                                constituents,
                                                ordered_phase.sublattices,
                                                substitutional_sublattice_idxs,
                                                )

                else:
                    shifted_subl_index = atom.sublattice_index + num_substitutional_sublattice_idxs - 1
                    variable_rename_dict[atom] = \
                        v.SiteFraction(ordered_phase_name, shifted_subl_index, atom.species)
                    
            self.mob_models['MQ'][c] = self._partitioned_expr(disordered_model.mob_models['MQ'][c], ordered_mobQ, variable_rename_dict, molefraction_dict)
            self.mob_models['MF'][c] = self._partitioned_expr(disordered_model.mob_models['MF'][c], ordered_mobF, variable_rename_dict, molefraction_dict)
            self.mob_models['DQ'][c] = self._partitioned_expr(disordered_model.mob_models['DQ'][c], ordered_diffQ, variable_rename_dict, molefraction_dict)
            self.mob_models['DF'][c] = self._partitioned_expr(disordered_model.mob_models['DF'][c], ordered_diffF, variable_rename_dict, molefraction_dict)

        return

def mobility_from_composition_set(composition_set, mobility_callables = None, mobility_correction = None, parameters = {}):
    '''
    Computes mobility from equilibrium results

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_callables : dict
        Pre-computed mobility callables for each element
    mobility_correction : dict (optional)
        Factor to multiply mobility by for each given element (defaults to 1)
    parameters : dict {str : float}
        List of parameters to override free symbols in the model

    Returns
    -------
    Array for floats for mobility of each element (alphabetical order)
    '''
    if mobility_callables is None:
        raise ValueError('mobility_callables is required')

    elements = list(composition_set.phase_record.nonvacant_elements)

    #Set mobility correction if not set
    if mobility_correction is None:
        mobility_correction = {A: 1 for A in elements}
    else:
        for A in elements:
            if A not in mobility_correction:
                mobility_correction[A] = 1

    #return np.array([mobility_correction[elements[A]] * mobility_callables[elements[A]](composition_set.dof) for A in range(len(elements))])
    param_keys, param_values = extract_parameters(parameters)
    if len(param_keys) > 0:
        callableInput = np.concatenate((composition_set.dof, param_values[0]), dtype=np.float_)
    else:
        callableInput = composition_set.dof
    return np.array([mobility_correction[elements[A]] * mobility_callables[elements[A]](callableInput) for A in range(len(elements))])

def compute_symbolic_expr_from_composition_set(composition_set, mobility_model, symbol, parameters = {}):
    '''
    Computes mobility symbol from equilibrium results using symbolic model

    NOTE: We don't build the callables for activation energy so this is a directly substitution
          into the symengine model. However, since this function isn't used often, the slower
          performance of this shouldn't be much of an issue

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_model : MobilityModel object
    symbol : str
        Name of symbol as symbol_el in model
    parameters : dict {str : float}
        List of parameters to override free symbols in the model

    Returns
    -------
    Array for floats for activation energy of each element (alphabetical order)
    '''
    elements = composition_set.phase_record.nonvacant_elements
    param_keys, param_values = extract_parameters(parameters)
    
    symbols = sorted([v.T, v.P, v.N, v.GE]) + mobility_model.site_fractions
    var_dict = {s:val for s,val in zip(symbols, np.array(composition_set.dof))}
    if len(param_keys) > 0:
        for p,val in zip(param_keys, param_values[0]):
            var_dict[p] = val

    return np.array([getattr(mobility_model, f'{symbol}_{elements[A]}').subs(var_dict).n(53, real=True) for A in range(len(elements))])

def mobility_from_composition_set_symbolic(composition_set, mobility_model, parameters = {}):
    '''
    Computes mobility from equilibrium results using symbolic model

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_model : MobilityModel object
    parameters : dict {str : float}
        List of parameters to override free symbols in the model

    Returns
    -------
    Array for floats for activation energy of each element (alphabetical order)
    '''
    return compute_symbolic_expr_from_composition_set(composition_set, mobility_model, 'MQ', parameters)
    
def activation_energy_from_composition_set(composition_set, mobility_model, parameters = {}):
    '''
    Computes activation energy for mobility from equilibrium results

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_model : MobilityModel object
    parameters : dict {str : float}
        List of parameters to override free symbols in the model

    Returns
    -------
    Array for floats for activation energy of each element (alphabetical order)
    '''
    return compute_symbolic_expr_from_composition_set(composition_set, mobility_model, 'MQa', parameters)

def prefactor_from_composition_set(composition_set, mobility_model, parameters = {}):
    '''
    Computes prefactor term, ln(M0), for mobility from equilibrium results

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    mobility_model : MobilityModel object
    parameters : dict {str : float}
        List of parameters to override free symbols in the model

    Returns
    -------
    Array for floats for activation energy of each element (alphabetical order)
    '''
    return compute_symbolic_expr_from_composition_set(composition_set, mobility_model, 'lnM0', parameters)

def tracer_diffusivity(composition_set, mobility_callables = None, mobility_correction = None, parameters = {}):
    '''
    Computes tracer diffusivity for given equilibrium results
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
    return R * T * mobility_from_composition_set(composition_set, mobility_callables, mobility_correction, parameters)

def mobility_matrix(composition_set, mobility_callables = None, mobility_correction = None, parameters = {}):
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
    elements = list(composition_set.phase_record.nonvacant_elements)
    X = composition_set.X

    #U-fraction - defined as U_a = X_a / sum(substitutionals)
    Usum = np.sum([X[A] for A in range(len(elements)) if elements[A] not in interstitials])
    U = X / Usum

    #Multiply mobility by U-fraction for ease of use when constructing the mobility matrix
    computedMob = mobility_from_composition_set(composition_set, mobility_callables, mobility_correction, parameters)
    mob = np.array([U[A] * computedMob[A] for A in range(len(elements))])

    #Find vacancy site fractions for multiplying with interstitials when making the mobility matrix
    #If vacancies are not found on the same sublattice, we'll defualt to 1 so there's at least some mobility and not 0
    #       A mobility of 0 would be quite unrealistic
    #       In addition, as we're working with interstitals, the vacancies are going to be close to 1, so this assumption wouldn't hurt
    vaTerms = {}            #Maps sublattice index to site fraction index for vacancies
    interstitialTerms = {}  #Maps interstitial to sublattice index
    index = len(composition_set.phase_record.state_variables)
    for i in range(len(composition_set.phase_record.variables)):
        if composition_set.phase_record.variables[i].species.name == 'VA':
            vaTerms[composition_set.phase_record.variables[i].sublattice_index] = composition_set.dof[index+i]
        if composition_set.phase_record.variables[i].species.name in interstitials:
            interstitialTerms[composition_set.phase_record.variables[i].species.name] = composition_set.phase_record.variables[i].sublattice_index

    # For interstitials
    #    M_aa = y_Va * M_a
    #    y_Va is taken from the same sublattice that a is on, where more vacancies on the sublattice implies faster diffusion
    # For substitutionals
    #    M_aa = (1-U_a) * U_a * M_a
    #    M_ab = -U_a * U_b * M_b
    # There are no entries for M_ab if one index is interstitial and the other is substitutional
    mobMatrix = np.zeros((len(elements), len(elements)))
    for a in range(len(elements)):
        for b in range(len(elements)):
            if a == b:
                if elements[a] in interstitials:
                    mobMatrix[a, a] = vaTerms.get(interstitialTerms[elements[a]], 1) * mob[a]
                else:
                    mobMatrix[a, b] = (1 - U[a]) * mob[b]
            else:
                if elements[b] not in interstitials:
                    mobMatrix[a, b] = -U[a] * mob[b]

    #Diffusivity requires dmu_a/dU_b; however, the free energy curvature gives dmu_a/dX_b
    #Assuming that Usum is constant and using chain-rule derivatives,
    #    the conversion from dmu_a/dX_b to dmu_a/dU_b can be done by multiplying the sum(substitutionals)
    mobMatrix *= Usum

    return mobMatrix

def chemical_diffusivity(chemical_potentials, composition_set, mobility_callables, mobility_correction = None, returnHessian = False, parameters = {}):
    '''
    Chemical diffusivity (D_kj)
        D_kj = sum((delta_ik - U_k) * U_i * M_i) * dmu_i/dU_j
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
    mobMatrix = mobility_matrix(composition_set, mobility_callables, mobility_correction, parameters)
    #print('mobMatrix', mobMatrix)
    Dkj = np.matmul(mobMatrix, dmudx)
    
    if returnHessian:
        return Dkj, dmudx
    else:
        return Dkj, None

def interdiffusivity(chemical_potentials, composition_set, refElement, mobility_callables = None, mobility_correction = None, returnHessian = False, parameters = {}):
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
    Dkj, hessian = chemical_diffusivity(chemical_potentials, composition_set, mobility_callables, mobility_correction, returnHessian, parameters)

    elements = list(composition_set.phase_record.nonvacant_elements)

    #Find index of reference element
    refIndex = 0
    for a in range(len(elements)):
        if elements[a] == refElement:
            refIndex = a
            break

    #Build Dnkj, skipping the reference element
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

def inverseMobility(chemical_potentials, composition_set, refElement, mobility_callables, mobility_correction = None, returnOther = True, parameters = {}):
    '''
    Inverse mobility matrix for determining interfacial composition from 
        Philippe and P. W. Voorhees, Acta Materialia 61 (2013) p. 4237

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
    Dnkj, _ = interdiffusivity(chemical_potentials, composition_set, refElement, mobility_callables, mobility_correction, False, parameters)
    totalH = dMudX(chemical_potentials, composition_set, refElement)
    #print('totalH', totalH)
    if returnOther:
        return Dnkj, totalH, np.matmul(totalH, np.linalg.inv(Dnkj))
    else:
        return None, None, np.matmul(totalH, np.linalg.inv(Dnkj))
    
def tracer_diffusivity_from_diff(composition_set, diffusivity_callables = None, diffusivity_correction = None, parameters = {}):
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

    #return np.array([diffusivity_correction[elements[A]] * diffusivity_callables[elements[A]](composition_set.dof) for A in range(len(elements))])

    param_keys, param_values = extract_parameters(parameters)
    if len(param_values) > 0:
        callableInput = np.concatenate((composition_set.dof, param_values[0]), dtype=np.float_)
    else:
        callableInput = composition_set.dof
    return np.array([diffusivity_correction[elements[A]] * diffusivity_callables[elements[A]](callableInput) for A in range(len(elements))])

def interdiffusivity_from_diff(composition_set, refElement, diffusivity_callables, diffusivity_correction = None, parameters = {}):
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

    param_keys, param_values = extract_parameters(parameters)
    if len(param_values) > 0:
        callableInput = np.concatenate((composition_set.dof, param_values[0]), dtype=np.float_)
    else:
        callableInput = composition_set.dof
    Dnkj = np.zeros((len(elements) - 1, len(elements) - 1))
    eleIndex = 0
    for a in range(len(elements) - 1):
        if elements[eleIndex] == refElement:
            eleIndex += 1

        #Daa = diffusivity_correction[elements[eleIndex]] * diffusivity_callables[elements[eleIndex]](composition_set.dof)
        Daa = diffusivity_correction[elements[eleIndex]] * diffusivity_callables[elements[eleIndex]](callableInput)
        Dnkj[a, a] = Daa

        eleIndex += 1

    return Dnkj

def inverseMobility_from_diffusivity(chemical_potentials, composition_set, refElement, diffusivity_callables, diffusivity_correction = None, returnOther = True, parameters = {}):
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
    Dnkj = interdiffusivity_from_diff(composition_set, refElement, diffusivity_callables, diffusivity_correction, parameters)
    totalH = dMudX(chemical_potentials, composition_set, refElement)

    if returnOther:
        return Dnkj, totalH, np.matmul(totalH, np.linalg.inv(Dnkj))
    else:
        return None, None, np.matmul(totalH, np.linalg.inv(Dnkj))


