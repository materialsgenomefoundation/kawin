import copy
from collections import namedtuple

import numpy as np
from tinydb import where

from pycalphad import Workspace, Model, Database, calculate, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.utils import extract_parameters

from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.thermo.FreeEnergyHessian import dMudX
from kawin.thermo.Mobility import MobilityModel, inverseMobility, inverseMobility_from_diffusivity, tracer_diffusivity, tracer_diffusivity_from_diff

DrivingForceCache = namedtuple('DrivingForceCache', 
                               ['conditions', 'matrix_eq', 'precipitate_eq', 'samples', 'ordered_samples'],
                               defaults=(None, None, None, None, None))

class ExtraFreeEnergyType(v.IndependentPotential):
    implementation_units = 'joules'
    display_units = 'joules'
    display_name = 'Extra Gibbs Free Energy'

    def __init__(self):
        super().__init__('GE')
    def __reduce__(self):
        return self.__class__, ()
    
setattr(v, 'GE', v.IndependentPotential('GE'))

class ExtraGibbsModel(Model):
    '''
    Subclass of pycalphad Model with extra variable GE
        GE represents any extra contribution to the Gibbs free energy
        such as the Gibbs-Thomson contribution
    '''
    energy = GM = property(lambda self: self.ast + v.GE)
    formulaenergy = G = property(lambda self: (self.ast + v.GE) * self._site_ratio_normalization)
    orderingContribution = OCM = property(lambda self: self.models['ord'])

class GeneralThermodynamics:
    '''
    Class for defining driving force and essential functions for
    binary and multicomponent systems using pycalphad for equilibrium
    calculations

    Parameters
    ----------
    database : Database or str
        pycalphad Database or file name for database
    elements : list
        Elements to consider
        Note: reference element must be the first index in the list
    phases : list
        Phases involved
        Note: matrix phase must be first index in the list
    drivingForceMethod : str (optional)
        Method used to calculate driving force
        Options are 'tangent' (default), 'approximate', 'sampling' and 'curvature' (not recommended)
    parameters : list [str] or dict {str : float} or None
        List of parameters to keep symbolic in the thermodynamic or mobility models
        If None, then parameters are fixed
    '''
    gOffset = 1      #Small value to add to precipitate phase for when order/disorder models are used

    def __init__(self, database, elements, phases, drivingForceMethod = 'tangent', parameters = None):
        if isinstance(database, str):
            database = Database(database)
        self.db = database

        self.elements = copy.copy(elements)
        if 'VA' not in self.elements:
            self.elements.append('VA')
        self._isBinary = len(self.elements) == 3
        
        if parameters is None:
            self._parameters = {}
        else:
            if isinstance(parameters, list):
                self._parameters = {p: 0 for p in parameters}
            else:
                self._parameters = parameters

        if type(phases) == str:  # check if a single phase was passed as a string instead of a list of phases.
            phases = [phases]
        self.phases = phases

        self._buildThermoModels()

        self.sampling_pDens = 2000      # Sampling density for driving force sampling method
        self.pDens = 500                # Sampling density for equilibrium

        self.setDrivingForceMethod(drivingForceMethod)
        self._buildMobilityModels()
        self.clearCache()

    def clearCache(self):
        '''
        Removes any cached data
        This is intended for surrogate training, where the cached data
        will be removed incase
        '''
        self._compset_cache = {}
        self._compset_cache_df = {}
        self._prev_cond = None
        self._parentEq = (None, None)

        self.diffusivity_cache = {}
        self.driving_force_cache = {}

        #Stored samples for precipitate phases at defined temperature
        self.points_cache = {}

        #This is so that these can be used again if the temperature has not changed since last usage
        self._prevTemperature = None

    def _buildThermoModels(self):
        '''
        Builds thermodynamic models for each phase

        This assumes that the first phase is the parent phase and the rest of the phases are precipitate phases
            For usage in a diffusion model, this won't affect anything

        For each precipitate phase, it checks whether the phase has an order/disorder contribution
            If so, then it checks if the disorder contribution comes from the parent phase (ex. gamma and gamma prime in Ni alloys)
            An ordering contribution phase record will be created to allow separated between the parent and the ordered phase
        '''
        self.orderedPhase = {self.phases[i]: False for i in range(1, len(self.phases))}
        for i in range(1, len(self.phases)):
            if self.db.phases[self.phases[i]].model_hints.get('disordered_phase') == self.phases[0]:
                self.orderedPhase[self.phases[i]] = True
                self._forceDisorder(self.phases[0])

        #Build phase models assuming first phase is parent phase and rest of precipitate phases
        #If the same phase is used for matrix and precipitate phase, then force the matrix phase to remove the ordering contribution
        #This may be unnecessary as already disordered phase models will not be affected, but I guess just in case the matrix phase happens to be an ordered solution
        param_keys, _ = extract_parameters(self._parameters)
        self.models = {self.phases[0]: Model(self.db, self.elements, self.phases[0], parameters=param_keys)}
        self.models[self.phases[0]].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)

        for i in range(1, len(self.phases)):
            self.models[self.phases[i]] = ExtraGibbsModel(self.db, self.elements, self.phases[i], parameters=param_keys)
            self.models[self.phases[i]].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)

        self.phase_records = PhaseRecordFactory(self.db, self.elements, 
                                                       self.models[self.phases[0]].state_variables, 
                                                       self.models, parameters=self._parameters)

    def _buildMobilityModels(self):
        '''
        Builds mobility models for phases that have model parameters
        '''
        self.mobModels = {p: None for p in self.phases}
        self.mobCallables = {p: None for p in self.phases}
        self.diffCallables = {p: None for p in self.phases}
        param_keys, _ = extract_parameters(self._parameters)
        phase_mob_params = {}
        phase_diff_params = {}
        self.mobModels = {}
        for p in self.phases:
            #Get mobility/diffusivity of phase p if exists
            param_search = self.db.search
            param_query_mob = (
                (where('phase_name') == p) & \
                (where('parameter_type') == 'MQ') | \
                (where('parameter_type') == 'MF')
            )

            param_query_diff = (
                (where('phase_name') == p) & \
                (where('parameter_type') == 'DQ') | \
                (where('parameter_type') == 'DF')
            )
            phase_mob_params[p] = param_search(param_query_mob)
            phase_diff_params[p] = param_search(param_query_diff)

            if len(phase_mob_params[p]) > 0 or len(phase_diff_params[p]) > 0:
                self.mobModels[p] = MobilityModel(self.db, self.elements, p, parameters=param_keys)
                self.mobModels[p].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)

        mob_phases = list(self.mobModels.keys())
        self.mob_phase_records = PhaseRecordFactory(self.db, self.elements, 
                                                    self.mobModels[mob_phases[0]].state_variables, 
                                                    self.mobModels, parameters=self._parameters)

        for p in self.mobModels:
            if len(phase_mob_params[p]) > 0:
                self.mobCallables[p] = {c: self.mob_phase_records.get_phase_property(p, 'MOB_'+c, include_grad=False, include_hess=False).func for c in self.phase_records[p].nonvacant_elements}
            if len(phase_diff_params[p]) > 0:
                self.diffCallables[p] = {c: self.mob_phase_records.get_phase_property(p, 'DIFF_'+c, include_grad=False, include_hess=False).func for c in self.phase_records[p].nonvacant_elements}

        #This applies to all phases since this is typically reflective of quenched-in vacancies
        self.mobility_correction = {A: 1 for A in self.elements}

    def updateParameters(self, parameters):
        '''
        Update parameter dictionary with new values

        Parameters
        ----------
        parameters : dict {str : float}
            Dictionary of parameters
            NOTE: this does not have to be the full list and can also have other parameters in it
                Only the parameters that are stored upon initialization will be changed
        '''
        for pm in parameters:
            if pm in self._parameters:
                self._parameters[pm] = parameters[pm]

        param_keys, param_values = extract_parameters(self._parameters)
        for p in self.phases:
            self.phase_records[p].parameters[:] = np.asarray(param_values, dtype=np.float64)

    def _forceDisorder(self, phase):
        '''
        For phases using an order/disorder model, pycalphad will neglect the disordered phase unless
        it is the only phase set active, so the order and disordered portion of the phase will use the same model

        For the Gibbs-Thomson effect to be applied, the ordered and disordered parts of the model will need to be kept separate
        As a fix, a new phase is added to the database that uses only the disordered part of the model
        '''
        newPhase = 'DIS_' + phase
        self.phases[0] = newPhase
        self.db.phases[newPhase] = copy.deepcopy(self.db.phases[phase])
        self.db.phases[newPhase].name = newPhase
        del self.db.phases[newPhase].model_hints['ordered_phase']
        del self.db.phases[newPhase].model_hints['disordered_phase']

        #Copy database parameters with new name
        param_query = where('phase_name') == phase
        params = self.db.search(param_query)
        for p in params:
            #We have to create a new dictionary since p is a TinyDB.Document
            newP = {}
            for entry in p:
                newP[entry] = p[entry]
            newP['phase_name'] = newPhase
            self.db._parameters.insert(newP)

    def setDrivingForceMethod(self, drivingForceMethod):
        '''
        Sets method for calculating driving force

        Parameters
        ----------
        drivingForceMethod - str
            Options are ['approximate', 'sampling', 'curvature']
        '''
        if drivingForceMethod == 'approximate':
            self._drivingForce = self._getDrivingForceApprox
        elif drivingForceMethod == 'sampling':
            self._drivingForce = self._getDrivingForceSampling
        elif drivingForceMethod == 'curvature':
            self._drivingForce = self._getDrivingForceCurvature
        elif drivingForceMethod == 'tangent':
            self._drivingForce = self._getDrivingForceTangent
        else:
            raise Exception('Driving force method must be either \'approximate\', \'sampling\', \'tangent\' or \'curvature\'')

    def setDFSamplingDensity(self, density):
        '''
        Sets sampling density for sampling method in driving
        force calculations

        Default upon initialization is 2000

        Parameters
        ----------
        density : int
            Number of samples to take per degree of freedom in the phase
        '''
        self.points_cache = {}
        self.sampling_pDens = density

    def setEQSamplingDensity(self, density):
        '''
        Sets sampling density for equilibrium calculations

        Default upon initialization is 500

        Parameters
        ----------
        density : int
            Number of samples to take per degree of freedom in the phase
        '''
        self.pDens = density

    def setMobility(self, mobility):
        '''
        Allows user to define mobility functions

        mobility : dict
            Dictionary of functions for each element (including reference)
            Each function takes in (v.T, v.P, v.N, v.GE, site fractions) and returns mobility

        Optional - only required for multicomponent systems where
            mobility terms are not defined in the TDB database
        '''
        self.mobCallables = mobility

    def setDiffusivity(self, diffusivity):
        '''
        Allows user to define diffusivity functions

        diffusivity : dict
            Dictionary of functions for each element (including reference)
            Each function takes in (v.T, v.P, v.N, v.GE, site fractions) and returns diffusivity

        Optional - only required for multicomponent systems where
            diffusivity terms are not defined in the TDB database
            and if mobility terms are not defined
        '''
        self.diffCallables = diffusivity

    def setMobilityCorrection(self, element, factor):
        '''
        Factor to multiply mobility by for each element

        Parameters
        ----------
        element : str
            Element to set factor for
            If 'all', factor will be set to all elements
        factor : float
            Scaling factor
        '''
        if element == 'all':
            for e in self.mobility_correction:
                self.mobility_correction[e] = factor
        else:
            self.mobility_correction[element] = factor

    def _process_x(self, x):
        '''
        Processes x to always be an array for len(elements) - 1
        If x in len(elements), then we assume that the first item is the solute
        '''
        x = np.atleast_1d(x)
        if len(x) == len(self.elements) - 1:
            x = x[1:]
        return x

    def _getConditions(self, x, T, gExtra = 0):
        '''
        Creates dictionary of conditions from composition, temperature and gExtra

        Parameters
        ----------
        x : list
            Composition (excluding reference element)
        T : float
            Temperature
        gExtra : float
            Gibbs free energy to add to phase
        '''
        x = self._process_x(x)
        cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        cond.update({v.GE: gExtra, v.N: 1, v.P: 101325, v.T: T})
        self._prev_cond = cond
        return cond
    
    def _setupSubModels(self, precPhase = None):
        """
        Creates a subset of phases and models and updates the phase records accordingly

        Options for precPhase
            -1           -> [parent phase]
            None         -> [parent phase, 1st precipitate phase]
            str          -> [parent phase, precPhase]
            list[phases] -> precPhase
        """
        phases = [self.phases[0]]
        if precPhase != -1:
            if precPhase is None:
                precPhase = self.phases[1]
            if isinstance(precPhase, str):
                phases.append(precPhase)
            else:
                # Copy list
                phases = [p for p in precPhase]

        sub_models = {p: self.models[p] for p in phases}

        # pycalphad seems to use the phase record model list to generate composition sets
        # so we need to make sure that the phase record models align with the subset of phases
        self.phase_records.models = sub_models
        return phases, sub_models

    def getEq(self, x, T, gExtra = 0, precPhase = None):
        '''
        Calculates equilibrium at specified x, T, gExtra

        This is separated from the interfacial composition function so that this can be used for getting curvature for interfacial composition from mobility

        Parameters
        ----------
        x : float or array
            Composition
            Needs to be array for multicomponent systems
        T : float
            Temperature
        gExtra : float
            Gibbs-Thomson contribution (if applicable)
        precPhase : str, int, list or None
            Precipitate phase (default is first precipitate)
            Options:
                None - first precipitate phase in phase list
                str - specific precipitate phase by name
                list - all phases by name in list
                -1 - no precipitate phase

        Returns
        -------
        Workspace object
        '''
        cond = self._getConditions(x, T, gExtra+self.gOffset)
        
        phases, sub_models = self._setupSubModels(precPhase)
        wks = Workspace(self.db, self.elements, phases, cond, 
                        models=sub_models, phase_record_factory=self.phase_records, 
                        calc_opts={'pdens': self.pDens})

        return wks

    def getLocalEq(self, x, T, gExtra = 0, precPhase = None, composition_sets = None):
        '''
        Calculates local equilibrium at specified x, T, gExtra

        Local equilibrium is defined as all phases have a single composition set
            Miscibility gaps are ignored
            In the case of a single phase with a miscibility gap, this returns the composition
            set at x

        Parameters
        ----------
        x : float or array
            Composition
            Needs to be array for multicomponent systems
        T : float
            Temperature
        gExtra : float
            Gibbs-Thomson contribution (if applicable)
        precPhase : str, int, list or None
            Precipitate phase (default is first precipitate)
            Options:
                None - first precipitate phase in phase list
                str - specific precipitate phase by name
                list - all phases by name in list
                -1 - no precipitate phase

        Returns
        -------
        result - equilibrium convergence and chemical potentials
        composition_sets - list of CompositionSet for phases in "equilibrium"
            Note - "equilibrium" in terms of the matrix and singled out precipitate phase (or just matrix if precPhase is -1)
        '''
        cond = self._getConditions(x, T, gExtra)
        phases, sub_models = self._setupSubModels(precPhase)
        return local_equilibrium(self.db, self.elements, phases, cond, 
                                 sub_models, self.phase_records, composition_sets=composition_sets)

    def getInterdiffusivity(self, x, T, removeCache = True, phase = None):
        '''
        Gets interdiffusivity at specified x and T
        Requires TDB database to have mobility or diffusivity parameters

        Parameters
        ----------
        x : float, array or 2D array
            Composition
            Float or array for binary systems
            Array or 2D array for multicomponent systems
        T : float or array
            Temperature
            If array, must be same length as x
                For multicomponent systems, must be same length as 0th axis
        removeCache : boolean
            If True, recalculates equilibrium to get interdiffusivity (default)
            If False, will use calculation from driving force calcs (if available) to compute diffusivity
        phase : str
            Phase to compute diffusivity for (defaults to first or matrix phase)
            This only needs to be used for multiphase diffusion simulations

        Returns
        -------
        interdiffusivity - will return array if T is an array
            For binary case - float or array of floats
            For multicomponent - matrix or array of matrices
        '''
        dnkj = []
        x = np.atleast_2d(x)
        if self._isBinary:
            x = x.T
        T = np.atleast_1d(T)
        dnkj = [self._interdiffusivitySingle(xi, Ti, removeCache, phase) for xi, Ti in zip(x, T)]
        return np.squeeze(dnkj)

    def _interdiffusivitySingle(self, x, T, removeCache = True, phase = None):
        '''
        Gets interdiffusivity at unique composition and temperature

        Parameters
        ----------
        x : float or array
            Composition
        T : float
            Temperature
        removeCache : boolean
        phase : str

        Returns
        -------
        Interdiffusivity as a matrix (will return float in binary case)
        '''
        phase = self.phases[0] if phase is None else phase
        comp_sets = self.diffusivity_cache.get(phase, None)
        result, comp_sets = self.getLocalEq(x, T, 0, [phase], composition_sets=comp_sets)
        cs_matrix = comp_sets[0]
        chemical_potentials = result.chemical_potentials

        if self.mobCallables[phase] is None:
            Dnkj, _, _ = inverseMobility_from_diffusivity(chemical_potentials, cs_matrix, 
                                                          self.elements[0], self.diffCallables[phase],
                                                          diffusivity_correction=self.mobility_correction,
                                                          parameters = self._parameters)
        else:
            Dnkj, _, _ = inverseMobility(chemical_potentials, cs_matrix, 
                                         self.elements[0], self.mobCallables[phase],
                                         mobility_correction=self.mobility_correction,
                                         parameters=self._parameters)

        if not self._isBinary:
            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)
            Dnkj = Dnkj[unsortIndices,:]
            Dnkj = Dnkj[:,unsortIndices]

        self.diffusivity_cache[phase] = None if removeCache else comp_sets
        return np.squeeze(Dnkj)

    def getTracerDiffusivity(self, x, T, removeCache = True, phase = None):
        '''
        Gets tracer diffusivity for element el at specified x and T
        Requires TDB database to have mobility or diffusivity parameters

        Parameters
        ----------
        x : float, array or 2D array
            Composition
            Float or array for binary systems
            Array or 2D array for multicomponent systems
        T : float or array
            Temperature
            If array, must be same length as x
                For multicomponent systems, must be same length as 0th axis
        removeCache : boolean
        phase : str

        Returns
        -------
        tracer diffusivity - will return array if T is an array
        '''
        td = []
        x = np.atleast_2d(x)
        if self._isBinary:
            x = x.T
        T = np.atleast_1d(T)
        td = [self._tracerDiffusivitySingle(xi, Ti, removeCache, phase) for xi, Ti in zip(x, T)]
        return np.squeeze(td)

    def _tracerDiffusivitySingle(self, x, T, removeCache = True, phase = None):
        '''
        Gets tracer diffusivity at unique composition and temperature

        Parameters
        ----------
        x : float or array
            Composition
        T : float
            Temperature
        el : str
            Element to calculate diffusivity
        
        Returns
        -------
        Tracer diffusivity as a float
        '''
        phase = self.phases[0] if phase is None else phase
        comp_sets = self.diffusivity_cache.get(phase, None)
        result, comp_sets = self.getLocalEq(x, T, 0, [phase], composition_sets=comp_sets)
        cs_matrix = comp_sets[0]

        if self.mobCallables[phase] is None:
            #NOTE: This is not tested yet
            Dtrace = tracer_diffusivity_from_diff(cs_matrix, self.diffCallables[phase], 
                                                  diffusivity_correction=self.mobility_correction, 
                                                  parameters=self._parameters)
        else:
            Dtrace = tracer_diffusivity(cs_matrix, self.mobCallables[phase], 
                                        mobility_correction=self.mobility_correction, 
                                        parameters=self._parameters)

        sortIndices = np.argsort(self.elements[:-1])
        unsortIndices = np.argsort(sortIndices)
        Dtrace = Dtrace[unsortIndices]

        self.diffusivity_cache[phase] = None if removeCache else comp_sets
        return Dtrace

    def getDrivingForce(self, x, T, precPhase = None, removeCache = False):
        '''
        Gets driving force using method defined upon initialization

        Parameters
        ----------
        x : float, array or 2D array
            Composition of minor element in bulk matrix phase
            For binary system, use an array for multiple compositions
            For multicomponent systems, use a 2D array for multiple compositions
                Where 0th axis is for indices of each composition
        T : float or array
            Temperature in K
            Must be same length as x if x is array or 2D array
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative
        '''
        x = np.atleast_2d(x)
        if self._isBinary:
            x = x.T
        T = np.atleast_1d(T)
        dgArray, compArray = zip(*[self._drivingForce(xi, Ti, precPhase, removeCache) for xi, Ti in zip(x, T)])
        return np.squeeze(dgArray), np.squeeze(compArray)

    def _getDrivingForceSampling(self, x, T, precPhase = None, removeCache = False):
        '''
        Gets driving force for nucleation by sampling

        Steps
            1. Compute local equilibrium at x and T of only the matrix phase
            2. Sample precipitate phase
                If ordered contribution to matrix phase, then sample ordering contribution
                and remove points on the matrix free energy surface
            3. Compute energy difference between precipitate samples and chemical potential hyperplane
            4. Find sample that maximizes energy difference and return sample composition and driving force

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)

        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative
        '''
        precPhase = self.phases[1] if precPhase is None else precPhase

        #Calculate equilibrium with only the parent phase -------------------------------------------------------------------------------------------
        cond = self._getConditions(x, T, 0)

        cs_results = self._getPrecCompositionSetSamplingDF(x, T, cond, precPhase, removeCache)
        if cs_results is None:
            return None, None
        
        dg, prec_cs = cs_results

        sortIndices = np.argsort(self.elements[:-1])
        unsortIndices = np.argsort(sortIndices)
        beta_x = np.array(prec_cs.X, dtype=np.float64)
        beta_x = beta_x[unsortIndices]

        return np.squeeze(dg), np.squeeze(beta_x[1:])

    def _getDrivingForceApprox(self, x, T, precPhase = None, removeCache = False):
        '''
        Approximate method of driving force calculation
        Assumes equilibrium composition of precipitate phase

        Sampling method is used if driving force is negative

        Steps:
            1. Compute equilibrium and get composition sets for matrix and precipitate phase
            2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
                If not, then resort to sampling method
            3. Compute equilibrium at matrix composition and get chemical potential hyperplane
            4. Driving force is the difference between the free energy of the precipitate (from step 1)
               and the free energy on the chemical potential hyperplane (from step 3) at the precipitate composition

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        cond = self._getConditions(x, T, 0)

        cs_results = self._getCompositionSetsForDF(x, T, cond, precPhase, removeCache=removeCache)
        if cs_results is None:
            return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache)
        
        chemical_potentials, cs_matrix, cs_precip = cs_results
        sortedElements = sorted(list(set(self.elements) - {'VA'}))
        refIndex = sortedElements.index(self.elements[0])

        #Equilibrium at matrix composition for only the parent phase
        comp_sets = self._parentEq[1]
        self._parentEq = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=comp_sets)
        result, comp_sets = self._parentEq

        #Check if equilibrium has converged and chemical potential can be obtained
        #If not, then return None for driving force
        if any(np.isnan(result.chemical_potentials)):
            return None, None

        sortIndices = np.argsort(self.elements[1:-1])
        unsortIndices = np.argsort(sortIndices)

        xP = np.array(cs_precip.X, dtype=np.float64)

        dg = np.sum(xP * result.chemical_potentials) - np.sum(xP * chemical_potentials)

        #Remove reference element
        xP = np.delete(xP, refIndex)

        return np.squeeze(dg), np.squeeze(xP[unsortIndices])

    def _getDrivingForceCurvature(self, x, T, precPhase = None, removeCache = False):
        '''
        Gets driving force from curvature of free energy function
        Assumes small saturation

        Steps:
            1. Compute equilibrium and get composition sets for matrix and precipitate phase
            2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
                If not, then resort to sampling method
            3. Get dmu/dx (free energy curvature)
            4. Compute (x_infty - x_matrix) * dmu/dx * (x_prec - x_matrix)^T
                This does a first (or second?) order approximation of the driving force based off the curvature at x_infty

        Sampling method is used if driving force is negative

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        x = self._process_x(x)
        cond = self._getConditions(x, T, 0)

        cs_results = self._getCompositionSetsForDF(x, T, cond, precPhase, removeCache=removeCache)
        if cs_results is None:
            return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache)
        
        chemical_potentials, cs_matrix, cs_precip = cs_results
        sortedElements = sorted(list(set(self.elements) - {'VA'}))
        refIndex = sortedElements.index(self.elements[0])
        x_matrix = np.array(cs_matrix.X, dtype=np.float64)
        x_precip = np.array(cs_precip.X, dtype=np.float64)

        #If in two phase region, then get curvature of parent phase and use it to calculate driving force ---------------------------------------
        sortIndices = np.argsort(self.elements[1:-1])
        unsortIndices = np.argsort(sortIndices)

        dMudxParent = dMudX(chemical_potentials, cs_matrix, self.elements[0])
        xM = np.delete(x_matrix, refIndex)
        xP = np.delete(x_precip, refIndex)
        xBar = np.array([xP - xM])

        x = np.array(x)[sortIndices]
        xD = np.array([x - xM])

        dg = np.matmul(xD, np.matmul(dMudxParent, xBar.T))

        return np.squeeze(dg), np.squeeze(xP[unsortIndices])

    def _getDrivingForceTangent(self, x, T, precPhase = None, removeCache = False):
        '''
        Gets driving force from parallel tangent calculation

        Steps
            1. Compute equilibrium to get composition sets (or used previous cached CS)
            2. Compute equilibrium of matrix phase at matrix composition
            3. Remove composition and extra free energy from conditions
            4. Add chemical potential for each component to conditions
            5. Compute equilibrium of precipitate phase with new conditions
                The calculated v.GE is the driving force

        This will work for positive and negative driving forces

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other
            
        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        cond = self._getConditions(x, T, self.gOffset)

        if self._compset_cache_df.get(precPhase, None) is None or removeCache:
            #This will calculate local equilibrium for the matrix phase and get the composition set for the precipitate phase
            # _parentEq is set in this function
            cs_results = self._getPrecCompositionSetSamplingDF(x, T, cond, precPhase, removeCache=removeCache)
            if cs_results is None:
                return None, None

            dg, _prec_cs = cs_results
            self._compset_cache_df[precPhase] = [_prec_cs]
        else:
            #If we already have a cache, then we just need equilibrium at the matrix phase
            comp_sets = self._parentEq[1]
            self._parentEq = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=comp_sets)
            
        result, comp_sets = self._parentEq
    
        #Check that equilibrium has converged
        #If not, then return None, None since driving force can't be obtained
        if any(np.isnan(result.chemical_potentials)):
            return None, None

        #Remove element conditions and free extra Gibbs energy conditions
        for e in self.elements:
            if v.X(e) in cond:
                cond.pop(v.X(e))
        if v.GE in cond:
            cond.pop(v.GE)

        #Add chemical potential conditions
        sortedEl = sorted(list(set(self.elements) - set(['VA'])))
        for i in range(len(sortedEl)):
            cond[v.MU(sortedEl[i])] = result.chemical_potentials[i]

        #Solving for local equilibrium on precipitate
        #The fixed conditions are T, P and MU, so this should solve for precipitate composition and GE
        #   Rather than solving for parallel tangent where the driving force is the difference between the chemical potentials of matrix and precipitate phase
        #   This instead solves for the offset in the precipitate energy surface to make the precipitate lie on the chemical potential hyperplane of the matrix phase
        phases, sub_models = self._setupSubModels([precPhase])
        _precEq, _prec_cs = local_equilibrium(self.db, self.elements, phases, cond,
                                                sub_models, self.phase_records, composition_sets=self._compset_cache_df[precPhase])

        #Check if precipitate composition at equilibrium is the matrix composition
        #This can occur in order/disordered models where the miscibility gap is small enough that the parallel tangent can only be found at the matrix composition
        #In this case, switch to sampling for the driving force
        #This still seems to be an improvement over approximate and curvature methods since this occurs after the driving force becomes negative
        prec_comps = np.array(_prec_cs[0].X)
        mat_comps = np.array(comp_sets[0].X)
        if np.allclose(prec_comps, mat_comps, 1e-6):
            self._compset_cache_df[precPhase] = None
            return self._getDrivingForceSampling(x, T, precPhase, removeCache=removeCache)

        self._compset_cache_df[precPhase] = _prec_cs

        #Check that equilibrium has converged
        #If not, then return None, None since driving force can't be obtained
        if any(np.isnan(_precEq.chemical_potentials)):
            return None, None
        
        dg = _precEq.x[0]
        xb = np.array(_prec_cs[0].X)

        sortIndices = np.argsort(self.elements[:-1])
        unsortIndices = np.argsort(sortIndices)
        xb = xb[unsortIndices]

        return np.squeeze(dg), np.squeeze(xb[1:])
    
    def _getCompositionSetsForDF(self, x, T, cond, precPhase = None, removeCache = False):
        '''
        Wrapper for getting composition set from x and T by either global equilibrium or local from a cached composition set

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        chemical_potentials
        cs_matrix - composition set of matrix phase
        cs_precip - composition set of precipitate phase
        '''
        if self._compset_cache_df.get(precPhase, None) is None or removeCache:
            return self._getCompositionSetsEq(x, T, cond, precPhase)
        else:
            return self._getCompositionSetsCache(x, T, cond, precPhase)
    
    def _getCompositionSetsEq(self, x, T, cond, precPhase = None):
        '''
        Gets composition set from x and T by global equilibrium

        Steps
            1. Compute equilibrium at x and T
                If equilibrium did not converge or matrix phase is not stable, then return None
            2. Get composition sets and add to cache
                If precipitate is not stable, the return None
            3. Resolve possible issues with miscibility gaps
            4. Return values

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)

        Returns
        -------
        chemical_potentials
        cs_matrix - composition set of matrix phase
        cs_precip - composition set of precipitate phase
        '''
        #Create cache of composition set if not done so already or if training a surrogate
        #Training points for surrogates may be far apart, so starting from a previous
        #   composition set could give a bad starting position for the minimizer
        #Calculate equilibrium ----------------------------------------------------------------------------------------------------------------------
        wks = self.getEq(x, T, 0, precPhase)
        cs_list_matrix = [cs for cs in wks.get_composition_sets() if cs.phase_record.phase_name == self.phases[0]]
        cs_list_precip = [cs for cs in wks.get_composition_sets() if cs.phase_record.phase_name == precPhase]

        chemical_potentials = np.squeeze(wks.eq.MU)

        if len(cs_list_matrix) == 0:
            return None
        
        if any(np.isnan(chemical_potentials)):
            return None
        
        if len(cs_list_precip) > 0:
            miscMatrix = len(cs_list_matrix) > 1
            miscPrec = len(cs_list_precip) > 1
            cs_matrix = cs_list_matrix[0]
            cs_precip = cs_list_precip[0]
            self._compset_cache_df[precPhase] = [cs_matrix, cs_precip]

            if miscMatrix or miscPrec:
                phases, sub_models = self._setupSubModels([self.phases[0], precPhase])
                result, composition_sets = local_equilibrium(self.db, self.elements, phases, cond,
                                                             sub_models, self.phase_records,
                                                             composition_sets=self._compset_cache_df[precPhase])
                self._compset_cache_df[precPhase] = composition_sets
                chemical_potentials = result.chemical_potentials
                
                cs_matrix = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]][0]
                cs_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
        else:
            return None
        
        return chemical_potentials, cs_matrix, cs_precip
    
    def _getCompositionSetsCache(self, x, T, cond, precPhase = None):
        '''
        Gets composition set from x and T by global equilibrium

        Steps
            1. Compute local equilibrium at x and T using previous composition sets
            2. Get composition sets and update cache
                If equilibrium did not converge, then return None
            3. Return values

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)

        Returns
        -------
        chemical_potentials
        cs_matrix - composition set of matrix phase
        cs_precip - composition set of precipitate phase
        x_precip - composition of precipitate phase
        '''
        phases, sub_models = self._setupSubModels([self.phases[0], precPhase])
        result, composition_sets = local_equilibrium(self.db, self.elements, phases, cond,
                                                         sub_models, self.phase_records,
                                                         composition_sets=self._compset_cache_df[precPhase])
        self._compset_cache_df[precPhase] = composition_sets
        chemical_potentials = result.chemical_potentials
        if any(np.isnan(chemical_potentials)):
            return None
        
        ph = [cs.phase_record.phase_name for cs in composition_sets if cs.NP > 0]
        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            cs_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
            cs_matrix = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]][0]
        else:
            return None
        
        return chemical_potentials, cs_matrix, cs_precip
    
    def _getPrecCompositionSetSamplingDF(self, x, T, cond, precPhase = None, removeCache = False):
        '''
        Gets samples for precipitate phase for use in sampling driving force method and returns driving force and precipitate composition

        This is also use in tangent driving force method for when equilibrium is not (yet) cached

        Steps
            1. Compute local equilibrium at x and T of only the matrix phase
            2. Sample precipitate phase
                If ordered contribution to matrix phase, then sample ordering contribution
                and remove points on the matrix free energy surface
            3. Compute energy difference between precipitate samples and chemical potential hyperplane
            4. Find sample that maximizes energy difference and return sample composition and driving force

        Parameters
        ----------
        x : float or array
            Composition of minor element in bulk matrix phase
            Use float for binary systems
            Use array for multicomponent systems
        T : float
            Temperature in K
        precPhase : str (optional)
            Precipitate phase to consider (default is first precipitate phase in list)
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        driving force - max free energy difference
        precipitate composition - corresponds to max driving force
        '''
        orderTol = -1e-8

        #Equilibrium at matrix composition for only the parent phase
        comp_sets = self._parentEq[1]
        self._parentEq = self.getLocalEq(x, T, 0, [self.phases[0]], composition_sets=comp_sets)
        result, comp_sets = self._parentEq

        if any(np.isnan(result.chemical_potentials)):
            return None
        
        #Sample precipitate phase and get driving force differences at all points -------------------------------------------------------------------
        #Sample points of precipitate phase
        phases, sub_models = self._setupSubModels([precPhase])
        sample_data = self.points_cache.get(precPhase, {})
        prevT = sample_data.get('T', None)
        precPoints = sample_data.get('precipitate_points', None)
        orderedPoints = sample_data.get('ordered_points', None)
        if precPoints is None or prevT != T:
            precPoints = calculate(self.db, self.elements, phases[0], P=101325, T=T, GE=self.gOffset, 
                                   pdens=self.sampling_pDens, model=sub_models, output='GM', 
                                   phase_records=self.phase_records, to_xarray=False)
            if self.orderedPhase[precPhase]:
                orderedPoints = calculate(self.db, self.elements, phases[0], P = 101325, T = T, GE=self.gOffset, 
                                          pdens=self.sampling_pDens, model=sub_models, output='OCM', 
                                          phase_records=self.phase_records, to_xarray=False)
        if not removeCache:
            self.points_cache[precPhase] = {'T': T, 'precipitate_points': precPoints, 'ordered_points': orderedPoints}

        #Get value of chemical potential hyperplane at composition of sampled points
        precComp = np.squeeze(precPoints.X)
        mu = np.array([result.chemical_potentials])
        mult = precComp * mu

        #Difference between the chemical potential hyperplane and the samples points
        #The max driving force is the same as when the chemical potentials of the two phases are parallel
        diff = np.sum(mult, axis=1) - np.squeeze(precPoints.GM)

        #For phases at fixed composition, there will only be 1 set of site fractions
        #So we make this 2d to properly index
        y = np.atleast_2d(np.squeeze(precPoints.Y))
            
        #Find maximum driving force and corresponding composition -----------------------------------------------------------------------------------
        #For phases with order/disorder transition, a filter is applied such that it will only use points that are below the disordered energy surface
        if self.orderedPhase[precPhase]:
            indices = np.squeeze(orderedPoints.OCM) < orderTol
            diff = diff[indices]
            y = y[indices]

        dg = np.amax(diff)
        idx = np.argmax(diff)

        prec_cs = CompositionSet(self.phase_records[precPhase])
        state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
        y = np.array(y[idx][:prec_cs.phase_record.phase_dof], dtype=np.float64)
        prec_cs.update(y, 1, state_variables)

        return dg, prec_cs
