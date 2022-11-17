import numpy as np
from numpy.lib.function_base import diff
import scipy.spatial as sps
from pycalphad import Model, Database, calculate, equilibrium, variables as v
from pycalphad.codegen.callables import build_callables, build_phase_records
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.utils import get_state_variables
from pycalphad.plot.utils import phase_legend
from kawin.Mobility import MobilityModel, interdiffusivity, interdiffusivity_from_diff, inverseMobility, inverseMobility_from_diffusivity, tracer_diffusivity, tracer_diffusivity_from_diff
from kawin.FreeEnergyHessian import dMudX
from kawin.LocalEquilibrium import local_equilibrium
import matplotlib.pyplot as plt
import copy
from tinydb import where

setattr(v, 'GE', v.StateVariable('GE'))

class ExtraGibbsModel(Model):
    '''
    Child of pycalphad Model with extra variable GE
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
        Options are 'approximate' (default), 'sampling' and 'curvature' (not recommended)
    '''

    gOffset = 1      #Small value to add to precipitate phase for when order/disorder models are used

    def __init__(self, database, elements, phases, drivingForceMethod = 'approximate'):
        if isinstance(database, str):
            database = Database(database)
        self.db = database
        self.elements = copy.copy(elements)

        if 'VA' not in self.elements:
            self.elements.append('VA')

        if type(phases) == str:  # check if a single phase was passed as a string instead of a list of phases.
            phases = [phases]
        self.phases = phases
        self.orderedPhase = {phases[i]: False for i in range(1, len(phases))}
        for i in range(1, len(phases)):
            if 'disordered_phase' in self.db.phases[phases[i]].model_hints:
                if self.db.phases[phases[i]].model_hints['disordered_phase'] == self.phases[0]:
                    self.orderedPhase[phases[i]] = True
                    self._forceDisorder(self.phases[0])

        #Build phase models assuming first phase is parent phase and rest of precipitate phases
        #If the same phase is used for matrix and precipitate phase, then force the matrix phase to remove the ordering contribution
        #This may be unnecessary as already disordered phase models will not be affected, but I guess just in case the matrix phase happens to be an ordered solution
        self.models = {self.phases[0]: Model(self.db, self.elements, self.phases[0])}
        self.models[self.phases[0]].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)

        for i in range(1, len(phases)):
            self.models[self.phases[i]] = ExtraGibbsModel(self.db, self.elements, self.phases[i])
            self.models[self.phases[i]].state_variables = sorted([v.T, v.P, v.N, v.GE], key=str)

        self.phase_records = build_phase_records(self.db, self.elements, self.phases,
                                                 self.models[self.phases[0]].state_variables,
                                                 self.models, build_gradients=True, build_hessians=True)

        self.OCMphase_records = {}
        for i in range(1, len(self.phases)):
            if self.orderedPhase[self.phases[i]]:
                self.OCMphase_records[self.phases[i]] = build_phase_records(self.db, self.elements, [self.phases[i]],
                                                                            self.models[self.phases[0]].state_variables,
                                                                            {self.phases[i]: self.models[self.phases[i]]},
                                                                            output='OCM', build_gradients=False, build_hessians=False)


        #Amount of points to sample per degree of freedom
        # sampling_pDens is for when using sampling method in driving force calculations
        # pDens is for equilibrium calculations
        self.sampling_pDens = 2000
        self.pDens = 500

        #Stored variables of last time the class was used
        #This is so that these can be used again if the temperature has not changed since last usage
        self._prevTemperature = None

        #Pertains to parent phase (composition, sampled points, equilibrium calculations)
        self._prevX = None
        self._parentEq = None

        #Pertains to precipitate phases (sampled points)
        self._pointsPrec = {self.phases[i]: None for i in range(1, len(self.phases))}
        self._orderingPoints = {self.phases[i]: None for i in range(1, len(self.phases))}

        self.setDrivingForceMethod(drivingForceMethod)

        self.mobModels = {p: None for p in self.phases}
        self.mobCallables = {p: None for p in self.phases}
        self.diffCallables = {p: None for p in self.phases}
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

            pMob = param_search(param_query_mob)
            pDiff = param_search(param_query_diff)

            if len(pMob) > 0 or len(pDiff) > 0:
                self.mobModels[p] = MobilityModel(self.db, self.elements, p)
                if len(pMob) > 0:
                    self.mobCallables[p] = {}
                    for c in self.phase_records[p].nonvacant_elements:
                        bcp = build_callables(self.db, self.elements, [p], {p: self.mobModels[p]},
                                            parameter_symbols=None, output='mob_'+c, build_gradients=False, build_hessians=False,
                                            additional_statevars=[v.T, v.P, v.N, v.GE])
                        self.mobCallables[p][c] = bcp['mob_'+c]['callables'][p]
                else:
                    self.diffCallables[p] = {}
                    for c in self.phase_records[p].nonvacant_elements:
                        bcp = build_callables(self.db, self.elements, [p], {p: self.mobModels[p]},
                                            parameter_symbols=None, output='diff_'+c, build_gradients=False, build_hessians=False,
                                            additional_statevars=[v.T, v.P, v.N, v.GE])
                        self.diffCallables[p][c] = bcp['diff_'+c]['callables'][p]

        #This applies to all phases since this is typically reflective of quenched-in vacancies
        self.mobility_correction = {A: 1 for A in self.elements}

        #Cached results
        self._compset_cache = {}
        self._compset_cache_df = {}
        self._matrix_cs = None

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

    def clearCache(self):
        '''
        Removes any cached data
        This is intended for surrogate training, where the cached data
        will be removed incase
        '''
        self._compset_cache = {}
        self._compset_cache_df = {}
        self._matrix_cs = None

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
        else:
            raise Exception('Driving force method must be either \'approximate\', \'sampling\' or \'curvature\'')

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
        self._pointsPrec = {self.phases[i]: None for i in range(1, len(self.phases))}
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
        cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        cond[v.P] = 101325
        cond[v.T] = T
        cond[v.GE] = gExtra
        cond[v.N] = 1
        return cond

    def _createCompositionSet(self, eq, state_variables, phase, phase_amounts, idx):
        '''
        Creates a pycalphad CompositionSet from equilibrium results

        Parameters
        ----------
        eq : pycalphad equilibrium result
        state_variables : list
            List of state variables
        phase : str
            Phase to create CompositionSet for
        phase_amounts : list
            Array of floats for phase fraction of each phase
        idx : ndarray
            Index array for the index of phase
        '''
        miscibility = False
        cs = CompositionSet(self.phase_records[phase])
        #If there's a miscibility gap in the matrix phase, then take the largest value
        if len(idx) > 1:
            idx = [idx[np.argmax(phase_amounts[idx])]]
            miscibility = True
        cs.update(eq.Y.isel(vertex=idx).values.ravel()[:cs.phase_record.phase_dof],
                        phase_amounts[idx], state_variables)

        return cs, miscibility

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
        precPhase : str
            Precipitate phase (default is first precipitate)

        Returns
        -------
        Dataset from pycalphad equilibrium results
        '''
        phases = [self.phases[0]]
        if precPhase != -1:
            if precPhase is None:
                precPhase = self.phases[1]
            if isinstance(precPhase, str):
                phases.append(precPhase)
            else:
                phases = [p for p in precPhase]
        phaseRec = {p: self.phase_records[p] for p in phases}

        if not hasattr(x, '__len__'):
            x = [x]

        #Remove first element if x lists composition of all elements
        if len(x) == len(self.elements) - 1:
            x = x[1:]

        cond = self._getConditions(x, T, gExtra+self.gOffset)

        eq = equilibrium(self.db, self.elements, phases, cond, model=self.models, 
                         phase_records=phaseRec, 
                         calc_opts={'pdens': self.pDens})
        return eq

    def getLocalEq(self, x, T, gExtra = 0, precPhase = None, composition_sets = None):
        phases = [self.phases[0]]
        if precPhase != -1:
            if precPhase is None:
                precPhase = self.phases[1]
            if isinstance(precPhase, str):
                phases.append(precPhase)
            else:
                phases = [p for p in precPhase]

        if not hasattr(x, '__len__'):
            x = [x]

        #Remove first element if x lists composition of all elements
        if len(x) == len(self.elements) - 1:
            x = x[1:]

        cond = self._getConditions(x, T, gExtra)
        result, composition_sets = local_equilibrium(self.db, self.elements, phases, cond,
                                                         self.models, self.phase_records,
                                                         composition_sets=composition_sets)
        return result, composition_sets

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

        if hasattr(T, '__len__'):
            for i in range(len(T)):
                dnkj.append(self._interdiffusivitySingle(x[i], T[i], removeCache, phase))
            return np.array(dnkj)
        else:
            return self._interdiffusivitySingle(x, T, removeCache, phase)

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
        if phase is None:
            phase = self.phases[0]

        if not hasattr(x, '__len__'):
            x = [x]

        #Remove first element if x lists composition of all elements
        if len(x) == len(self.elements) - 1:
            x = x[1:]

        cond = self._getConditions(x, T, 0)

        if removeCache:
            self._matrix_cs = None
            self._parentEq, self._matrix_cs = local_equilibrium(self.db, self.elements, [phase], cond,
                                                        self.models, self.phase_records,
                                                        composition_sets=self._matrix_cs)

        cs_matrix = [cs for cs in self._matrix_cs if cs.phase_record.phase_name == phase][0]
        chemical_potentials = self._parentEq.chemical_potentials

        if self.mobCallables[phase] is None:
            Dnkj, _, _ = inverseMobility_from_diffusivity(chemical_potentials, cs_matrix,
                                                                            self.elements[0], self.diffCallables[phase],
                                                                            diffusivity_correction=self.mobility_correction)
        else:
            Dnkj, _, _ = inverseMobility(chemical_potentials, cs_matrix, self.elements[0],
                                                        self.mobCallables[phase],
                                                        mobility_correction=self.mobility_correction)

        if len(x) == 1:
            return Dnkj.ravel()[0]
        else:
            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)
            Dnkj = Dnkj[unsortIndices,:]
            Dnkj = Dnkj[:,unsortIndices]
            return Dnkj


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

        if hasattr(T, '__len__'):
            for i in range(len(T)):
                td.append(self._tracerDiffusivitySingle(x[i], T[i], removeCache, phase))
            return np.array(td)
        else:
            return self._tracerDiffusivitySingle(x, T, removeCache, phase)

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
        if phase is None:
            phase = self.phases[0]

        if not hasattr(x, '__len__'):
            x = [x]

        #Remove first element if x lists composition of all elements
        if len(x) == len(self.elements) - 1:
            x = x[1:]

        cond = self._getConditions(x, T, 0)

        if removeCache:
            self._matrix_cs = None
            self._parentEq, self._matrix_cs = local_equilibrium(self.db, self.elements, [phase], cond,
                                                        self.models, self.phase_records,
                                                        composition_sets=self._matrix_cs)

        cs_matrix = [cs for cs in self._matrix_cs if cs.phase_record.phase_name == phase][0]

        if self.mobCallables[phase] is None:
            #NOTE: This is note tested yet
            Dtrace = tracer_diffusivity_from_diff(cs_matrix, self.diffCallables[phase], diffusivity_correction=self.mobility_correction)
        else:
            Dtrace = tracer_diffusivity(cs_matrix, self.mobCallables[phase], mobility_correction=self.mobility_correction)

        sortIndices = np.argsort(self.elements[:-1])
        unsortIndices = np.argsort(sortIndices)

        Dtrace = Dtrace[unsortIndices]

        return Dtrace

    def getDrivingForce(self, x, T, precPhase = None, returnComp = False, training = False):
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
        returnComp : bool (optional)
            Whether to return composition of precipitate (defaults to False)

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative or returnComp is False
        '''
        if hasattr(T, '__len__'):
            dgArray = []
            compArray = []
            for i in range(len(T)):
                dg, comp = self._drivingForce(x[i], T[i], precPhase, returnComp, training)
                dgArray.append(dg)
                compArray.append(comp)
            dgArray = np.array(dgArray)
            compArray = np.array(compArray)
            return dgArray, compArray
        else:
            return self._drivingForce(x, T, precPhase, returnComp, training)

    def _getDrivingForceSampling(self, x, T, precPhase = None, returnComp = False, training = False):
        '''
        Gets driving force for nucleation by sampling

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
        returnComp : bool (optional)
            Whether to return composition of precipitate (defaults to False)

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative or returnComp is False
        '''
        precPhase = self.phases[1] if precPhase is None else precPhase

        #Calculate equilibrium with only the parent phase -------------------------------------------------------------------------------------------
        if not hasattr(x, '__len__'):
            x = [x]
        cond = self._getConditions(x, T, 0)
        self._prevX = x

        #Equilibrium at matrix composition for only the parent phase
        self._parentEq, self._matrix_cs = local_equilibrium(self.db, self.elements, [self.phases[0]], cond,
                                                            self.models, self.phase_records,
                                                            composition_sets = self._matrix_cs)

        #Remove cache when training
        if training:
            self._matrix_cs = None

        #Check if equilibrium has converged and chemical potential can be obtained
        #If not, then return None for driving force
        if any(np.isnan(self._parentEq.chemical_potentials)):
            return None, None

        #Sample precipitate phase and get driving force differences at all points -------------------------------------------------------------------
        #Sample points of precipitate phase
        if self._pointsPrec[precPhase] is None or self._prevTemperature != T:
            self._pointsPrec[precPhase] = calculate(self.db, self.elements, precPhase, P = 101325, T = T, GE=self.gOffset, pdens = self.sampling_pDens, model=self.models, output='GM', phase_records=self.phase_records)
            if self.orderedPhase[precPhase]:
                self._orderingPoints[precPhase] = calculate(self.db, self.elements, precPhase, P = 101325, T = T, GE=self.gOffset, pdens = self.sampling_pDens, model=self.models, output='OCM', phase_records=self.OCMphase_records[precPhase])
            self._prevTemperature = T

        #Get value of chemical potential hyperplane at composition of sampled points
        precComp = self._pointsPrec[precPhase].X.values.ravel()
        precComp = precComp.reshape((int(len(precComp) / (len(self.elements) - 1)), len(self.elements) - 1))
        mu = np.array([self._parentEq.chemical_potentials])
        mult = precComp * mu

        #Difference between the chemical potential hyperplane and the samples points
        #The max driving force is the same as when the chemical potentials of the two phases are parallel
        diff = np.sum(mult, axis=1) - self._pointsPrec[precPhase].GM.values.ravel()

        #Find maximum driving force and corresponding composition -----------------------------------------------------------------------------------
        #For phases with order/disorder transition, a filter is applied such that it will only use points that are below the disordered energy surface
        if self.orderedPhase[precPhase]:
            diff = diff[self._orderingPoints[precPhase].OCM.values.ravel() < -1e-8]

        if returnComp:
            g = np.amax(diff)

            if g < 0:
                return g, None
            else:
                #Get all compositions for each point and grab the composition corresponding to max driving force
                #For ordered compounds, the composition needs to be filtered to remove any disordered points (corresponding to matrix phase)
                #This only has to be done for composition since 'diff' is already filtered
                if len(x) == 1:
                    betaX = self._pointsPrec[precPhase].X.sel(component=self.elements[1]).values.ravel()
                    if self.orderedPhase[precPhase]:
                        betaX = betaX[self._orderingPoints[precPhase].OCM.values < -1e-8]
                    comp = betaX[np.argmax(diff)]
                else:
                    betaX = [self._pointsPrec[precPhase].X.sel(component=self.elements[i+1]).values for i in range(len(x))]
                    if self.orderedPhase[precPhase]:
                        for i in range(len(x)):
                            betaX[i] = betaX[i][self._orderingPoints[precPhase].OCM.values < -1e-8]
                    comp = [betaX[i][np.argmax(diff)] for i in range(len(x))]

                return g, comp
        else:
            return np.amax(diff), None

    def _getDrivingForceApprox(self, x, T, precPhase = None, returnComp = False, training = False):
        '''
        Approximate method of driving force calculation
        Assumes equilibrium composition of precipitate phase

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
        returnComp : bool (optional)
            Whether to return composition of precipitate (defaults to False)

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative or returnComp is False
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        if not hasattr(x, '__len__'):
            x = [x]
        cond = self._getConditions(x, T, 0)
        self._prevX = x

        #Create cache of composition set if not done so already or if training a surrogate
        #Training points for surrogates may be far apart, so starting from a previous
        #   composition set could give a bad starting position for the minimizer
        if self._compset_cache_df.get(precPhase, None) is None or training:
            #Calculate equilibrium ----------------------------------------------------------------------------------------------------------------------
            eq = self.getEq(x, T, 0, precPhase)
            #Cast values in state_variables to double for updating composition sets
            state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
            stable_phases = eq.Phase.values.ravel()
            phase_amounts = eq.NP.values.ravel()
            matrix_idx = np.where(stable_phases == self.phases[0])[0]
            precip_idx = np.where(stable_phases == precPhase)[0]
            chemical_potentials = eq.MU.values.ravel()
            x_precip = eq.isel(vertex=precip_idx).X.values.ravel()

            #If matrix phase is not stable, then use sampling method
            #   This may occur during surrogate training of interfacial composition,
            #   where we're trying to calculate the driving force at the precipitate composition
            #   In this case, the conditions will be at th precipitate composition which can result in
            #   only that phase being stable
            if len(matrix_idx) == 0:
                return self._getDrivingForceSampling(x, T, precPhase, returnComp)

            #Test that precipitate phase is stable and that we're not training a surrogate
            #If not, then there's no composition set to cache
            if len(precip_idx) > 0:
                cs_matrix, miscMatrix = self._createCompositionSet(eq, state_variables, self.phases[0], phase_amounts, matrix_idx)
                cs_precip, miscPrec = self._createCompositionSet(eq, state_variables, precPhase, phase_amounts, precip_idx)
                x_precip = np.array(cs_precip.X)

                composition_sets = [cs_matrix, cs_precip]
                self._compset_cache_df[precPhase] = composition_sets

                if miscMatrix or miscPrec:
                    result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                            self.models, self.phase_records,
                                                            composition_sets=self._compset_cache_df[precPhase])
                    self._compset_cache_df[precPhase] = composition_sets
                    chemical_potentials = result.chemical_potentials
                    cs_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
                    x_precip = np.array(cs_precip.X)

            ph = np.unique(stable_phases[stable_phases != ''])
            ele = eq.component.values.ravel()
        else:
            result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                         self.models, self.phase_records,
                                                         composition_sets=self._compset_cache_df[precPhase])
            self._compset_cache_df[precPhase] = composition_sets
            chemical_potentials = result.chemical_potentials
            ph = [cs.phase_record.phase_name for cs in composition_sets if cs.NP > 0]
            if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                cs_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
                x_precip = np.array(cs_precip.X)
                ele = list(cs_precip.phase_record.nonvacant_elements)

        #Check that equilibrium has converged
        #If not, then return None, None since driving force can't be obtained
        if any(np.isnan(chemical_potentials)):
            return None, None

        #If in two phase region, then calculate equilibrium using only parent phase and find free energy difference between chemical potential and free energy of preciptiate
        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            for i in range(len(ele)):
                if ele[i] == self.elements[0]:
                    refIndex = i
                    break

            #Equilibrium at matrix composition for only the parent phase
            self._parentEq, self._matrix_cs = local_equilibrium(self.db, self.elements, [self.phases[0]], cond,
                                                                self.models, self.phase_records,
                                                                composition_sets=self._matrix_cs)


            #Remove caching if training surrogate in case training points are far apart
            if training:
                self._matrix_cs = None

            #Check if equilibrium has converged and chemical potential can be obtained
            #If not, then return None for driving force
            if any(np.isnan(self._parentEq.chemical_potentials)):
                return None, None

            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)

            xP = x_precip

            dg = np.sum(xP * self._parentEq.chemical_potentials) - np.sum(xP * chemical_potentials)

            #Remove reference element
            xP = np.delete(xP, refIndex)

            if returnComp:
                if len(x) == 1:
                    return dg.ravel()[0], xP[unsortIndices][0]
                else:
                    return dg.ravel()[0], xP[unsortIndices]
            else:
                return dg.ravel()[0], None
        else:
            #If driving force is negative, then use sampling method ---------------------------------------------------------------------------------
            return self._getDrivingForceSampling(x, T, precPhase, returnComp)

    def _getDrivingForceCurvature(self, x, T, precPhase = None, returnComp = False, training = False):
        '''
        Gets driving force from curvature of free energy function
        Assumes small saturation

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
        returnComp : bool (optional)
            Whether to return composition of precipitate (defaults to False)

        Returns
        -------
        (driving force, precipitate composition)
        Driving force is positive if precipitate can form
        Precipitate composition will be None if driving force is negative or returnComp is False
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        if not hasattr(x, '__len__'):
            x = [x]
        cond = self._getConditions(x, T, 0)
        self._prevX = x

        #Create cache of composition set if not done so already or if training a surrogate
        #Training points for surrogates may be far apart, so starting from a previous
        #   composition set could give a bad starting position for the minimizer
        if self._compset_cache_df.get(precPhase, None) is None or training:
            #Calculate equilibrium ----------------------------------------------------------------------------------------------------------------------
            eq = self.getEq(x, T, 0, precPhase)
            #Cast values in state_variables to double for updating composition sets
            state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
            stable_phases = eq.Phase.values.ravel()
            phase_amounts = eq.NP.values.ravel()
            matrix_idx = np.where(stable_phases == self.phases[0])[0]
            precip_idx = np.where(stable_phases == precPhase)[0]
            chemical_potentials = eq.MU.values.ravel()
            x_precip = eq.isel(vertex=precip_idx).X.values.ravel()
            x_matrix = eq.isel(vertex=matrix_idx).X.values.ravel()

            #If matrix phase is not stable, then use sampling method
            #   This may occur during surrogate training of interfacial composition,
            #   where we're trying to calculate the driving force at the precipitate composition
            #   In this case, the conditions will be at th precipitate composition which can result in
            #   only that phase being stable
            if len(matrix_idx) == 0:
                return self._getDrivingForceSampling(x, T, precPhase, returnComp)

            #Test that precipitate phase is stable and that we're not training a surrogate
            #If not, then there's no composition set to cache
            if len(precip_idx) > 0:
                cs_matrix, miscMatrix = self._createCompositionSet(eq, state_variables, self.phases[0], phase_amounts, matrix_idx)
                cs_precip, miscPrec = self._createCompositionSet(eq, state_variables, precPhase, phase_amounts, precip_idx)
                x_matrix = np.array(cs_matrix.X)
                x_precip = np.array(cs_precip.X)
                
                composition_sets = [cs_matrix, cs_precip]
                self._compset_cache_df[precPhase] = composition_sets

                if miscMatrix or miscPrec:
                    result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                         self.models, self.phase_records,
                                                         composition_sets=self._compset_cache_df[precPhase])
                    self._compset_cache_df[precPhase] = composition_sets
                    chemical_potentials = result.chemical_potentials
                    cs_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
                    x_precip = np.array(cs_precip.X)

                    cs_matrix = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]][0]
                    x_matrix = np.array(cs_matrix.X)

            ph = np.unique(stable_phases[stable_phases != ''])
            ele = eq.component.values.ravel()
        else:
            result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                         self.models, self.phase_records,
                                                         composition_sets=self._compset_cache_df[precPhase])
            self._compset_cache_df[precPhase] = composition_sets
            chemical_potentials = result.chemical_potentials
            ph = [cs.phase_record.phase_name for cs in composition_sets if cs.NP > 0]
            if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                cs_precip = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
                x_precip = np.array(cs_precip.X)

                cs_matrix = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]][0]
                x_matrix = np.array(cs_matrix.X)

                ele = list(cs_precip.phase_record.nonvacant_elements)

        #Check that equilibrium has converged
        #If not, then return None, None since driving force can't be obtained
        if any(np.isnan(chemical_potentials)):
            return None, None

        if not hasattr(x, '__len__'):
            x = [x]

        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            for i in range(len(ele)):
                if ele[i] == self.elements[0]:
                    refIndex = i
                    break

            #If in two phase region, then get curvature of parent phase and use it to calculate driving force ---------------------------------------
            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)

            dMudxParent = dMudX(chemical_potentials, composition_sets[0], self.elements[0])
            xM = np.delete(x_matrix, refIndex)

            xP = np.delete(x_precip, refIndex)
            xBar = np.array([xP - xM])

            x = np.array(x)[sortIndices]
            xD = np.array([x - xM])

            dg = np.matmul(xD, np.matmul(dMudxParent, xBar.T))

            if returnComp:
                if len(x) == 1:
                    return dg.ravel()[0], xP[unsortIndices][0]
                else:
                    return dg.ravel()[0], xP[unsortIndices]
            else:
                return dg.ravel()[0], None
        else:
            #If driving force is negative, then use sampling method ---------------------------------------------------------------------------------
            return self._getDrivingForceSampling(x, T, precPhase, returnComp)

class BinaryThermodynamics (GeneralThermodynamics):
    '''
    Class for defining driving force and interfacial composition functions
    for a binary system using pyCalphad and thermodynamic databases

    Parameters
    ----------
    database : str
        File name for database
    elements : list
        Elements to consider
        Note: reference element must be the first index in the list
    phases : list
        Phases involved
        Note: matrix phase must be first index in the list
    drivingForceMethod : str (optional)
        Method used to calculate driving force
        Options are 'approximate' (default), 'sampling' and 'curvature' (not recommended)
    interfacialCompMethod: str (optional)
        Method used to calculate interfacial composition
        Options are 'eq' (default) and 'curvature' (not recommended)
    '''
    def __init__(self, database, elements, phases, drivingForceMethod = 'approximate', interfacialCompMethod = 'equilibrium'):
        super().__init__(database, elements, phases, drivingForceMethod)

        if self.elements[1] < self.elements[0]:
            self.reverse = True
        else:
            self.reverse = False

        #Guess composition for when finding tieline
        self._guessComposition = {self.phases[i]: (0, 1, 0.1) for i in range(1, len(self.phases))}

        self.setInterfacialMethod(interfacialCompMethod)


    def setInterfacialMethod(self, interfacialCompMethod):
        '''
        Changes method for caluclating interfacial composition

        Parameters
        ----------
        interfacialCompMethod - str
            Options are ['equilibrium', 'curvature']
        '''
        if interfacialCompMethod == 'equilibrium':
            self._interfacialComposition = self._interfacialCompositionFromEq
        elif interfacialCompMethod == 'curvature':
            self._interfacialComposition = self._interfacialCompositionFromCurvature
        else:
            raise Exception('Interfacial composition method must be either \'equilibrium\' or \'curvature\'')

    def setGuessComposition(self, conditions):
        '''
        Sets initial composition when calculating equilibrium for interfacial energy

        Parameters
        ----------
        conditions : float, tuple or dict
            Guess composition(s) to solve equilibrium for
            This should encompass the region where a tieline can be found
            between the matrix and precipitate phases
            Options:    float - will set to all precipitate phases
                        tuple - (min, max dx) will set to all precipitate phases
                        dictionary {phase name: scalar or tuple}
        '''
        if isinstance(conditions, dict):
            #Iterating over conditions dictionary in case not all precipitate phases are listed
            for p in conditions:
                self._guessComposition[p] = conditions[p]
        #If not dictionary, then set to all phases
        else:
            for i in range(1, len(self.phases)):
                self._guessComposition[self.phases[i]] = conditions

    def getInterfacialComposition(self, T, gExtra = 0, precPhase = None):
        '''
        Gets interfacial composition accounting for Gibbs-Thomson effect

        Parameters
        ----------
        T : float or array
            Temperature in K
        gExtra : float or array (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        Note: for multiple conditions, only gExtra has to be an array
            This will calculate compositions for multiple gExtra at the input Temperature

            If T is also an array, then T and gExtra must be the same length
            where each index will pertain to a single condition

        Returns
        -------
        (parent composition, precipitate composition)
        Both will be either float or array based off shape of gExtra
        Will return (None, None) if precipitate is unstable
        '''
        if hasattr(gExtra, '__len__'):
            if not hasattr(T, '__len__'):
                caArray, cbArray = self._interfacialComposition(T, gExtra, precPhase)
            else:
                #If T is also an array, then iterate through T and gExtra
                #Otherwise, pycalphad will create a cartesian product of the two
                caArray = []
                cbArray = []
                for i in range(len(gExtra)):
                    ca, cb = self._interfacialComposition(T[i], gExtra[i], precPhase)
                    caArray.append(ca)
                    cbArray.append(cb)
                caArray = np.array(caArray)
                cbArray = np.array(cbArray)

            return caArray, cbArray
        else:
            return self._interfacialComposition(T, gExtra, precPhase)


    def _interfacialCompositionFromEq(self, T, gExtra = 0, precPhase = None):
        '''
        Gets interfacial composition by calculating equilibrum with Gibbs-Thomson effect

        Parameters
        ----------
        T : float
            Temperature in K
        gExtra : float (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        Returns
        -------
        (parent composition, precipitate composition)
        Both will be either float or array based off shape of gExtra
        Will return (None, None) if precipitate is unstable
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        if hasattr(gExtra, '__len__'):
            gExtra = np.array(gExtra)
        else:
            gExtra = np.array([gExtra])
        gExtra += self.gOffset

        #Compute equilibrium at guess composition
        cond = {v.X(self.elements[1]): self._guessComposition[precPhase], v.T: T, v.P: 101325, v.GE: gExtra}
        eq = equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond, model=self.models,
                        phase_records={self.phases[0]: self.phase_records[self.phases[0]], precPhase: self.phase_records[precPhase]},
                        calc_opts = {'pdens': self.pDens})

        xParentArray = np.zeros(len(gExtra))
        xPrecArray = np.zeros(len(gExtra))
        for g in range(len(gExtra)):
            eqG = eq.where(eq.GE == gExtra[g], drop=True)
            gm = eqG.GM.values.ravel()
            for i in range(len(gm)):
                eqSub = eqG.where(eqG.GM == gm[i], drop=True)

                ph = eqSub.Phase.values.ravel()
                ph = ph[ph != '']

                #Check if matrix and precipitate phase are stable, and check if there's no miscibility gaps
                if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                    #Get indices for each phase
                    eqPa = eqSub.where(eqSub.Phase == self.phases[0], drop=True)
                    eqPr = eqSub.where(eqSub.Phase == precPhase, drop=True)

                    cParent = eqPa.X.values.ravel()
                    cPrec = eqPr.X.values.ravel()

                    #Get composition of element, use element index of 1 is the parent index is first alphabetically
                    if self.reverse:
                        xParent = cParent[0]
                        xPrec = cPrec[0]
                    else:
                        xParent = cParent[1]
                        xPrec = cPrec[1]

                    xParentArray[g] = xParent
                    xPrecArray[g] = xPrec
                    break
            if xParentArray[g] == 0:
                xParentArray[g] = -1
                xPrecArray[g] = -1

        if len(gExtra) == 1:
            return xParentArray[0], xPrecArray[0]
        else:
            return xParentArray, xPrecArray


    def _interfacialCompositionFromCurvature(self, T, gExtra = 0, precPhase = None):
        '''
        Gets interfacial composition using free energy curvature
        G''(x - xM)(xP-xM) = 2*y*V/R

        Parameters
        ----------
        T : float
            Temperature in K
        gExtra : float (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        Returns
        -------
        (parent composition, precipitate composition)
        Both will be either float or array based off shape of gExtra
        Will return (None, None) if precipitate is unstable
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        if hasattr(gExtra, '__len__'):
            gExtra = np.array(gExtra)
        else:
            gExtra = np.array([gExtra])

        #Compute equilibrium at guess composition
        cond = {v.X(self.elements[1]): self._guessComposition[precPhase], v.T: T, v.P: 101325, v.GE: self.gOffset}
        eq = equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond, model=self.models,
                        phase_records={self.phases[0]: self.phase_records[self.phases[0]], precPhase: self.phase_records[precPhase]},
                        calc_opts = {'pdens': self.pDens})

        gm = eq.GM.values.ravel()
        for g in gm:
            eqSub = eq.where(eq.GM == g, drop=True)

            ph = eqSub.Phase.values.ravel()
            ph = ph[ph != '']

            #Check if matrix and precipitate phase are stable, and check if there's no miscibility gaps
            if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                #Cast values in state_variables to double for updating composition sets
                state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
                stable_phases = eqSub.Phase.values.ravel()
                phase_amounts = eqSub.NP.values.ravel()
                matrix_idx = np.where(stable_phases == self.phases[0])[0]
                precip_idx = np.where(stable_phases == precPhase)[0]

                cs_matrix = CompositionSet(self.phase_records[self.phases[0]])
                if len(matrix_idx) > 1:
                    matrix_idx = [matrix_idx[np.argmax(phase_amounts[matrix_idx])]]
                cs_matrix.update(eqSub.Y.isel(vertex=matrix_idx).values.ravel()[:cs_matrix.phase_record.phase_dof],
                                    phase_amounts[matrix_idx], state_variables)
                cs_precip = CompositionSet(self.phase_records[precPhase])
                if len(precip_idx) > 1:
                    precip_idx = [precip_idx[np.argmax(phase_amounts[precip_idx])]]
                cs_precip.update(eqSub.Y.isel(vertex=precip_idx).values.ravel()[:cs_precip.phase_record.phase_dof],
                                    phase_amounts[precip_idx], state_variables)

                chemical_potentials = eqSub.MU.values.ravel()
                cPrec = eqSub.isel(vertex=precip_idx).X.values.ravel()
                cParent = eqSub.isel(vertex=matrix_idx).X.values.ravel()

                dMudxParent = dMudX(chemical_potentials, cs_matrix, self.elements[0])
                dMudxPrec = dMudX(chemical_potentials, cs_precip, self.elements[0])

                #Get composition of element, use element index of 1 is the parent index is first alphabetically
                if self.reverse:
                    xParentEq = cParent[0]
                    xPrecEq = cPrec[0]
                else:
                    xParentEq = cParent[1]
                    xPrecEq = cPrec[1]

                dMudxParent = dMudxParent[0,0]
                dMudxPrec = dMudxPrec[0,0]

                if dMudxParent != 0:
                    xParent = gExtra / dMudxParent / (xPrecEq - xParentEq) + xParentEq
                else:
                    xParent = xParentEq

                if dMudxPrec != 0:
                    xPrec = dMudxParent * (xParent - xParentEq) / dMudxPrec + xPrecEq
                else:
                    xPrec = xPrecEq

                xParent[xParent < 0] = 0
                xParent[xParent > 1] = 1
                xPrec[xPrec < 0] = 0
                xPrec[xPrec > 1] = 1

                if len(gExtra) == 1:
                    return xParent[0], xPrec[0]
                else:
                    return xParent, xPrec

        if len(gExtra) == 1:
            return -1, -1
        else:
            return -1*np.ones(len(gExtra)), -1*np.ones(len(gExtra))


    def plotPhases(self, ax, T, gExtra = 0, plotGibbsOffset = False, *args, **kwargs):
        '''
        Plots sampled points from the parent and precipitate phase

        Parameters
        ----------
        ax : Axis
        T : float
            Temperature in K
        gExtra : float (optional)
            Extra contributions to the Gibbs free energy of precipitate
            Defaults to 0
        plotGibbsOffset : bool (optional)
            If True and gExtra is not 0, the sampled points of the
                precipitate phase will be plotted twice with gExtra and
                with no extra Gibbs free energy contributions
            Defualts to False
        '''
        points = calculate(self.db, self.elements, self.phases[0], P=101325, T=T, GE=0, model=self.models, phase_records=self.phase_records, output='GM')
        ax.scatter(points.X.sel(component=self.elements[1]), points.GM / 1000, label=self.phases[0], *args, **kwargs)

        #Add gExtra to precipitate phase
        for i in range(1, len(self.phases)):
            points = calculate(self.db, self.elements, self.phases[i], P=101325, T=T, GE=0, model=self.models, phase_records=self.phase_records, output='GM')
            ax.scatter(points.X.sel(component=self.elements[1]), (points.GM + gExtra) / 1000, label=self.phases[i], *args, **kwargs)

            #Plot non-offset precipitate phase
            if plotGibbsOffset and gExtra != 0:
                ax.scatter(points.X.sel(component=self.elements[1]), points.GM / 1000, color='silver', alpha=0.3, *args, **kwargs)

        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_xlabel('Composition ' + self.elements[1])
        ax.set_ylabel('Gibbs Free Energy (kJ/mol)')


class MulticomponentThermodynamics (GeneralThermodynamics):
    '''
    Class for defining driving force and (possibly) interfacial composition functions
    for a multicomponent system using pyCalphad and thermodynamic databases

    Parameters
    ----------
    database : str
        File name for database
    elements : list
        Elements to consider
        Note: reference element must be the first index in the list
    phases : list
        Phases involved
        Note: matrix phase must be first index in the list
    drivingForceMethod : str (optional)
        Method used to calculate driving force
        Options are 'approximate' (default), 'sampling' and 'curvature' (not recommended)
    '''
    def __init__(self, database, elements, phases, drivingForceMethod = 'approximate'):
        super().__init__(database, elements, phases, drivingForceMethod)

        #Previous variables for curvature terms
        #Near saturation, pycalphad may detect only a single phase (if sampling density is too low)
        #When this occurs, this will assume that the system is on the same tie-line and
        #use the previously calculated values
        self._prevDc = {p: None for p in phases[1:]}
        self._prevMc = {p: None for p in phases[1:]}
        self._prevGba = {p: None for p in phases[1:]}
        self._prevBeta = {p: None for p in phases[1:]}
        self._prevCa = {p: None for p in phases[1:]}
        self._prevCb = {p: None for p in phases[1:]}

    def getInterfacialComposition(self, x, T, gExtra = 0, precPhase = None):
        '''
        Gets interfacial composition by calculating equilibrum with Gibbs-Thomson effect

        Parameters
        ----------
        T : float or array
            Temperature in K
        gExtra : float or array (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        Note: for multiple conditions, only gExtra has to be an array
            This will calculate compositions for multiple gExtra at the input Temperature

            If T is also an array, then T and gExtra must be the same length
            where each index will pertain to a single condition

        Returns
        -------
        (parent composition, precipitate composition)
        Both will be either float or array based off shape of gExtra
        Will return (None, None) if precipitate is unstable
        '''
        if hasattr(gExtra, '__len__'):
            if not hasattr(T, '__len__'):
                T = T * np.ones(len(gExtra))

            caArray = []
            cbArray = []
            for i in range(len(gExtra)):
                ca, cb = self._interfacialComposition(x, T[i], gExtra[i], precPhase)
                caArray.append(ca)
                cbArray.append(cb)
            caArray = np.array(caArray)
            cbArray = np.array(cbArray)
            return caArray, cbArray
        else:
            return self._interfacialComposition(x, T, gExtra, precPhase)


    def _interfacialComposition(self, x, T, gExtra = 0, precPhase = None):
        '''
        Gets interfacial composition, will return None, None if composition is in single phase region

        Parameters
        ----------
        T : float
            Temperature in K
        gExtra : float (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        Returns
        -------
        (parent composition, precipitate composition)
        Both will be either float or array based off shape of gExtra
        Will return (None, None) if precipitate is unstable
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        eq = self.getEq(x, T, gExtra, precPhase)

        #Check for convergence, return None if not converged
        if np.any(np.isnan(eq.MU.values.ravel())):
            return None, None

        ph = eq.Phase.values.ravel()
        ph = ph[ph != '']

        #Check if matrix and precipitate phase are stable, and check if there's no miscibility gaps
        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            sortIndices = np.argsort(self.elements[:-1])
            unsortIndices = np.argsort(sortIndices)

            mu = eq.MU.values.ravel()
            mu = mu[unsortIndices]

            eqPh = eq.where(eq.Phase == self.phases[0], drop=True)
            xM = eqPh.X.values.ravel()
            xM = xM[unsortIndices]

            eqPh = eq.where(eq.Phase == precPhase, drop=True)
            xP = eqPh.X.values.ravel()
            xP = xP[unsortIndices]

            return xM, xP

        return None, None

    def _curvatureFactorFromEq(self, chemical_potentials, composition_sets, precPhase=None, training = False):
        '''
        Curvature factor (from Phillipes and Voorhees - 2013)

        Parameters
        ----------
        chemical_potentials : 1-D float64 array
        composition_sets : List[pycalphad.composition_set.CompositionSet]
        precPhase : str (optional)
            Precipitate phase (defaults to first precipitate in list)
        training : bool (optional)
            For surrogate training, will return None rather than previous results
            if 2-phase region is not detected in equilibrium calculation

        Returns
        -------
        {D-1 dCbar / dCbar^T M-1 dCbar} - for calculating interfacial composition of matrix
        {1 / dCbar^T M-1 dCbar} - for calculating growth rate
        {Gb^-1 Ga} - for calculating precipitate composition
        beta - Impingement rate
        Ca - interfacial composition of matrix phase
        Cb - interfacial composition of precipitate phase

        Will return (None, None, None, None, None, None) if single phase
        '''
        if precPhase is None:
            precPhase = self.phases[1]

        #Check if input equilibrium has converged
        if np.any(np.isnan(chemical_potentials)):
            if training:
                return None, None, None, None, None, None
            else:
                print('Warning: equilibrum was not able to be solved for, using results of previous calculation')
                return self._prevDc[precPhase], self._prevMc[precPhase], self._prevGba[precPhase], self._prevBeta[precPhase], self._prevCa[precPhase], self._prevCb[precPhase]

        ele = list(composition_sets[0].phase_record.nonvacant_elements)
        refIndex = ele.index(self.elements[0])

        ph = [cs.phase_record.phase_name for cs in composition_sets]

        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)

            matrix_cs = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]][0]

            if self.mobCallables[self.phases[0]] is None:
                Dnkj, dMudxParent, invMob = inverseMobility_from_diffusivity(chemical_potentials, matrix_cs,
                                                                             self.elements[0], self.diffCallables[self.phases[0]],
                                                                             diffusivity_correction=self.mobility_correction)

                #NOTE: This is note tested yet
                Dtrace = tracer_diffusivity_from_diff(matrix_cs, self.diffCallables[self.phases[0]], diffusivity_correction=self.mobility_correction)
            else:
                Dnkj, dMudxParent, invMob = inverseMobility(chemical_potentials, matrix_cs, self.elements[0],
                                                            self.mobCallables[self.phases[0]],
                                                            mobility_correction=self.mobility_correction)
                Dtrace = tracer_diffusivity(matrix_cs, self.mobCallables[self.phases[0]], mobility_correction=self.mobility_correction)

            xMFull = np.array(matrix_cs.X)
            xM = np.delete(xMFull, refIndex)

            precip_cs = [cs for cs in composition_sets if cs.phase_record.phase_name == precPhase][0]
            dMudxPrec = dMudX(chemical_potentials, precip_cs, self.elements[0])
            xPFull = np.array(precip_cs.X)
            xP = np.delete(xPFull, refIndex)
            xBarFull = np.array([xPFull - xMFull])
            xBar = np.array([xP - xM])

            num = np.matmul(np.linalg.inv(Dnkj), xBar.T).flatten()

            #Denominator should be a scalar since its V * M * V^T
            den = np.matmul(xBar, np.matmul(invMob, xBar.T)).flatten()[0]

            if np.linalg.matrix_rank(dMudxPrec) == dMudxPrec.shape[0]:
                Gba = np.matmul(np.linalg.inv(dMudxPrec), dMudxParent)
                Gba = Gba[unsortIndices,:]
                Gba = Gba[:,unsortIndices]
            else:
                Gba = np.zeros(dMudxPrec.shape)

            betaNum = xBarFull**2
            betaDen = Dtrace * xMFull.flatten()
            bsum = np.sum(betaNum / betaDen)
            if bsum == 0:
                beta = self._prevBeta[precPhase]
            else:
                beta = 1 / bsum

            self._prevDc[precPhase] = num[unsortIndices] / den
            self._prevMc[precPhase] = 1 / den
            self._prevGba[precPhase] = Gba
            self._prevBeta[precPhase] = beta
            self._prevCa[precPhase] = xM[unsortIndices]
            self._prevCb[precPhase] = xP[unsortIndices]

            return self._prevDc[precPhase], self._prevMc[precPhase], self._prevGba[precPhase], self._prevBeta[precPhase], self._prevCa[precPhase], self._prevCb[precPhase]
        else:
            if training:
                return None, None, None, None, None, None
            else:
                #print('Warning: only a single phase detected in equilibrium, using results of previous calculation')
                #return self._prevDc[precPhase], self._prevMc[precPhase], self._prevGba[precPhase], self._prevBeta[precPhase], self._prevCa[precPhase], self._prevCb[precPhase]

                #If two-phase equilibrium is not found, then the temperature may have changed to where the precipitate is unstable
                #Return None in this case
                return None, None, None, self._prevBeta[precPhase], None, None


    def curvatureFactor(self, x, T, precPhase = None, training = False):
        '''
        Curvature factor (from Phillipes and Voorhees - 2013) from composition and temperature
        This is the same as curvatureFactorEq, but will calculate equilibrium from x and T first

        Parameters
        ----------
        x : array
            Composition of solutes
        T : float
            Temperature
        precPhase : str (optional)
            Precipitate phase (defaults to first precipitate in list)

        Returns
        -------
        {D-1 dCbar / dCbar^T M-1 dCbar} - for calculating interfacial composition of matrix
        {1 / dCbar^T M-1 dCbar} - for calculating growth rate
        {Gb^-1 Ga} - for calculating precipitate composition
        beta - Impingement rate
        Ca - interfacial composition of matrix phase
        Cb - interfacial composition of precipitate phase

        Will return (None, None, None, None, None, None) if single phase
        '''
        if precPhase is None:
            precPhase = self.phases[1]
        if not hasattr(x, '__len__'):
            x = [x]

        #Remove first element if x lists composition of all elements
        if len(x) == len(self.elements) - 1:
            x = x[1:]
        cond = self._getConditions(x, T, 0)

        #Perform equilibrium from scratch if cache not set or when training surrogate
        if self._compset_cache.get(precPhase, None) is None or training:
            eq = self.getEq(x, T, 0, precPhase)
            state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
            stable_phases = eq.Phase.values.ravel()
            phase_amounts = eq.NP.values.ravel()
            matrix_idx = np.where(stable_phases == self.phases[0])[0]
            precip_idx = np.where(stable_phases == precPhase)[0]

            #If matrix phase is not stable (why?), then return previous values
            #Curvature can't be calculated if matrix phase isn't present
            if len(matrix_idx) == 0:
                if training:
                    return None, None, None, None, None, None
                else:
                    print('Warning: matrix phase not detected, using results of previous calculation')
                    return self._prevDc, self._prevMc, self._prevGba, self._prevBeta, self._prevCa, self._prevCb

            cs_matrix, miscMatrix = self._createCompositionSet(eq, state_variables, self.phases[0], phase_amounts, matrix_idx)

            chemical_potentials = eq.MU.values.ravel()

            #If precipitate phase is not stable, then only store matrix phase in composition sets
            #Checks for single phase regions are done in _curvatureFactorFromEq,
            # so this will allow to fail there
            if len(precip_idx) == 0:
                composition_sets = [cs_matrix]
                self._compset_cache[precPhase] = None
            else:
                cs_precip, miscPrec = self._createCompositionSet(eq, state_variables, precPhase, phase_amounts, precip_idx)

                composition_sets = [cs_matrix, cs_precip]
                self._compset_cache[precPhase] = composition_sets

                if miscMatrix or miscPrec:
                    result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                            self.models, self.phase_records,
                                                            composition_sets=self._compset_cache[precPhase])
                    self._compset_cache[precPhase] = composition_sets
                    chemical_potentials = result.chemical_potentials

        else:
            result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                         self.models, self.phase_records,
                                                         composition_sets=self._compset_cache[precPhase])
            self._compset_cache[precPhase] = composition_sets
            chemical_potentials = result.chemical_potentials

        result = self._curvatureFactorFromEq(chemical_potentials, composition_sets, precPhase, training)
        return result

    def getGrowthAndInterfacialComposition(self, x, T, dG, R, gExtra, precPhase = None, training = False):
        '''
        Returns growth rate and interfacial compostion given Gibbs-Thomson contribution

        Parameters
        ----------
        x : array
            Composition of solutes
        T : float
            Temperature
        dG : float
            Driving force at given x and T
        R : float or array
            Precipitate radius
        gExtra : float or array
            Gibbs-Thomson contribution (must be same shape as R)
        precPhase : str (optional)
            Precipitate phase (defaults to first precipitate in list)

        Returns
        -------
        (growth rate, matrix composition, precipitate composition, equilibrium matrix comp, equilibrium precipitate comp)
        growth rate will be float or array based off shape of R
        matrix and precipitate composition will be array or 2D array based
            off shape of R
        '''
        if hasattr(R, '__len__'):
            R = np.array(R)
        if hasattr(gExtra, '__len__'):
            gExtra = np.array(gExtra)

        dc, mc, gba, beta, ca, cb = self.curvatureFactor(x, T, precPhase, training)
        if dc is None:
            return None, None, None, None, None

        Rdiff = (dG - gExtra)

        gr = (mc / R) * Rdiff

        if hasattr(Rdiff, '__len__'):
            calpha = np.zeros((len(Rdiff), len(self.elements[1:-1])))
            dca = np.zeros((len(Rdiff), len(self.elements[1:-1])))
            cbeta = np.zeros((len(Rdiff), len(self.elements[1:-1])))
            for i in range(len(self.elements[1:-1])):
                calpha[:,i] = x[i] - dc[i] * Rdiff
                dca[:,i] = calpha[:,i] - ca[i]

            dcb = np.matmul(gba, dca.T)
            for i in range(len(self.elements[1:-1])):
                cbeta[:,i] = cb[i] + dcb[i,:]

            calpha[calpha < 0] = 0
            calpha[calpha > 1] = 1
            cbeta[cbeta < 0] = 0
            cbeta[cbeta > 1] = 1

            return gr, calpha, cbeta, ca, cb
        else:
            calpha = x - dc * Rdiff
            cbeta = cb + np.matmul(gba, (calpha - ca)).flatten()

            calpha[calpha < 0] = 0
            calpha[calpha > 1] = 1
            cbeta[cbeta < 0] = 0
            cbeta[cbeta > 1] = 1

            return gr, calpha, cbeta, ca, cb

    def impingementFactor(self, x, T, precPhase = None, training = False):
        '''
        Returns impingement factor for nucleation rate calculations

        Parameters
        ----------
        x : array
            Composition of solutes
        T : float
            Temperature
        precPhase : str (optional)
            Precipitate phase (defaults to first precipitate in list)
        '''
        dc, mc, gba, beta, ca, cb = self.curvatureFactor(x, T, precPhase, training)
        return beta
