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

    gOffset = 1      #Small value to add to precipitate phase for when order/disorder models are used

    def __init__(self, database, elements, phases, drivingForceMethod = 'approximate'):
        self.db = Database(database)
        self.elements = elements

        if 'VA' not in self.elements:
            self.elements.append('VA')
            
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
        self._pointsParent = None
        self._parentEq = None

        #Pertains to precipitate phases (sampled points)
        self._pointsPrec = {self.phases[i]: None for i in range(1, len(self.phases))}
        self._orderingPoints = {self.phases[i]: None for i in range(1, len(self.phases))}

        self.setDrivingForceMethod(drivingForceMethod)

        #Get mobility/diffusivity if exists
        param_search = self.db.search
        param_query_mob = (
            (where('phase_name') == self.phases[0]) & \
            (where('parameter_type') == 'MQ') | \
            (where('parameter_type') == 'MF')
        )

        param_query_diff = (
            (where('phase_name') == self.phases[0]) & \
            (where('parameter_type') == 'DQ') | \
            (where('parameter_type') == 'DF')
        )

        pMob = param_search(param_query_mob)
        pDiff = param_search(param_query_diff)
        self.mobModels = None
        self.mobCallables = {} if len(pMob) > 0 else None
        self.diffCallables = {} if len(pDiff) > 0 else None
        if len(pMob) > 0 or len(pDiff) > 0:
            self.mobModels = MobilityModel(self.db, self.elements, self.phases[0])
            if len(pMob) > 0:
                for c in self.phase_records[self.phases[0]].nonvacant_elements:
                    bcp = build_callables(self.db, self.elements, [self.phases[0]], {self.phases[0]: self.mobModels},
                                          parameter_symbols=None, output='mob_'+c, build_gradients=False, build_hessians=False,
                                          additional_statevars=[v.T, v.P, v.N, v.GE])
                    self.mobCallables[c] = bcp['mob_'+c]['callables'][self.phases[0]]
            else:
                for c in self.phase_records[self.phases[0]].nonvacant_elements:
                    bcp = build_callables(self.db, self.elements, [self.phases[0]], {self.phases[0]: self.mobModels},
                                          parameter_symbols=None, output='diff_'+c, build_gradients=False, build_hessians=False,
                                          additional_statevars=[v.T, v.P, v.N, v.GE])
                    self.diffCallables[c] = bcp['diff_'+c]['callables'][self.phases[0]]

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
        self._pointsParent = None
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
        if precPhase is None:
            precPhase = self.phases[1]

        if hasattr(x, '__len__'):
            #Remove first element if x lists composition of all elements
            if len(x) == len(self.elements) - 1:
                x = x[1:]

            cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        else:
            cond = {v.X(self.elements[1]): x}
        cond[v.P] = 101325
        cond[v.T] = T
        cond[v.GE] = gExtra + self.gOffset

        eq = equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond, model=self.models, 
                         phase_records={self.phases[0]: self.phase_records[self.phases[0]], precPhase: self.phase_records[precPhase]}, 
                         calc_opts={'pdens': self.pDens})
        return eq

    def getInterdiffusivity(self, x, T):
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

        Returns
        -------
        interdiffusivity - will return array if T is an array
            For binary case - float or array of floats
            For multicomponent - matrix or array of matrices
        '''
        dnkj = []

        if hasattr(T, '__len__'):
            for i in range(len(T)):
                dnkj.append(self._interdiffusivitySingle(x[i], T[i]))
            return np.array(dnkj)
        else:
            return self._interdiffusivitySingle(x, T)

    def _interdiffusivitySingle(self, x, T):
        '''
        Gets interdiffusivity at unique composition and temperature

        Parameters
        ----------
        x : float or array
            Composition
        T : float
            Temperature
        
        Returns
        -------
        Interdiffusivity as a matrix (will return float in binary case)
        '''
        if not hasattr(x, '__len__'):
            x = [x]
        cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        cond[v.P] = 101325
        cond[v.T] = T
        cond[v.GE] = 0
        cond['N'] = 1
        eqPh = equilibrium(self.db, self.elements, self.phases[0], cond, model=self.models, phase_records={self.phases[0]: self.phase_records[self.phases[0]]}, calc_opts = {'pdens': self.pDens})

        state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
        stable_phases = eqPh.Phase.values.ravel()
        phase_amounts = eqPh.NP.values.ravel()
        matrix_idx = np.where(stable_phases == self.phases[0])[0]

        cs_matrix = CompositionSet(self.phase_records[self.phases[0]])
        cs_matrix.update(eqPh.Y.isel(vertex=matrix_idx).values.ravel()[:cs_matrix.phase_record.phase_dof],
                            phase_amounts[matrix_idx], state_variables)
        chemical_potentials = eqPh.MU.values.ravel()

        if self.mobCallables is None:
            Dnkj, _, _ = inverseMobility_from_diffusivity(chemical_potentials, cs_matrix,
                                                                            self.elements[0], self.diffCallables,
                                                                            diffusivity_correction=self.mobility_correction)
        else:
            Dnkj, _, _ = inverseMobility(chemical_potentials, cs_matrix, self.elements[0],
                                                        self.mobCallables,
                                                        mobility_correction=self.mobility_correction)

        if len(x) == 1:
            return Dnkj.ravel()[0]
        else:
            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)
            Dnkj = Dnkj[unsortIndices,:]
            Dnkj = Dnkj[:,unsortIndices]
            return Dnkj
            

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
        cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        cond[v.P] = 101325
        cond[v.T] = T
        cond[v.GE] = 0
        cond['N'] = 1

        #Equilibrium at matrix composition for only the parent phase
        self._parentEq, self._matrix_cs = local_equilibrium(self.db, self.elements, [self.phases[0]], cond, 
                                                            self.models, self.phase_records, 
                                                            composition_sets = self._matrix_cs)
        #Remove cache when training
        if training:
            self._matrix_cs = None

        self._prevX = x

        #Check if equilibrium has converged and chemical potential can be obtained
        #If not, then return None for driving force
        #if any(np.isnan(self._parentEq.MU.values.ravel())):
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
        #mu = np.array([self._parentEq.MU.values.ravel()])
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
        cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        cond[v.P] = 101325
        cond[v.T] = T
        cond[v.GE] = 0
        cond[v.N] = 1

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
                cs_matrix = CompositionSet(self.phase_records[self.phases[0]])
                cs_matrix.update(eq.Y.isel(vertex=matrix_idx).values.ravel()[:cs_matrix.phase_record.phase_dof],
                                phase_amounts[matrix_idx], state_variables)
                cs_precip = CompositionSet(self.phase_records[precPhase])
                cs_precip.update(eq.Y.isel(vertex=precip_idx).values.ravel()[:cs_precip.phase_record.phase_dof],
                                phase_amounts[precip_idx], state_variables)
                composition_sets = [cs_matrix, cs_precip]
                self._compset_cache_df[precPhase] = composition_sets

            chemical_potentials = eq.MU.values.ravel()
            ph = stable_phases[stable_phases != '']
            x_precip = eq.isel(vertex=precip_idx).X.values.ravel()
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

            self._prevX = x

            #Check if equilibrium has converged and chemical potential can be obtained
            #If not, then return None for driving force
            #if any(np.isnan(self._parentEq.MU.values.ravel())):
            if any(np.isnan(self._parentEq.chemical_potentials)):
                return None, None

            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)

            xP = x_precip
            
            #dg = np.sum(xP * self._parentEq.MU.values.ravel()) - np.sum(xP * eqPh.MU.values.ravel())
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
        cond = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        cond[v.P] = 101325
        cond[v.T] = T
        cond[v.GE] = 0
        cond[v.N] = 1

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
                cs_matrix = CompositionSet(self.phase_records[self.phases[0]])
                cs_matrix.update(eq.Y.isel(vertex=matrix_idx).values.ravel()[:cs_matrix.phase_record.phase_dof],
                                phase_amounts[matrix_idx], state_variables)
                cs_precip = CompositionSet(self.phase_records[precPhase])
                cs_precip.update(eq.Y.isel(vertex=precip_idx).values.ravel()[:cs_precip.phase_record.phase_dof],
                                phase_amounts[precip_idx], state_variables)
                composition_sets = [cs_matrix, cs_precip]
                self._compset_cache_df[precPhase] = composition_sets

            chemical_potentials = eq.MU.values.ravel()
            ph = stable_phases[stable_phases != '']
            x_precip = eq.isel(vertex=precip_idx).X.values.ravel()
            x_matrix = eq.isel(vertex=matrix_idx).X.values.ravel()
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
                T = T * np.ones(len(gExtra))

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
        
        #Compute equilibrium at guess composition
        cond = {v.X(self.elements[1]): self._guessComposition[precPhase], v.T: T, v.P: 101325, v.GE: gExtra + self.gOffset}
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
                
                return xParent, xPrec

        return None, None


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
                cs_matrix.update(eqSub.Y.isel(vertex=matrix_idx).values.ravel()[:cs_matrix.phase_record.phase_dof],
                                    phase_amounts[matrix_idx], state_variables)
                cs_precip = CompositionSet(self.phase_records[precPhase])
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
                
                return xParent, xPrec

        return None, None
        
        
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
        points = calculate(self.db, self.elements, self.phases[0], P=101325, T=T, GE=0, phase_records=self.phase_records, output='GM')
        ax.scatter(points.X.sel(component=self.elements[1]), points.GM / 1000, label=self.phases[0], *args, **kwargs)
        
        #Add gExtra to precipitate phase
        for i in range(1, len(self.phases)):
            points = calculate(self.db, self.elements, self.phases[i], P=101325, T=T, GE=0, phase_records=self.phase_records, output='GM')
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
        self._prevDc = None
        self._prevMc = None
        self._prevGba = None
        self._prevBeta = None
        self._prevCa = None
        self._prevCb = None

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

        else:
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
        #Check if input equilibrium has converged
        if np.any(np.isnan(chemical_potentials)):
            if training:
                return None, None, None, None, None, None
            else:
                print('Warning: equilibrum was not able to be solved for, using results of previous calculation')
                return self._prevDc, self._prevMc, self._prevGba, self._prevBeta, self._prevCa, self._prevCb

        if precPhase is None:
            precPhase = self.phases[1]

        ele = list(composition_sets[0].phase_record.nonvacant_elements)
        refIndex = ele.index(self.elements[0])

        ph = [cs.phase_record.phase_name for cs in composition_sets]

        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            sortIndices = np.argsort(self.elements[1:-1])
            unsortIndices = np.argsort(sortIndices)

            matrix_cs = [cs for cs in composition_sets if cs.phase_record.phase_name == self.phases[0]][0]

            if self.mobCallables is None:
                Dnkj, dMudxParent, invMob = inverseMobility_from_diffusivity(chemical_potentials, matrix_cs,
                                                                             self.elements[0], self.diffCallables,
                                                                             diffusivity_correction=self.mobility_correction)

                #NOTE: This is note tested yet
                Dtrace = tracer_diffusivity_from_diff(matrix_cs, self.diffCallables, diffusivity_correction=self.mobility_correction)
            else:
                Dnkj, dMudxParent, invMob = inverseMobility(chemical_potentials, matrix_cs, self.elements[0],
                                                            self.mobCallables,
                                                            mobility_correction=self.mobility_correction)
                Dtrace = tracer_diffusivity(matrix_cs, self.mobCallables, mobility_correction=self.mobility_correction)

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
            beta = 1 / np.sum(betaNum / betaDen)

            self._prevDc = num[unsortIndices] / den
            self._prevMc = 1 / den
            self._prevGba = Gba
            self._prevBeta = beta
            self._prevCa = xM[unsortIndices]
            self._prevCb = xP[unsortIndices]

            return self._prevDc, self._prevMc, self._prevGba, self._prevBeta, self._prevCa, self._prevCb
        else:
            if training:
                return None, None, None, None, None, None
            else:
                print('Warning: only a single phase detected in equilibrium, using results of previous calculation')
                return self._prevDc, self._prevMc, self._prevGba, self._prevBeta, self._prevCa, self._prevCb


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
        if hasattr(x, '__len__'):
            #Remove first element if x lists composition of all elements
            if len(x) == len(self.elements) - 1:
                x = x[1:]

            conds = {v.X(self.elements[i+1]): x[i] for i in range(len(x))}
        else:
            conds = {v.X(self.elements[1]): x}
        conds.update({v.N: 1, v.P: 1e5, v.GE: 0, v.T: T})

        #Perform equilibrium from scratch if cache not set or when training surrogate
        if self._compset_cache.get(precPhase, None) is None or training:
            eq = self.getEq(x, T, 0, precPhase)
            state_variables = np.array([conds[v.GE], conds[v.N], conds[v.P], conds[v.T]], dtype=np.float64)
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

            cs_matrix = CompositionSet(self.phase_records[self.phases[0]])
            cs_matrix.update(eq.Y.isel(vertex=matrix_idx).values.ravel()[:cs_matrix.phase_record.phase_dof],
                             phase_amounts[matrix_idx], state_variables)

            #If precipitate phase is not stable, then only store matrix phase in composition sets
            #Checks for single phase regions are done in _curvatureFactorFromEq,
            # so this will allow to fail there
            if len(precip_idx) == 0:
                composition_sets = [cs_matrix]
                self._compset_cache[precPhase] = None
            else:
                cs_precip = CompositionSet(self.phase_records[precPhase])
                cs_precip.update(eq.Y.isel(vertex=precip_idx).values.ravel()[:cs_precip.phase_record.phase_dof],
                                phase_amounts[precip_idx], state_variables)
                composition_sets = [cs_matrix, cs_precip]
                self._compset_cache[precPhase] = composition_sets

            chemical_potentials = eq.MU.values.ravel()
        else:
            result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], conds,
                                                         self.models, self.phase_records,
                                                         composition_sets=self._compset_cache[precPhase])
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
        (growth rate, matrix composition, precipitate composition)
        growth rate will be float or array based off shape of R
        matrix and precipitate composition will be array or 2D array based
            off shape of R
        '''
        if hasattr(R, '__len__'):
            R = np.array(R)
        if hasattr(gExtra, '__len__'):
            gExtra = np.array(gExtra)

        dc, mc, gba, beta, ca, cb = self.curvatureFactor(x, T, precPhase, training)

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
            cbeta[cbeta < 0] = 0

            return gr, calpha, cbeta
        else:
            calpha = x - dc * Rdiff
            cbeta = cb + np.matmul(gba, (calpha - ca)).flatten()

            calpha[calpha < 0] = 0
            cbeta[cbeta < 0] = 0

            return gr, calpha, cbeta

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
