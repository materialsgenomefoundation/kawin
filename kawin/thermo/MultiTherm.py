from kawin.thermo.Thermodynamics import GeneralThermodynamics
import numpy as np
from pycalphad import variables as v
from kawin.thermo.Mobility import inverseMobility, inverseMobility_from_diffusivity, tracer_diffusivity, tracer_diffusivity_from_diff
from kawin.thermo.FreeEnergyHessian import dMudX
from kawin.thermo.LocalEquilibrium import local_equilibrium

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
        Options are 'tangent' (default), 'approximate', 'sampling' and 'curvature' (not recommended)
    parameters : list [str] or dict {str : float}
        List of parameters to keep symbolic in the thermodynamic or mobility models
    '''
    def __init__(self, database, elements, phases, drivingForceMethod = 'tangent', parameters = None):
        super().__init__(database, elements, phases, drivingForceMethod, parameters)

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

    def _curvatureFactorFromEq(self, chemical_potentials, composition_sets, precPhase=None):
        '''
        Curvature factor (from Phillipes and Voorhees - 2013)

        Steps
            1. Check that there is 2 phases in equilibrium, one being the matrix and the other being precipitate
            2. Get Dnkj, dmu/dx and inverse mobility term from composition set of matrix phase
            3. Get dmu/dx of precipitate phase
            4. Get difference in matrix and precipitate phase composition (we use a second order approximation to get precipitate composition as function of R)
            5. Compute numerator, denominator, Gba and beta term
                Denominator (X_bar^T * invMob * X_bar) is used for growth rate (eq 28)
                Numerator (D^-1 * X_bar), denominator is used for matrix interfacial composition (eq 31)
                Gba and matrix interfacial composition is used for precipitate interfacial composition (eq 36)
                    Gba here is (dmu/dx_beta)^-1 * dmu/dx_alpha
                Note: these equations have a term X_bar^T * dmu/dx_alpha * X_bar_infty, but this is just the driving force so we don't need to calculate it here

        Parameters
        ----------
        chemical_potentials : 1-D float64 array
        composition_sets : List[pycalphad.composition_set.CompositionSet]
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
                                                                             diffusivity_correction=self.mobility_correction, parameters=self._parameters)

                #NOTE: This is note tested yet
                Dtrace = tracer_diffusivity_from_diff(matrix_cs, self.diffCallables[self.phases[0]], diffusivity_correction=self.mobility_correction, parameters=self._parameters)
            else:
                Dnkj, dMudxParent, invMob = inverseMobility(chemical_potentials, matrix_cs, self.elements[0],
                                                            self.mobCallables[self.phases[0]],
                                                            mobility_correction=self.mobility_correction, parameters=self._parameters)
                Dtrace = tracer_diffusivity(matrix_cs, self.mobCallables[self.phases[0]], mobility_correction=self.mobility_correction, parameters=self._parameters)

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
            return None
            # if training:
            #     return None
            # else:
            #     #print('Warning: only a single phase detected in equilibrium, using results of previous calculation')
            #     #return self._prevDc[precPhase], self._prevMc[precPhase], self._prevGba[precPhase], self._prevBeta[precPhase], self._prevCa[precPhase], self._prevCb[precPhase]

            #     #If two-phase equilibrium is not found, then the temperature may have changed to where the precipitate is unstable
            #     #Return None in this case
            #     return None


    def curvatureFactor(self, x, T, precPhase = None, training = False, searchDir = None):
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
        searchDir : None or array
            If two-phase equilibrium is not present, then move x towards this composition to find two-phase equilibria
        training : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

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
            cs_results = self._getCompositionSetsForCurvature(x, T, precPhase)
            if cs_results is None:
                return None
            
            chemical_potentials, composition_sets = cs_results
        else:
            result, composition_sets = local_equilibrium(self.db, self.elements, [self.phases[0], precPhase], cond,
                                                         self.models, self.phase_records,
                                                         composition_sets=self._compset_cache[precPhase])
            self._compset_cache[precPhase] = composition_sets
            chemical_potentials = result.chemical_potentials

        #Check if input equilibrium has converged
        if np.any(np.isnan(chemical_potentials)):
            if training:
                return None
            else:
                print('Warning: equilibrum was not able to be solved for, using results of previous calculation')
                return self._prevDc[precPhase], self._prevMc[precPhase], self._prevGba[precPhase], self._prevBeta[precPhase], self._prevCa[precPhase], self._prevCb[precPhase]

        ph = [cs.phase_record.phase_name for cs in composition_sets]
        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            return self._curvatureFactorFromEq(chemical_potentials, composition_sets, precPhase)
        #If in a singl phase region, we want to go along a search direction to find the nearest two phase region
        #    We then use this two-phase region to calculate growth rate (which should all be negative for dissolution)
        #    In PrecipitateModel, searchDir is the previous precipitate nucleate composition
        #    We performe a rouch search 
        elif searchDir is not None:
            currX = np.array(x)
            searchDir = np.array(searchDir)
            currX = 0.5 * currX + 0.5 * searchDir
            foundTwoPhases = False
            maxIt = 15
            currIt = 0
            while not foundTwoPhases:
                cs_results = self._getCompositionSetsForCurvature(currX, T, precPhase)
                if cs_results is None:
                    return None
                chemical_potentials, composition_sets = cs_results
                ph = [cs.phase_record.phase_name for cs in composition_sets]
                if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                    foundTwoPhases = True
                elif len(ph) == 1 and self.phases[0] in ph:
                    #Only matrix is stable, move closer to searchDir
                    currX = 0.5*currX + 0.5*searchDir
                elif len(ph) == 1 and precPhase in ph:
                    #Only precipitate is stable, move closer to original x
                    currX = 0.5*currX + 0.5*np.array(x)

                #More than likely, this is not needed, but just in case
                #MaxIt is 15, which refers to a 6e-5 difference in test composition between the 14th and 15th iteration
                #    Which is probably more than enough to find a two-phase region
                currIt += 1
                if currIt > maxIt:
                    return None
            
            chemical_potentials, composition_sets = cs_results
            return self._curvatureFactorFromEq(chemical_potentials, composition_sets, precPhase)
        else:
            return None

    def getGrowthAndInterfacialComposition(self, x, T, dG, R, gExtra, precPhase = None, training = False, searchDir = None):
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
        searchDir : None or array
            If two-phase equilibrium is not present, then move x towards this composition to find a two-phase region
        training : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

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

        curv_results = self.curvatureFactor(x, T, precPhase, training, searchDir)
        if curv_results is None:
            return None, None, None, None, None
    
        dc, mc, gba, beta, ca, cb = curv_results

        #dc, mc, gba, beta, ca, cb = self.curvatureFactor(x, T, precPhase, training, searchDir)
        #if dc is None:
        #    return None, None, None, None, None

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
        training : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other
        '''
        curv_results = self.curvatureFactor(x, T, precPhase, training)
        if curv_results is None:
            return self._prevBeta[precPhase]
        dc, mc, gba, beta, ca, cb = curv_results
        return beta
    
    def _getCompositionSetsForCurvature(self, x, T, precPhase):
        '''
        Create composition sets from equilibrium to be used for curvature factor

        Parameters
        ----------
        x : array
            Composition of solutes
        T : float
            Temperature
        precPhase : str (optional)
            Precipitate phase (defaults to first precipitate in list)
        '''
        cond = self._getConditions(x, T, 0)
        eq = self.getEq(x, T, 0, precPhase)
        state_variables = np.array([cond[v.GE], cond[v.N], cond[v.P], cond[v.T]], dtype=np.float64)
        stable_phases = eq.Phase.values.ravel()
        phase_amounts = eq.NP.values.ravel()
        matrix_idx = np.where(stable_phases == self.phases[0])[0]
        precip_idx = np.where(stable_phases == precPhase)[0]

        #If matrix phase is not stable (why?), then return previous values
        #Curvature can't be calculated if matrix phase isn't present
        if len(matrix_idx) == 0:
            return None

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

        return chemical_potentials, composition_sets
    
    def _curvatureWithSearch(self, x, T, precPhase = None, training = True):
        '''
        Performs driving force calculation to get xb, which can be used to find
        curvature factors when driving force is negative. Main use is for the surrogate model
        to train on all points

        Parameters
        ----------
        x : array
            Composition of solutes
        T : float
            Temperature
        precPhase : str (optional)
            Precipitate phase (defaults to first precipitate in list)
        training : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other
        '''
        dg, xb = self.getDrivingForce(x, T, precPhase, returnComp = True, training = training)
        return self.curvatureFactor(x, T, precPhase, training = training, searchDir=xb)