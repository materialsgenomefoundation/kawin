from collections import namedtuple

import numpy as np

from pycalphad import variables as v

from kawin.thermo.Thermodynamics import GeneralThermodynamics
from kawin.thermo.LocalEquilibrium import local_equilibrium
from kawin.thermo.FreeEnergyHessian import dMudX
from kawin.thermo.Mobility import inverseMobility, inverseMobility_from_diffusivity, tracer_diffusivity, tracer_diffusivity_from_diff

# mc = 1 / (dx_bar^T * inv(M) * dx_bar)
# dc = inv(Dnkj) * dx_bar / mc
# gba = inv(d2G_beta/dx2) * d2G_alpha/dx2
# c_eq_alpha = equilibrium composition of alpha phase
# c_eq_beta = equilibrium composition of beta phase
CurvatureOutput = namedtuple('CurvatureOutput',
                            ['dc', 'mc', 'gba', 'beta', 'c_eq_alpha', 'c_eq_beta'],
                            defaults=(None, None, None, None, None, None))

# growth_rate = dR/dt
# c_alpha = interfacial composition of alpha phase
# c_beta = interfacial composition of beta phase
# c_eq_alpha = equilibrium composition of alpha phase
# c_eq_beta = equilibrium composition of beta phase
GrowthRateOutput = namedtuple('GrowthRateOutput',
                             ['growth_rate', 'c_alpha', 'c_beta', 'c_eq_alpha', 'c_eq_beta'],
                             defaults=(None, None, None, None, None))


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
        self._curvature_outputs = {p: CurvatureOutput() for p in phases[1:]}

    def clearCache(self):
        super().clearCache()
        self._compset_cache_curvature = {}

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
        gExtra = np.atleast_1d(gExtra)
        T = np.atleast_1d(T)
        if len(T) == 1:
            T = T*np.ones(gExtra.shape, dtype=np.float64)
        
        caArray, cbArray = zip(*[self._interfacialComposition(x, T[i], gExtra[i], precPhase) for i in range(len(gExtra))])
        return np.squeeze(caArray), np.squeeze(cbArray)

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
        precPhase = self.phases[1] if precPhase is None else precPhase

        wks = self.getEq(x, T, gExtra, precPhase)
        mu = np.squeeze(wks.eq.MU)

        #Check for convergence, return None if not converged
        if np.any(np.isnan(mu)):
            return -1*np.ones(len(self.elements[:-1]), dtype=np.float64), -1*np.ones(len(self.elements[:-1]), dtype=np.float64)
        
        cs_list = wks.get_composition_sets()
        ph = [cs.phase_record.phase_name for cs in cs_list]

        #Check if matrix and precipitate phase are stable, and check if there's no miscibility gaps
        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            cs_matrix = cs_list[ph.index(self.phases[0])]
            cs_precip = cs_list[ph.index(precPhase)]

            xM = np.array(cs_matrix.X, dtype=np.float64)
            xP = np.array(cs_precip.X, dtype=np.float64)

            sortIndices = np.argsort(self.elements[:-1])
            unsortIndices = np.argsort(sortIndices)
            return xM[unsortIndices], xP[unsortIndices]

        return -1*np.ones(len(self.elements[:-1]), dtype=np.float64), -1*np.ones(len(self.elements[:-1]), dtype=np.float64)
    
    def _curvatureFactorFromEq(self, chemical_potentials, cs_matrix, cs_precip, precPhase=None):
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
        precPhase = self.phases[1] if precPhase is None else precPhase
        non_va_elements = list(cs_matrix.phase_record.nonvacant_elements)
        refIndex = non_va_elements.index(self.elements[0])

        sortIndices = np.argsort(self.elements[1:-1])
        unsortIndices = np.argsort(sortIndices)

        if self.mobCallables[self.phases[0]] is None:
            Dnkj, dMudxParent, invMob = inverseMobility_from_diffusivity(chemical_potentials, cs_matrix,
                                                                            self.elements[0], self.diffCallables[self.phases[0]],
                                                                            diffusivity_correction=self.mobility_correction, parameters=self._parameters)

            #NOTE: This is note tested yet
            Dtrace = tracer_diffusivity_from_diff(cs_matrix, self.diffCallables[self.phases[0]], diffusivity_correction=self.mobility_correction, parameters=self._parameters)
        else:
            Dnkj, dMudxParent, invMob = inverseMobility(chemical_potentials, cs_matrix, self.elements[0],
                                                        self.mobCallables[self.phases[0]],
                                                        mobility_correction=self.mobility_correction, parameters=self._parameters)
            Dtrace = tracer_diffusivity(cs_matrix, self.mobCallables[self.phases[0]], mobility_correction=self.mobility_correction, parameters=self._parameters)

        xMFull = np.array(cs_matrix.X)
        xM = np.delete(xMFull, refIndex)

        dMudxPrec = dMudX(chemical_potentials, cs_precip, self.elements[0])
        xPFull = np.array(cs_precip.X)
        xP = np.delete(xPFull, refIndex)
        xBarFull = np.array([xPFull - xMFull])
        xBar = np.array([xP - xM])

        # Numerator term for Eq 31 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
        # D^-1 * dCbar
        num = np.matmul(np.linalg.inv(Dnkj), xBar.T).flatten()

        # Denominator term for Eq 28 and 31 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
        # Cbar^T * M^-1 * Cbar
        #Denominator should be a scalar since its V * M * V^T
        den = np.matmul(xBar, np.matmul(invMob, xBar.T)).flatten()[0]

        # Matrix/precipitate curvature term for Eq 36 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
        # (d2G_beta/dx2)^-1 * (d2G_alpha/dx2)
        if np.linalg.matrix_rank(dMudxPrec) == dMudxPrec.shape[0]:
            Gba = np.matmul(np.linalg.inv(dMudxPrec), dMudxParent)
            Gba = Gba[unsortIndices,:]
            Gba = Gba[:,unsortIndices]
        else:
            Gba = np.zeros(dMudxPrec.shape)

        # We compute the impingment rate here since we get tracer diffusivity
        # and equilibrium composition as we're computing the other terms
        betaNum = xBarFull**2
        betaDen = Dtrace * xMFull.flatten()
        bsum = np.sum(betaNum / betaDen)
        if bsum == 0:
            beta = self._curvature_outputs[precPhase].beta
        else:
            beta = 1 / bsum

        self._curvature_outputs[precPhase] = CurvatureOutput(dc=num[unsortIndices]/den, 
                                                             mc=1/den, 
                                                             gba=Gba, 
                                                             beta=beta, 
                                                             c_eq_alpha=xM[unsortIndices], 
                                                             c_eq_beta=xP[unsortIndices])

        return self._curvature_outputs[precPhase]

    def curvatureFactor(self, x, T, precPhase = None, removeCache = False, searchDir = None):
        '''
        Curvature factor (from Phillipes and Voorhees - 2013) from composition and temperature
        This will find equilibrium and composition sets for x and T and call _curvatureFactorFromEq

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
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        CurvatureCache object
            {D-1 dCbar / dCbar^T M-1 dCbar} - for calculating interfacial composition of matrix
            {1 / dCbar^T M-1 dCbar} - for calculating growth rate
            {Gb^-1 Ga} - for calculating precipitate composition
            beta - Impingement rate
            Ca - interfacial composition of matrix phase
            Cb - interfacial composition of precipitate phase

        Will return (None, None, None, None, None, None) if single phase
        '''
        # If equilibrium is invalid, then we return the previous values if available
        # Else, we return None
        def _process_invalid_eq():
            if removeCache:
                self._compset_cache_curvature[precPhase] = None
            if self._compset_cache_curvature.get(precPhase) is None:
                return None
            else:
                print('Warning: equilibrum was not able to be solved for, using results of previous calculation')
                return self._curvature_outputs[precPhase]
            
        # Get composition sets for equilibrium between matrix and precipitate
        precPhase = self.phases[1] if precPhase is None else precPhase
        eq_results = self._getCompositionSetsEq(x, T, precPhase, self._compset_cache_curvature)
        if eq_results is None:
            return _process_invalid_eq()
        
        chemical_potentials, cs_matrix, cs_precip = eq_results

        # If either the matrix of precipitate is unstable, then search for two-phase equilibria
        if cs_matrix is None or cs_precip is None:
            eq_results = self._searchForTwoPhaseEq(x, T, precPhase, searchDir)
            if eq_results is None:
                return _process_invalid_eq()
                
            chemical_potentials, cs_matrix, cs_precip = eq_results

        self._compset_cache_curvature[precPhase] = None if removeCache else [cs_matrix, cs_precip]
        return self._curvatureFactorFromEq(chemical_potentials, cs_matrix, cs_precip, precPhase)
        
    def _searchForTwoPhaseEq(self, x, T, precPhase, searchDir = None):
        '''
        Given x and a search direction (which should correspond to the composition of precipitate from driving force calc)
        Iteratively search between x and search direction until two phase equilibria is found

        If two phase equilibria is not found, then return None
        else return chemical_potential, cs_matrix, cs_precip
        '''
        if searchDir is None:
            return None
        
        searchDir = np.array(searchDir)
        currX = 0.5 * np.array(x) + 0.5 * searchDir
        currIt = 0
        
        #MaxIt is 15, which refers to a maximum of 6e-5 difference in test composition between the 14th and 15th iteration
        #    Which is probably more than enough to find a two-phase region
        maxIt = 15
        while currIt < maxIt:
            eq_results = self._getCompositionSetsEq(currX, T, precPhase)
            if eq_results is None:
                return None
            chemical_potentials, cs_matrix, cs_precip = eq_results
            # If matrix and precipitate are both stable, then return composition sets
            if cs_matrix is not None and cs_precip is not None:
                return chemical_potentials, cs_matrix, cs_precip
            # If only matrix is stable, then move currX towards searchDir
            elif cs_precip is None:
                currX = 0.5*currX + 0.5*searchDir
            # If only precipitate is stable, then move currX towards x
            elif cs_matrix is None:
                currX = 0.5*currX + 0.5*np.array(x)
            else:
                return None
            
            currIt += 1

        return None

    def getGrowthAndInterfacialComposition(self, x, T, dG, R, gExtra, precPhase = None, removeCache = False, searchDir = None):
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
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other

        Returns
        -------
        (growth rate, matrix composition, precipitate composition, equilibrium matrix comp, equilibrium precipitate comp)
        growth rate will be float or array based off shape of R
        matrix and precipitate composition will be array or 2D array based
            off shape of R
        '''
        x = self._process_x(x)
        R = np.atleast_1d(R)
        gExtra = np.atleast_1d(gExtra)

        curv_results = self.curvatureFactor(x, T, precPhase, removeCache, searchDir)
        if curv_results is None:
            return None

        # Eq 28 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
        # Rdiff is (dCbar^T * d2G/dx2 * C^inf - 2yVm/R) = (driving force - Gibbs Thompson energy)
        Rdiff = (dG - gExtra)
        gr = (curv_results.mc / R) * Rdiff

        # Eq 31 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
        # calpha and cbeta are shape (len(gExtra), len(elements)-1)
        calpha = x[np.newaxis,:] - np.outer(Rdiff, curv_results.dc)

        # Eq 36 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
        dca = calpha - curv_results.c_eq_alpha[np.newaxis,:]
        dcb = np.matmul(curv_results.gba, dca.T).T
        cbeta = curv_results.c_eq_beta[np.newaxis,:] + dcb

        calpha = np.clip(calpha, 0, 1)
        cbeta = np.clip(cbeta, 0, 1)

        return GrowthRateOutput(growth_rate=np.squeeze(gr), 
                                c_alpha=np.squeeze(calpha), 
                                c_beta=np.squeeze(cbeta), 
                                c_eq_alpha=np.squeeze(curv_results.c_eq_alpha), 
                                c_eq_beta=np.squeeze(curv_results.c_eq_beta))

    def impingementFactor(self, x, T, precPhase = None, removeCache = False):
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
        removeCache : bool (optional)
            If True, this will not cache any equilibrium
            This is used for training since training points may not be near each other
        '''
        curv_results = self.curvatureFactor(x, T, precPhase, removeCache)
        if curv_results is None:
            return self._curvature_outputs[precPhase].beta
        return curv_results.beta
    
    def _curvatureWithSearch(self, x, T, precPhase = None, removeCache = True):
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
        _, xb = self.getDrivingForce(x, T, precPhase, removeCache = removeCache)
        return self.curvatureFactor(x, T, precPhase, removeCache = removeCache, searchDir=xb)