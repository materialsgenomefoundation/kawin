from kawin.thermo.Thermodynamics import GeneralThermodynamics
import numpy as np
from pycalphad import equilibrium, calculate, variables as v
from pycalphad.core.composition_set import CompositionSet
from kawin.thermo.FreeEnergyHessian import dMudX

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
        Options are 'tangent' (default), 'approximate', 'sampling', and 'curvature' (not recommended)
    interfacialCompMethod: str (optional)
        Method used to calculate interfacial composition
        Options are 'equilibrium' (default) and 'curvature' (not recommended)
    parameters : list [str] or dict {str : float}
        List of parameters to keep symbolic in the thermodynamic or mobility models
    '''
    def __init__(self, database, elements, phases, drivingForceMethod = 'tangent', interfacialCompMethod = 'equilibrium', parameters = None):
        super().__init__(database, elements, phases, drivingForceMethod, parameters)

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

                #dmudx are scalars here
                dMudxParent = dMudxParent[0,0]
                dMudxPrec = dMudxPrec[0,0]

                if dMudxParent != 0:
                    xParent = gExtra / dMudxParent / (xPrecEq - xParentEq) + xParentEq
                else:
                    xParent = xParentEq*np.ones(len(gExtra))

                if dMudxPrec != 0:
                    xPrec = dMudxParent * (xParent - xParentEq) / dMudxPrec + xPrecEq
                else:
                    xPrec = xPrecEq*np.ones(len(gExtra))

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