from kawin.thermo.Thermodynamics import GeneralThermodynamics
import numpy as np
from pycalphad import Workspace, equilibrium, calculate, variables as v
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
        self.reverse = self.elements[1] < self.elements[0]

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
        gExtra = np.atleast_1d(gExtra)
        T = np.atleast_1d(T)
        precPhase = self._getPrecipitatePhase(precPhase)
        if len(T) == 1:
            caArray, cbArray = self._interfacialComposition(T[0], gExtra, precPhase)
        else:
            caArray, cbArray = zip(*[self._interfacialComposition(T[i], gExtra[i], precPhase) for i in range(len(T))])
        return np.squeeze(caArray), np.squeeze(cbArray)

    def _interfacialCompositionFromEq(self, T, gExtra, precPhase):
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
        gExtra = np.atleast_1d(gExtra)
        gExtra += self.gOffset

        #Compute equilibrium at guess composition
        cond = {v.X(self.elements[1]): self._guessComposition[precPhase], v.T: T, v.P: 101325, v.GE: gExtra}
        phases, sub_models = self._setupSubModels(precPhase)
        wks = Workspace(self.db, self.elements, phases, cond, models=sub_models,
                        phase_record_factory = self.phase_records, calc_opts={'pdens': self.pDens})
        
        coord_keys = list(wks.eq.coords.keys())
        ge_var_idx = coord_keys.index('GE')

        xMatrixArray = -1*np.ones(gExtra.shape, dtype=np.float64)
        xPrecipArray = -1*np.ones(gExtra.shape, dtype=np.float64)
        gIndex = 0
        # cs_idx is a 5 len tuple of (GE, N, P, T, X)
        # where it iterates by X first, then by GE
        for cs_idx, cs_list in wks.enumerate_composition_sets():
            # If the GE index is greater than the current record index, then
            # update the current index. This occurs if we did not find any
            # two-phase regions at a given value of GE and now we want to move
            # on to the next value of GE
            if cs_idx[ge_var_idx] > gIndex:
                gIndex = cs_idx[ge_var_idx]

            # Only check for two-phase region if the cs list corresponds
            # to the current index of GE that we're looking at
            # If the current index of GE is greater than what's in
            # cs_idx, then we want to iterate the composition sets until we
            # catch up to gIndex
            if cs_idx[ge_var_idx] == gIndex:
                ph = [cs.phase_record.phase_name for cs in cs_list]
                if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                    cs_matrix = [cs for cs in cs_list if cs.phase_record.phase_name == self.phases[0]][0]
                    cs_precip = [cs for cs in cs_list if cs.phase_record.phase_name == precPhase][0]

                    c_idx = 0 if self.reverse else 1
                    xMatrixArray[gIndex] = cs_matrix.X[c_idx]
                    xPrecipArray[gIndex] = cs_precip.X[c_idx]
                    gIndex += 1

        return np.squeeze(xMatrixArray), np.squeeze(xPrecipArray)

    def _interfacialCompositionFromCurvature(self, T, gExtra, precPhase):
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
        gExtra = np.atleast_1d(gExtra)

        #Compute equilibrium at guess composition
        cond = {v.X(self.elements[1]): self._guessComposition[precPhase], v.T: T, v.P: 101325, v.GE: self.gOffset, v.N: 1}
        phases, sub_models = self._setupSubModels(precPhase)
        wks = Workspace(self.db, self.elements, phases, cond, models=sub_models,
                        phase_record_factory = self.phase_records, calc_opts={'pdens': self.pDens})
        chemical_potentials = np.squeeze(wks.eq.MU)

        idx = 0
        for _, cs_list in wks.enumerate_composition_sets():
            ph = [cs.phase_record.phase_name for cs in cs_list]
            if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
                cs_matrix = [cs for cs in cs_list if cs.phase_record.phase_name == self.phases[0]][0]
                cs_precip = [cs for cs in cs_list if cs.phase_record.phase_name == precPhase][0]
                chem_pot = chemical_potentials[idx]

                #Free energy curvature
                dMudxMatrix = np.squeeze(dMudX(chem_pot, cs_matrix, self.elements[0]))
                dMudxPrecip = np.squeeze(dMudX(chem_pot, cs_precip, self.elements[0]))

                c_idx = 0 if self.reverse else 1
                xMatrixEq = cs_matrix.X[c_idx]
                xPrecipEq = cs_precip.X[c_idx]

                #Compute composition of matrix and precipitate phase
                # x_meq, x_peq - matrix and precipitate composition at bulk equilibrum
                # x_m, x_p - matrix and precipitate composition for particle
                # (x_m - x_meq) * dmu/dx * (x_peq - x_meq) = gExtra
                # (x_m - x_meq) * dmu_m/dx = (x_p - x_peq) * dmu_p/dx
                # If dmu/dx is 0 (or undefined), then we use the equilibrium compositions
                if dMudxMatrix != 0:
                    xMatrix = gExtra / dMudxMatrix / (xPrecipEq - xMatrixEq) + xMatrixEq
                else:
                    xMatrix = xMatrixEq*np.ones(gExtra.shape)

                if dMudxPrecip != 0:
                    xPrecip = dMudxMatrix * (xMatrix - xMatrixEq) / dMudxPrecip + xPrecipEq
                else:
                    xPrecip = xPrecipEq*np.ones(gExtra.shape)

                xMatrix = np.clip(xMatrix, 0, 1, dtype=np.float64)
                xPrecip = np.clip(xPrecip, 0, 1, dtype=np.float64)
                return np.squeeze(xMatrix), np.squeeze(xPrecip)

            idx += 1

        return -1*np.ones(gExtra.shape), -1*np.ones(gExtra.shape)

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
        phases, sub_models = self._setupSubModels([self.phases[0]])
        points = calculate(self.db, self.elements, phases[0], P=101325, T=T, GE=0, model=sub_models, phase_records=self.phase_records, output='GM')
        ax.scatter(points.X.sel(component=self.elements[1]), points.GM / 1000, label=self.phases[0], *args, **kwargs)

        #Add gExtra to precipitate phase
        for i in range(1, len(self.phases)):
            phases, sub_models = self._setupSubModels([self.phases[i]])
            points = calculate(self.db, self.elements, phases[0], P=101325, T=T, GE=0, model=sub_models, phase_records=self.phase_records, output='GM')
            ax.scatter(points.X.sel(component=self.elements[1]), (points.GM + gExtra) / 1000, label=self.phases[i], *args, **kwargs)

            #Plot non-offset precipitate phase
            if plotGibbsOffset and gExtra != 0:
                ax.scatter(points.X.sel(component=self.elements[1]), points.GM / 1000, color='silver', alpha=0.3, *args, **kwargs)

        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_xlabel('Composition ' + self.elements[1])
        ax.set_ylabel('Gibbs Free Energy (kJ/mol)')