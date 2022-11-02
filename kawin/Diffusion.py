import numpy as np
import matplotlib.pyplot as plt
from kawin.Mobility import mobility_from_composition_set
import time
import csv
import copy
from itertools import zip_longest

class DiffusionModel:
    #Boundary conditions
    FLUX = 0
    COMPOSITION = 1

    def __init__(self, zlim, N, elements = ['A', 'B'], phases = ['alpha']):
        '''
        Class for defining a 1-dimensional mesh

        Parameters
        ----------
        zlim : tuple
            Z-bounds of mesh (lower, upper)
        N : int
            Number of nodes
        elements : list of str
            Elements in system (first element will be assumed as the reference element)
        phases : list of str
            Number of phases in the system
        '''
        if isinstance(phases, str):
            phases = [phases]
        self.zlim, self.N = zlim, N
        self.allElements, self.elements = elements, elements[1:]
        self.phases = phases
        self.therm = None

        self.z = np.linspace(zlim[0], zlim[1], N)
        self.dz = self.z[1] - self.z[0]

        self.reset()

        self.LBC, self.RBC = self.FLUX*np.ones(len(self.elements)), self.FLUX*np.ones(len(self.elements))
        self.LBCvalue, self.RBCvalue = np.zeros(len(self.elements)), np.zeros(len(self.elements))

        self.cache = True
        self.setHashSensitivity(4)
        self.minComposition = 1e-8

        self.maxCompositionChange = 0.002

    def reset(self):
        '''
        Resets model

        This involves clearing any caches in the Thermodynamics object and this model
        as well as resetting the composition and phase profiles
        '''
        if self.therm is not None:
            self.therm.clearCache()
        
        self.x = np.zeros((len(self.elements), self.N))
        self.p = np.ones((1,self.N)) if len(self.phases) == 1 else np.zeros((len(self.phases), self.N))
        self.hashTable = {}
        self.isSetup = False

    def setThermodynamics(self, thermodynamics):
        '''
        Defines thermodynamics object for the diffusion model

        Parameters
        ----------
        thermodynamics : Thermodynamics object
            Requires the elements in the Thermodynamics and DiffusionModel objects to have the same order
        '''
        self.therm = thermodynamics

    def setTemperature(self, T):
        '''
        Sets iso-thermal temperature

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        '''
        self.T = T

    def save(self, filename, compressed = False, toCSV = False):
        '''
        Saves mesh, composition and phases

        Parameters
        ----------
        filename : str
            File to save to
        compressed : bool
            Whether to compress data if saving to numpy binary format (toCSV = False)
        toCSV : bool
            Whether to output data to a .CSV file format
        '''
        if toCSV:
            headers = ['Distance(m)']
            arrays = [self.z]
            for i in range(len(self.allElements)):
                headers.append('x(' + self.allElements[i] + ')')
                if i == 0:
                    arrays.append(1 - np.sum(self.x, axis=0))
                else:
                    arrays.append(self.x[i-1,:])
            for i in range(len(self.phases)):
                headers.append('f(' + self.phases[i] + ')')
                arrays.append(self.p[i,:])
            rows = zip_longest(*arrays, fillvalue='')
            if '.csv' not in filename.lower():
                filename = filename + '.csv'
            with open(filename, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
                csv.writer(f).writerows(rows)
        else:
            variables = ['zlim', 'N', 'allElements', 'phases', 'z', 'x', 'p']
            vDict = {v: getattr(self, v) for v in variables}
            if compressed:
                np.savez_compressed(filename, **vDict, allow_pickle=True)
            else:
                np.savez(filename, **vDict, allow_pickle=True)

    def load(filename):
        '''
        Loads a previously saved model

        filename : str
            File name to load model from, must include file extension
        '''
        if '.np' in filename.lower():
            data = np.load(filename, allow_pickle=True)
            model = DiffusionModel(data['zlim'], data['N'], data['allElements'], data['phases'])
            model.z = data['z']
            model.x = data['x']
            model.p = data['p']
        else:
            with open(filename, 'r') as csvFile:
                data = csv.reader(csvFile, delimiter=',')
                i = 0
                headers = []
                columns = {}
                for row in data:
                    if i == 0:
                        headers = row
                        columns = {h: [] for h in headers}
                    else:
                        for j in range(len(row)):
                            if row[j] != '':
                                columns[headers[j]].append(float(row[j]))
                    i += 1
            
            elements, phases = [], []
            x, p = [], []
            for h in headers:
                if 'Distance' in h:
                    z = columns[h]
                elif 'x' in h:
                    elements.append(h[2:-1])
                    x.append(columns[h])
                elif 'f' in h:
                    phases.append(h[2:-1])
                    p.append(columns[h])
            model = DiffusionModel([z[0], z[-1]], len(z), elements, phases)
            model.z = np.array(z)
            model.x = np.array(x)[1:,:]
            model.p = np.array(p)
        return model   

    def setHashSensitivity(self, s):
        '''
        Sets sensitivity of the hash table by significant digits

        For example, if a composition set is (0.5693, 0.2937) and s = 3, then
        the hash will be stored as (0.569, 0.294)

        Lower s values will give faster simulation times at the expense of accuracy

        Parameters
        ----------
        s : int
            Number of significant digits to keep for the hash table
        '''
        self.hashSensitivity = np.power(10, int(s))

    def _getHash(self, x):
        '''
        Gets hash value for a composition set

        Parameters
        ----------
        x : list of floats
            Composition set to create hash
        '''
        return hash(tuple((x*self.hashSensitivity).astype(np.int32)))
        #return int(np.sum(np.power(self.hashSensitivity, 1+np.arange(len(x))) * x))

    def useCache(self, use):
        '''
        Whether to use the hash table

        Parameters
        ----------
        use : bool
            If True, then the hash table will be used
        '''
        self.cache = use

    def clearCache(self):
        '''
        Clears hash table
        '''
        self.hashTable = {}

    def _getElementIndex(self, element = None):
        '''
        Gets index of element in self.elements

        Parameters
        ----------
        element : str
            Specified element, will return first element if None
        '''
        if element is None:
            return 0
        else:
            return self.elements.index(element)

    def _getPhaseIndex(self, phase = None):
        '''
        Gets index of phase in self.phases

        Parameters
        ----------
        phase : str
            Specified phase, will return first phase if None
        '''
        if phase is None:
            return 0
        else:
            return self.phases.index(phase)

    def setBC(self, LBCtype = 0, LBCvalue = 0, RBCtype = 0, RBCvalue = 0, element = None):
        '''
        Set boundary conditions

        Parameters
        ----------
        LBCtype : int
            Left boundary condition type
                Mesh1D.FLUX - constant flux
                Mesh1D.COMPOSITION - constant composition
        LBCvalue : float
            Value of left boundary condition
        RBCtype : int
            Right boundary condition type
                Mesh1D.FLUX - constant flux
                Mesh1D.COMPOSITION - constant composition
        RBCvalue : float
            Value of right boundary condition
        element : str
            Specified element to apply boundary conditions on
        '''
        eIndex = self._getElementIndex(element)
        self.LBC[eIndex] = LBCtype
        self.LBCvalue[eIndex] = LBCvalue
        if LBCtype == self.COMPOSITION:
            self.x[eIndex,0] = LBCvalue

        self.RBC[eIndex] = RBCtype
        self.RBCvalue[eIndex] = RBCvalue
        if RBCtype == self.COMPOSITION:
            self.x[eIndex,-1] = RBCvalue

    def setCompositionLinear(self, Lvalue, Rvalue, element = None):
        '''
        Sets composition as a linear function between ends of the mesh

        Parameters
        ----------
        Lvalue : float
            Value at left boundary
        Rvalue : float
            Value at right boundary
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        self.x[eIndex] = np.linspace(Lvalue, Rvalue, self.N)

    def setCompositionStep(self, Lvalue, Rvalue, z, element = None):
        '''
        Sets composition as a step-wise function

        Parameters
        ----------
        Lvalue : float
            Value on left side of mesh
        Rvalue : float
            Value on right side of mesh
        z : float
            Position on mesh where composition switches from Lvalue to Rvalue
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        Lindices = self.z <= z
        self.x[eIndex,Lindices] = Lvalue
        self.x[eIndex,~Lindices] = Rvalue

    def setCompositionSingle(self, value, z, element = None):
        '''
        Sets single node to specified composition

        Parameters
        ----------
        value : float
            Composition
        z : float
            Position to set value to (will use closest node to z)
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        zIndex = np.argmin(np.abs(self.z-z))
        self.x[eIndex,zIndex] = value

    def setCompositionInBounds(self, value, Lbound, Rbound, element = None):
        '''
        Sets single node to specified composition

        Parameters
        ----------
        value : float
            Composition
        Lbound : float
            Position of left bound
        Rbound : float
            Position of right bound
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        indices = (self.z >= Lbound) & (self.z <= Rbound)
        self.x[eIndex,indices] = value

    def setCompositionFunction(self, func, element = None):
        '''
        Sets composition as a function of z

        Parameters
        ----------
        func : function
            Function taking in z and returning composition
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        self.x[eIndex,:] = func(self.z)

    def setCompositionProfile(self, z, x, element = None):
        '''
        Sets composition profile by linear interpolation

        Parameters
        ----------
        z : array
            z-coords of composition profile
        x : array
            Composition profile
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        z = np.array(z)
        x = np.array(x)
        sortIndices = np.argsort(z)
        z = z[sortIndices]
        x = x[sortIndices]
        self.x[eIndex,:] = np.interp(self.z, z, x)

    def setup(self):
        '''
        General setup function for all diffusio models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if self.therm is not None:
            self.therm.clearCache()
        xsum = np.sum(self.x, axis=0)
        if any(xsum > 1):
            print('Compositions add up to above 1 between z = [{:.3e}, {:.3e}]'.format(np.amin(self.z[xsum>1]), np.amax(self.z[xsum>1])))
            raise Exception('Some compositions sum up to above 1')
        self.x[self.x > self.minComposition] = self.x[self.x > self.minComposition] - len(self.allElements) * self.minComposition
        self.x[self.x < self.minComposition] = self.minComposition
        self.isSetup = True

    def getFluxes(self):
        '''
        "Virtual" function to be implemented by child objects
        '''
        return [], []

    def updateMesh(self):
        '''
        "Virtual" function to be implemented by child objects
        '''
        pass

    def update(self):
        '''
        Updates the mesh by a given dt that is calculated for numerical stability
        '''
        #Get fluxes
        fluxes, dt = self.getFluxes()

        if self.t + dt > self.tf:
            dt = self.tf - self.t

        #Update mesh
        self.updateMesh(fluxes, dt)
        self.x[self.x < self.minComposition] = self.minComposition
        self.t += dt

    def solve(self, simTime, verbose=False, vIt=10):
        '''
        Solves the model by updated the mesh until the final simulation time is met
        '''
        self.setup()

        self.t = 0
        self.tf = simTime
        i = 0
        t0 = time.time()
        if verbose:
            print('Iteration\tSim Time (h)\tRun time (s)')
        while self.t < self.tf:
            if verbose and i % vIt == 0:
                tf = time.time()
                print(str(i) + '\t\t{:.3f}\t\t{:.3f}'.format(self.t/3600, tf-t0))
            self.update()
            i += 1

        tf = time.time()
        print(str(i) + '\t\t{:.3f}\t\t{:.3f}'.format(self.t/3600, tf-t0))

    def getX(self, element):
        '''
        Gets composition profile of element
        
        Parameters
        ----------
        element : str
            Element to get profile of
        '''
        if element in self.allElements and element not in self.elements:
            return 1 - np.sum(self.x, axis=0)
        else:
            e = self._getElementIndex(element)
            return self.x[e]

    def getP(self, phase):
        '''
        Gets phase profile

        Parameters
        ----------
        phase : str
            Phase to get profile of
        '''
        p = self._getPhaseIndex(phase)
        return self.p[p]

    def plot(self, ax, plotReference = True, plotElement = None, zScale = 1, *args, **kwargs):
        '''
        Plots composition profile

        Parameters
        ----------
        ax : matplotlib Axes object
            Axis to plot on
        plotReference : bool
            Whether to plot reference element (composition = 1 - sum(composition of rest of elements))
        plotElement : None or str
            Plots single element if it is defined, otherwise, all elements are plotted
        zScale : float
            Scale factor for z-coordinates
        '''
        if not self.isSetup:
            self.setup()

        if plotElement is not None:
            if plotElement not in self.elements and plotElement in self.allElements:
                x = 1 - np.sum(self.x, axis=0)
            else:
                e = self._getElementIndex(plotElement)
                x = self.x[e]
            ax.plot(self.z/zScale, x, *args, **kwargs)
        else:
            if plotReference:
                refE = 1 - np.sum(self.x, axis=0)
                ax.plot(self.z/zScale, refE, label=self.allElements[0], *args, **kwargs)
            for e in range(len(self.elements)):
                ax.plot(self.z/zScale, self.x[e], label=self.elements[e], *args, **kwargs)
            
        ax.set_xlim([self.zlim[0]/zScale, self.zlim[1]/zScale])
        ax.legend()
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Composition (at.%)')

    def plotTwoAxis(self, axL, Lelements, Relements, zScale = 1, *args, **kwargs):
        '''
        Plots composition profile with two y-axes

        Parameters
        ----------
        axL : matplotlib Axes object
            Left axis to plot on
        Lelements : list of str
            Elements to plot on left axis
        Relements : list of str
            Elements to plot on right axis
        zScale : float
            Scale factor for z-coordinates
        '''
        if not self.isSetup:
            self.setup()

        if type(Lelements) is str:
            Lelements = [Lelements]
        if type(Relements) is str:
            Relements = [Relements]

        ci = 0
        refE = 1 - np.sum(self.x, axis=0)
        axR = axL.twinx()
        for e in range(len(Lelements)):
            if Lelements[e] in self.elements:
                eIndex = self._getElementIndex(Lelements[e])
                axL.plot(self.z/zScale, self.x[eIndex], label=self.elements[eIndex], color = 'C' + str(ci), *args, **kwargs)
                ci = ci+1 if ci <= 9 else 0
            elif Lelements[e] in self.allElements:
                axL.plot(self.z/zScale, refE, label=self.allElements[0], color = 'C' + str(ci), *args, **kwargs)
                ci = ci+1 if ci <= 9 else 0
        for e in range(len(Relements)):
            if Relements[e] in self.elements:
                eIndex = self._getElementIndex(Relements[e])
                axR.plot(self.z/zScale, self.x[eIndex], label=self.elements[eIndex], color = 'C' + str(ci), *args, **kwargs)
                ci = ci+1 if ci <= 9 else 0
            elif Relements[e] in self.allElements:
                axR.plot(self.z/zScale, refE, label=self.allElements[0], color = 'C' + str(ci), *args, **kwargs)
                ci = ci+1 if ci <= 9 else 0

        
        axL.set_xlim([self.zlim[0]/zScale, self.zlim[1]/zScale])
        axL.set_xlabel('Distance (m)')
        axL.set_ylabel('Composition (at.%) ' + str(Lelements))
        axR.set_ylabel('Composition (at.%) ' + str(Relements))
        
        lines, labels = axL.get_legend_handles_labels()
        lines2, labels2 = axR.get_legend_handles_labels()
        axR.legend(lines+lines2, labels+labels2, framealpha=1)

        return axL, axR

    def plotPhases(self, ax, plotPhase = None, zScale = 1, *args, **kwargs):
        '''
        Plots phase fractions over z

        Parameters
        ----------
        ax : matplotlib Axes object
            Axis to plot on
        plotPhase : None or str
            Plots single phase if it is defined, otherwise, all phases are plotted
        zScale : float
            Scale factor for z-coordinates
        '''
        if not self.isSetup:
            self.setup()

        if plotPhase is not None:
            p = self._getPhaseIndex(plotPhase)
            ax.plot(self.z/zScale, self.p[p], *args, **kwargs)
        else:
            for p in range(len(self.phases)):
                ax.plot(self.z/zScale, self.p[p], label=self.phases[p], *args, **kwargs)
        ax.set_xlim([self.zlim[0]/zScale, self.zlim[1]/zScale])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Phase Fraction')
        ax.legend()

class SinglePhaseModel(DiffusionModel):
    def getFluxes(self):
        '''
        Gets fluxes at the boundary of each nodes

        Returns
        -------
        fluxes : (e-1, n+1) array of floats
            e - number of elements including reference element
            n - number of nodes
        dt : float
            Maximum calculated time interval for numerical stability
        '''
        xMid = (self.x[:,1:] + self.x[:,:-1]) / 2

        if len(self.elements) == 1:
            d = np.zeros(self.N-1)
        else:
            d = np.zeros((self.N-1, len(self.elements), len(self.elements)))
        if self.cache:
            for i in range(self.N-1):
                hashValue = self._getHash(xMid[:,i])
                if hashValue not in self.hashTable:
                    self.hashTable[hashValue] = self.therm.getInterdiffusivity(xMid[:,i], self.T, phase=self.phases[0])
                d[i] = self.hashTable[hashValue]
        else:
            d = self.therm.getInterdiffusivity(xMid.T, self.T*np.ones(self.N-1), phase=self.phases[0])

        dxdz = (self.x[:,1:] - self.x[:,:-1]) / self.dz
        fluxes = np.zeros((len(self.elements), self.N+1))
        if len(self.elements) == 1:
            fluxes[0,1:-1] = -d * dxdz
        else:
            dxdz = np.expand_dims(dxdz, axis=0)
            fluxes[:,1:-1] = -np.matmul(d, np.transpose(dxdz, (2,1,0)))[:,:,0].T
        for e in range(len(self.elements)):
            fluxes[e,0] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else fluxes[e,1]
            fluxes[e,-1] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else fluxes[e,-2]

        dt = 0.4 * self.dz**2 / np.amax(np.abs(d))

        return fluxes, dt

    def updateMesh(self, fluxes, dt):
        '''
        Updates mesh using fluxes by time increment dt

        Parameters
        ----------
        fluxes : 2D array
            Fluxes for each element between each node. Size must be (E, N-1)
                E - number of elements (NOT including reference element)
                N - number of nodes
            Boundary conditions will automatically be applied
        dt : float
            Time increment
        '''
        for e in range(len(self.elements)):
            self.x[e] += -(fluxes[e,1:] - fluxes[e,:-1]) * dt / self.dz

class HomogenizationModel(DiffusionModel):
    def __init__(self, zlim, N, elements = ['A', 'B'], phases = ['alpha']):
        super().__init__(zlim, N, elements, phases)

        self.mobilityFunction = self.wienerUpper
        self.defaultMob = 0
        self.eps = 0.05

        self.sortIndices = np.argsort(self.allElements)
        self.unsortIndices = np.argsort(self.sortIndices)
        self.labFactor = 1

    def reset(self):
        '''
        Resets model

        This also includes chemical potential and pycalphad CompositionSets for each node
        '''
        super().reset()
        self.mu = np.zeros((len(self.elements)+1, self.N))
        self.compSets = [None for _ in range(self.N)]

    def setMobilityFunction(self, function):
        '''
        Sets averaging function to use for mobility

        Default mobility value should be that a phase of unknown mobility will be ignored for average mobility calcs

        Parameters
        ----------
        function : str
            Options - 'upper wiener', 'lower wiener', 'upper hashin-shtrikman', 'lower hashin-strikman', 'labyrinth'
        '''
        #np.finfo(dtype).max - largest representable value
        #np.finfo(dtype).tiny - smallest positive usable value
        if 'upper' in function and 'wiener' in function:
            self.mobilityFunction = self.wienerUpper
            self.defaultMob = np.finfo(np.float64).tiny
        elif 'lower' in function and 'wiener' in function:
            self.mobilityFunction = self.wienerLower
            self.defaultMob = np.finfo(np.float64).max
        elif 'upper' in function and 'hashin' in function:
            self.mobilityFunction = self.hashin_shtrikmanUpper
            self.defaultMob = np.finfo(np.float64).tiny
        elif 'lower' in function and 'hashin' in function:
            self.mobilityFunction = self.hashin_shtrikmanLower
            self.defaultMob = np.finfo(np.float64).max
        elif 'lab' in function:
            self.mobilityFunction = self.labyrinth
            self.defaultMob = np.finfo(np.float64).tiny

    def setLabyrinthFactor(self, n):
        '''
        Labyrinth factor

        Parameters
        ----------
        n : int
            Either 1 or 2
            Note: n = 1 will the same as the weiner upper bounds
        '''
        if n < 1:
            n = 1
        if n > 2:
            n = 2
        self.labFactor = n

    def setup(self):
        '''
        Sets up model

        This also includes getting the CompositionSets for each node
        '''
        super().setup()
        #self.midX = 0.5 * (self.x[:,1:] + self.x[:,:-1])
        self.p = self.updateCompSets(self.x)

    def _newEqCalc(self, x):
        '''
        Calculates equilibrium and returns a CompositionSet
        '''
        eq = self.therm.getEq(x, self.T, 0, self.phases)
        state_variables = np.array([0, 1, 101325, self.T], dtype=np.float64)
        stable_phases = eq.Phase.values.ravel()
        phase_amounts = eq.NP.values.ravel()
        comp = []
        for p in stable_phases:
            if p != '':
                idx = np.where(stable_phases == p)[0]
                cs, misc = self.therm._createCompositionSet(eq, state_variables, p, phase_amounts, idx)
                comp.append(cs)

        if len(comp) == 0:
            comp = None

        return self.therm.getLocalEq(x, self.T, 0, self.phases, comp)

    def updateCompSets(self, xarray):
        '''
        Updates the array of CompositionSets

        If an equilibrium calculation is already done for a given composition, 
        the CompositionSet will be taken out of the hash table

        Otherwise, a new equilibrium calculation will be performed

        Parameters
        ----------
        xarray : (e-1, N) array
            Composition for each node
            e is number of elements
            N is number of nodes

        Returns
        -------
        parray : (p, N) array
            Phase fractions for each node
            p is number of phases
        '''
        parray = np.zeros((len(self.phases), xarray.shape[1]))
        for i in range(parray.shape[1]):
            if self.cache:
                hashValue = self._getHash(xarray[:,i])
                if hashValue not in self.hashTable:
                    result, comp = self._newEqCalc(xarray[:,i])
                    #result, comp = self.therm.getLocalEq(xarray[:,i], self.T, 0, self.phases, self.compSets[i])
                    self.hashTable[hashValue] = (result, comp, None)
                else:
                    result, comp, _ = self.hashTable[hashValue]
                results, self.compSets[i] = copy.copy(result), copy.copy(comp)
            else:
                if self.compSets[i] is None:
                    results, self.compSets[i] = self._newEqCalc(xarray[:,i])
                else:
                    results, self.compSets[i] = self.therm.getLocalEq(xarray[:,i], self.T, 0, self.phases, self.compSets[i])
            self.mu[:,i] = results.chemical_potentials[self.unsortIndices]
            cs_phases = [cs.phase_record.phase_name for cs in self.compSets[i]]
            for p in range(len(cs_phases)):
                parray[self._getPhaseIndex(cs_phases[p]), i] = self.compSets[i][p].NP
        
        return parray

    def getMobility(self, xarray):
        '''
        Gets mobility of all phases

        Returns
        -------
        (p, e+1, N) array - p is number of phases, e is number of elements, N is number of nodes
        '''
        mob = self.defaultMob * np.ones((len(self.phases), len(self.elements)+1, xarray.shape[1]))
        for i in range(xarray.shape[1]):
            if self.cache:
                hashValue = self._getHash(xarray[:,i])
                _, _, mTemp = self.hashTable[hashValue]
            else:
                mTemp = None
            if mTemp is None or not self.cache:
                maxPhaseAmount = 0
                maxPhaseIndex = 0
                for p in range(len(self.phases)):
                    if self.p[p,i] > 0:
                        if self.p[p,i] > maxPhaseAmount:
                            maxPhaseAmount = self.p[p,i]
                            maxPhaseIndex = p
                        if self.phases[p] in self.therm.mobCallables and self.therm.mobCallables[self.phases[p]] is not None:
                            #print(self.phases, self.phases[p], xarray[:,i], self.p[:,i], i, self.compSets[i])
                            compset = [cs for cs in self.compSets[i] if cs.phase_record.phase_name == self.phases[p]][0]
                            mob[p,:,i] = mobility_from_composition_set(compset, self.therm.mobCallables[self.phases[p]], self.therm.mobility_correction)[self.unsortIndices]
                            mob[p,:,i] *= np.concatenate(([1-np.sum(xarray[:,i])], xarray[:,i]))
                        else:
                            mob[p,:,i] = -1
                for p in range(len(self.phases)):
                    if any(mob[p,:,i] == -1) and not all(mob[p,:,i] == -1):
                        mob[p,:,i] = mob[maxPhaseIndex,:,i]
                    if all(mob[p,:,i] == -1):
                        mob[p,:,i] = self.defaultMob
                if self.cache:
                    self.hashTable[hashValue] = (self.hashTable[hashValue][0], self.hashTable[hashValue][1], copy.copy(mob[:,:,i]))
            else:
                mob[:,:,i] = mTemp

        return mob

    def wienerUpper(self, xarray):
        '''
        Upper wiener bounds for average mobility

        Returns
        -------
        (e+1, N) mobility array - e is number of elements, N is number of nodes
        '''
        mob = self.getMobility(xarray)
        avgMob = np.sum(np.multiply(self.p[:,np.newaxis], mob), axis=0)
        return avgMob

    def wienerLower(self, xarray):
        '''
        Lower wiener bounds for average mobility

        Returns
        -------
        (e+1, N) mobility array - e is number of elements, N is number of nodes
        '''
        #(p, e, N)
        mob = self.getMobility(xarray)
        avgMob = 1/np.sum(np.multiply(self.p[:,np.newaxis], 1/mob), axis=0)
        return avgMob

    def labyrinth(self, xarray):
        '''
        Labyrinth mobility

        Returns
        -------
        (e+1, N) mobility array - e is number of elements, N is number of nodes
        '''
        mob = self.getMobility(xarray)
        avgMob = np.sum(np.multiply(np.power(self.p[:,np.newaxis], self.labFactor), mob), axis=0)
        return avgMob

    def hashin_shtrikmanUpper(self, xarray):
        '''
        Upper hashin shtrikman bounds for average mobility

        Returns
        -------
        (e+1, N) mobility array - e is number of elements, N is number of nodes
        '''
        #self.p                                 #(p,N)
        mob = self.getMobility(xarray)          #(p,e+1,N)
        maxMob = np.amax(mob, axis=0)           #(e+1,N)

        # 1 / ((1 / mPhi - mAlpha) + 1 / (3mAlpha)) = 3mAlpha * (mPhi - mAlpha) / (2mAlpha + mPhi)
        Ak = 3 * maxMob * (mob - maxMob) / (2*maxMob + mob)
        Ak = Ak * self.p[:,np.newaxis]
        Ak = np.sum(Ak, axis=0)
        avgMob = maxMob + Ak / (1 - Ak / (3*maxMob))
        return avgMob

    def hashin_shtrikmanLower(self, xarray):
        '''
        Lower hashin shtrikman bounds for average mobility

        Returns
        -------
        (e, N) mobility array - e is number of elements, N is number of nodes
        '''
        #self.p                                 #(p,N)
        mob = self.getMobility(xarray)          #(p,e+1,N)
        minMob = np.amin(mob, axis=0)           #(e+1,N)

        #This prevents an infinite mobility which could cause the time interval to be 0
        minMob[minMob == np.inf] = 0

        # 1 / ((1 / mPhi - mAlpha) + 1 / (3mAlpha)) = 3mAlpha * (mPhi - mAlpha) / (2mAlpha + mPhi)
        Ak = 3 * minMob * (mob - minMob) / (2*minMob + mob)

        Ak = Ak * self.p[:,np.newaxis]
        Ak = np.sum(Ak, axis=0)
        avgMob = minMob + Ak / (1 - Ak / (3*minMob))
        return avgMob

    def getFluxes(self):
        '''
        Return fluxes and time interval for the current iteration
        '''
        self.p = self.updateCompSets(self.x)

        #Get average mobility between nodes
        avgMob = self.mobilityFunction(self.x)
        avgMob = 0.5 * (avgMob[:,1:] + avgMob[:,:-1])

        #Composition between nodes
        avgX = 0.5 * (self.x[:,1:] + self.x[:,:-1])
        avgX = np.concatenate(([1-np.sum(avgX, axis=0)], avgX), axis=0)

        #Chemical potential gradient
        dmudz = (self.mu[:,1:] - self.mu[:,:-1]) / self.dz

        #Composition gradient (we need to calculate gradient for reference element)
        dxdz = (self.x[:,1:] - self.x[:,:-1]) / self.dz
        dxdz = np.concatenate(([0-np.sum(dxdz, axis=0)], dxdz), axis=0)

        # J = -M * dmu/dz
        # Ideal contribution: J_id = -eps * M*R*T / x * dx/dz
        fluxes = np.zeros((len(self.elements)+1, self.N-1))
        fluxes = -avgMob * dmudz
        nonzeroComp = avgX != 0
        fluxes[nonzeroComp] += -self.eps * avgMob[nonzeroComp] * 8.314 * self.T * dxdz[nonzeroComp] / avgX[nonzeroComp]

        #Flux in a volume fixed frame: J_vi = J_i - x_i * sum(J_j)
        vfluxes = np.zeros((len(self.elements), self.N+1))
        vfluxes[:,1:-1] = fluxes[1:,:] - avgX[1:,:] * np.sum(fluxes, axis=0)

        #Boundary conditions
        for e in range(len(self.elements)):
            vfluxes[e,0] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else vfluxes[e,1]
            vfluxes[e,-1] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else vfluxes[e,-2]

        #Time increment
        #This is done by finding the time interval such that the composition
        # change caused by the fluxes will be lower than self.maxCompositionChange
        dJ = np.abs(vfluxes[:,1:] - vfluxes[:,:-1]) / self.dz
        dt = self.maxCompositionChange / np.amax(dJ[dJ!=0])

        return vfluxes, dt
        
    def updateMesh(self, fluxes, dt):
        '''
        Updates the mesh based off the fluxes and time interval
        '''
        for e in range(len(self.elements)):
            self.x[e] += -(fluxes[e,1:] - fluxes[e,:-1]) * dt / self.dz