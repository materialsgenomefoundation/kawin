import numpy as np
import matplotlib.pyplot as plt
from kawin.Mobility import mobility_from_composition_set
import time
import csv
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

        self.setHashSensitivity(4)
        self.minComposition = 1e-9

    def reset(self):
        if self.therm is not None:
            self.therm.removeCache()
        
        self.x = np.zeros((len(self.elements), self.N))
        self.p = np.ones((1,self.N)) if len(self.phases) == 1 else np.zeros((len(self.phases), self.N))
        self.hashTable = {}

    def setThermodynamics(self, thermodynamics):
        self.therm = thermodynamics

    def setTemperature(self, T):
        self.T = T

    def save(self, filename, compressed = False, toCSV = False):
        #Saves mesh, composition and phases
        if toCSV:
            headers = ['Distance (m)']
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
        self.hashSensitivity = np.power(10, int(s))

    def _getHash(self, x):
        return int(np.sum(np.power(self.hashSensitivity, 1+np.arange(len(x))) * x))

    def _getElementIndex(self, element = None):
        '''
        Gets index of element in self.elements

        Parameters
        ----------
        elements : str
            Specified element, will return first element if None
        '''
        if element is None:
            return 0
        else:
            return self.elements.index(element)

    def _getPhaseIndex(self, phase = None):
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

    def setup(self):
        self.x[self.x < self.minComposition] = self.minComposition

    def getFluxes(self):
        return [], []

    def updateMesh(self):
        pass

    def update(self):
        #Get fluxes
        fluxes, dt = self.getFluxes()

        if self.t + dt > self.tf:
            dt = self.tf - self.t

        #Update mesh
        self.updateMesh(fluxes, dt)
        self.x[self.x < self.minComposition] = self.minComposition
        self.t += dt

    def solve(self, simTime, verbose=False, vIt=10):
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

    def plot(self, ax, plotReference = True, zScale = 1):
        '''
        Plots composition profile

        Parameters
        ----------
        ax : matplotlib Axes object
            Axis to plot on
        plotReference : bool
            Whether to plot reference element (composition = 1 - sum(composition of rest of elements))
        '''
        if plotReference:
            refE = 1 - np.sum(self.x, axis=0)
            ax.plot(self.z/zScale, refE, label=self.allElements[0])
        for e in range(len(self.elements)):
            ax.plot(self.z/zScale, self.x[e], label=self.elements[e])
            
        ax.set_xlim([self.zlim[0]/zScale, self.zlim[1]/zScale])
        ax.legend()
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Composition (at.%)')

    def plotTwoAxis(self, axL, Lelements, Relements, zScale = 1):
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
                axL.plot(self.z/zScale, self.x[eIndex], label=self.elements[eIndex], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0
            elif Lelements[e] in self.allElements:
                axL.plot(self.z/zScale, refE, label=self.allElements[0], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0
        for e in range(len(Relements)):
            if Relements[e] in self.elements:
                eIndex = self._getElementIndex(Relements[e])
                axR.plot(self.z/zScale, self.x[eIndex], label=self.elements[eIndex], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0
            elif Relements[e] in self.allElements:
                axR.plot(self.z/zScale, refE, label=self.allElements[0], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0

        
        axL.set_xlim([self.zlim[0]/zScale, self.zlim[1]/zScale])
        axL.set_xlabel('Distance (m)')
        axL.set_ylabel('Composition (at.%) ' + str(Lelements))
        axR.set_ylabel('Composition (at.%) ' + str(Relements))
        
        lines, labels = axL.get_legend_handles_labels()
        lines2, labels2 = axR.get_legend_handles_labels()
        axR.legend(lines+lines2, labels+labels2, framealpha=1)

        return axL, axR

    def plotPhases(self, ax, zScale = 1):
        for p in range(len(self.phases)):
            ax.plot(self.z * zScale, self.p[p], label=self.phases[p])
        ax.set_xlim([self.zlim[0]/zScale, self.zlim[1]/zScale])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Phase Fraction')
        ax.legend()

class SinglePhaseModel(DiffusionModel):
    def getFluxes(self):
        xMid = (self.x[:,1:] + self.x[:,:-1]) / 2

        if len(self.elements) == 1:
            d = np.zeros(self.N-1)
        else:
            d = np.zeros((self.N-1, len(self.elements), len(self.elements)))
        for i in range(self.N-1):
            hashValue = self._getHash(xMid[:,i])
            if hashValue not in self.hashTable:
                self.hashTable[hashValue] = self.therm.getInterdiffusivity(xMid[:,i], self.T, phase=self.phases[0])
            d[i] = self.hashTable[hashValue]
        #d = self.therm.getInterdiffusivity(xMid.T, self.T*np.ones(self.N-1), phase=self.phases[0])

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
    def __init__(self, *args):
        super().__init__(*args)

        self.mu = np.zeros((len(self.elements)+1, self.N))
        self.compSets = [None for _ in range(self.N)]
        self.mobilityFunction = self.wienerUpper
        self.defaultMob = 0
        self.eps = 0.05

        self.sortIndices = np.argsort(self.allElements)
        self.unsortIndices = np.argsort(self.sortIndices)

    def setMobilityFunction(self, function):
        if 'upper' in function and 'wiener' in function:
            self.mobilityFunction = self.wienerUpper
            self.defaultMob = 0
        elif 'lower' in function and 'wiener' in function:
            self.mobilityFunction = self.wienerLower
            self.defaultMob = 1
        elif 'upper' in function and 'hasin' in function:
            self.mobilityFunction = self.hashin_shtrikmanUpper
            self.defaultMob = 0
        elif 'lower' in function and 'hashin' in function:
            self.mobilityFunction = self.hashin_shtrikmanLower
            self.defaultMob = 1

    def setup(self):
        super().setup()
        self.updateCompSets()

    def updateCompSets(self):
        self.p = np.zeros((len(self.phases), self.N))
        for i in range(self.N):
            if self.compSets[i] is None:
                eq = self.therm.getEq(self.x[:,i], self.T, 0, self.phases)
                state_variables = np.array([0, 1, 101325, self.T], dtype=np.float64)
                stable_phases = eq.Phase.values.ravel()
                phase_amounts = eq.NP.values.ravel()
                self.compSets[i] = []
                for p in stable_phases:
                    if p != '':
                        idx = np.where(stable_phases == p)[0]
                        cs, misc = self.therm._createCompositionSet(eq, state_variables, p, phase_amounts, idx)
                        self.compSets[i].append(cs)
            results, self.compSets[i] = self.therm.getLocalEq(self.x[:,i], self.T, 0, self.phases, self.compSets[i])
            self.mu[:,i] = results.chemical_potentials[self.unsortIndices]
            cs_phases = [cs.phase_record.phase_name for cs in self.compSets[i]]
            for p in range(len(cs_phases)):
                self.p[self._getPhaseIndex(cs_phases[p]), i] = self.compSets[i][p].NP

    def getMobility(self):
        '''
        Gets mobility of all phases

        Returns
        -------
        (p, e+1, N) array - p is number of phases, e is number of elements, N is number of nodes
        '''
        mob = self.defaultMob * np.ones((len(self.phases), len(self.elements)+1, self.N))
        for i in range(self.N):
            maxPhaseAmount = 0
            maxPhaseIndex = 0
            for p in range(len(self.phases)):
                if self.p[p,i] > 0:
                    if self.p[p,i] > maxPhaseAmount:
                        maxPhaseAmount = self.p[p,i]
                        maxPhaseIndex = p
                    if self.phases[p] in self.therm.mobCallables:
                        compset = [cs for cs in self.compSets[i] if cs.phase_record.phase_name == self.phases[p]][0]
                        mob[p,:,i] = mobility_from_composition_set(compset, self.therm.mobCallables[self.phases[p]], self.therm.mobility_correction)[self.unsortIndices]
                        mob[p,:,i] *= np.concatenate(([1-np.sum(self.x[:,i])], self.x[:,i]))
                    else:
                        mob[p,:,i] = -1
            for p in range(len(self.phases)):
                if any(mob[p,:,i] == -1):
                    mob[p,:,i] = mob[maxPhaseIndex,:,i]

        return mob

    def wienerUpper(self):
        '''
        Upper wiener bounds for average mobility

        Returns
        -------
        (e+1, N) mobility array - e is number of elements, N is number of nodes
        '''
        mob = self.getMobility()
        avgMob = np.sum(np.multiply(self.p[:,np.newaxis], mob), axis=0)
        return avgMob

    def wienerLower(self):
        '''
        Lower wiener bounds for average mobility

        Returns
        -------
        (e+1, N) mobility array - e is number of elements, N is number of nodes
        '''
        #(p, e, N)
        mob = self.getMobility()
        avgMob = 1/np.sum(np.multiply(self.p[:,np.newaxis], 1/mob), axis=0)
        return avgMob

    def hashin_shtrikmanUpper(self):
        '''
        Upper hashin shtrikman bounds for average mobility

        Returns
        -------
        (e, N) mobility array - e is number of elements, N is number of nodes
        '''
        pass

    def hashin_shtrikmanLower(self):
        '''
        Lower hashin shtrikman bounds for average mobility

        Returns
        -------
        (e, N) mobility array - e is number of elements, N is number of nodes
        '''
        pass

    def getFluxes(self):
        #Get average mobility between nodes
        avgMob = self.mobilityFunction()
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
        #HOW DO WE DO THIS!?!?!?
        #nonzero = (vfluxes[:,1:-1] != 0) & (dxdz[1:,:] != 0)
        #D = vfluxes[:,1:-1][nonzero] / dxdz[1:,:][nonzero]
        #dt = 0.1 * self.dz**2 / np.amax(np.abs(D))
        #print(dt, np.amax(np.abs(D)), np.amax(vfluxes[:,1:-1][nonzero]), np.amin(dxdz[1:,:][nonzero]))

        dJ = np.abs(vfluxes[:,1:] - vfluxes[:,:-1]) / self.dz
        dt = 0.005 / np.amax(dJ[dJ!=0])
        print(dt)
        #nonzero = (self.x != 0) & (dJ != 0)
        #dt = 0.05 * np.amin(self.x[nonzero]) / np.amax(dJ[nonzero])
        #dt = 0.01 * np.amin(self.x[nonzero] / dJ[nonzero])
        #dt = 10

        self.updateCompSets()

        return vfluxes, dt
        

    def updateMesh(self, fluxes, dt):
        for e in range(len(self.elements)):
            self.x[e] += -(fluxes[e,1:] - fluxes[e,:-1]) * dt / self.dz