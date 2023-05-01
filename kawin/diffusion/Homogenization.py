import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.Mobility import mobility_from_composition_set
import copy

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

    def _newEqCalc(self, x, T):
        '''
        Calculates equilibrium and returns a CompositionSet
        '''
        eq = self.therm.getEq(x, T, 0, self.phases)
        state_variables = np.array([0, 1, 101325, T], dtype=np.float64)
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

        return self.therm.getLocalEq(x, T, 0, self.phases, comp)

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
                hashValue = self._getHash(xarray[:,i], self.T[i])
                if hashValue not in self.hashTable:
                    result, comp = self._newEqCalc(xarray[:,i], self.T[i])
                    #result, comp = self.therm.getLocalEq(xarray[:,i], self.T, 0, self.phases, self.compSets[i])
                    self.hashTable[hashValue] = (result, comp, None)
                else:
                    result, comp, _ = self.hashTable[hashValue]
                results, self.compSets[i] = copy.copy(result), copy.copy(comp)
            else:
                if self.compSets[i] is None:
                    results, self.compSets[i] = self._newEqCalc(xarray[:,i], self.T[i])
                else:
                    results, self.compSets[i] = self.therm.getLocalEq(xarray[:,i], self.T[i], 0, self.phases, self.compSets[i])
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
                hashValue = self._getHash(xarray[:,i], self.T[i])
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
        self.T = self.Tfunc(self.z, self.t)
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
        Tmid = (self.T[1:] + self.T[:-1]) / 2
        Tmidfull = Tmid[np.newaxis,:]
        for i in range(fluxes.shape[0]-1):
            Tmidfull = np.concatenate((Tmidfull, Tmid[np.newaxis,:]), axis=0)
        fluxes[nonzeroComp] += -self.eps * avgMob[nonzeroComp] * 8.314 * Tmidfull[nonzeroComp] * dxdz[nonzeroComp] / avgX[nonzeroComp]

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