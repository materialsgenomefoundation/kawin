import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.Mesh import geometricMean
from kawin.thermo.Mobility import mobility_from_composition_set
import copy

class HomogenizationModel(DiffusionModel):
    def __init__(self, zlim, N, elements = ['A', 'B'], phases = ['alpha'], mesh=None, record = True):
        super().__init__(zlim, N, elements, phases, mesh, record)

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
        #self.mu = np.zeros((len(self.elements)+1, self.N))
        self.mu = np.zeros((self.N, len(self.elements)+1))
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
        wks = self.therm.getEq(x, T, 0, self.phases)
        chemical_potentials = np.squeeze(wks.eq.MU)
        composition_sets = wks.get_composition_sets()
        return chemical_potentials, composition_sets

    def updateCompSets(self, xarray):
        '''
        Updates the array of CompositionSets

        If an equilibrium calculation is already done for a given composition, 
        the CompositionSet will be taken out of the hash table

        Otherwise, a new equilibrium calculation will be performed

        Parameters
        ----------
        xarray : (N, e-1) array
            Composition for each node
            e is number of elements
            N is number of nodes

        Returns
        -------
        parray : (n, p) array
            Phase fractions for each node
            p is number of phases
        '''
        #parray = np.zeros((len(self.phases), xarray.shape[1]))
        parray = np.zeros((xarray.shape[0], len(self.phases)))
        for i in range(parray.shape[0]):
            if self.cache:
                hashValue = self._getHash(xarray[i], self.T[i])
                if hashValue not in self.hashTable:
                    chemical_potentials, comp = self._newEqCalc(xarray[i], self.T[i])
                    self.hashTable[hashValue] = (chemical_potentials, comp, None)
                else:
                    chemical_potentials, comp, _ = self.hashTable[hashValue]
                chemical_potentials, self.compSets[i] = copy.copy(chemical_potentials), copy.copy(comp)
            else:
                chemical_potentials, self.compSets[i] = self._newEqCalc(xarray[i], self.T[i])
            self.mu[i] = chemical_potentials[self.unsortIndices]
            cs_phases = [cs.phase_record.phase_name for cs in self.compSets[i]]
            for p in range(len(cs_phases)):
                parray[i, self._getPhaseIndex(cs_phases[p])] = self.compSets[i][p].NP
        
        return parray

    def getMobility(self, xarray):
        '''
        Gets mobility of all phases

        Returns
        -------
        (N, p, e+1) array - p is number of phases, e is number of elements, N is number of nodes
        '''
        #mob = self.defaultMob * np.ones((len(self.phases), len(self.elements)+1, xarray.shape[1]))
        mob = self.defaultMob * np.ones((xarray.shape[0], len(self.phases), len(self.elements)+1))
        for i in range(xarray.shape[0]):
            if self.cache:
                hashValue = self._getHash(xarray[i], self.T[i])
                _, _, mTemp = self.hashTable[hashValue]
            else:
                mTemp = None
            if mTemp is None or not self.cache:
                maxPhaseAmount = 0
                maxPhaseIndex = 0
                for p in range(len(self.phases)):
                    if self.p[i,p] > 0:
                        if self.p[i,p] > maxPhaseAmount:
                            maxPhaseAmount = self.p[i,p]
                            maxPhaseIndex = p
                        if self.phases[p] in self.therm.mobCallables and self.therm.mobCallables[self.phases[p]] is not None:
                            #print(self.phases, self.phases[p], xarray[:,i], self.p[:,i], i, self.compSets[i])
                            compset = [cs for cs in self.compSets[i] if cs.phase_record.phase_name == self.phases[p]][0]
                            mob[i,p] = mobility_from_composition_set(compset, self.therm.mobCallables[self.phases[p]], self.therm.mobility_correction)[self.unsortIndices]
                            mob[i,p] *= np.concatenate(([1-np.sum(xarray[i])], xarray[i]))
                        else:
                            mob[i,p] = -1
                for p in range(len(self.phases)):
                    if any(mob[i,p] == -1) and not all(mob[i,p] == -1):
                        mob[i,p] = mob[i,maxPhaseIndex]
                    if all(mob[i,p] == -1):
                        mob[i,p] = self.defaultMob
                if self.cache:
                    self.hashTable[hashValue] = (self.hashTable[hashValue][0], self.hashTable[hashValue][1], copy.copy(mob[i]))
            else:
                mob[i] = mTemp

        return mob

    def wienerUpper(self, xarray):
        '''
        Upper wiener bounds for average mobility

        Returns
        -------
        (N, e+1) mobility array - e is number of elements, N is number of nodes
        '''
        mob = self.getMobility(xarray)
        avgMob = np.sum(np.multiply(self.p[:,:,np.newaxis], mob), axis=1)
        return avgMob

    def wienerLower(self, xarray):
        '''
        Lower wiener bounds for average mobility

        Returns
        -------
        (N, e+1) mobility array - e is number of elements, N is number of nodes
        '''
        #(p, e, N)
        mob = self.getMobility(xarray)
        avgMob = 1/np.sum(np.multiply(self.p[:,:,np.newaxis], 1/mob), axis=1)
        return avgMob

    def labyrinth(self, xarray):
        '''
        Labyrinth mobility

        Returns
        -------
        (N, e+1) mobility array - e is number of elements, N is number of nodes
        '''
        mob = self.getMobility(xarray)
        avgMob = np.sum(np.multiply(np.power(self.p[:,:,np.newaxis], self.labFactor), mob), axis=1)
        return avgMob

    def hashin_shtrikmanUpper(self, xarray):
        '''
        Upper hashin shtrikman bounds for average mobility

        Returns
        -------
        (N, e+1) mobility array - e is number of elements, N is number of nodes
        '''
        #self.p                                 #(p,N)
        mob = self.getMobility(xarray)          #(N,p,e+1)
        maxMob = np.amax(mob, axis=1)[:,np.newaxis,:]           #(N,1,e+1)

        # 1 / ((1 / mPhi - mAlpha) + 1 / (3mAlpha)) = 3mAlpha * (mPhi - mAlpha) / (2mAlpha + mPhi)
        Ak = 3 * maxMob * (mob - maxMob) / (2*maxMob + mob)
        Ak = Ak * self.p[:,:,np.newaxis]
        Ak = np.sum(Ak, axis=1)
        avgMob = maxMob[:,0,:] + Ak / (1 - Ak / (3*maxMob[:,0,:]))
        return avgMob

    def hashin_shtrikmanLower(self, xarray):
        '''
        Lower hashin shtrikman bounds for average mobility

        Returns
        -------
        (N, e+1) mobility array - e is number of elements, N is number of nodes
        '''
        #self.p                                 #(p,N)
        mob = self.getMobility(xarray)          #(N,p,e+1)
        minMob = np.amin(mob, axis=1)[:,np.newaxis,:]           #(N,1,e+1)

        #This prevents an infinite mobility which could cause the time interval to be 0
        minMob[minMob == np.inf] = 0

        # 1 / ((1 / mPhi - mAlpha) + 1 / (3mAlpha)) = 3mAlpha * (mPhi - mAlpha) / (2mAlpha + mPhi)
        Ak = 3 * minMob * (mob - minMob) / (2*minMob + mob)
        Ak = Ak * self.p[:,:,np.newaxis]
        Ak = np.sum(Ak, axis=1)
        avgMob = minMob[:,0,:] + Ak / (1 - Ak / (3*minMob[:,0,:]))
        return avgMob
    
    def _getFluxes(self, t, x_curr):
        '''
        Return fluxes and time interval for the current iteration

        Steps:
            1. Get average mobility from homogenization function. Interpolate to get mobility (M) at cell boundaries
            2. Interpolate composition to get composition (x) at cell boundaries
            3. Calculate chemical potential gradient (dmu/dz) at cell boundaries
            4. Calculate composition gradient (dx/dz) at cell boundaries
            5. Calculate homogenization flux = -M / dmu/dz
            6. Calculate ideal contribution = -eps * M*R*T / x * dx/dz
            7. Apply boundary conditions for fluxes at ends of mesh
                If fixed flux condition (Neumann) - then use the flux defined in the condition
                If fixed composition condition (Dirichlet) - then use nearby flux (this will keep the composition fixed after apply the fluxes)

        TODO: If using RK4, I believe the phase fraction will be from the last step of the RK4 iteration. May not make sense to do that
        '''
        x = x_curr[0]
        self.T = self.Tfunc(self.z, t)
        self.p = self.updateCompSets(x)

        #Get average mobility between nodes
        avgMob = self.mobilityFunction(x)
        avgMob = 0.5 * (avgMob[1:] + avgMob[:-1])

        #Composition between nodes
        avgX = 0.5 * (x[1:] + x[:-1])
        avgX = np.concatenate((1-np.sum(avgX, axis=1)[:,np.newaxis], avgX), axis=1)

        #Chemical potential gradient
        dmudz = (self.mu[1:] - self.mu[:-1]) / self.mesh.dz

        #Composition gradient (we need to calculate gradient for reference element)
        dxdz = (x[1:] - x[:-1]) / self.mesh.dz
        dxdz = np.concatenate((0-np.sum(dxdz, axis=1)[:,np.newaxis], dxdz), axis=1)

        # J = -M * dmu/dz
        # Ideal contribution: J_id = -eps * M*R*T / x * dx/dz
        #fluxes = np.zeros((len(self.elements)+1, self.N-1))
        fluxes = np.zeros((self.N-1, len(self.elements)+1))
        fluxes = -avgMob * dmudz
        nonzeroComp = avgX != 0
        Tmid = (self.T[1:] + self.T[:-1]) / 2
        Tmidfull = Tmid[:,np.newaxis]
        for i in range(fluxes.shape[1]-1):
            Tmidfull = np.concatenate((Tmidfull, Tmid[:,np.newaxis]), axis=1)
        fluxes[nonzeroComp] += -self.eps * avgMob[nonzeroComp] * 8.314 * Tmidfull[nonzeroComp] * dxdz[nonzeroComp] / avgX[nonzeroComp]

        #Flux in a volume fixed frame: J_vi = J_i - x_i * sum(J_j)
        #vfluxes = np.zeros((len(self.elements), self.N+1))
        vfluxes = np.zeros((self.N+1, len(self.elements)))
        vfluxes[1:-1] = fluxes[:,1:] - avgX[:,1:] * np.sum(fluxes, axis=1)[:,np.newaxis]

        #Boundary conditions
        for e in range(len(self.elements)):
            vfluxes[0,e] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else vfluxes[1,e]
            vfluxes[-1,e] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else vfluxes[-2,e]

        return vfluxes

    def getFluxes(self):
        '''
        Return fluxes and time interval for the current iteration
        '''
        vfluxes = self._getFluxes(self.currentTime, [self.x])
        dJ = np.abs(vfluxes[1:] - vfluxes[:-1]) / self.mesh.dz
        dt = self.maxCompositionChange / np.amax(dJ[dJ!=0])
        return vfluxes, dt
    
    def getDt(self, dXdt):
        '''
        Time increment
        This is done by finding the time interval such that the composition
            change caused by the fluxes will be lower than self.maxCompositionChange
        '''
        return self.maxCompositionChange / np.amax(np.abs(dXdt[0][dXdt[0]!=0]))
    
    def getdXdt(self, t, x):
        '''
        dXdt is defined as -dJ/dz
        '''
        fluxes = self._getFluxes(t, x)
        return [-(fluxes[1:,:] - fluxes[:-1,:])/self.mesh.dz]