import numpy as np

from kawin.precipitation.non_ideal.EffectiveDiffusion import EffectiveDiffusionFunctions
from kawin.precipitation.non_ideal.ShapeFactors import ShapeFactor
from kawin.precipitation.non_ideal.ElasticFactors import StrainEnergy
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors

GAS_CONSTANT = 8.314
AVOGADROS_NUMBER = 6.022e23
BOLTZMANN_CONSTANT = GAS_CONSTANT / AVOGADROS_NUMBER

class PrecipitationData:
    # Strings for each attributes to make it easier to loop accessing these terms
    ATTRIBUTES = [
        'time', 'temperature',
        'composition', 'xEqAlpha', 'xEqBeta',
        'drivingForce', 'impingement', 'Gcrit', 'Rcrit', 'nucRate', 'precipitateDensity',
        'Rnuc',  'Ravg', 'ARavg', 'volFrac', 'fconc'
    ]
    
    def __init__(self, phases: list[str], elements: list[int], N: int = 1):
        self.phases = phases
        self.elements = elements
        self.reset(N)

    def reset(self, N: int = 1):
        self.n = N-1
        self.time = np.zeros(N)
        self.temperature = np.zeros(N)
        self.composition = np.zeros((N, len(self.elements)))
        self.xEqAlpha = np.zeros((N, len(self.phases), len(self.elements)))
        self.xEqBeta = np.zeros((N, len(self.phases), len(self.elements)))
        self.drivingForce = np.zeros((N, len(self.phases)))
        self.impingement = np.zeros((N, len(self.phases)))
        self.Gcrit = np.zeros((N, len(self.phases)))
        self.Rcrit = np.zeros((N, len(self.phases)))
        self.Rnuc = np.zeros((N, len(self.phases)))
        self.nucRate = np.zeros((N, len(self.phases)))
        self.precipitateDensity = np.zeros((N, len(self.phases)))
        self.Ravg = np.zeros((N, len(self.phases)))
        self.ARavg = np.zeros((N, len(self.phases)))
        self.volFrac = np.zeros((N, len(self.phases)))
        self.fconc = np.zeros((N, len(self.phases), len(self.elements)))

    def appendToArrays(self, newData):
        '''
        Appends data from another PrecipitationData object to current one
        '''
        for name in self.ATTRIBUTES:
            setattr(self, name, np.concatenate([getattr(self, name), getattr(newData, name)], axis=0))
        self.n = len(self.time) - 1

    def copySlice(self, N: int = 0):
        sliceData = PrecipitationData(self.phases, self.elements, N=1)
        for name in self.ATTRIBUTES:
            getattr(sliceData, name)[0] = getattr(self, name)[N]
        return sliceData
    
    def setSlice(self, sliceData, N: int = 0):
        for name in self.ATTRIBUTES:
            getattr(self, name)[N] = getattr(sliceData, name)[0]

    def print(self, N: int = 0):
        for name in self.ATTRIBUTES:
            print(f'{name}: {getattr(self, name)[N]}')

class VolumeParameter:
    MOLAR_VOLUME = 0
    ATOMIC_VOLUME = 1
    LATTICE_PARAMETER = 2

    def __init__(self):
        self.a = None
        self.Va = None
        self.Vm = None
        self.atomsPerCell = None

    def setVolume(self, value, volumeType, atomsPerCell):
        '''
        Function to set lattice parameter, atomic volume and molar volume

        Parameters
        ----------
        value : float
            Value for volume parameters (lattice parameter, atomic (unit cell) volume or molar volume)
        valueType : VolumeParameter
            States what volume term that value is
        atomsPerCell : int
            Number of atoms in the unit cell
        '''
        self.atomsPerCell = atomsPerCell
        if volumeType == self.MOLAR_VOLUME:
            self.Vm = value
            self.Va = atomsPerCell * self.Vm / AVOGADROS_NUMBER
            self.a = np.cbrt(self.Va)
        elif volumeType == self.ATOMIC_VOLUME:
            self.Va = value
            self.Vm = self.Va * AVOGADROS_NUMBER / atomsPerCell
            self.a = np.cbrt(self.Va)
        elif volumeType == self.LATTICE_PARAMETER:
            self.a = value
            self.Va = self.a**3
            self.Vm = self.Va * AVOGADROS_NUMBER / atomsPerCell

class NucleationParameters:
    def __init__(self, grainSize = 100, aspectRatio = 1, dislocationDensity = 5e12, bulkN0 = None):
        self.setNucleationDensity(grainSize, aspectRatio, dislocationDensity, bulkN0)

        self._parametersSet = False
        self.GBareaN0 = None
        self.GBedgeN0 = None
        self.GBcornerN0 = None
        self.dislocationN0 = None

    def setNucleationDensity(self, grainSize = 100, aspectRatio = 1, dislocationDensity = 5e12, bulkN0 = None):
        '''
        Sets grain size and dislocation density which determines the available nucleation sites
        
        Parameters
        ----------
        grainSize : float (optional)
            Average grain size in microns (default at 100um if this function is not called)
        aspectRatio : float (optional)
            Aspect ratio of grains (default at 1)
        dislocationDensity : float (optional)
            Dislocation density (m/m3) (default at 5e12)
        bulkN0 : float (optional)
            This allows for the use to override the nucleation site density for bulk precipitation
            By default (None), this is calculated by the number of lattice sites containing a solute atom
            However, for calibration purposes, it may be better to set the nucleation site density manually
        '''
        self.grainSize = grainSize * 1e-6
        self.grainAspectRatio = aspectRatio
        self.dislocationDensity = dislocationDensity
        self.bulkN0 = bulkN0

    def bulkSites(self, x0, VmAlpha):
        #Set bulk nucleation site to the number of solutes per unit volume
        #   This is the represent that any solute atom can be a nucleation site
        #NOTE: some texts will state the bulk nucleation sites to just be the number
        #       of lattice sites per unit volume. The justification for this would be 
        #       the solutes can diffuse around to any lattice site and nucleate there
        return np.amin(x0) * (AVOGADROS_NUMBER / VmAlpha)

    def dislocationSites(self, VmAlpha):
        return self.dislocationDensity * (AVOGADROS_NUMBER / VmAlpha)**(1/3)

    def grainBoundaryArea(self, grainSize, grainAspectRatio):
        gbArea = (6 * np.sqrt(1 + 2 * grainAspectRatio**2) + 1 + 2 * grainAspectRatio)
        gbArea /= (4 * grainAspectRatio * grainSize)
        return gbArea
    
    def grainBoundarySites(self, grainSize, grainAspectRatio, VmAlpha):
        gbAreaN0 = self.grainBoundaryArea(grainSize, grainAspectRatio)
        gbAreaN0 *= (AVOGADROS_NUMBER / VmAlpha)**(2/3)
        return gbAreaN0

    def grainEdgeLength(self, grainSize, grainAspectRatio):
        gbEdge = 2 * (np.sqrt(2) + 2*np.sqrt(1 + grainAspectRatio**2))
        gbEdge /= (grainAspectRatio * grainSize**2)
        return gbEdge

    def grainEdgeSites(self, grainSize, grainAspectRatio, VmAlpha):
        gbEdgeN0 = self.grainEdgeLength(grainSize, grainAspectRatio)
        gbEdgeN0 *= (AVOGADROS_NUMBER / VmAlpha)**(1/3)
        return gbEdgeN0
    
    def grainCornerAmount(self, grainSize, grainAspectRatio):
        gbCornerAmount = 12 / (grainAspectRatio * grainSize**3)
        return gbCornerAmount

    def grainCornerSites(self, grainSize, grainAspectRatio, VmAlpha):
        gbCornerN0 = self.grainCornerAmount(grainSize, grainAspectRatio)
        return gbCornerN0

    def setupNucleationDensity(self, x0, VmAlpha):
        if self.bulkN0 is None:
            self.bulkN0 = self.bulkSites(x0, VmAlpha)
        self.dislocationN0 = self.dislocationSites(VmAlpha)
        
        if self.grainSize != np.inf:
            self.GBareaN0 = self.grainBoundarySites(self.grainSize, self.grainAspectRatio, VmAlpha)
            self.GBedgeN0 = self.grainEdgeSites(self.grainSize, self.grainAspectRatio, VmAlpha)
            self.GBcornerN0 = self.grainCornerSites(self.grainSize, self.grainAspectRatio, VmAlpha)
        else:
            self.GBareaN0 = 0
            self.GBedgeN0 = 0
            self.GBcornerN0 = 0

class TemperatureParameters:
    def __init__(self, *args):
        self.setTemperatureParameters(args)
        self._isIsothermal = True

    def setTemperatureParameters(self, *args):
        if len(args) == 2:
            print(args)
            self.setTemperatureArray(*args)
        elif len(args) == 1:
            if callable(args[0]):
                self.setTemperatureFunction(args[0])
            else:
                self.setIsothermalTemperature(args[0])
        else:
            self.Tparameters = None
            self.Tfunction = None

    def setIsothermalTemperature(self, T: float):
        self._isIsothermal = True
        self.Tparameters = T
        self.Tfunction = lambda t: self.Tparameters

    def setTemperatureArray(self, times: list[float], temperatures: list[float]):
        self._isIsothermal = False
        self.Tparameters = (times, temperatures)
        self.Tfunction = lambda t: np.interp(t/3600, self.Tparameters[0], self.Tparameters[1], self.Tparameters[1][0], self.Tparameters[1][-1])

    def setTemperatureFunction(self, func):
        self._isIsothermal = False
        self.Tparameters = func
        self.Tfunction = func

    def __call__(self, t):
        return self.Tfunction(t)

class MatrixParameters:
    def __init__(self):
        self.effDiffFuncs = EffectiveDiffusionFunctions()
        self.effDiffDistance = self.effDiffFuncs.effectiveDiffusionDistance
        self.GBenergy = 0.3
        self.volume = VolumeParameter()
        self.nucleation = NucleationParameters()
        self.theta = 2
        self.initComposition = None

class PrecipitateParameters:
    '''
    Parameters for a single precipitate
    '''
    def __init__(self, name, phase = None):
        # Name is the print/output name of the phase and phase is the actual name in the database
        # If phase isn't supplied, then phase = name
        self.name = str(name)
        if phase is None:
            phase = name
        self.phase = str(phase)

        self.strainEnergy = StrainEnergy()
        self.shapeFactor = ShapeFactor()
        self.GBfactor = GBFactors()
        self.volume = VolumeParameter()
        self.gamma = None
        self.calculateAspectRatio = False
        self.RdrivingForceLimit = 0
        self.infinitePrecipitateDiffusion = True
        self.parentPhases = []
        self.Rmin = None

    def setup(self, gbEnergy = 0.3, minRadius = 3e-10):
        if self.GBfactor.isGrainBoundaryNucleation:
            self.shapeFactor.setSpherical()
        self.GBfactor.setFactors(gbEnergy, self.gamma)

        self.strainEnergy.setup()
        if self.strainEnergy.type != StrainEnergy.CONSTANT:
            if self.shapeFactor.particleType == ShapeFactor.SPHERE:
                self.strainEnergy.setSpherical()
            elif self.shapeFactor.particleType == ShapeFactor.CUBIC:
                self.strainEnergy.setCuboidal()
            else:
                self.strainEnergy.setEllipsoidal()

        self.Rmin = minRadius
        
    def computeStrainEnergyFromR(self, r):
        return self.strainEnergy.strainEnergy(self.shapeFactor.normalRadii(r))
    
    def computeGibbsThomsonContribution(self, r):
        vmbeta = self.volume.Vm
        strain = self.computeStrainEnergyFromR(r)
        thermoFactor = self.shapeFactor.thermoFactor(r)
        return vmbeta * (strain + 2*thermoFactor*self.gamma / r)

class Constraints:
    def __init__(self):
        self.reset()

    def reset(self):
        self.minRadius = 3e-10
        self.maxTempChange = 1

        self.maxDTFraction = 1e-2
        self.minDTFraction = 1e-5

        #Constraints on maximum time step
        self.checkTemperature = True
        self.maxNonIsothermalDT = 1

        #Maximum dissolution as volume fraction of the particle size distribution
        self.checkPSD = True
        self.maxDissolution = 1e-3

        #Maximum change in critical radius by relative increase
        self.checkRcrit = True
        self.maxRcritChange = 0.01

        #Maximum change in nucleation rate as a ratio
        self.checkNucleation = True
        self.maxNucleationRateChange = 0.5
        self.minNucleationRate = 1e-5

        #Maximum volume change from nucleation
        self.checkVolumePre = True
        self.maxVolumeChange = 0.001
        
        self.minComposition = 0
        self.minNucleateDensity = 1e-10

        #TODO: may want to test more to see if this value should be lower or higher
        #This will attempt to increase the time by 0.1%
        #This also only affects the sim if the calculated dt is extremely large
        #So probably only when nucleation rate is 0 will this matter
        #This roughly corresponds to 1e4 steps over 5-7 orders of magnitude on a log time scale
        self.dtScale = 1e-3

    def computeDTfromPSD(self, n, temperatures, PBMs, growth, dissolutionIndex, phases, dtMax):
        if self.checkPSD:
            dtPBM = [dtMax]
            # TODO: do we really need to check if we're in isothermal state? I believe this was
            #       added before dissolution rate from negative driving forces were possible
            if n > 0 and temperatures[n] == temperatures[n-1]:
                dtPBM += [PBMs[p].getDTEuler(dtMax, growth[p], dissolutionIndex[p]) for p in range(len(phases))]
            return np.amin(dtPBM)
        else:
            return dtMax

    def computeDTfromNucleationRate(self, n, nucRate, phases, dtPrev, dtMax):
        if self.checkNucleation:
            dtNuc = dtMax * np.ones(len(phases))
            if n > 0:
                nRateCurr = nucRate[n]
                nRatePrev = nucRate[n-1]
                for p in range(len(phases)):
                    if nRateCurr[p] > self.minNucleationRate and nRatePrev[p] > self.minNucleationRate and nRatePrev[p] != nRateCurr[p]:
                        dtNuc[p] = self.maxNucleationRateChange * dtPrev / np.abs(np.log10(nRatePrev[p] / nRateCurr[p]))
            else:
                for p in range(len(phases)):
                    if nucRate[n,p] * dtPrev > 1e5:
                        dtNuc[p] = 1e5 / nucRate[n,p]
            return np.amin(dtNuc)
        else:
            return dtMax
        
    def computeDTfromTemperature(self, n, temperatures, dtPrev, dtMax):
        if self.checkTemperature and n > 0:
            Tchange = temperatures[n] - temperatures[n-1]
            dtTemp = dtMax
            if Tchange > self.maxNonIsothermalDT:
                dtTemp = self.maxNonIsothermalDT * dtPrev / Tchange
            return dtTemp
        else:
            return dtMax
        
    def computeDTfromRcrit(self, n, Rcrit, dGs, phases, dtPrev, dtMax):
        if self.checkRcrit and n > 0:
            dtRad = dtMax * np.ones(len(phases))
            if not all((Rcrit[n-1,:] == 0) & (Rcrit[n,:] - Rcrit[n-1,:] == 0) & (dGs[n,:] <= 0)):
                indices = (Rcrit[n-1,:] > 0) & (Rcrit[n,:] - Rcrit[n-1,:] != 0) & (dGs[n,:] > 0)
                dtRad[indices] = self.maxRcritChange * dtPrev / np.abs((Rcrit[n,:][indices] - Rcrit[n-1,:][indices]) / Rcrit[n-1,:][indices])
            return np.amin(dtRad)
        else:
            return dtMax
        
    def computeDTfromVolume(self, n, nucRate, nucRadius, PBMs, growths, VmAlpha, VmBeta, GB, phases, dtMax):
        if self.checkVolumePre:
            dV = np.zeros(len(phases))
            for p in range(len(phases)):
                #Calculate estimate volume change based off growth rate and nucleated particles
                #TODO: account for non-spherical precipitates
                dVi = PBMs[p].PSD * PBMs[p].PSDsize**2 * 0.5 * (growths[p][1:] + growths[p][:-1])
                dVi[dVi < 0] = 0
                dV = VmAlpha / VmBeta[p] * (GB[p].areaFactor * np.sum(dVi) + GB[p].volumeFactor * nucRate[n,p] * nucRadius[n,p]**3)

            dtVol = dtMax * np.ones(len(phases))
            for p in range(len(phases)):
                if dV != 0:
                    dtVol[p] = self.maxVolumeChange / (2 * np.abs(dV))
            return np.amin(dtVol)
        else:
            return dtMax


