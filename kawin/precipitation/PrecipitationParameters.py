import numpy as np

from kawin.precipitation.parameters.Volume import VolumeParameter
from kawin.precipitation.parameters.EffectiveDiffusion import EffectiveDiffusionFunctions
from kawin.precipitation.parameters.ShapeFactors import ShapeFactor, SphereDescription, CuboidalDescription
from kawin.precipitation.parameters.ElasticFactors import StrainEnergy, SphericalEnergyDescription, CuboidalEnergyDescription, EllipsoidalEnergyDescription, ConstantEnergyDescription
from kawin.precipitation.parameters.Nucleation import NucleationBarrierParameters, NucleationSiteParameters, DislocationDescription

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

    def toDict(self):
        data = {name: getattr(self, name) for name in self.ATTRIBUTES}
        return data
    
    def fromDict(self, data):
        for name in self.ATTRIBUTES:
            setattr(self, name, data[name])
        self.n = len(self.time) - 1

class TemperatureParameters:
    def __init__(self, *args):
        if len(args) == 2:
            print(args)
            self.setTemperatureArray(*args)
        elif len(args) == 1:
            if isinstance(args[0], TemperatureParameters):
                self._isIsothermal = args[0]._isIsothermal
                self.Tparameters = args[0].Tparameters
                self.Tfunction = args[0].Tfunction
            elif callable(args[0]):
                self.setTemperatureFunction(args[0])
            else:
                self.setIsothermalTemperature(args[0])
        else:
            self._isIsothermal = True
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
    def __init__(self, solutes):
        self.solutes = solutes
        self._initComposition = None

        self.volume = VolumeParameter()
        self.volume._updateCallbacks.append(self.update)
        self.nucleationSites = NucleationSiteParameters()
        self.GBenergy = 0.3
        
        self.effectiveDiffusion = EffectiveDiffusionFunctions()
        self.theta = 2

    @property
    def initComposition(self):
        return self._initComposition
    
    @initComposition.setter
    def initComposition(self, value):
        self._initComposition = value
        self.update()

    def update(self):
        # update nucleation site volume and bulkN0
        #   only do this once both volume and composition is defined
        # if bulkN0 is set before composition or volume, then we leave to the user defined bulkN0
        self.nucleationSites.VmAlpha = self.volume.Vm
        if self._initComposition is not None and self.nucleationSites.VmAlpha is not None:
            if self.nucleationSites._compositionDependentBulkN0:
                self.nucleationSites.setBulkDensityFromComposition(self._initComposition)

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
        
        self._gamma = None

        self.volume = VolumeParameter()

        self.shapeFactor = ShapeFactor(precipitateShape=SphereDescription(), ar=1)
        self.shapeFactor._updateCallbacks.append(self.validate)
        self.strainEnergy = StrainEnergy(shape=ConstantEnergyDescription())
        self.calculateAspectRatio = False

        self.nucleation = NucleationBarrierParameters(site=DislocationDescription(), gbEnergy=0.3)
        self.nucleation._updateCallbacks.append(self.validate)
        self.parentPhases = []

        self.RdrivingForceLimit = 0
        self.infinitePrecipitateDiffusion = True
        self.Rmin = 3e-10

        self.validate()

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self.validate()
        
    def validate(self):
        self.nucleation.gamma = self.gamma
        
        if self.nucleation.description.isGrainBoundaryNucleation and not isinstance(self.shapeFactor.description, SphereDescription):
            raise ValueError('Nucleation is set to grain boundary nucleation and shape factor not set to spherical. \
                             If using GB nucleation, shape factor should be spherical. If shape factor is spherical, nucleation should be bulk or dislocations')
        
        # If strain energy is not constant, then switch to description that matches shapeFactor
        # TODO: this is really awkward especially if the user switches the strain energy description, which would not update the shape factor
        # I guess the alternative would be that the call back for shape factors would validate nucleation and update strain energy
        # Then the callback for strain energy would update the shape factor
        # Thus, the final shape of the precipitate would depend on whatever attribute is last called
        if not isinstance(self.strainEnergy.description, ConstantEnergyDescription):
            if isinstance(self.shapeFactor.description, SphereDescription):
                self.strainEnergy.setShape(SphericalEnergyDescription())
            elif isinstance(self.shapeFactor.description, CuboidalDescription):
                self.strainEnergy.setShape(CuboidalEnergyDescription())
            else:
                self.strainEnergy.setShape(EllipsoidalEnergyDescription())
        
    def computeStrainEnergyFromR(self, r):
        return self.strainEnergy.compute(self.shapeFactor.normalRadii(r))
    
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


