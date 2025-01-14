'''
Factors for nucleation on bulk/dislocations (homogenous) 
or grain boundaries/edges/corners (heterogenous)

Heterogenous nucleation affects precipitation in two ways
    Modifies the critical nucleation barrier (precipitate property)
    Modifies the number of nucleation sites (matrix property)

Changes to the nucleation barrier is taken from P.J. Clemm, J.C. Fisher,
"The influence of grain boundaries on the nucleation of secondary phases"
Acta Metallurgica 3 (1955) 70
doi:10.1016/0001-6160(55)90014-6

Changes to the number of nucleation sites is taken from 2024b TC-Prisma user guide
    Density is defined as amount of n-d space / volume (length^d / length^3)
    Where d is the dimension of the site type: dislocations (1d), area (2d), edges (1d), corners (0d)
    To get total nucleation sites, we assume the n-d space is 1 atom thick, and multiply by
    the number of atoms / dimension (N_A / V_m)^(d/3)
'''
import numpy as np
from kawin.Constants import AVOGADROS_NUMBER

class NucleationDescriptionBase:
    name = 'ABSTRACT NUCLEATION DESCRIPTION'
    maxRatio = np.inf

    def _createArrays(self, gbk):
        '''
        Creates arrays for grain boundary factors, this is done here to allow limits
        to be placed on the grain boundary ratio
        '''
        gbk = np.atleast_1d(gbk)
        indices = gbk < self.maxRatio
        valid_gbk = gbk[indices]

        values = -1*np.ones(gbk.shape, dtype=np.float64)
        
        return gbk, valid_gbk, indices, values
    
    def _formatArray(self, values, indices, setInvalidToNan = True):
        '''
        By default, setInvalidToNan will be true, will will set all invalid indices to nan
            This is for the user API so that it's easy to tell when a gb energy / interfacial energy
            ratio is invalid
        When setting factors internally, this will be set to false so it's easier to compare
        when a value is invalid (since np.nan == np.nan will return False)
        '''
        if setInvalidToNan:
            values[~indices] = np.nan
        return np.squeeze(values)
    
    def gbRatio(self, gbEnergy, gamma):
        '''
        Grain boundary to interfacial energy ratio
        '''
        return gbEnergy / (2*gamma)
    
    def gbRemoval(self, gbk = None, setInvalidToNan = True):
        gbk, valid_gbk, indices, values = self._createArrays(gbk)
        values[indices] = self._gbRemoval(valid_gbk)
        return self._formatArray(values, indices, setInvalidToNan)
    
    def _gbRemoval(self, gbk):
        raise NotImplementedError()
    
    def areaFactor(self, gbk = None, setInvalidToNan = True):
        gbk, valid_gbk, indices, values = self._createArrays(gbk)
        values[indices] = self._areaFactor(valid_gbk)
        return self._formatArray(values, indices, setInvalidToNan)
    
    def _areaFactor(self, gbk):
        raise NotImplementedError()
    
    def volumeFactor(self, gbk = None, setInvalidToNan = True):
        gbk, valid_gbk, indices, values = self._createArrays(gbk)
        values[indices] = self._volumeFactor(valid_gbk)
        return self._formatArray(values, indices, setInvalidToNan)
    
    def _volumeFactor(self, gbk):
        raise NotImplementedError()
    
    def areaRemoval(self, gbk = None, setInvalidToNan = True):
        gbk, valid_gbk, indices, values = self._createArrays(gbk)
        values[indices] = self._areaRemoval(valid_gbk)
        return self._formatArray(values, indices, setInvalidToNan)
    
    def _areaRemoval(self, gbk):
        return np.sqrt(self._gbRemoval(gbk) / np.pi)
        
    @property
    def isGrainBoundaryNucleation(self):
        return True

class BulkDescription(NucleationDescriptionBase):
    name = 'BULK'
    maxRatio = np.inf

    def _gbRemoval(self, gbk):
        return np.zeros(gbk.shape)
    
    def _areaFactor(self, gbk):
        return 4*np.pi * np.ones(gbk.shape)

    def _volumeFactor(self, gbk):
        return 4*np.pi/3 * np.ones(gbk.shape)
    
    def _areaRemoval(self, gbk):
        return np.ones(gbk.shape)
    
    @property
    def isGrainBoundaryNucleation(self):
        return False
    
class DislocationDescription(BulkDescription):
    name = 'DISLOCATIONS'
    
class GrainBoundaryDescription(NucleationDescriptionBase):
    name = 'GRAIN BOUNDARIES'
    maxRatio = 1

    def _gbRemoval(self, gbk):
        return np.pi * (1 - gbk**2)
    
    def _areaFactor(self, gbk):
        return 4*np.pi * (1 - gbk)

    def _volumeFactor(self, gbk):
        return (2*np.pi/3) * (2 - 3*gbk + gbk**3)
    
class GrainEdgeDescription(NucleationDescriptionBase):
    name = 'GRAIN EDGES'
    maxRatio = np.sqrt(3)/2

    def alpha(self, gbk):
        return np.arcsin(1 / (2*np.sqrt(1 - gbk**2)))
    
    def beta(self, gbk):
        return np.arccos(gbk / np.sqrt(3*(1 - gbk**2)))

    def _gbRemoval(self, gbk):
        beta = self.beta(gbk)
        return 3*beta * (1 - gbk**2) - gbk*np.sqrt(3 - 4*gbk**2)
    
    def _areaFactor(self, gbk):
        alpha = self.alpha(gbk)
        beta = self.beta(gbk)
        return 12 * (np.pi/2 - alpha - gbk*beta)

    def _volumeFactor(self, gbk):
        alpha = self.alpha(gbk)
        beta = self.beta(gbk)
        return 2 * (np.pi - 2*alpha + (gbk**2/3) * np.sqrt(3 - 4*gbk**2) - beta*gbk*(3 - gbk**2))

class GrainCornerDescription(NucleationDescriptionBase):
    name = 'GRAIN CORNERS'
    maxRatio = np.sqrt(2/3)

    def K(self, gbk):
        return (4/3)*np.sqrt(3/2 - 2*gbk**2) - 2*gbk/3
    
    def phi(self, gbk):
        return np.arcsin(self.K(gbk) / (2*np.sqrt(1 - gbk**2)))
    
    def delta(self, gbk):
        return np.arccos((np.sqrt(2) - gbk*np.sqrt(3 - self.K(gbk)**2)) / (self.K(gbk)*np.sqrt(1 - gbk**2)))

    def _gbRemoval(self, gbk):
        K = self.K(gbk)
        phi = self.phi(gbk)
        return 3*(2*phi*(1 - gbk**2) - K*(np.sqrt(1 - gbk**2 - K**2 / 4) - K**2 / np.sqrt(8)))
    
    def _areaFactor(self, gbk):
        phi = self.phi(gbk)
        delta = self.delta(gbk)
        return 24*(np.pi/3 - gbk*phi - delta)

    def _volumeFactor(self, gbk):
        K = self.K(gbk)
        phi = self.phi(gbk)
        delta = self.delta(gbk)
        return 2*(4*(np.pi/3 - delta) + gbk*K*(np.sqrt(1 - gbk**2 - K**2 / 4) - K**2 / np.sqrt(8)) - 2*gbk*phi*(3 - gbk**2))

class NucleationBarrierParameters:
    '''
    Defines nucleation factors for bulk, dislocation, 
    surface, edge and corner nucleation

    This includes: surface area, volume, GB area removal, Rcrit and Gcrit

    Attributes
    ----------
    gbRemoval - factor to multiply by r**2 to get area of grain boundary removed
    areaFactor - factor to multiply by r**2 to get surface area of precipitate
    volumeFactor - factor to multiply by r**3 to get volume of precipitate
    areaRemoval - factor to multiply by r to get radius of eliminated grain boundary area
    '''
    def __init__(self, site='dislocations', gamma = None, gbEnergy=0.3):
        self._gbEnergy = gbEnergy
        self._gamma = gamma
        self._updateCallbacks = []

        self.setNucleationType(site)

        self._resetFactors()

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, value):
        self._description = value
        self._resetFactors()
        for callback in self._updateCallbacks:
            callback()

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._resetFactors()

    @property
    def gbEnergy(self):
        return self._gbEnergy
    
    @gbEnergy.setter
    def gbEnergy(self, value):
        self._gbEnergy = value
        self._resetFactors()

    def _resetFactors(self):
        self._GBk = None
        self._areaFactor = None
        self._volumeFactor = None
        self._gbRemoval = None
        self._areaRemoval = None

    def setNucleationType(self, site):
        '''
        Sets nucleation site type

        Parameters
        ----------
        site - int
            Index for site type based off list on top of this file
        '''
        descriptionDict = {
            BulkDescription.name.upper(): BulkDescription(),
            DislocationDescription.name.upper(): DislocationDescription(),
            GrainBoundaryDescription.name.upper(): GrainBoundaryDescription(),
            GrainEdgeDescription.name.upper(): GrainEdgeDescription(),
            GrainCornerDescription.name.upper(): GrainCornerDescription(),
        }
        if isinstance(site, str):
            site = site.upper()
        newDescription = descriptionDict.get(site, site)
        if not isinstance(newDescription, NucleationDescriptionBase):
            validValues = ', '.join(list(descriptionDict.keys()))
            raise ValueError(f"Unknown value '{site}'. Value must be: {validValues} or an instance of NucleationDescriptionBase")
        self.description = newDescription

    def _validateInputs(self):
        if self.gamma is None or self.gamma == 0:
            raise ValueError(f"Interfacial energy (gamma) is not set. NucleationBarrierParameters.gamma = {self.gamma}")
        if self.gbEnergy is None:
            raise ValueError(f"Grain boundary energy (gbEnergy) is not set. NucleationBarrierParameters.gbEnergy = {self.gbEnergy}")
    
    def _validateGBk(self):
        if self.GBk > self.description.maxRatio:
            errorString = f'Warning: Grain boundary to interfacial energy ratio is too large for nucleation barrer on {self.description.name}. '
            errorString += f'For nucleation on {self.description.name}. y_gb / 2*y_int must be below {self.description.maxRatio:.3f}, but is {self.GBk:.3f}'
            raise ValueError(errorString)

    # It may be possible to use an lru cache for these terms, then clear the cache
    # whenever the description, gamma or gbEnergy is modified
    # https://stackoverflow.com/questions/55497353/clear-cache-of-property-methods-python
    @property
    def GBk(self):
        if self._GBk is None:
            self._validateInputs()
            self._GBk = self.description.gbRatio(self.gbEnergy, self.gamma)
        return self._GBk
    
    @property
    def areaFactor(self):
        if self._areaFactor is None:
            self._validateGBk()
            self._areaFactor = self.description.areaFactor(self.GBk, setInvalidToNan=False)
        return self._areaFactor
    
    @property
    def volumeFactor(self):
        if self._volumeFactor is None:
            self._validateGBk()
            self._volumeFactor = self.description.volumeFactor(self.GBk, setInvalidToNan=False)
        return self._volumeFactor

    @property
    def gbRemoval(self):
        if self._gbRemoval is None:
            self._validateGBk()
            self._gbRemoval = self.description.gbRemoval(self.GBk, setInvalidToNan=False)
        return self._gbRemoval

    @property
    def areaRemoval(self):
        if self._areaRemoval is None:
            self._validateGBk()
            self._areaRemoval = self.description.areaRemoval(self.GBk, setInvalidToNan=False)
        return self._areaRemoval

    def Rcrit(self, dG):
        '''
        Critical radius for nucleation
        This is only done for GB nucleation
            Bulk and dislocation nucleation needs to account for shape factors

        R* = 2 * (b * \gamma_{\alpha \alpha} - a * \gamma_{\alpha \beta}) / (3 * c * dG)

        Parameters
        ----------
        dG - float
            Volumetric driving force
        '''
        return (2 * (self.areaFactor * self.gamma - self.gbRemoval * self.gbEnergy)) / (3 * self.volumeFactor * dG)

    def Gcrit(self, dG, Rcrit):
        '''
        Critical driving force for nucleation

        G* = 4/27 * (b * \gamma_{\alpha \alpha} - a * \gamma_{\alpha \beta})^3 / (c * dG)^2
        
        or in terms of R*
        G* = -R*^2 * (b * \gamma_{\alpha \alpha} - a * \gamma_{\alpha \beta} + c * r* * dG)

        This is calculated in terms of R* since there is a check in the nucleation calculation that will
        increase R* to minimum radius if R* is too small (very low nucleation barriers)

        Parameters
        ----------
        dG - float
            Volumetric driving force
        Rcrit - float
            Critical radius for nucleation
        '''
        return Rcrit**2 * ((self.areaFactor * self.gamma - self.gbRemoval * self.gbEnergy) - self.volumeFactor * dG * Rcrit)
    
class NucleationSiteParameters:
    def __init__(self, grainSize = 100, aspectRatio = 1, dislocationDensity = 5e12):
        self._grainSize = grainSize
        self._grainAspectRatio = aspectRatio
        self._dislocationDensity = dislocationDensity

        self.VmAlpha = None

        self._bulkN0 = None
        self._compositionDependentBulkN0 = True
        self._GBareaN0 = None
        self._GBedgeN0 = None
        self._GBcornerN0 = None
        self._dislocationN0 = None

    @property
    def grainSize(self):
        return self._grainSize
    
    @grainSize.setter
    def grainSize(self, value):
        self._grainSize = value
        self._GBareaN0 = None
        self._GBedgeN0 = None
        self._GBcornerN0 = None

    @property
    def grainAspectRatio(self):
        return self._grainAspectRatio
    
    @grainAspectRatio.setter
    def grainAspectRatio(self, value):
        self._grainAspectRatio = value
        self._GBareaN0 = None
        self._GBedgeN0 = None
        self._GBcornerN0 = None

    @property
    def dislocationDensity(self):
        return self._dislocationDensity
    
    @dislocationDensity.setter
    def dislocationDensity(self, value):
        self._dislocationDensity = value
        self._dislocationN0 = None

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
        if bulkN0 is not None:
            self.bulkN0 = bulkN0

    def setGrainSize(self, grainSize = 100, aspectRatio = 1):
        self.grainSize = grainSize * 1e-6
        self.grainAspectRatio = aspectRatio

    def setDislocationDensity(self, dislocationDensity):
        self.dislocationDensity = dislocationDensity

    def setBulkDensity(self, bulkN0):
        self.bulkN0 = bulkN0

    def setBulkDensityFromComposition(self, x0):
        #Set bulk nucleation site to the number of solutes per unit volume
        #   This is the represent that any solute atom can be a nucleation site
        #NOTE: some texts will state the bulk nucleation sites to just be the number
        #       of lattice sites per unit volume. The justification for this would be 
        #       the solutes can diffuse around to any lattice site and nucleate there
        self._validateVolume('bulkN0 from composition')
        self.bulkN0 = np.amin(x0) * (AVOGADROS_NUMBER / self.VmAlpha)
        self._compositionDependentBulkN0 = True

    @property
    def bulkN0(self):
        return self._bulkN0
    
    @bulkN0.setter
    def bulkN0(self, value):
        self._bulkN0 = value
        self._compositionDependentBulkN0 = False

    @property
    def dislocationN0(self):
        if self._dislocationN0 is None:
            self._validateVolume('dislocationN0')
            self._dislocationN0 = self.dislocationSites(self.VmAlpha)
        return self._dislocationN0
    
    @property
    def GBareaN0(self):
        if self._GBareaN0 is None:
            self._validateVolume('GBareaN0')
            self._GBareaN0 = self.grainBoundarySites(self.grainSize, self.grainAspectRatio, self.VmAlpha)
        return self._GBareaN0
    
    @property
    def GBedgeN0(self):
        if self._GBedgeN0 is None:
            self._validateVolume('GBedgeN0')
            self._GBedgeN0 = self.grainEdgeSites(self.grainSize, self.grainAspectRatio, self.VmAlpha)
        return self._GBedgeN0
    
    @property
    def GBcornerN0(self):
        if self._GBcornerN0 is None:
            self._GBcornerN0 = self.grainCornerSites(self.grainSize, self.grainAspectRatio, self.VmAlpha)
        return self._GBcornerN0

    def dislocationSites(self, VmAlpha):
        return self.dislocationDensity * (AVOGADROS_NUMBER / VmAlpha)**(1/3)

    def grainBoundaryDensity(self, grainSize, grainAspectRatio):
        rho = (6 * np.sqrt(1 + 2 * grainAspectRatio**2) + 1 + 2 * grainAspectRatio)
        rho /= (4 * grainAspectRatio * grainSize)
        return rho
    
    def grainBoundarySites(self, grainSize, grainAspectRatio, VmAlpha):
        rho = self.grainBoundaryDensity(grainSize, grainAspectRatio)
        rho *= (AVOGADROS_NUMBER / VmAlpha)**(2/3)
        return rho

    def grainEdgeDensity(self, grainSize, grainAspectRatio):
        rho = 2 * (np.sqrt(2) + 2*np.sqrt(1 + grainAspectRatio**2))
        rho /= (grainAspectRatio * grainSize**2)
        return rho

    def grainEdgeSites(self, grainSize, grainAspectRatio, VmAlpha):
        rho = self.grainEdgeDensity(grainSize, grainAspectRatio)
        rho *= (AVOGADROS_NUMBER / VmAlpha)**(1/3)
        return rho
    
    def grainCornerDensity(self, grainSize, grainAspectRatio):
        rho = 12 / (grainAspectRatio * grainSize**3)
        return rho

    def grainCornerSites(self, grainSize, grainAspectRatio, VmAlpha):
        rho = self.grainCornerDensity(grainSize, grainAspectRatio)
        return rho
    
    def _validateVolume(self, term):
        if self.VmAlpha is None:
            raise ValueError(f'NucleationSiteParameters.VMalpha must be set to compute {term}.')

