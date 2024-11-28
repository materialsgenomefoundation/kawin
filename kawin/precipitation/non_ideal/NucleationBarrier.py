from collections import namedtuple

import numpy as np

NucleationFactorData = namedtuple('NucleationFactorData',
                    ['area_factor', 'volume_factor', 'gb_removal'])

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
    '''
    #Nucleation site type
    BULK = 0
    DISLOCATION = 1
    GRAIN_BOUNDARIES = 2
    GRAIN_EDGES = 3
    GRAIN_CORNERS = 4

    def __init__(self):
        self.gbEnergy = 0
        self.gamma = 0
        self.nucleationSiteType = self.BULK
        self.GBk = 0

        # Since we default to bulk nucleation, the correction factors will not depend on GB or interfacial energy
        self.setFactors(0, 1)

    def setNucleationType(self, site):
        '''
        Sets nucleation site type

        Parameters
        ----------
        site - int
            Index for site type based off list on top of this file
        '''
        if isinstance(site, str):
            if site.upper() == 'GRAIN_BOUNDARIES':
                self.nucleationSiteType = self.GRAIN_BOUNDARIES
            elif site.upper() == 'GRAIN_EDGES':
                self.nucleationSiteType = self.GRAIN_EDGES
            elif site.upper() == 'GRAIN_CORNERS':
                self.nucleationSiteType = self.GRAIN_CORNERS
            elif site.upper() == 'DISLOCATIONS':
                self.nucleationSiteType = self.DISLOCATION
            else:
                self.nucleationSiteType = self.BULK
        else:
            self.nucleationSiteType = site

    def getGBRatio(self, gbEnergy, gamma):
        '''
        Grain boundary to interfacial energy ratio
        '''
        return gbEnergy / (2*gamma)
    
    def _createArrays(self, gbk, limit = np.inf):
        '''
        Creates arrays for grain boundary factors, this is done here to allow limits
        to be placed on the grain boundary ratio
        '''
        gbk = np.atleast_1d(gbk)
        indices = gbk < limit
        valid_gbk = gbk[indices]

        gbRemoval = -1*np.ones(gbk.shape, dtype=np.float64)
        areaFactor = -1*np.ones(gbk.shape, dtype=np.float64)
        volumeFactor = -1*np.ones(gbk.shape, dtype=np.float64)
        
        return gbk, valid_gbk, indices, gbRemoval, areaFactor, volumeFactor
    
    def _createNucleationFactors(self, areaFactor, volumeFactor, gbRemoval, indices, setInvalidToNan = True):
        '''
        Creates a NucleationFactorData object

        By default, setInvalidToNan will be true, will will set all invalid indices to nan
            This is for the user API so that it's easy to tell when a gb energy / interfacial energy
            ratio is invalid
        When setting factors internally, this will be set to false so it's easier to compare
        when a value is invalid (since np.nan == np.nan will return False)
        '''
        if setInvalidToNan:
            areaFactor[~indices] = np.nan
            volumeFactor[~indices] = np.nan
            gbRemoval[~indices] = np.nan
        return NucleationFactorData(area_factor=np.squeeze(areaFactor),
                      volume_factor=np.squeeze(volumeFactor),
                      gb_removal=np.squeeze(gbRemoval))


    def bulkFactors(self, gbk = None, setInvalidToNan = True):
        '''
        Factors for bulk nucleation. This assumes precipitate is spherical 
        and no GB is removed
        '''
        gbk, valid_gbk, indices, gbRemoval, areaFactor, volumeFactor = self._createArrays(gbk, np.inf)

        gbRemoval[indices] = 0
        areaFactor[indices] = 4 * np.pi
        volumeFactor[indices] = 4 * np.pi / 3
        
        return self._createNucleationFactors(areaFactor, volumeFactor, gbRemoval, indices, setInvalidToNan)

    def grainBoundaryFactors(self, gbk, setInvalidToNan = True):
        '''
        Factors for grain boundary nucleation
        '''
        gbk, valid_gbk, indices, gbRemoval, areaFactor, volumeFactor = self._createArrays(gbk, 1)

        gbRemoval[indices] = np.pi * (1 - valid_gbk**2)
        areaFactor[indices] = 4 * np.pi * (1 - valid_gbk)
        volumeFactor[indices] = (2 * np.pi / 3) * (2 - 3 * valid_gbk + valid_gbk**3)
        
        return self._createNucleationFactors(areaFactor, volumeFactor, gbRemoval, indices, setInvalidToNan)

    def grainEdgeFactors(self, gbk, setInvalidToNan = True):
        '''
        Factors for grain edge nucleation
        '''
        gbk, valid_gbk, indices, gbRemoval, areaFactor, volumeFactor = self._createArrays(gbk, np.sqrt(3)/2)

        alpha = np.arcsin(1 / (2 * np.sqrt(1 - valid_gbk**2)))
        beta = np.arccos(valid_gbk / np.sqrt(3 * (1 - valid_gbk**2)))

        gbRemoval[indices] = 3 * beta * (1 - valid_gbk**2) - valid_gbk * np.sqrt(3 - 4 * valid_gbk**2)
        areaFactor[indices] = 12 * (np.pi / 2 - alpha - valid_gbk * beta)
        volumeFactor[indices] = 2 * (np.pi - 2 * alpha + (valid_gbk**2 / 3) * np.sqrt(3 - 4 * valid_gbk**2) - beta * valid_gbk * (3 - valid_gbk**2))
        
        return self._createNucleationFactors(areaFactor, volumeFactor, gbRemoval, indices, setInvalidToNan)
    
    def grainCornerFactors(self, gbk, setInvalidToNan = True):
        '''
        Factors for grain corners nucleation
        '''
        gbk, valid_gbk, indices, gbRemoval, areaFactor, volumeFactor = self._createArrays(gbk, np.sqrt(2/3))

        K = (4 / 3) * np.sqrt(3 / 2 - 2 * valid_gbk**2) - 2 * valid_gbk / 3
        phi = np.arcsin(K / (2 * np.sqrt(1 - valid_gbk**2)))
        delta = np.arccos((np.sqrt(2) - valid_gbk * np.sqrt(3 - K**2)) / (K * np.sqrt(1 - valid_gbk**2)))

        gbRemoval[indices] = 3 * (2 * phi * (1 - valid_gbk**2) - K * (np.sqrt(1 - valid_gbk**2 - K**2 / 4) - K**2 / np.sqrt(8)))
        areaFactor[indices] = 24 * (np.pi / 3 - valid_gbk * phi - delta)
        volumeFactor[indices] = 2 * (4 * (np.pi / 3 - delta) + valid_gbk * K * (np.sqrt(1 - valid_gbk**2 - K**2 / 4) - K**2 / np.sqrt(8)) - 2 * valid_gbk * phi * (3 - valid_gbk**2))
        
        return self._createNucleationFactors(areaFactor, volumeFactor, gbRemoval, indices, setInvalidToNan)

    def setFactors(self, gbEnergy, gamma, nucleationSiteType = None):
        '''
        Calculated area, volume and GB removal factors

        Parameters
        ----------
        gbEnergy - float
            Grain boundary energy
        gamma - float
            Interfacial energy between precipitate and bulk
        '''
        if nucleationSiteType is not None:
            self.setNucleationType(nucleationSiteType)

        #Redundant storage of GB and interfacial energy, 
        #but will help in defining Rcrit and Gcrit later
        self.gbEnergy = gbEnergy
        self.gamma = gamma
        self.GBk = self.getGBRatio(self.gbEnergy, self.gamma)

        if not self.isGrainBoundaryNucleation:
            gbData = self.bulkFactors(self.GBk, setInvalidToNan=False)
        elif self.nucleationSiteType == self.GRAIN_BOUNDARIES:
            gbData = self.grainBoundaryFactors(self.GBk, setInvalidToNan=False)
        elif self.nucleationSiteType == self.GRAIN_EDGES:
            gbData = self.grainEdgeFactors(self.GBk, setInvalidToNan=False)
        elif self.nucleationSiteType == self.GRAIN_CORNERS:
            gbData = self.grainCornerFactors(self.GBk, setInvalidToNan=False)

        self.areaFactor = gbData.area_factor
        self.volumeFactor = gbData.volume_factor
        self.gbRemoval = gbData.gb_removal

        #Otherwise, set back to bulk nucleation
        if any([self.areaFactor == -1, self.volumeFactor == -1, self.gbRemoval == -1]):
            site = ''
            ratio = 1
            if self.nucleationSiteType == self.GRAIN_BOUNDARIES:
                site = 'grain boundaries'
                ratio = 1
            elif self.nucleationSiteType == self.GRAIN_EDGES:
                site = 'grain edges'
                ratio = np.sqrt(3)/2
            elif self.nucleationSiteType == self.GRAIN_CORNERS:
                site = 'grain corners'
                ratio = np.sqrt(2/3)

            #Check grain boundary energy to interfacial energy ratio
            #If the ratio is too large to create a nucleation barrier, set nucleation site back to bulk
            #TODO: If the ratio is too large, then no nucleation barrier exists for nucleation on grain boundaries
            #      We need a way to handle this case rather than currently avoiding it
            print('Warning: Grain boundary to interfacial energy ratio is too large for nucleation barrier on {}'.format(site.upper()))
            print('For nucleation on {}, gamma_GB / 2 * gamma must be below {:.3f}, but is currently {:.3f}'.format(site.upper(), ratio, self.GBk))
            print('Setting nucleation site to bulk.')
            gbData = self.bulkFactors(self.GBk)
            self.areaFactor = gbData.area_factor
            self.volumeFactor = gbData.volume_factor
            self.gbRemoval = gbData.gb_removal

    @property
    def areaRemoval(self):
        if not self.isGrainBoundaryNucleation:
            return 1
        else:
            return np.sqrt(self.gbRemoval / np.pi)
        
    @property
    def isGrainBoundaryNucleation(self):
        return self.nucleationSiteType != self.BULK and self.nucleationSiteType != self.DISLOCATION

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

