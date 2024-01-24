import numpy as np

class GBFactors:
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

    def setNucleationType(self, site):
        '''
        Sets nucleation site type

        Parameters
        ----------
        site - int
            Index for site type based off list on top of this file
        '''
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

    def setFactors(self, gbEnergy, gamma):
        '''
        Calculated area, volume and GB removal factors

        Parameters
        ----------
        gbEnergy - float
            Grain boundary energy
        gamma - float
            Interfacial energy between precipitate and bulk
        '''
        #Redundant storage of GB and interfacial energy, 
        #but will help in defining Rcrit and Gcrit later
        self.gbEnergy = gbEnergy
        self.gamma = gamma
        self.GBk = gbEnergy / (2 * gamma)

        #Bulk nucleation
        if self.nucleationSiteType == self.BULK or self.nucleationSiteType == self.DISLOCATION:
            self.gbRemoval = 0
            self.areaFactor = 4 * np.pi
            self.volumeFactor = 4 * np.pi / 3
        
        #Grain boundary area nucleation
        elif self.nucleationSiteType == self.GRAIN_BOUNDARIES:
            self.gbRemoval = np.pi * (1 - self.GBk**2)
            self.areaFactor = 4 * np.pi * (1 - self.GBk)
            self.volumeFactor = (2 * np.pi / 3) * (2 - 3 * self.GBk + self.GBk**3)

        #Grain edge nucleation
        elif self.nucleationSiteType == self.GRAIN_EDGES and self.GBk < np.sqrt(3) / 2:
            alpha = np.arcsin(1 / (2 * np.sqrt(1 - self.GBk**2)))
            beta = np.arccos(self.GBk / np.sqrt(3 * (1 - self.GBk**2)))
            self.gbRemoval = 3 * beta * (1 - self.GBk**2) - self.GBk * np.sqrt(3 - 4 * self.GBk**2)
            self.areaFactor = 12 * (np.pi / 2 - alpha - self.GBk * beta)
            self.volumeFactor = 2 * (np.pi - 2 * alpha + (self.GBk**2 / 3) * np.sqrt(3 - 4 * self.GBk**2) - beta * self.GBk * (3 - self.GBk**2))
                
        #Grain corner nucleation
        elif self.nucleationSiteType == self.GRAIN_CORNERS and self.GBk < np.sqrt(2 / 3):
            K = (4 / 3) * np.sqrt(3 / 2 - 2 * self.GBk**2) - 2 * self.GBk / 3
            phi = np.arcsin(K / (2 * np.sqrt(1 - self.GBk**2)))
            delta = np.arccos((np.sqrt(2) - self.GBk * np.sqrt(3 - K**2)) / (K * np.sqrt(1 - self.GBk**2)))
            self.gbRemoval = 3 * (2 * phi * (1 - self.GBk**2) - K * (np.sqrt(1 - self.GBk**2 - K**2 / 4) - K**2 / np.sqrt(8)))
            self.areaFactor = 24 * (np.pi / 3 - self.GBk * phi - delta)
            self.volumeFactor = 2 * (4 * (np.pi / 3 - delta) + self.GBk * K * (np.sqrt(1 - self.GBk**2 - K**2 / 4) - K**2 / np.sqrt(8)) - 2 * self.GBk * phi * (3 - self.GBk**2))
        
        #Otherwise, set back to bulk nucleation
        else:
            site = ''
            if self.nucleationSiteType == self.GRAIN_BOUNDARIES:
                site = 'grain boundaries'
            elif self.nucleationSiteType == self.GRAIN_EDGES:
                site = 'grain edges'
            elif self.nucleationSiteType == self.GRAIN_CORNERS:
                site = 'grain corners'

            #Check grain boundary energy to interfacial energy ratio
            #If the ratio is too large to create a nucleation barrier, set nucleation site back to bulk
            #TODO: If the ratio is too large, then no nucleation barrier exists for nucleation on grain boundaries
            #      We need a way to handle this case rather than currently avoiding it
            print('Warning: Grain boundary to interfacial energy ratio is too large for nucleation barrier on {}'.format(site.upper()))
            if self.nucleationSiteType == self.GRAIN_BOUNDARIES:
                ratio = 1
            elif self.nucleationSiteType == self.GRAIN_EDGES:
                ratio = np.sqrt(3) / 2
            elif self.nucleationSiteType == self.GRAIN_CORNERS:
                ratio = np.sqrt(2 / 3)
            print('For nucleation on {}, gamma_GB / 2 * gamma must be below {:.3f}, but is currently {:.3f}'.format(site.upper(), ratio, self.GBk))
            print('Setting nucleation site to bulk.')
            self.gbRemoval = 0
            self.areaFactor = 4 * np.pi
            self.volumeFactor = 4 * np.pi / 3

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

