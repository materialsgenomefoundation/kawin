'''
Shape factors for non-spherical precipitates are defined for the following:
    equivalent spherical radius
        For a characteristic length of 1, this gives the radius of a sphere with
        the equivalent volume
    normal radii
        This gives the radii along the three axes that gives a volume of 1
    kinetic factor
        Correction term to the growth rate
    thermo factor
        Correction term to the Gibbs-Thomson coefficient

The default assumption for the KWN model is a spherical precipitate, so 
the equivalent radius, kinetic factor and thermo factor all return 1

equivalent radius, normal radii, and kinetic factor for needle, plate and cuboidal taken from 
K. Wu, Q. Chen, P. Mason, "Simulation of precipitation kinetics with non-spherical particles"
Journal of Phase Equilibria and Diffusion 39 (2018) 571
doi:10.1007/s11669-018-0644-1

thermo factor taken from
B. Holmedal, E. Osmundsen and Q. Du, "Precipitation of Non-spherical particles in aluminum alloys
part I: Generalization of the Kampmann-Wagner numerical model" Metallurgical and Materials
Transactions A 47A (2016) 581
doi:10.1007/s1161-015-3197-5

The thermo factor is also defined in Wu et al. The differences between these two papers is that
Holmedal et al uses the surface area correction of the particle to modify the Gibbs-Thomson 
contribution while Wu et al uses the volume correction
'''

import numpy as np

class ShapeDescriptionBase:
    '''
    Defines functions to describe a precipitate shape

    Must implement (as a function of aspect ratio)
        _eqRadius - equivalent spherical radius (radius of sphere that gives the same volume)
        _normalRadii - radius along normals to give a spherical volume of 1
        _thermoFactor - factor that modifies the  Gibbs-Thomson contribution
        _kineticFactor - factor that modifies the growth rate
    '''
    name = 'ABSTRACT SHAPE DESCRIPTION'

    def __init__(self):
        # Factors for when aspect ratio = 1
        self.eqRadiusFactorMin = 1
        self.kineticFactorMin = 1
        self.thermoFactorMin = 1

    def _processAspectRatio(self, ar):
        ar = np.atleast_1d(ar)
        ar[ar < 1] = 1
        return ar

    def eccentricity(self, ar):
        '''
        Eccentricity given the aspect ratio (for needle and plate shaped precipitates)
        '''
        return np.sqrt(1 - 1 / ar**2)
    
    def normalRadii(self, ar):
        '''
        Radius along the 3 axis to give a spherical volume of 1

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        3 length array for each axis if R is scalar
        n x 3 array if R is an array
        '''
        ar = self._processAspectRatio(ar)
        return np.squeeze(self._normalRadii(ar))

    def eqRadiusFactor(self, ar):
        '''
        Equivalent spherical radius factor for generic precipitates

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        Eq. radius factor with same shape as R
        '''
        ar = self._processAspectRatio(ar)
        factor = self.eqRadiusFactorMin * np.ones(ar.shape)
        factor[ar > 1] = self._eqRadius(ar[ar > 1])
        return np.squeeze(factor)
        
    def kineticFactor(self, ar):
        '''
        Kinetic factor for generic precipitates

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        Kinetic factor with same shape as R
        '''
        ar = self._processAspectRatio(ar)
        factor = self.kineticFactorMin * np.ones(ar.shape)
        factor[ar > 1] = self._kineticFactor(ar[ar > 1])
        return np.squeeze(factor)
                
    def thermoFactor(self, ar):
        '''
        Thermodynamic factor for generic precipitates

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        Thermodynamic factor with same shape as R
        '''
        ar = self._processAspectRatio(ar)
        factor = self.thermoFactorMin * np.ones(ar.shape)
        factor[ar > 1] = self._thermoFactor(ar[ar > 1])
        return np.squeeze(factor)
    
    def _eqRadius(self, ar):
        raise NotImplementedError()
    
    def _normalRadii(self, ar):
        raise NotImplementedError()
    
    def _kineticFactor(self, ar):
        raise NotImplementedError()
    
    def _thermoFactor(self, ar):
        raise NotImplementedError()
    
class SphereDescription(ShapeDescriptionBase):
    name = 'SPHERE'

    def _eqRadius(self, ar):
        '''
        Equivalent radius for a sphere (returns 1)
        '''
        return np.ones(ar.shape)
    
    def _normalRadii(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        return np.cbrt((3 / (4 * np.pi))) * np.ones((len(ar), 3))
    
    def _kineticFactor(self, ar):
        '''
        Kinetic factor for a sphere (returns 1)
        '''
        return np.ones(ar.shape)
    
    def _thermoFactor(self, ar):
        '''
        Thermodynamic factor for a sphere (returns 1)
        '''
        return np.ones(ar.shape)
    
class NeedleDescription(ShapeDescriptionBase):
    name = 'NEEDLE'

    def _eqRadius(self, ar):
        '''
        Equivalent radius for needle shaped precipitate
        '''
        return np.cbrt(ar)

    def _normalRadii(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        scale = np.cbrt(1 / ar)
        return np.cbrt((3 / (4 * np.pi))) * np.array([scale, scale, scale * ar]).T
    
    def _kineticFactor(self, ar):
        '''
        Kinetic factor for needle shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return 2 * np.cbrt(ar**2) * ecc / (np.log(1 + ecc) - np.log(1 - ecc))
    
    def _thermoFactor(self, ar):
        '''
        Thermodynamic factor for needle shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return (1 / (2 * ar**(2/3))) * (1 + ar / ecc * np.arcsin(ecc))
    
class PlateDescription(ShapeDescriptionBase):
    name = 'PLATE'

    def _eqRadius(self, ar):
        '''
        Equivalent radius for plate shaped precipitate
        '''
        return np.cbrt(ar**2)
    
    def _normalRadii(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        scale = np.cbrt(1 / ar**2)
        return np.cbrt((3 / (4 * np.pi))) * np.array([scale * ar, scale * ar, scale]).T
    
    def _kineticFactor(self, ar):
        '''
        Kinetic factor for plate shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return ecc * np.cbrt(ar) / (np.pi/2 - np.arccos(ecc))
        #return ecc * np.cbrt(ar) / (np.arccos(0) - np.arccos(ecc))
        #return ecc * np.cbrt(ar) / np.arccos(1/ar)

    def _thermoFactor(self, ar):
        '''
        Thermodynamic factor for plate shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return (1 / (2 * ar**(4/3))) * (ar**2 + (1 / (2 * ecc)) * np.log((1 + ecc) / (1 - ecc)))
    
class CuboidalDescription(ShapeDescriptionBase):
    name = 'CUBIC'
    
    def __init__(self):
        super().__init__()
        self.eqRadiusFactorMin = self.eqRadiusFactor(1)
        self.kineticFactorMin = self.kineticFactor(1.0001)
        self.thermoFactorMin = self.thermoFactor(1)

    def _eqRadius(self, ar):
        '''
        Equivalent radius for cuboidal shaped precipitate
        '''
        return np.cbrt(3 * ar / (4 * np.pi))
    
    def _normalRadii(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        scale = np.cbrt(1 / ar)
        return np.array([scale, scale, scale * ar]).T
    
    def _kineticFactor(self, ar):
        '''
        Kinetic factor for cuboidal shaped precipitate
        '''
        return 0.1 * np.exp(-0.091 * (ar - 1)) + 1.736 * np.sqrt(ar**2 - 1) / (np.cbrt(ar) * np.log(2 * ar**2 + 2 * ar * np.sqrt(ar**2 - 1) - 1))

    def _thermoFactor(self, ar):
        '''
        Thermodynamic factor for cuboidal shaped precipitate
        '''
        return (2 * ar + 1) / (2 * np.pi) * (4 * np.pi / (3 * ar))**(2/3)

class ShapeFactor:
    '''
    Defines functions for shape factors of needle, plate and cuboidal precipitates
    Shape factors involve the following:
        Aspect Ratio - ratio of long axis to short axis
        Eccentricity - characteristic value of the precipitate shape that depends on the aspect ratio
        Equivalent radius factor - correction factor to the radius to give the radius of a sphere with equivalent volume
        Kinetic factor - correction factor for the growth rate of the particle
        Thermodynamic factor / Area factor - correction factor for the critical driving force/radius for nucleation
        
    NOTE:
    normalRadaii, eqRadiusFactor, kineticFactor and thermoFactor are functions of radius
        ShapeFactor.function(radius) -> factor

    The equivalent functions in the description are functions of aspect ratio
        ShapeFactor.description(aspect ratio) -> factor
    '''
    def __init__(self, precipitateShape='sphere', ar=1):
        self._description = SphereDescription()
        self._updateCallbacks = []
        self.tol = 1e-3

        self.setPrecipitateShape(precipitateShape, ar)

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, value):
        self._description = value
        for callback in self._updateCallbacks:
            callback()

    def setPrecipitateShape(self, precipitateShape, ar = 1):
        '''
        General shape setting function

        Defaults to spherical
        '''
        descriptionDict = {
            SphereDescription.name.upper(): SphereDescription(),
            NeedleDescription.name.upper(): NeedleDescription(),
            PlateDescription.name.upper(): PlateDescription(),
            CuboidalDescription.name.upper(): CuboidalDescription(),
        }
        if isinstance(precipitateShape, str):
            precipitateShape = precipitateShape.upper()
        newDescription = descriptionDict.get(precipitateShape, precipitateShape)
        if not isinstance(newDescription, ShapeDescriptionBase):
            validValues = ', '.join(list(descriptionDict.keys()))
            raise ValueError(f"Unknown value '{precipitateShape}'. Value must be: {validValues} or an instance of ShapeDescriptionBase")
        self.description = newDescription

        # Override aspect ratio for sphere description to be 1
        if isinstance(precipitateShape, SphereDescription):
            ar = 1
        self.setAspectRatio(ar)
            
    def setSpherical(self, ar = 1):
        '''
        Sets factors for spherical precipitates
        '''
        self.setPrecipitateShape(SphereDescription(), 1)
        
    def setNeedleShape(self, ar = 1):
        '''
        Factors for needle shaped precipitates
        '''
        self.setPrecipitateShape(NeedleDescription(), ar)
        
    def setPlateShape(self, ar = 1):
        '''
        Factors for plate shaped precipitates
        '''
        self.setPrecipitateShape(PlateDescription(), ar)
        
    def setCuboidalShape(self, ar = 1):
        '''
        Factors for cuboidal shaped precipitates
        '''
        self.setPrecipitateShape(CuboidalDescription(), ar)
        
    def setAspectRatio(self, ar):
        '''
        Sets aspect ratio as a function of R (equivalent spherical radius)
        
        Parameters
        ----------
        ar : float or function
            Aspect ratio
            If function, must take in radius and return aspect ratio
        '''
        if np.isscalar(ar):
            self._aspectRatioScalar = ar
            self.aspectRatio = self._scalarAspectRatioEquation
            self.findRcrit = self._findRcritScalar
        else:
            self.aspectRatio = ar
            self.findRcrit = self._findRcrit
            
    def _scalarAspectRatioEquation(self, R):
        '''
        Aspect ratio as a function of radius if the aspect ratio is set as a scalar
        '''
        R = np.atleast_1d(R)
        return np.squeeze(self._aspectRatioScalar * np.ones(R.shape))

    def normalRadii(self, R):
        '''
        Radius along the 3 axis to give a spherical volume of 1

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        3 length array for each axis if R is scalar
        n x 3 array if R is an array
        '''
        ar = self.aspectRatio(R)
        return self.description.normalRadii(ar)

    def eqRadiusFactor(self, R):
        '''
        Equivalent spherical radius factor for generic precipitates

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        Eq. radius factor with same shape as R
        '''
        ar = self.aspectRatio(R)
        return self.description.eqRadiusFactor(ar)
        
    def kineticFactor(self, R):
        '''
        Kinetic factor for generic precipitates

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        Kinetic factor with same shape as R
        '''
        ar = self.aspectRatio(R)
        return self.description.kineticFactor(ar)
                
    def thermoFactor(self, R):
        '''
        Thermodynamic factor for generic precipitates

        Parameters
        ----------
        R : float or array
            Equivalent spherical radius

        Returns
        -------
        Thermodynamic factor with same shape as R
        '''
        ar = self.aspectRatio(R)
        return self.description.thermoFactor(ar)
    
    def _findRcritScalar(self, RcritSphere, Rmax):
        '''
        Critical radius given a scalar aspect ratio
        '''
        return RcritSphere * self.thermoFactor(RcritSphere)
        #return RcritSphere * self.eqRadiusFactor(RcritSphere)
    
    def _findRcrit(self, RcritSphere, Rmax):
        '''
        Critical radius given aspect ratio as a function of radius
        Found by bisection method
        '''
        minR = RcritSphere
        maxR = Rmax
        midR = (minR + maxR) / 2
        
        #Objective function is R = R_sphere * thermoFactor(ar(R))
        #Or R / (R_sphere * thermoFactor(ar(R))) - 1 = 0, this requires that the radius is within tol percent of true value
        fMin = minR / (RcritSphere * self.thermoFactor(minR)) - 1
        fMax = maxR / (RcritSphere * self.thermoFactor(maxR)) - 1
        fMid = midR / (RcritSphere * self.thermoFactor(midR)) - 1

        n = 0
        while np.abs(fMid) > self.tol:
            if fMin * fMid >= 0:
                minR = midR
                fMin = fMid
            else:
                maxR = midR
                fMax = fMid
                
            midR = (minR + maxR) / 2
            fMid = midR / (RcritSphere * self.thermoFactor(midR)) - 1
            
            n += 1
            if n == 100:
                return RcritSphere
                
        return midR
        