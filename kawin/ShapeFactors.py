import numpy as np

class ShapeFactor:
    '''
    Defines functions for shape factors of needle, plate and cuboidal precipitates
    Shape factors involve the following:
        Aspect Ratio - ratio of long axis to short axis
        Eccentricity - characteristic value of the precipitate shape that depends on the aspect ratio
        Equivalent radius factor - correction factor to the radius to give the radius of a sphere with equivalent volume
        Kinetic factor - correction factor for the growth rate of the particle
        Thermodynamic factor / Area factor - correction factor for the critical driving force/radius for nucleation
        
        For a sphere (or needle and plate precipitates with aspect ratio of 1):
            Eccentricity = 0
            Equivalent radius factor = 1
            Kinetic factor = 1
            Thermodynamic factor = 1

    NOTE:
    normalRadaii, eqRadiusFactor, kineticFactor and thermoFactor 
        take in R and output the corresponding factors

    _normalRadaiiEquation, __eqRadiusEquation, _kineticEquation and _thermoEquation
        take in aspect ratio and output the corresponding factors
    '''
    
    #Particle shape types
    SPHERE = 0
    NEEDLE = 1
    PLATE = 2
    CUBIC = 3

    def __init__(self):
        self.setSpherical()
        self.setAspectRatio(1)

        #This is the same for all equations
        self.eqRadiusFactorMin = 1
        self.tol = 1e-3
        
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
        if hasattr(R, '__len__'):
            return self._aspectRatioScalar * np.ones(len(R))
        else:
            return self._aspectRatioScalar
            
    def setSpherical(self, ar = 1):
        '''
        Sets factors for spherical precipitates
        '''
        self.particleType = self.SPHERE
        self._eqRadiusEquation = self._eqRadiusFactorSphere
        self._normalRadiiEquation = self._normalRadiiSphere
        self._kineticEquation = self._kineticFactorSphere
        self._thermoEquation = self._thermoFactorSphere
        self.setAspectRatio(1)
        
        self.kineticFactorMin = 1
        self.thermoFactorMin = 1
        
    def setNeedleShape(self, ar = 1):
        '''
        Factors for needle shaped precipitates
        '''
        self.particleType = self.NEEDLE
        self.setAspectRatio(ar)
        self._eqRadiusEquation = self._eqRadiusFactorNeedle
        self._normalRadiiEquation = self._normalRadiiNeedle
        self._kineticEquation = self._kineticFactorEquationNeedle
        self._thermoEquation = self._thermoFactorEquationNeedle
        
        self.kineticFactorMin = 1
        self.thermoFactorMin = 1
        
    def setPlateShape(self, ar = 1):
        '''
        Factors for plate shaped precipitates
        '''
        self.particleType = self.PLATE
        self.setAspectRatio(ar)
        self._eqRadiusEquation = self._eqRadiusFactorPlate
        self._normalRadiiEquation = self._normalRadiiPlate
        self._kineticEquation = self._kineticFactorEquationPlate
        self._thermoEquation = self._thermoFactorEquationPlate
        
        self.kineticFactorMin = 1
        self.thermoFactorMin = 1
        
    def setCuboidalShape(self, ar = 1):
        '''
        Factors for cuboidal shaped precipitates
        '''
        self.particleType = self.CUBIC
        self.setAspectRatio(ar)
        self._eqRadiusEquation = self._eqRadiusFactorCuboidal
        self._normalRadiiEquation = self._normalRadiiCuboidal
        self._kineticEquation = self._kineticFactorEquationCuboidal
        self._thermoEquation = self._thermoFactorEquationCuboidal
        
        self.kineticFactorMin = 1
        
        #Thermodynamic factor for cuboidal precipitates is not 1 when aspect ratio is 1
        self.thermoFactorMin = self._thermoFactorEquationCuboidal(1)

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

        #Keep aspect ratio above 1
        if hasattr(ar, '__len__'):
            ar[ar < 1] = 1
        else:
            ar = 1 if ar < 1 else ar
            
        return self._normalRadiiEquation(ar)

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
        if hasattr(ar, '__len__'):
            factor = self.eqRadiusFactorMin * np.ones(len(ar))
            factor[ar > 1] = self._eqRadiusEquation(ar[ar > 1])
            return factor
        else:
            if ar <= 1:
                return self.eqRadiusFactorMin
            else:
                return self._eqRadiusEquation(ar)
        
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
        if hasattr(ar, '__len__'):
            factor = self.kineticFactorMin * np.ones(len(ar))
            factor[ar > 1] = self._kineticEquation(ar[ar > 1])
            return factor
        else:
            if ar <= 1:
                return self.kineticFactorMin
            else:
                return self._kineticEquation(ar)
                
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
        if hasattr(ar, '__len__'):
            factor = self.thermoFactorMin * np.ones(len(ar))
            factor[ar > 1] = self._thermoEquation(ar[ar > 1])
            return factor
        else:
            if ar <= 1:
                return self.thermoFactorMin
            else:
                return self._thermoEquation(ar)
                
    def eccentricity(self, ar):
        '''
        Eccentricity given the aspect ratio (for needle and plate shaped precipitates)
        '''
        return np.sqrt(1 - 1 / ar**2)
        
    # Equivalent spherical radius ----------------------------------------
    def _eqRadiusFactorSphere(self, ar):
        '''
        Equivalent radius for a sphere (returns 1)
        '''
        return 1

    def _eqRadiusFactorNeedle(self, ar):
        '''
        Equivalent radius for needle shaped precipitate
        '''
        return np.cbrt(ar)

    def _eqRadiusFactorPlate(self, ar):
        '''
        Equivalent radius for plate shaped precipitate
        '''
        return np.cbrt(ar**2)

    def _eqRadiusFactorCuboidal(self, ar):
        '''
        Equivalent radius for cuboidal shaped precipitate
        '''
        return np.cbrt(3 * ar / (4 * np.pi))

    # Normalized radaii --------------------------------------------------
    def _normalRadiiSphere(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        if hasattr(ar, '__len__'):
            return np.cbrt((3 / 4 * np.pi)) * np.ones((len(ar), 3))
        else:
            return np.cbrt((3 / (4 * np.pi))) * np.ones(3)

    def _normalRadiiNeedle(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        scale = np.cbrt(1 / ar)
        if hasattr(ar, '__len__'):
            return np.cbrt((3 / (4 * np.pi))) * np.array([scale, scale, scale * ar]).T
        else:
            return np.cbrt((3 / (4 * np.pi))) * np.array([scale, scale, scale * ar])

    def _normalRadiiPlate(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        scale = np.cbrt(1 / ar**2)
        if hasattr(ar, '__len__'):
            return np.cbrt((3 / (4 * np.pi))) * np.array([scale * ar, scale * ar, scale]).T
        else:
            return np.cbrt((3 / (4 * np.pi))) * np.array([scale * ar, scale * ar, scale])

    def _normalRadiiCuboidal(self, ar):
        '''
        Returns radius along the 3-axis for a volume of 1
        '''
        scale = np.cbrt(1 / ar)
        if hasattr(ar, '__len__'):
            return np.array([scale, scale, scale * ar]).T
        else:
            return np.array([scale, scale, scale * ar])

    # Kinetic factor -----------------------------------------------------        
    def _kineticFactorSphere(self, ar):
        '''
        Kinetic factor for a sphere (returns 1)
        '''
        return 1

    def _kineticFactorEquationNeedle(self, ar):
        '''
        Kinetic factor for needle shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return 2 * np.cbrt(ar**2) * ecc / (np.log(1 + ecc) - np.log(1 - ecc))

    def _kineticFactorEquationPlate(self, ar):
        '''
        Kinetic factor for plate shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return ecc * np.cbrt(ar) / (np.arccos(0) - np.arccos(ecc))

    def _kineticFactorEquationCuboidal(self, ar):
        '''
        Kinetic factor for cuboidal shaped precipitate
        '''
        return 0.1 * np.exp(-0.091 * (ar - 1)) + 1.736 * np.sqrt(ar**2 - 1) / (np.cbrt(ar) * np.log(2 * ar**2 + 2 * ar * np.sqrt(ar**2 - 1) - 1))

    # Thermodynamic factor -----------------------------------------------
    def _thermoFactorSphere(self, ar):
        '''
        Thermodynamic factor for a sphere (returns 1)
        '''
        return 1
        
    def _thermoFactorEquationNeedle(self, ar):
        '''
        Thermodynamic factor for needle shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return (1 / (2 * ar**(2/3))) * (1 + ar / ecc * np.arcsin(ecc))
        
    def _thermoFactorEquationPlate(self, ar):
        '''
        Thermodynamic factor for plate shaped precipitate
        '''
        ecc = self.eccentricity(ar)
        return (1 / (2 * ar**(4/3))) * (ar**2 + (1 / (2 * ecc)) * np.log((1 + ecc) / (1 - ecc)))
        
    def _thermoFactorEquationCuboidal(self, ar):
        '''
        Thermodynamic factor for cuboidal shaped precipitate
        '''
        return (2 * ar + 1) / (2 * np.pi) * (4 * np.pi / (3 * ar))**(2/3)
    
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
        mid = (minR + maxR) / 2
        
        #Objective function is R = R_sphere * thermoFactor(ar(R))
        #Or R / (R_sphere * thermoFactor(ar(R))) - 1 = 0, this requires that the radius is within tol percent of true value
        fMin = minR / (RcritSphere * self.thermoFactor(minR)) - 1
        fMax = maxR / (RcritSphere * self.thermoFactor(maxR)) - 1
        fMid = mid / (RcritSphere * self.thermoFactor(mid)) - 1

        #fMin = min / (RcritSphere * self.eqRadiusFactor(min)) - 1
        #fMax = max / (RcritSphere * self.eqRadiusFactor(max)) - 1
        #fMid = mid / (RcritSphere * self.eqRadiusFactor(mid)) - 1

        n = 0
        while np.abs(fMid) > self.tol:
            if fMin * fMid >= 0:
                minR = mid
                fMin = fMid
            else:
                maxR = mid
                fMax = fMid
                
            mid = (minR + maxR) / 2
            fMid = mid / (RcritSphere * self.thermoFactor(mid)) - 1
            #fMid = mid / (RcritSphere * self.eqRadiusFactor(mid)) - 1
            
            n += 1
            if n == 100:
                return RcritSphere
                
        return mid
        