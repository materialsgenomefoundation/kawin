import numpy as np
from numpy.testing import assert_allclose

from kawin.precipitation import ShapeFactor

Rsingle = 2
Rarray = np.linspace(1, 2, 10)

def test_SphericalOutput():
    '''
    Tests output of spherical shape factors given a single radius or list of radii
    Since these factors are constant, all factors should give a scaler value

    The aspectRatio, eqRadius, kineticFactor and thermoFactor methods returns
        a) a scalar for a single radius
        b) a list or corresponding length for a list of radii

    The normalRadii method returns
        a) a list of length 3 for a single radius
        b) an array of size (n x 3) for a list of radii of length n
    '''
    sf = ShapeFactor()
    sf.setSpherical()

    arSingle = sf.aspectRatio(Rsingle)
    arArray = sf.aspectRatio(Rarray)

    eqRsingle = sf.eqRadiusFactor(Rsingle)
    eqRarray = sf.eqRadiusFactor(Rarray)

    radiiSingle = sf.normalRadii(Rsingle)
    radiiArray = sf.normalRadii(Rarray)

    thermoSingle = sf.thermoFactor(Rsingle)
    thermoArray = sf.thermoFactor(Rarray)

    kineticSingle = sf.kineticFactor(Rsingle)
    kineticArray = sf.kineticFactor(Rarray)

    assert np.isscalar(arSingle) or (type(arSingle) == np.ndarray and arSingle.ndim == 0)
    assert arArray.shape == (10,)
    assert np.isscalar(eqRsingle) or (type(eqRsingle) == np.ndarray and eqRsingle.ndim == 0)
    assert eqRarray.shape == (10,)
    assert radiiSingle.shape == (3,)
    assert radiiArray.shape == (10,3)
    assert np.isscalar(thermoSingle) or (type(thermoSingle) == np.ndarray and thermoSingle.ndim == 0)
    assert thermoArray.shape == (10,)
    assert np.isscalar(kineticSingle) or (type(kineticSingle) == np.ndarray and kineticSingle.ndim == 0)
    assert kineticArray.shape == (10,)

    assert_allclose(arSingle, 1.0, rtol=1e-3)
    assert_allclose(eqRsingle, 1.0, rtol=1e-3)
    assert_allclose(radiiSingle, 0.62035*np.ones(3), rtol=1e-3)
    assert_allclose(thermoSingle, 1.0, rtol=1e-3)
    assert_allclose(kineticSingle, 1.0, rtol=1e-3)

def test_NeedleOutput():
    '''
    Tests output of needle shape factors given a single radius or list of radii

    The aspectRatio, eqRadius, kineticFactor and thermoFactor methods returns
        a) a scalar for a single radius
        b) a list or corresponding length for a list of radii

    The normalRadii method returns
        a) a list of length 3 for a single radius
        b) an array of size (n x 3) for a list of radii of length n
    '''
    sf = ShapeFactor()
    sf.setNeedleShape(ar=2)

    arSingle = sf.aspectRatio(Rsingle)
    arArray = sf.aspectRatio(Rarray)

    eqRsingle = sf.eqRadiusFactor(Rsingle)
    eqRarray = sf.eqRadiusFactor(Rarray)

    radiiSingle = sf.normalRadii(Rsingle)
    radiiArray = sf.normalRadii(Rarray)

    thermoSingle = sf.thermoFactor(Rsingle)
    thermoArray = sf.thermoFactor(Rarray)

    kineticSingle = sf.kineticFactor(Rsingle)
    kineticArray = sf.kineticFactor(Rarray)

    assert np.isscalar(arSingle) or (type(arSingle) == np.ndarray and arSingle.ndim == 0)
    assert arArray.shape == (10,)
    assert np.isscalar(eqRsingle) or (type(eqRsingle) == np.ndarray and eqRsingle.ndim == 0)
    assert eqRarray.shape == (10,)
    assert radiiSingle.shape == (3,)
    assert radiiArray.shape == (10,3)
    assert np.isscalar(thermoSingle) or (type(thermoSingle) == np.ndarray and thermoSingle.ndim == 0)
    assert thermoArray.shape == (10,)
    assert np.isscalar(kineticSingle) or (type(kineticSingle) == np.ndarray and kineticSingle.ndim == 0)
    assert kineticArray.shape == (10,)

    assert_allclose(arSingle, 2.0, rtol=1e-3)
    assert_allclose(eqRsingle, 1.259921, rtol=1e-3)
    assert_allclose(radiiSingle, [0.49237, 0.49237, 0.98474], rtol=1e-3)
    assert_allclose(thermoSingle, 1.076728, rtol=1e-3)
    assert_allclose(kineticSingle, 1.043867, rtol=1e-3)

def test_PlateOutput():
    '''
    Tests output of plate shape factors given a single radius or list of radii

    The aspectRatio, eqRadius, kineticFactor and thermoFactor methods returns
        a) a scalar for a single radius
        b) a list or corresponding length for a list of radii

    The normalRadii method returns
        a) a list of length 3 for a single radius
        b) an array of size (n x 3) for a list of radii of length n
    '''
    sf = ShapeFactor()
    sf.setPlateShape(ar=2)

    arSingle = sf.aspectRatio(Rsingle)
    arArray = sf.aspectRatio(Rarray)

    eqRsingle = sf.eqRadiusFactor(Rsingle)
    eqRarray = sf.eqRadiusFactor(Rarray)

    radiiSingle = sf.normalRadii(Rsingle)
    radiiArray = sf.normalRadii(Rarray)

    thermoSingle = sf.thermoFactor(Rsingle)
    thermoArray = sf.thermoFactor(Rarray)

    kineticSingle = sf.kineticFactor(Rsingle)
    kineticArray = sf.kineticFactor(Rarray)

    assert np.isscalar(arSingle) or (type(arSingle) == np.ndarray and arSingle.ndim == 0)
    assert arArray.shape == (10,)
    assert np.isscalar(eqRsingle) or (type(eqRsingle) == np.ndarray and eqRsingle.ndim == 0)
    assert eqRarray.shape == (10,)
    assert radiiSingle.shape == (3,)
    assert radiiArray.shape == (10,3)
    assert np.isscalar(thermoSingle) or (type(thermoSingle) == np.ndarray and thermoSingle.ndim == 0)
    assert thermoArray.shape == (10,)
    assert np.isscalar(kineticSingle) or (type(kineticSingle) == np.ndarray and kineticSingle.ndim == 0)
    assert kineticArray.shape == (10,)

    assert_allclose(arSingle, 2.0, rtol=1e-3)
    assert_allclose(eqRsingle, 1.587401, rtol=1e-3)
    assert_allclose(radiiSingle, [0.78159, 0.78159, 0.39079], rtol=1e-3)
    assert_allclose(thermoSingle, 1.095444, rtol=1e-3)
    assert_allclose(kineticSingle, 1.041946, rtol=1e-3)

def test_CuboidalOutput():
    '''
    Tests output of cuboidal shape factors given a single radius or list of radii

    The aspectRatio, eqRadius, kineticFactor and thermoFactor methods returns
        a) a scalar for a single radius
        b) a list or corresponding length for a list of radii

    The normalRadii method returns
        a) a list of length 3 for a single radius
        b) an array of size (n x 3) for a list of radii of length n
    '''
    sf = ShapeFactor()
    sf.setCuboidalShape(ar=2)

    arSingle = sf.aspectRatio(Rsingle)
    arArray = sf.aspectRatio(Rarray)

    eqRsingle = sf.eqRadiusFactor(Rsingle)
    eqRarray = sf.eqRadiusFactor(Rarray)

    radiiSingle = sf.normalRadii(Rsingle)
    radiiArray = sf.normalRadii(Rarray)

    thermoSingle = sf.thermoFactor(Rsingle)
    thermoArray = sf.thermoFactor(Rarray)

    kineticSingle = sf.kineticFactor(Rsingle)
    kineticArray = sf.kineticFactor(Rarray)

    assert np.isscalar(arSingle) or (type(arSingle) == np.ndarray and arSingle.ndim == 0)
    assert arArray.shape == (10,)
    assert np.isscalar(eqRsingle) or (type(eqRsingle) == np.ndarray and eqRsingle.ndim == 0)
    assert eqRarray.shape == (10,)
    assert radiiSingle.shape == (3,)
    assert radiiArray.shape == (10,3)
    assert np.isscalar(thermoSingle) or (type(thermoSingle) == np.ndarray and thermoSingle.ndim == 0)
    assert thermoArray.shape == (10,)
    assert np.isscalar(kineticSingle) or (type(kineticSingle) == np.ndarray and kineticSingle.ndim == 0)
    assert kineticArray.shape == (10,)

    assert_allclose(arSingle, 2.0, rtol=1e-3)
    assert_allclose(eqRsingle, 0.781592, rtol=1e-3)
    assert_allclose(radiiSingle, [0.7937, 0.7937, 1.5874], rtol=1e-3)
    assert_allclose(thermoSingle, 1.302654, rtol=1e-3)
    assert_allclose(kineticSingle, 0.99737, rtol=1e-3)

def test_AspectRatio():
    '''
    Tests that aspect ratio can be as a function
    '''
    arFunc = lambda R : 0.5 * R + 3
    sf = ShapeFactor()
    sf.setAspectRatio(arFunc)
    Rarray = sf.aspectRatio(np.array([1, 2]))

    assert_allclose(Rarray, np.array([3.5, 4]), rtol=1e-5)

def test_RcritScalar():
    sf = ShapeFactor()
    sf.setNeedleShape(2)

    RcritSphere = 1e-9
    Rcrit = sf._findRcritScalar(RcritSphere, 0)
    assert_allclose(Rcrit, 1.076728e-9, rtol=1e-3)

def test_RcritSearch():
    sf = ShapeFactor()
    arFunc = lambda r: 2.3 * (r/1e-9)**1.1
    sf.setPrecipitateShape('needle', arFunc)

    RcritSphere =1e-9
    Rmax = 1e-8
    Rcrit = sf._findRcrit(RcritSphere, Rmax)
    assert_allclose(Rcrit, 1.149414e-9, rtol=1e-3)







