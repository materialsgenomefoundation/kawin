from numpy.testing import assert_allclose
import numpy as np
from kawin.ShapeFactors import ShapeFactor
from kawin.ElasticFactors import StrainEnergy

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

def test_AspectRatio():
    '''
    Tests that aspect ratio can be as a function
    '''
    arFunc = lambda R : 0.5 * R + 3
    sf = ShapeFactor()
    sf.setAspectRatio(arFunc)
    Rarray = sf.aspectRatio(np.array([1, 2]))

    assert_allclose(Rarray, np.array([3.5, 4]), rtol=1e-5)

def test_StrainOutput():
    '''
    Tests that strain energy outputs:
        a) a scalar for a single set of radii (3 length array)
        b) a list for a list of radii (n x 3 array)
    '''
    se = StrainEnergy()
    se.setEigenstrain(0.01)
    se.setElasticConstants(150e9, 100e9, 75e9)

    rSingle = np.random.random(3)
    rArray = np.random.random((10, 3))

    elSingle = se.strainEnergy(rSingle)
    elArray = se.strainEnergy(rArray)

    assert np.isscalar(elSingle) or (type(elSingle) == np.ndarray and elSingle.ndim == 0)
    assert elArray.shape == (10,)