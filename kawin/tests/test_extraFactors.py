from numpy.testing import assert_allclose
import numpy as np

from kawin.precipitation import ShapeFactor
from kawin.precipitation import StrainEnergy
from kawin.precipitation.parameters.ElasticFactors import convert2To4rankTensor

import itertools

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

    elSingle = se.compute(rSingle)
    elArray = se.compute(rArray)

    assert np.isscalar(elSingle) or (type(elSingle) == np.ndarray and elSingle.ndim == 0)
    assert elArray.shape == (10,)

def test_StrainValues():
    '''
    Test strain energy calculation of arbitrary system

    Parameters are taken from 11_Extra_Factors for the Cu-Ti system
    '''
    se = StrainEnergy()
    se.setEllipsoidal()
    se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)
    se.setEigenstrain([0.022, 0.022, 0.003])
    #se.setup()

    aspect = 1.5
    rSph = 4e-9 / np.cbrt(aspect)
    r = np.array([rSph, rSph, aspect*rSph])
    E = se.compute(r)

    assert_allclose(E, 1.22956765e-17, rtol=1e-3)

def test_AspectRatioFromStrainEnergy():
    '''
    Test eq aspect ratio calculation of arbitrary system

    Parameters are taken from 11_Extra_Factors for the IN718 system
    '''
    se = StrainEnergy()
    se.setEigenstrain([6.67e-3, 6.67e-3, 2.86e-2])
    se.setModuli(G=57.1e9, nu=0.33)
    se.setEllipsoidal()
    #se.setup()

    sf = ShapeFactor()
    sf.setPlateShape()

    gamma = 0.02375
    Rsph = np.array([5e-10])
    eqAR = se.eqAR_bySearch(Rsph, gamma, sf)
    R = 2*Rsph*eqAR / np.cbrt(eqAR**2)

    assert_allclose(R, [1.13444719e-9], rtol=1e-3)
    assert_allclose(eqAR, [1.46], rtol=1e-3)

def test_different_strain_energy_inputs():
    '''
    Make sure the elastic tensor is the same for different types of inputs
    
    Following options in kawin are:
        2nd rank elastic tensor (6x6 matrix)
        Elastic constants c11, c12 and c44
        2 different moduli (from E, nu, G, lambda, K, or M)

    This will use the G and nu parameters from 11_Extra_Factors for the IN718 example
        Any values should work though since we're just checking that the elastic tensor is the same
    '''
    G = 57.1e9                      #Shear modulus
    nu = 0.33                       #Poisson ratio
    E = 2*G*(1+nu)                  #Elastic modulus
    lam = 2*G*nu / (1-2*nu)         #Lame's first parameter
    K = 2*G*(1+nu)/(3*(1-2*nu))     #Bulk modulus
    M = 2*G*(1-nu)/(1-2*nu)         #P-wave modulus

    c11 = E*(1-nu)/((1+nu)*(1-2*nu))
    c12 = E*nu/((1+nu)*(1-2*nu))
    c44 = G

    se = StrainEnergy()

    r2Tensor = np.array([[c11, c12, c12, 0, 0, 0], [c12, c11, c12, 0, 0, 0], [c12, c12, c11, 0, 0, 0], [0, 0, 0, c44, 0, 0], [0, 0, 0, 0, c44, 0], [0, 0, 0, 0, 0, c44]])
    r4Tensor = convert2To4rankTensor(r2Tensor)
    
    #Test 2nd rank tensor input
    se.setElasticTensor(r2Tensor)
    assert_allclose(se.params.cMatrix_4th, r4Tensor, rtol=1e-3)

    #Test elastic constants input
    se.setElasticConstants(c11, c12, c44)
    assert_allclose(se.params.cMatrix_4th, r4Tensor, rtol=1e-3)

    #This is in the order of the if statements in StrainEnergy._setModuli so it's easier to debug
    moduli = {'E': E, 'nu': nu, 'G': G, 'lam': lam, 'K': K, 'M': M}
    moduli_names = moduli.keys()

    #Test each pair of moduli as inputs
    for pair in itertools.combinations(moduli_names, 2):
        moduli_input = {m: moduli[m] for m in pair}
        se.setModuli(**moduli_input)
        assert_allclose(se.params.cMatrix_4th, r4Tensor, rtol=1e-3)






