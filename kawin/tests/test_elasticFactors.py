import itertools
import numpy as np
from numpy.testing import assert_allclose

from kawin.precipitation import StrainEnergy, ShapeFactor
from kawin.precipitation.parameters.ElasticFactors import convert2To4rankTensor, convert4To2rankTensor, invert4rankTensor, convertVecTo2rankTensor, convert2rankToVec, rotateRank2Tensor, rotateRank4Tensor

def test_tensorConversions():
    c2 = np.array([
        [1, 7, 8, 0, 0, 0],
        [7, 2, 9, 0, 0, 0],
        [8, 9, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 6],
        ])
    
    c4 = convert2To4rankTensor(c2)
    assert_allclose([c4[0,0,0,0]], [1], rtol=1e-3)
    assert_allclose([c4[1,1,1,1]], [2], rtol=1e-3)
    assert_allclose([c4[2,2,2,2]], [3], rtol=1e-3)
    assert_allclose([c4[0,0,1,1], c4[1,1,0,0]], [7, 7], rtol=1e-3)
    assert_allclose([c4[0,0,2,2], c4[2,2,0,0]], [8, 8], rtol=1e-3)
    assert_allclose([c4[2,2,1,1], c4[1,1,2,2]], [9, 9], rtol=1e-3)
    assert_allclose([c4[0,2,0,2], c4[0,2,2,0], c4[2,0,0,2], c4[2,0,2,0]], [5, 5, 5, 5], rtol=1e-3)
    assert_allclose([c4[0,1,0,1], c4[0,1,1,0], c4[1,0,0,1], c4[1,0,1,0]], [6, 6, 6, 6], rtol=1e-3)
    assert_allclose([c4[1,2,1,2], c4[1,2,2,1], c4[2,1,1,2], c4[2,1,2,1]], [4, 4, 4, 4], rtol=1e-3)

    c2_back = convert4To2rankTensor(c4)
    assert_allclose(c2, c2_back, rtol=1e-3)

    vec = np.array([1, 2, 3, 4, 5, 6])
    c2 = convertVecTo2rankTensor(vec)
    assert_allclose([c2[0,0]], [1], rtol=1e-3)
    assert_allclose([c2[1,1]], [2], rtol=1e-3)
    assert_allclose([c2[2,2]], [3], rtol=1e-3)
    assert_allclose([c2[0,1], c2[1,0]], [6, 6], rtol=1e-3)
    assert_allclose([c2[0,2], c2[2,0]], [5, 5], rtol=1e-3)
    assert_allclose([c2[1,2], c2[1,2]], [4, 4], rtol=1e-3)

    vec_back = convert2rankToVec(c2)
    assert_allclose(vec, vec_back, rtol=1e-3)

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

def test_StrainSphere():
    se = StrainEnergy('sphere')
    se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)
    se.setEigenstrain(0.01)
    rSph = 4e-9
    E = se.compute([rSph, rSph, rSph])
    assert_allclose(E, 5.375748e-18, rtol=1e-3)

def test_StrainCube():
    se = StrainEnergy('cube')
    se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)
    se.setEigenstrain(0.01)
    rSph = 4e-9
    E = se.compute([rSph, rSph, rSph])
    assert_allclose(E, 3.316448e-18, rtol=1e-3)

def test_StrainConstant():
    se = StrainEnergy()
    se.setConstantElasticEnergy(2e7)
    rSph = 4e-9
    E = se.compute([rSph, rSph, rSph])
    assert_allclose(E, 5.361651e-18, rtol=1e-3)

def test_StrainEllipse():
    '''
    Test strain energy calculation of arbitrary system

    Parameters are taken from 11_Extra_Factors for the Cu-Ti system
    '''
    se = StrainEnergy()
    se.setEllipsoidal()
    se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)
    se.setEigenstrain([0.022, 0.022, 0.003])

    aspect = 1.5
    rSph = 4e-9 / np.cbrt(aspect)
    r = np.array([rSph, rSph, aspect*rSph])
    E = se.compute(r)

    assert_allclose(E, 1.22956765e-17, rtol=1e-3)

    # test the other strain energy methods to make sure they compute the same
    E4 = se.description.strainEnergyEllipsoid(r)
    E4stress = se.description.strainEnergyEllipsoidWithStress(r)
    E2 = se.description.strainEnergyEllipsoid2ndRank(r)
    E4bohm = se.description.strainEnergyBohm(r)
    E2bohm = se.description.strainEnergyBohm2ndRank(r)

    assert_allclose([E4, E4stress, E2, E4bohm, E2bohm], 1.22956765e-17*np.ones(5), rtol=1e-3)

    # Since elastic constants are isotropic, we can test special case for ohm (cubic symmetry)
    n = se.description._n(0.2, 0.2)
    ohm = se.description._OhmGeneral(n, se.description.params.cMatrix_4th)
    ohmCubic = se.description._OhmCubic(n, se.description.params.cMatrix_4th)
    assert_allclose(ohm, ohmCubic, rtol=1e-3)

def test_StrainEllipseDifferentPrecConstants():
    '''
    Test strain energy calculation for precipitate having different elastic constants

    I think the elastic constants and eigenstrain shouldn't matter in this case and the ratio is determined by
    the nu and E/Eprec, but I'll need to look into this more
    '''
    se = StrainEnergy('ellipsoid')
    se.setModuli(E=168.4e9, nu=0.3)
    se.setEigenstrain(0.01)

    sePrec = StrainEnergy('ellipsoid')
    sePrec.setModuli(E=168.4e9, nu=0.3)
    sePrec.setModuliPrecipitate(E=3*168.4e9, nu=0.3)
    sePrec.setEigenstrain(0.01)

    aspect = 2
    rSph = 4e-9 / np.cbrt(aspect)
    r = np.array([rSph, rSph, aspect*rSph])
    energy = se.compute(r)
    energyPrec = sePrec.compute(r)

    assert_allclose(energyPrec/energy, 1.414644, rtol=1e-3)

def test_AspectRatioFromStrainEnergy():
    '''
    Test eq aspect ratio calculation of arbitrary system

    Parameters are taken from 11_Extra_Factors for the IN718 system
    '''
    se = StrainEnergy()
    se.setEigenstrain([6.67e-3, 6.67e-3, 2.86e-2])
    se.setModuli(G=57.1e9, nu=0.33)
    se.setShape('ellipsoid')

    sf = ShapeFactor('plate')

    gamma = 0.02375
    Rsph = np.array([5e-10])
    eqAR = se.eqAR_bySearch(Rsph, gamma, sf)
    R = 2*Rsph*eqAR / np.cbrt(eqAR**2)

    assert_allclose(R, [1.13444719e-9], rtol=1e-3)
    assert_allclose(eqAR, [1.46], rtol=1e-3)

    eqAR = se.eqAR_byGR(Rsph, gamma, sf)
    R = 2*Rsph*eqAR / np.cbrt(eqAR**2)
    assert_allclose(R, [1.13396e-9], rtol=1e-3)
    assert_allclose(eqAR, [1.4581], rtol=1e-3)

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