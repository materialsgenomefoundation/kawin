from numpy.testing import assert_allclose
import numpy as np
import os
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics, BinarySurrogate, MulticomponentSurrogate
from kawin.tests.datasets import *

AlZrTherm = BinaryThermodynamics(ALZR_TDB, ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='approximate')
NiCrAlTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='approximate')

#Set constant sampling densities for each Thermodynamics object
#pycalphad equilibrium results may change based off sampling density, so this is to make sure
#tests won't failed unneccesarily because the default sampling densities are modified
AlZrTherm.setDFSamplingDensity(2000)
AlZrTherm.setEQSamplingDensity(500)
NiCrAlTherm.setDFSamplingDensity(2000)
NiCrAlTherm.setEQSamplingDensity(500)

def test_Surr_binary_DG_output():
    '''
    Tests output of binary surrogate driving force function
    This should give the same response as corresponding functions
    in Thermodynamics

    Ex. for f(x, T) -> (dg, xP)
        (scalar, scalar) input -> scalar
        (array, array) input -> array
    '''
    surr = BinarySurrogate(AlZrTherm)
    T = 673.15
    xtrain = np.logspace(-5, -2, 5)
    surr.trainDrivingForce(xtrain, T, scale='log')

    dg, xP = surr.getDrivingForce(xtrain[3], 673.15)
    dgT, xPT = AlZrTherm.getDrivingForce(xtrain[3], 673.15, training = True)
    dgarray, xParray = surr.getDrivingForce([0.004, 0.005], [673.15, 683.15])

    assert np.isscalar(dg) or (type(dg) == np.ndarray and dg.ndim == 0)
    assert np.isscalar(xP) or (type(xP) == np.ndarray and xP.ndim == 0)
    assert hasattr(dgarray, '__len__') and len(dgarray) == 2
    assert hasattr(xParray, '__len__') and len(xParray) == 2

    #Compare to Thermodynamics, high tolerance since we're just checking that functions are interchangeable
    assert_allclose(dg, dgT, atol=0, rtol=1e-1)
    assert_allclose(xP, xPT, atol=0, rtol=1e-1)

def test_Surr_binary_IC_output():
    '''
    Tests output of binary surrogate composition function
    This should give the same response as corresponding functions
    in Thermodynamics

    Ex. for f(T, g) -> (xM, xP)
        (scalar, scalar) -> (scalar, scalar)
        (array, array) -> (array, array)
        (scalar, array) -> (array, array)   Special case where T is scalar
    '''
    surr = BinarySurrogate(AlZrTherm)
    T = 673.15

    gExtra = np.linspace(100, 10000, 5)
    surr.trainInterfacialComposition(T, gExtra)

    xm, xp = surr.getInterfacialComposition(673.15, gExtra[3])
    xmT, xpT = AlZrTherm.getInterfacialComposition(673.15, gExtra[3])
    xmarray, xparray = surr.getInterfacialComposition([673.15, 683.15], [5000, 10000])
    xmarray2, xparray2 = surr.getInterfacialComposition(673.15, [5000, 10000])

    assert np.isscalar(xm) or (type(xm) == np.ndarray and xm.ndim == 0)
    assert np.isscalar(xp) or (type(xp) == np.ndarray and xp.ndim == 0)
    assert hasattr(xmarray, '__len__') and len(xmarray) == 2
    assert hasattr(xparray, '__len__') and len(xparray) == 2
    assert hasattr(xmarray2, '__len__') and len(xmarray2) == 2
    assert hasattr(xparray2, '__len__') and len(xparray2) == 2

    #Compare to Thermodynamics, high tolerance since we're just checking that functions are interchangeable
    assert_allclose(xm, xmT, atol=0, rtol=1e-1)
    assert_allclose(xp, xpT, atol=0, rtol=1e-1)


def test_Surr_binary_Diff_output():
    '''
    Tests output of binary surrogate diffusivity function
    This should give the same response as corresponding functions
    in Thermodynamics

    Ex. for diffusivity
        f(x, T) = diff
        (scalar, scalar) -> scalar
        (array, array) -> array
    '''
    surr = BinarySurrogate(AlZrTherm)
    T = 673.15
    xtrain = np.logspace(-5, -2, 5)

    surr.trainInterdiffusivity(xtrain, [T, T + 100])

    dnkj = surr.getInterdiffusivity(xtrain[3], 673.15)
    dnkjT = AlZrTherm.getInterdiffusivity(xtrain[3], 673.15)
    dnkjarray = surr.getInterdiffusivity([0.004, 0.005], [673.15, 683.15])
    assert np.isscalar(dnkj) or (type(dnkj) == np.ndarray and dnkj.ndim == 0)
    assert hasattr(dnkjarray, '__len__') and len(dnkjarray) == 2

    #Compare to Thermodynamics, high tolerance since we're just checking that functions are interchangeable
    assert_allclose(dnkj, dnkjT, atol=0, rtol=1e-1)

def test_Surr_binary_save():
    '''
    Checks that binary surrogate can be saved and loaded to get same values
    '''
    surr = BinarySurrogate(AlZrTherm)
    T = 673.15
    xtrain = np.logspace(-5, -2, 5)
    surr.trainDrivingForce(xtrain, T, scale='log')

    gExtra = np.linspace(100, 1000, 5)
    surr.trainInterfacialComposition(T, gExtra)

    surr.trainInterdiffusivity(xtrain, [T, T + 100])

    a, b = surr.getDrivingForce(0.004, T)
    c, d = surr.getInterfacialComposition(T, 500)
    e = surr.getInterdiffusivity(0.1, T + 50)

    surr.save('kawin/tests/alzr')

    surr2 = BinarySurrogate.load('kawin/tests/alzr')
    a2, b2 = surr2.getDrivingForce(0.004, T)
    c2, d2 = surr2.getInterfacialComposition(T, 500)
    e2 = surr2.getInterdiffusivity(0.1, T + 50)

    os.remove('kawin/tests/alzr')

    assert_allclose([a, b, c, d, e], [a2, b2, c2, d2, e2], rtol=1e-3)

def test_Surr_binary_save_missing():
    '''
    Checks that load function will not fail if one of the three surrogates are not trained yet
    '''
    surr = BinarySurrogate(AlZrTherm)
    T = 673.15
    xtrain = np.logspace(-5, -2, 5)
    surr.trainDrivingForce(xtrain, T, scale='log')

    a, b = surr.getDrivingForce(0.004, T)

    surr.save('kawin/tests/alzr')

    surr2 = BinarySurrogate.load('kawin/tests/alzr')
    a2, b2 = surr2.getDrivingForce(0.004, T)

    os.remove('kawin/tests/alzr')

    assert_allclose(a, a2, atol=0, rtol=1e-3)

def test_Surr_ternary_DG_output():
    '''
    Tests output of multicomponent surrogate driving force function
    This should give the same response as corresponding functions
    in Thermodynamics

    Ex. for f(x, T) -> (dg, xP)
        (array, scalar) -> scalar
        (2D array, array) -> array
    '''
    surr = MulticomponentSurrogate(NiCrAlTherm)
    T = [1073.15, 1123.15]
    x = [[0.06, 0.08], [0.06, 0.1], [0.06, 0.12], [0.08, 0.08], [0.08, 0.1], [0.08, 0.12], [0.1, 0.08], [0.1, 0.1], [0.1, 0.12]]
    surr.trainDrivingForce(x, T)

    dg, xP = surr.getDrivingForce(x[5], 1073.15)
    dgT, xPT = NiCrAlTherm.getDrivingForce(x[5], 1073.15, training = True)
    dgarray, xParray = surr.getDrivingForce([[0.08, 0.1], [0.085, 0.1], [0.09, 0.1]], [1073.15, 1078.15, 1083.15])

    assert np.isscalar(dg) or (type(dg) == np.ndarray and dg.ndim == 0)
    assert xP.ndim == 1 and len(xP) == 2
    assert hasattr(dgarray, '__len__')
    assert xParray.shape == (3, 2)

    #Compare to Thermodynamics, high tolerance since we're just checking that functions are interchangeable
    assert_allclose(dg, dgT, atol=0, rtol=1e-1)
    assert_allclose(xP, xPT, atol=0, rtol=1e-1)

def test_Surr_ternary_IC_output():
    '''
    Tests output of multicomponent surrogate interfacial composition function
    This should give the same response as corresponding functions
    in Thermodynamics

    Ex. f(x, T, dG, R, gE) -> (gr, xM, xP, xM_EQ, xP_EQ)
        (array, scalar, scalar, scalar, scalar) -> (scalar, array, array, array, array)
        (array, scalar, scalar, array, array) -> (array, 2D array, 2D array, array, array)
    '''
    surr = MulticomponentSurrogate(NiCrAlTherm)
    T = [1073.15, 1123.15]
    x = [[0.06, 0.08], [0.06, 0.1], [0.06, 0.12], [0.08, 0.08], [0.08, 0.1], [0.08, 0.12], [0.1, 0.08], [0.1, 0.1], [0.1, 0.12]]
    surr.trainCurvature(x, T)

    g, ca, cb, caEQ, cbEQ = surr.getGrowthAndInterfacialComposition(x[5], 1073.15, 900, 1e-9, 1000)
    gT, caT, cbT, _, _ = NiCrAlTherm.getGrowthAndInterfacialComposition(x[5], 1073.15, 900, 1e-9, 1000, training = True)
    garray, caarray, cbarray, caEQarray, cbEQarray = surr.getGrowthAndInterfacialComposition([0.08, 0.1], 1073.15, 900, [0.5e-9, 1e-9, 2e-9], [2000, 1000, 500])

    assert np.isscalar(g) or (type(g) == np.ndarray and g.ndim == 0)
    assert hasattr(ca, '__len__') and len(ca) == 2
    assert hasattr(cb, '__len__') and len(cb) == 2
    assert hasattr(caEQ, '__len__') and len(caEQ) == 2
    assert hasattr(cbEQ, '__len__') and len(cbEQ) == 2
    assert hasattr(garray, '__len__') and len(garray) == 3
    assert caarray.shape == (3, 2)
    assert cbarray.shape == (3, 2)
    assert hasattr(caEQarray, '__len__') and len(caEQarray) == 2
    assert hasattr(caEQarray, '__len__') and len(caEQarray) == 2

    #Compare to Thermodynamics, high tolerance since we're just checking that functions are interchangeable
    assert_allclose(g, gT, atol=0, rtol=1e-1)
    assert_allclose(ca, caT, atol=0, rtol=1e-1)
    assert_allclose(cb, cbT, atol=0, rtol=1e-1)

def test_Surr_ternary_save():
    '''
    Checks that multicomponent surrogate can be saved and loaded
    '''
    surr = MulticomponentSurrogate(NiCrAlTherm)
    T = [1073.15, 1123.15]
    x = [[0.06, 0.08], [0.06, 0.1], [0.06, 0.12], [0.08, 0.08], [0.08, 0.1], [0.08, 0.12], [0.1, 0.08], [0.1, 0.1], [0.1, 0.12]]
    surr.trainDrivingForce(x, T)

    surr.trainCurvature(x, T)

    a, b = surr.getDrivingForce([0.08, 0.1], T[0]+25)
    g, ca, cb, _, _ = surr.getGrowthAndInterfacialComposition([0.08, 0.1], T[0]+25, 900, 1e-9, 1000)
    beta = surr.impingementFactor([0.08, 0.1], T[0]+25)

    surr.save('kawin/tests/nicral')

    surr2 = MulticomponentSurrogate.load('kawin/tests/nicral')
    a2, b2 = surr2.getDrivingForce([0.08, 0.1], T[0]+25)
    g2, ca2, cb2, _, _ = surr2.getGrowthAndInterfacialComposition([0.08, 0.1], T[0]+25, 900, 1e-9, 1000)
    beta2 = surr2.impingementFactor([0.08, 0.1], T[0]+25)

    os.remove('kawin/tests/nicral')

    assert_allclose([a, b[0], b[1], g, ca[0], ca[1], cb[0], cb[1], beta], [a2, b2[0], b2[1], g2, ca2[0], ca2[1], cb2[0], cb2[1], beta2], atol=0, rtol=1e-3)

def test_Surr_ternary_save_missing():
    '''
    Checks that load function will not fail if one of the three surrogates are not trained yet
    '''
    surr = MulticomponentSurrogate(NiCrAlTherm)
    T = [1073.15, 1123.15]
    x = [[0.06, 0.08], [0.06, 0.1], [0.06, 0.12], [0.08, 0.08], [0.08, 0.1], [0.08, 0.12], [0.1, 0.08], [0.1, 0.1], [0.1, 0.12]]
    surr.trainDrivingForce(x, T)

    a, b = surr.getDrivingForce([0.08, 0.1], T[0]+25)
    surr.save('kawin/tests/nicral')

    surr2 = MulticomponentSurrogate.load('kawin/tests/nicral')
    a2, b2 = surr2.getDrivingForce([0.08, 0.1], T[0]+25)
    os.remove('kawin/tests/nicral')

    assert_allclose([a, b[0], b[1]], [a2, b2[0], b2[1]], atol=0, rtol=1e-3)
