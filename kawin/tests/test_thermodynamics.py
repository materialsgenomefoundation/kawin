import numpy as np
from numpy.testing import assert_allclose
from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.tests.datasets import *
from pycalphad import Database

#Default driving force method will be 'tangent'
AlZrTherm = BinaryThermodynamics(ALZR_TDB, ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='tangent')
NiCrAlTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')
NiCrAlThermDiff = MulticomponentThermodynamics(NICRAL_TDB_DIFF, ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')
NiAlCrTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'AL', 'CR'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')
NiAlCrThermDiff = MulticomponentThermodynamics(NICRAL_TDB_DIFF, ['NI', 'AL', 'CR'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')
AlCrNiTherm = MulticomponentThermodynamics(NICRAL_TDB, ['AL', 'CR', 'NI'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')

#Set constant sampling densities for each Thermodynamics object
#pycalphad equilibrium results may change based off sampling density, so this is to make sure
#tests won't failed unneccesarily because the default sampling densities are modified
AlZrTherm.setDFSamplingDensity(2000)
AlZrTherm.setEQSamplingDensity(500)
NiCrAlTherm.setDFSamplingDensity(2000)
NiCrAlTherm.setEQSamplingDensity(500)
NiCrAlThermDiff.setDFSamplingDensity(2000)
NiCrAlThermDiff.setEQSamplingDensity(500)
NiAlCrTherm.setDFSamplingDensity(2000)
NiAlCrTherm.setEQSamplingDensity(500)
NiAlCrThermDiff.setDFSamplingDensity(2000)
NiAlCrThermDiff.setEQSamplingDensity(500)
AlCrNiTherm.setDFSamplingDensity(2000)
AlCrNiTherm.setEQSamplingDensity(500)

def test_DG_binary():
    '''
    Checks value of binary driving force calculation

    Driving force value was updated due to switch from approximate to tangent method
    '''
    dg, _ = AlZrTherm.getDrivingForce(0.004, 673.15, removeCache = True)
    assert_allclose(dg, 6346.930428, atol=0, rtol=1e-3)

def test_DG_binary_output():
    '''
    Checks that output of binary driving force calculation follows input
    Ex. for f(x, T) -> (dg, xP)
        (scalar, scalar) input -> scalar
        (array, array) input -> array
    '''
    methods = ['sampling', 'approximate', 'curvature', 'tangent']
    for m in methods:
        AlZrTherm.setDrivingForceMethod(m)
        dg, xP = AlZrTherm.getDrivingForce(0.004, 673.15, removeCache = True)
        dgarray, xParray = AlZrTherm.getDrivingForce([0.004, 0.005], [673.15, 683.15], removeCache = True)

        assert np.isscalar(dg) or (type(dg) == np.ndarray and dg.ndim == 0)
        assert np.isscalar(xP) or (type(xP) == np.ndarray and xP.ndim == 0)
        assert hasattr(dgarray, '__len__') and len(dgarray) == 2
        assert hasattr(xParray, '__len__') and len(xParray) == 2

    AlZrTherm.setDrivingForceMethod('tangent')

def test_DG_ternary():
    '''
    Checks value of ternary driving force calculation

    Driving force value was updated due to switch from approximate to tangent method
    '''
    dg, _ = NiCrAlTherm.getDrivingForce([0.08, 0.1], 1073.15, removeCache = True)
    assert_allclose(dg, 265.779087, atol=0, rtol=1e-3)

def test_DG_ternary_output():
    '''
    Checks that output of ternary driving force calculations follow input
    Ex. for f(x, T) -> (dg, xP)
        (array, scalar) -> scalar
        (2D array, array) -> array
    '''
    methods = ['sampling', 'approximate', 'curvature', 'tangent']
    for m in methods:
        NiCrAlTherm.setDrivingForceMethod(m)
        dg, xP = NiCrAlTherm.getDrivingForce([0.08, 0.1], 1073.15, removeCache = True)
        dgarray, xParray = NiCrAlTherm.getDrivingForce([[0.08, 0.1], [0.085, 0.1], [0.09, 0.1]], [1073.15, 1078.15, 1083.15], removeCache = True)
        assert np.isscalar(dg) or (type(dg) == np.ndarray and dg.ndim == 0)
        assert xP.ndim == 1 and len(xP) == 2
        assert hasattr(dgarray, '__len__')
        assert xParray.shape == (3, 2)

    NiCrAlTherm.setDrivingForceMethod('tangent')

def test_DG_ternary_order():
    '''
    Check that driving force is the same given that the order of input elements and composition are the same
    Ex. Input elements as [Ni, Cr, Al] should require composition to be [Cr, Al]
        Input elements as [Ni, Al, Cr] should require composition to be [Al, Cr]
        Input elements as [Al, Cr, Ni] should require composition to be [Cr, Ni]
    '''
    dg1, _ = NiCrAlTherm.getDrivingForce([0.08, 0.1], 1073.15, removeCache = True)
    dg2, _ = NiAlCrTherm.getDrivingForce([0.1, 0.08], 1073.15, removeCache = True)
    dg3, _ = AlCrNiTherm.getDrivingForce([0.08, 0.82], 1073.15, removeCache = True)
    assert_allclose(dg1, dg2, atol=0, rtol=1e-3)
    assert_allclose(dg2, dg3, atol=0, rtol=1e-3)

def test_IC_binary():
    '''
    Check value of interfacial composition for binary case
    '''
    xm, xp = AlZrTherm.getInterfacialComposition(673.15, 10000)
    assert_allclose(xm, 0.0233507, atol=0, rtol=1e-3)

def test_IC_unstable():
    '''
    Checks that (-1, -1) is returned for unstable precipitate
    '''
    xm, xp = AlZrTherm.getInterfacialComposition(673.15, 50000)
    assert(xm == -1 and xp == -1)

def test_IC_binary_output():
    '''
    Checks that output of interfacial composition follows input
    Ex. For f(T, g) -> (xM, xP)
        (scalar, scalar) -> (scalar, scalar)
        (array, array) -> (array, array)
        (scalar, array) -> (array, array)   Special case where T is scalar
    '''
    methods = ['curvature', 'equilibrium']
    for m in methods:
        AlZrTherm.setInterfacialMethod(m)
        xm, xp = AlZrTherm.getInterfacialComposition(673.15, 5000)
        xmarray, xparray = AlZrTherm.getInterfacialComposition([673.15, 683.15], [5000, 50000])
        xmarray2, xparray2 = AlZrTherm.getInterfacialComposition(673.15, [5000, 50000])

        assert np.isscalar(xm) or (type(xm) == np.ndarray and xm.ndim == 0)
        assert np.isscalar(xp) or (type(xp) == np.ndarray and xp.ndim == 0)
        assert hasattr(xmarray, '__len__') and len(xmarray) == 2
        assert hasattr(xparray, '__len__') and len(xparray) == 2
        assert hasattr(xmarray2, '__len__') and len(xmarray2) == 2
        assert hasattr(xparray2, '__len__') and len(xparray2) == 2

    AlZrTherm.setInterfacialMethod('equilibrium')

def test_Mob_binary():
    '''
    Checks value of binary interdiffusvity calculation
    '''
    dnkj = AlZrTherm.getInterdiffusivity(0.004, 673.15)
    assert_allclose(dnkj, 1.280344e-20, atol=0, rtol=1e-3)

def test_Mob_binary_output():
    '''
    Checks output of binary mobility follows input
    Ex. f(x, T) = diff
        (scalar, scalar) -> scalar
        (array, array) -> array
    '''
    dnkj = AlZrTherm.getInterdiffusivity(0.004, 673.15)
    dnkjarray = AlZrTherm.getInterdiffusivity([0.004, 0.005], [673.15, 683.15])
    assert np.isscalar(dnkj) or (type(dnkj) == np.ndarray and dnkj.ndim == 0)
    assert hasattr(dnkjarray, '__len__') and len(dnkjarray) == 2

def test_Mob_ternary():
    '''
    Checks value of ternary interdiffusivity calculation
    '''
    dnkj = NiCrAlTherm.getInterdiffusivity([0.08, 0.1], 1073.15)
    assert_allclose(dnkj, [[8.239509e-18, 4.433713e-18], [2.339385e-17, 5.049116e-17]], atol=0, rtol=1e-3)

def test_Mob_ternary_output():
    '''
    Checks output of multicomponent mobility follows input
    Ex. f(x, T) = diff
        (array, scalar) -> 2D array
        (2D array, array) -> 3D array
    '''
    dnkj = NiCrAlTherm.getInterdiffusivity([0.08, 0.1], 1073.15)
    dnkjarray = NiCrAlTherm.getInterdiffusivity([[0.08, 0.1], [0.085, 0.1], [0.09, 0.1]], [1073.15, 1078.15, 1083.15])

    assert dnkj.shape == (2, 2)
    assert dnkjarray.shape == (3, 2, 2)

def test_Mob_order():
    '''
    Test diffusivity matrix is given in correct order as input elements
    Ex. [Ni, Cr, Al] should give diffusivity matrix of [[D_CrCr, D_CrAl], [D_AlCr, D_AlAl]]
        and [Ni, Al, Cr] should give [[D_AlAl, D_AlCr], [D_CrAl, D_CrCr]]
    '''
    dnkj1 = NiCrAlTherm.getInterdiffusivity([0.08, 0.1], 1073.15)
    dnkj2 = NiAlCrTherm.getInterdiffusivity([0.1, 0.08], 1073.15)
    dnkj2[:,[0,1]] = dnkj2[:,[1,0]]
    dnkj2[[0,1],:] = dnkj2[[1,0],:]
    assert_allclose(dnkj1, dnkj2, atol=0, rtol=1e-3)

def test_Curv_ternary():
    '''
    Checks that order of elements does not matter for curvature calculations
    '''
    n1, d1, g1, b1, ca1, cb1 = NiCrAlTherm.curvatureFactor([0.08, 0.1], 1073.15, removeCache = True)
    n2, d2, g2, b2, ca2, cb2 = NiAlCrTherm.curvatureFactor([0.1, 0.08], 1073.15, removeCache = True)
    n3, d3, g3, b3, ca3, cb3 = AlCrNiTherm.curvatureFactor([0.08, 0.82], 1073.15, removeCache = True)

    n2[[0,1]] = n2[[1,0]]
    g2[[0,1],:] = g2[[1,0],:]
    g2[:,[0,1]] = g2[:,[1,0]]
    ca2[[0,1]] = ca2[[1,0]]
    cb2[[0,1]] = cb2[[1,0]]
    ca3change = [ca3[0], 1-ca3[0]-ca3[1]]
    cb3change = [cb3[0], 1-cb3[0]-cb3[1]]

    #Will only test d3, b3 and ca3,cb3
    #n3 and g3 cannot be directly compared to n1 and g1
    assert_allclose(n1, n2, atol=0, rtol=1e-3)
    assert_allclose(d1, d2, atol=0, rtol=1e-3)
    assert_allclose(d1, d3, atol=0, rtol=1e-3)
    assert_allclose(g1, g2, atol=0, rtol=1e-3)
    assert_allclose(b1, b2, atol=0, rtol=1e-3)
    assert_allclose(b1, b3, atol=0, rtol=1e-3)
    assert_allclose(ca1, ca2, atol=0, rtol=1e-3)
    assert_allclose(ca1, ca3change, atol=0, rtol=1e-3)
    assert_allclose(cb1, cb2, atol=0, rtol=1e-3)
    assert_allclose(cb1, cb3change, atol=0, rtol=1e-3)

def test_IC_ternary():
    '''
    Checks that order does not matter for growth and interfacial composition calculations
    Ignore equilibrium compositions since growth and interfacial compositions depend on them anyways
        If growth and interfacial compositions are correct, then equilibrium compositions are also correct
    '''
    growth1 = NiCrAlTherm.getGrowthAndInterfacialComposition([0.08, 0.1], 1073.15, 900, 1e-9, 1000, removeCache = True)
    growth2 = NiAlCrTherm.getGrowthAndInterfacialComposition([0.1, 0.08], 1073.15, 900, 1e-9, 1000, removeCache = True)
    growth3 = AlCrNiTherm.getGrowthAndInterfacialComposition([0.08, 0.82], 1073.15, 900, 1e-9, 1000, removeCache = True)
    #g3, ca3, cb3, _, _ = AlCrNiTherm.getGrowthAndInterfacialComposition([0.08, 0.82], 1073.15, 900, 1e-9, 1000, removeCache = True)

    g1, ca1, cb1 = growth1.growth_rate, growth1.c_alpha, growth1.c_beta
    g2, ca2, cb2 = growth2.growth_rate, growth2.c_alpha, growth2.c_beta
    g3, ca3, cb3 = growth3.growth_rate, growth3.c_alpha, growth3.c_beta

    #Change ca2,cb2 from [AL, CR] to [CR, AL]
    ca2[[0,1]] = ca2[[1,0]]
    cb2[[0,1]] = cb2[[1,0]]

    #Change ca3,cb3 from [CR, NI] to [CR, AL]
    ca3change = [ca3[0], 1-ca3[0]-ca3[1]]
    cb3change = [cb3[0], 1-cb3[0]-cb3[1]]

    assert_allclose(g1, -1.618827e-09, atol=0, rtol=1e-3)

    assert_allclose(g1, g2, atol=0, rtol=1e-3)
    assert_allclose(g1, g3, atol=0, rtol=1e-3)
    assert_allclose(ca1, ca2, atol=0, rtol=1e-3)
    assert_allclose(ca1, ca3change, atol=0, rtol=1e-3)
    assert_allclose(cb1, cb2, atol=0, rtol=1e-3)
    assert_allclose(cb1, cb3change, atol=0, rtol=1e-3)

def test_IC_ternary_output():
    '''
    Checks that output of IC follows input
    Ex. f(x, T, dG, R, gE) -> (gr, xM, xP)
        (array, scalar, scalar, scalar, scalar) -> (scalar, array, array)
        (array, scalar, scalar, array, array) -> (array, 2D array, 2D array)
    '''
    growth = NiCrAlTherm.getGrowthAndInterfacialComposition([0.08, 0.1], 1073.15, 900, 1e-9, 1000, removeCache = True)
    growth_array= NiCrAlTherm.getGrowthAndInterfacialComposition([0.08, 0.1], 1073.15, 900, [0.5e-9, 1e-9, 2e-9], [2000, 1000, 500], removeCache = True)

    g, ca, cb = growth.growth_rate, growth.c_alpha, growth.c_beta
    garray, caarray, cbarray = growth_array.growth_rate, growth_array.c_alpha, growth_array.c_beta

    assert np.isscalar(g) or (type(g) == np.ndarray and g.ndim == 0)
    assert hasattr(ca, '__len__') and len(ca) == 2
    assert hasattr(cb, '__len__') and len(cb) == 2
    assert hasattr(garray, '__len__') and len(garray) == 3
    assert caarray.shape == (3, 2)
    assert cbarray.shape == (3, 2)


def test_initialize_with_pycalphad_database():
    """
    Checks that a pycalphad Database object can be passed to the kawin.GeneralThermodynamics class.
    """
    GeneralThermodynamics(Database(ALZR_TDB), ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='approximate')

def test_initialize_with_single_phase():
    """
    Checks if a single phase was passed as a string instead of multiple phases passed as a list of strings.
    """
    GeneralThermodynamics(ALZR_TDB, ['AL', 'ZR'], 'FCC_A1')

def test_Mob_tracer_ternary():
    '''
    Checks value of ternary tracer diffusivity calculation
    '''
    td = NiCrAlTherm.getTracerDiffusivity([0.08, 0.1], 1073.15)
    assert_allclose(td, [8.039466e-18, 5.465542e-18, 1.520994e-17], rtol=1e-3)

def test_Mob_tracer_ternary_output():
    '''
    Checks output of multicomponent mobility follows input
    Ex. f(x, T) = diff
        (array, scalar) -> 2D array
        (2D array, array) -> 3D array
    '''
    td = NiCrAlTherm.getTracerDiffusivity([0.08, 0.1], 1073.15)
    tdarray = NiCrAlTherm.getTracerDiffusivity([[0.08, 0.1], [0.085, 0.1]], [1073.15, 1078.15])

    assert td.shape == (3,)
    assert tdarray.shape == (2, 3)

def test_Diff_ternary():
    '''
    Checks value of ternary interdiffusivity calculation
    '''
    dnkj = NiCrAlThermDiff.getInterdiffusivity([0.08, 0.1], 1073.15)
    assert_allclose(dnkj, [[3.099307e-8, 0], [0, 1.958226e-8]], atol=0, rtol=1e-3)

def test_Diff_ternary_output():
    '''
    Checks output of multicomponent mobility follows input
    Ex. f(x, T) = diff
        (array, scalar) -> 2D array
        (2D array, array) -> 3D array
    '''
    dnkj = NiCrAlThermDiff.getInterdiffusivity([0.08, 0.1], 1073.15)
    dnkjarray = NiCrAlThermDiff.getInterdiffusivity([[0.08, 0.1], [0.085, 0.1], [0.09, 0.1]], [1073.15, 1078.15, 1083.15])

    assert dnkj.shape == (2, 2)
    assert dnkjarray.shape == (3, 2, 2)

def test_Diff_order():
    '''
    Test diffusivity matrix is given in correct order as input elements
    Ex. [Ni, Cr, Al] should give diffusivity matrix of [[D_CrCr, D_CrAl], [D_AlCr, D_AlAl]]
        and [Ni, Al, Cr] should give [[D_AlAl, D_AlCr], [D_CrAl, D_CrCr]]
    '''
    dnkj1 = NiCrAlThermDiff.getInterdiffusivity([0.08, 0.1], 1073.15)
    dnkj2 = NiAlCrThermDiff.getInterdiffusivity([0.1, 0.08], 1073.15)
    dnkj2[:,[0,1]] = dnkj2[:,[1,0]]
    dnkj2[[0,1],:] = dnkj2[[1,0],:]
    assert_allclose(dnkj1, dnkj2, atol=0, rtol=1e-3)

def test_Diff_tracer_ternary():
    '''
    Checks value of ternary tracer diffusivity calculation
    '''
    td = NiCrAlThermDiff.getTracerDiffusivity([0.08, 0.1], 1073.15)
    assert_allclose(td, [7.357088e-18, 3.099307e-8, 1.958226e-8], atol=0, rtol=1e-3)

def test_Diff_tracer_ternary_output():
    '''
    Checks output of multicomponent mobility follows input
    Ex. f(x, T) = diff
        (array, scalar) -> 2D array
        (2D array, array) -> 3D array
    '''
    td = NiCrAlTherm.getTracerDiffusivity([0.08, 0.1], 1073.15)
    tdarray = NiCrAlTherm.getTracerDiffusivity([[0.08, 0.1], [0.085, 0.1]], [1073.15, 1078.15])

    assert td.shape == (3,)
    assert tdarray.shape == (2, 3)
