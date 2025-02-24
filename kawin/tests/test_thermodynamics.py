import numpy as np
from numpy.testing import assert_allclose

from pycalphad import Database

from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.tests.databases import *

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

def test_load_database_without_mobility():
    therm_nomob = BinaryThermodynamics(ALZR_TDB_NO_MOB, ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'])

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
    outputs = {
        'sampling': (6345.930428259977, 
                     0.25, 
                     np.array([6345.93042826, 6612.95334136], dtype=np.float64), 
                     np.array([0.25, 0.25], dtype=np.float64)),
        'approximate': (6345.930428259977, 
                        0.25, 
                        np.array([6345.93042826, 6612.95334136], dtype=np.float64), 
                        np.array([0.25, 0.25], dtype=np.float64)),
        'curvature': (107614.58901380893, 
                      0.25, 
                      np.array([107614.58901381, 118502.7444224], dtype=np.float64), 
                      np.array([0.25, 0.25], dtype=np.float64)),
        'tangent': (6345.930428259977, 
                    0.25, 
                    np.array([6345.93042826, 6612.95334136], dtype=np.float64), 
                    np.array([0.25, 0.25], dtype=np.float64)),
    }
    for m in methods:
        AlZrTherm.setDrivingForceMethod(m)
        dg, xP = AlZrTherm.getDrivingForce(0.004, 673.15, removeCache = True)
        dgarray, xParray = AlZrTherm.getDrivingForce([0.004, 0.005], [673.15, 683.15], removeCache = True)

        assert_allclose(dg, outputs[m][0], atol=0, rtol=1e-3)
        assert_allclose(xP, outputs[m][1], atol=0, rtol=1e-3)
        assert_allclose(dgarray, outputs[m][2], atol=0, rtol=1e-3)
        assert_allclose(xParray, outputs[m][3], atol=0, rtol=1e-3)

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
    outputs = {
        'sampling': (149.3270234841475, 
                     np.array([0.05152576, 0.19847424], dtype=np.float64), 
                     np.array([149.32702348, 177.46915929, 201.31219334], dtype=np.float64), 
                     np.array([[0.05152576, 0.19847424], [0.05215108, 0.19784892], [0.05265133, 0.19734867]], dtype=np.float64)),
        'approximate': (244.01202650912455, 
                        np.array([0.06160901, 0.17315461], dtype=np.float64), 
                        np.array([244.01202651, 266.01999011, 284.72675077], dtype=np.float64), 
                        np.array([[0.06160901, 0.17315461], [0.06381704, 0.17185646], [0.06596566, 0.17071728]], dtype=np.float64)),
        'curvature': (260.3790192099628, 
                      np.array([0.06160901, 0.17315461], dtype=np.float64), 
                      np.array([260.37901921, 286.87854217, 310.28588738], dtype=np.float64), 
                      np.array([[0.06160901, 0.17315461], [0.06381704, 0.17185646], [0.06596566, 0.17071728]], dtype=np.float64)),
        'tangent': (265.7790871298183, 
                    np.array([0.05474014, 0.18714878], dtype=np.float64), 
                    np.array([265.77908713, 292.39934163, 315.57844117], dtype=np.float64), 
                    np.array([[0.05474014, 0.18714878], [0.05580252, 0.18724615], [0.05681591, 0.18734315]], dtype=np.float64)),
    }
    methods = ['sampling', 'approximate', 'curvature', 'tangent']
    for m in methods:
        NiCrAlTherm.setDrivingForceMethod(m)
        dg, xP = NiCrAlTherm.getDrivingForce([0.08, 0.1], 1073.15, removeCache = True)
        dgarray, xParray = NiCrAlTherm.getDrivingForce([[0.08, 0.1], [0.085, 0.1], [0.09, 0.1]], [1073.15, 1078.15, 1083.15], removeCache = True)
        
        assert_allclose(dg, outputs[m][0], atol=0, rtol=1e-3)
        assert_allclose(xP, outputs[m][1], atol=0, rtol=1e-3)
        assert_allclose(dgarray, outputs[m][2], atol=0, rtol=1e-3)
        assert_allclose(xParray, outputs[m][3], atol=0, rtol=1e-3)

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
    outputs = {
        'curvature': (0.00023491993105882053,
                      0.25,
                      np.array([0.00023492, 0.00214397], dtype=np.float64),
                      np.array([0.25, 0.25], dtype=np.float64),
                      np.array([0.00023492, 0.00188604], dtype=np.float64),
                      np.array([0.25, 0.25], dtype=np.float64)),
        'equilibrium': (0.0016991452009952736,
                      0.25,
                      np.array([0.00169915, -1], dtype=np.float64),
                      np.array([0.25, -1], dtype=np.float64),
                      np.array([0.00169915, -1], dtype=np.float64),
                      np.array([0.25, -1], dtype=np.float64))
    }
    for m in methods:
        AlZrTherm.setInterfacialMethod(m)
        xm, xp = AlZrTherm.getInterfacialComposition(673.15, 5000)
        xmarray, xparray = AlZrTherm.getInterfacialComposition([673.15, 683.15], [5000, 50000])
        xmarray2, xparray2 = AlZrTherm.getInterfacialComposition(673.15, [5000, 50000])

        assert_allclose(xm, outputs[m][0], atol=0, rtol=1e-3)
        assert_allclose(xp, outputs[m][1], atol=0, rtol=1e-3)
        assert_allclose(xmarray, outputs[m][2], atol=0, rtol=1e-3)
        assert_allclose(xparray, outputs[m][3], atol=0, rtol=1e-3)
        assert_allclose(xmarray2, outputs[m][4], atol=0, rtol=1e-3)
        assert_allclose(xparray2, outputs[m][5], atol=0, rtol=1e-3)

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
    assert_allclose(dnkj, 1.2803441194011191e-20, atol=0, rtol=1e-3)
    assert_allclose(dnkjarray, np.array([1.28034412e-20, 2.41102630e-20], dtype=np.float64), atol=0, rtol=1e-3)

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
    dnkj_desired = np.array([[8.23950865e-18, 4.43371333e-18], [2.33938485e-17, 5.04911642e-17]], dtype=np.float64)
    dnkjarray_desired = np.array([[[8.23950865e-18, 4.43371333e-18], [2.33938485e-17, 5.04911642e-17]],
                                  [[9.88803167e-18, 5.48849162e-18], [2.70880749e-17, 5.90341424e-17]], 
                                  [[1.18344487e-17, 6.75930949e-18], [3.13266687e-17, 6.89291270e-17]]], dtype=np.float64)
    assert_allclose(dnkj, dnkj_desired, atol=0, rtol=1e-3)
    assert_allclose(dnkjarray, dnkjarray_desired, atol=0, rtol=1e-3)

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

    assert_allclose(n1, np.array([-8.01953749e-05,  6.86043210e-05], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(d1, 1.6188271145399225e-20, atol=0, rtol=1e-3)
    assert_allclose(g1, np.array([[0.19083364, -0.60826741], [ 0.31993136, 1.59959791]], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(b1, 1.3680751771342657e-16, atol=0, rtol=1e-3)
    assert_allclose(ca1, np.array([0.08275715, 0.08903276], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(cb1, np.array([0.06160901, 0.17315461], dtype=np.float64), atol=0, rtol=1e-3)

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

def test_Curv_in_single_phase():
    '''
    Checks that curvature factors returns None in a single phase region
    But, if a search direction is supplied, then a two-phase region can be found

    We will not check the values here since the search direction is mainly to get
    an approximation for the curvature factor when its undefined and to get a
    negative growth rate for the precipitates
    '''
    NiCrAlTherm.setDrivingForceMethod('tangent')
    x0 = [0.01, 0.01]
    T = 1073.15
    curv_factors_invalid = NiCrAlTherm.curvatureFactor(x0, T, removeCache = True)

    dg, xb = NiCrAlTherm.getDrivingForce(x0, T, removeCache=True)
    curv_factors = NiCrAlTherm.curvatureFactor(x0, T, removeCache=True, searchDir=xb)

    assert curv_factors_invalid is None
    assert curv_factors is not None

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
    assert_allclose(ca1, np.array([0.07198046, 0.10686043], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(cb1, np.array([0.04870846, 0.19822392], dtype=np.float64), atol=0, rtol=1e-3)

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

    assert_allclose(g, -1.6188271145399285e-09, atol=0, rtol=1e-3)
    assert_allclose(ca, np.array([0.07198046, 0.10686043], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(cb, np.array([0.04870846, 0.19822392], dtype=np.float64), atol=0, rtol=1e-3)

    assert_allclose(garray, np.array([-3.56141965e-08, -1.61882711e-09, 3.23765423e-09], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(caarray, np.array([[0, 0.17546475],
                                       [0.07198046, 0.10686043],
                                       [0.11207815, 0.07255827]], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(cbarray, np.array([[0, 0.28230623],
                                       [0.04870846, 0.19822392],
                                       [0.07722533, 0.15618276]], dtype=np.float64), atol=0, rtol=1e-3)

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

    assert_allclose(td, np.array([8.03946597e-18, 5.46554241e-18, 1.52099350e-17], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(tdarray, np.array([[8.03946597e-18, 5.46554241e-18, 1.52099350e-17],
                                       [9.33087557e-18, 6.51277012e-18, 1.78317544e-17]], dtype=np.float64), atol=0, rtol=1e-3)

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

    assert_allclose(dnkj, np.array([[3.09930669e-08, 0.00000000e+00], [0.00000000e+00, 1.95822648e-08]], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(dnkjarray, np.array([[[3.09930669e-08, 0.00000000e+00], [0.00000000e+00, 1.95822648e-08]],
                                         [[3.16573173e-08, 0.00000000e+00], [0.00000000e+00, 2.03078351e-08]],
                                         [[3.23294741e-08, 0.00000000e+00], [0.00000000e+00, 2.10532167e-08]]], dtype=np.float64), atol=0, rtol=1e-3)

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

    assert_allclose(td, np.array([8.03946597e-18, 5.46554241e-18, 1.52099350e-17], dtype=np.float64), atol=0, rtol=1e-3)
    assert_allclose(tdarray, np.array([[8.03946597e-18, 5.46554241e-18, 1.52099350e-17],
                                       [9.33087557e-18, 6.51277012e-18, 1.78317544e-17]], dtype=np.float64), atol=0, rtol=1e-3)

