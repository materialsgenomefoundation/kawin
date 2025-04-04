import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from kawin.diffusion import SinglePhaseModel, HomogenizationModel, TemperatureParameters
from kawin.diffusion.mesh import Cartesian1D, Cylindrical1D, Spherical1D, Cartesian2D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh import ProfileBuilder, StepProfile1D, LinearProfile1D, DiracDeltaProfile, ConstantProfile, GaussianProfile, ExperimentalProfile1D, BoundedEllipseProfile, BoundedRectangleProfile
from kawin.diffusion.DiffusionParameters import computeMobility, TemperatureParameters
from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
from kawin.thermo import GeneralThermodynamics
from kawin.tests.datasets import *

N = 100
NiCrTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
NiCrAlTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])
FeCrNiTherm = GeneralThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])

def test_compositionInput():
    '''
    Tests that after setting up a model, all components are greater than 0
    
    In practice, this greatly speeds up simulation time without sacrificing accuracy since it avoids 
    performing equilibrium calculations with a composition of 0 for any given component

    The composition and setup functions for both models inherit from the
    base diffusion model class, so any model can be used here
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(StepProfile1D(0, [0.2, 0.8], [1, 0]), ['CR', 'AL'])
    
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    model = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1200+273.15))
    model.setup()

    assert(mesh.y[25,0] + mesh.y[25,1] < 1)
    assert(mesh.y[75,0] + mesh.y[75,1] < 1)
    assert(1 - (mesh.y[75,0] + mesh.y[75,1]) >= model.constraints.minComposition)
    assert(1 - (mesh.y[25,0] + mesh.y[25,1]) >= model.constraints.minComposition)
    assert(mesh.y[75,1] >= model.constraints.minComposition)

def test_singlePhaseFluxes_shape():
    '''
    Tests the dimensions of the single phase fluxes function

    Should be (E-1, N+1) where E is the number of elements and
    N is the number of points
    '''
    profile = ProfileBuilder([(StepProfile1D(0, 0.2, 0.8), 'CR')])
    mesh = Cartesian1D(['CR'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    model = SinglePhaseModel(mesh, ['NI', 'CR'], ['FCC_A1'], NiCrTherm, TemperatureParameters(1073))
    model.setup()
    f = model.getdXdt(model.currentTime, model.getCurrentX())
    assert(f[0].shape == (N,1))

    profile = ProfileBuilder([(StepProfile1D(0, 0.2, 0.4), 'CR'), (StepProfile1D(0, 0.4, 0.4), 'AL')])
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    model = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1073))
    model.setup()
    f = model.getdXdt(model.currentTime, model.getCurrentX())
    assert(f[0].shape == (N,2))

def test_homogenizationMobility():
    '''
    Tests the dimensions of the homogenization mobility function

    Should be (P, E, N) where P is the number of phases,
    E is the number of elements and
    N is the number of points
    '''
    N = 10

    homogenizationParameters = HomogenizationParameters()
    x = np.linspace(0.2, 0.3, N)
    T = 1073*np.ones(N)
    mobBinary, _ = computeHomogenizationFunction(NiCrTherm, x, T, homogenizationParameters)
    assert(mobBinary.shape == (N,2))

    homogenizationParameters = HomogenizationParameters()
    x_cr = np.linspace(0.2, 0.3, N)
    x_al = np.linspace(0.3, 0.2, N)
    x = np.array([x_cr, x_al]).T
    T = 1073*np.ones(N)
    mobTernary, _ = computeHomogenizationFunction(NiCrAlTherm, x, T, homogenizationParameters)
    assert(mobTernary.shape == (N,3))

def test_homogenizationSinglePhaseMobility():
    '''
    Tests that in a single phase region, any of the mobility functions will give 
    the same mobility of the single phase itself
    '''
    x = [0.05, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.labyrinthFactor = 2
    mob_data = computeMobility(NiCrAlTherm, x, T)

    mob_funcs = [HomogenizationParameters.WIENER_UPPER, HomogenizationParameters.WIENER_LOWER, 
                 HomogenizationParameters.HASHIN_UPPER, HomogenizationParameters.HASHIN_LOWER, 
                 HomogenizationParameters.LABYRINTH]
    for f in mob_funcs:
        homogenizationParameters.setHomogenizationFunction(f)
        mob, _ = computeHomogenizationFunction(NiCrAlTherm, x, T, homogenizationParameters)
        assert(np.allclose(np.squeeze(mob), np.squeeze(mob_data.mobility[0]), atol=0, rtol=1e-3))

def test_homogenization_wiener_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.WIENER_UPPER)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [5.422604e-22, 1.416420e-22, 2.327880e-22], atol=0, rtol=1e-3)

def test_homogenization_wiener_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.WIENER_LOWER)
    
    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [4.090531e-21, 1.068474e-21, 1.756032e-21], atol=0, rtol=1e-3)

def test_homogenization_hashin_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.HASHIN_UPPER)
    
    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [4.114414e-22, 1.074712e-22, 1.766285e-22], atol=0, rtol=1e-3)

def test_homogenization_hashin_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.HASHIN_LOWER)
    
    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [9.292913e-21, 2.427370e-21, 3.989373e-21], atol=0, rtol=1e-3)

def test_homogenization_lab():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    homogenizationParameters = HomogenizationParameters()
    homogenizationParameters.setHomogenizationFunction(HomogenizationParameters.LABYRINTH)
    
    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x1, T, homogenizationParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = computeHomogenizationFunction(NiCrAlTherm, x2, T, homogenizationParameters)
    assert_allclose(mob, [5.422604e-22, 1.416420e-22, 2.327880e-22], atol=0, rtol=1e-3)

def test_single_phase_dxdt():
    '''
    Check dxdt values of arbitrary single phase model problem

    We spot check a few points on dxdt rather than checking the entire array

    This uses the parameters from 06_Single_Phase_Diffusion example with the composition
    being linear rather then step functions
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(0, [0.077, 0.054], 2e-3, [0.359, 0.062]), ['CR', 'AL'])
    meshes = [
        Cartesian1D(['CR', 'AL'], [0, 2e-3], 20),
        Cylindrical1D(['CR', 'AL'], [0, 2e-3], 20),
        Spherical1D(['CR', 'AL'], [0, 2e-3], 20)
    ]

    vals5 = [
        np.array([1.63031418e-9, 5.87290513e-10]),
        np.array([1.34386193e-8, 8.02074750e-9]),
        np.array([2.52603986e-8, 1.54590585e-8]),
    ]
    vals10 = [
        np.array([1.54252455e-9, 1.08801591e-9]),
        np.array([8.47547223e-9, 5.37498213e-9]),
        np.array([1.54119182e-8, 9.66441601e-9]),
    ]
    vals15 = [
        np.array([1.59190900e-9, 1.79208543e-9]),
        np.array([6.79212648e-9, 5.15379480e-9]),
        np.array([1.19940010e-8, 8.51736977e-9]),
    ]
    # dt should be the same for all three meshes for single phase model since it only depends on D
    dts = [
        25902.839039,
        25902.839039,
        25902.839039,
    ]
    for i, mesh in enumerate(meshes):
        mesh.setResponseProfile(profile)
        m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
        m.setup()
        dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
        dt = m.getDt(dxdt)
        
        print(dxdt[0][5], dxdt[0][10], dxdt[0][15], dt)
        assert_allclose(dxdt[0][5], vals5[i], rtol=1e-3)
        assert_allclose(dxdt[0][10], vals10[i], rtol=1e-3)
        assert_allclose(dxdt[0][15], vals15[i], rtol=1e-3)
        assert_allclose(dt, dts[i], rtol=1e-3)

def test_diffusion_x_shape():
    '''
    Check the flatten and unflatten behavior for Diffusion model

    SinglePhaseModel and Homogenization model follows the same path for these functions
    since we just deal with fluxes for elements

    For this setup:
        getCurrentX will return a single element array with the element having a shape of (2,20)
        flattenX will return a 1D array of length 40 (2x20)
        unflattenX should take the output of flattenX and getCurrentX to bring the (40,) array to [(2,20)]
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh.setResponseProfile(profile)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1473.15))
    m.setup()

    x = m.getCurrentX()
    origShape = x[0].shape
    
    x_flat = m.flattenX(x)
    flatShape = x_flat.shape

    x_restore = m.unflattenX(x_flat, x)
    unflatShape = x_restore[0].shape

    assert(len(x) == 1)
    assert(origShape == unflatShape)
    assert(flatShape == (np.prod(origShape),))
    assert(len(x_restore) == 1)

def test_homogenization_dxdt():
    '''
    Check flux values of arbitrary homogenization model problem

    We spot check a few points on dxdt rather than checking the entire array
    
    We'll only test using the hashin lower homogenization function since there's already tests for 
    the output of each homogenization function

    This uses the parameters from 07_Homogenization_Model example with the compositions
    being linear rather than stepwise functions

    Note: values are changed slightly here since the mesh constructs the bounds to be slightly smaller than before
        Old implementation - ends of the mesh is at the node centers
        New implementation - ends of the mesh is at the node edges (node width is 1/(N-1) times smaller)
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-5e-4, [0.257, 0.065], 5e-4, [0.423, 0.276]), ['CR', 'NI'])
    mesh = Cartesian1D(['CR', 'NI'], [-5e-4, 5e-4], 20)
    mesh.setResponseProfile(profile)
    homogenizationParameters = HomogenizationParameters(HomogenizationParameters.HASHIN_LOWER, eps=0.01)

    m = HomogenizationModel(mesh,  ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'], 
                            thermodynamics=FeCrNiTherm, temperature=TemperatureParameters(1373.15),
                            homogenizationParameters=homogenizationParameters)
    m.constraints.maxCompositionChange = 0.002

    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)
    
    # #Index 5
    ind5, vals5 = 5, np.array([-1.42139140e-09, 1.23814781e-09])

    # #Index 10
    ind10, vals10 = 10, np.array([-9.42102782e-10, 1.66848272e-09])

    # #Index 15
    ind15, vals15 = 15, np.array([-3.9239687e-10, 7.66269736e-10])

    print(dxdt[0][ind5], dxdt[0][ind10], dxdt[0][ind15], dt)
    assert_allclose(dxdt[0][ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 62333.021201, rtol=1e-3)


    mesh = Cartesian1D(['CR', 'NI'], [-5e-4, 5e-4], 20)
    mesh.setResponseProfile(profile)
    homogenizationParameters = HomogenizationParameters(HomogenizationParameters.HASHIN_UPPER, eps=0.01)
    m = HomogenizationModel(mesh,  ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'], 
                            thermodynamics=FeCrNiTherm, temperature=TemperatureParameters(1373.15),
                            homogenizationParameters=homogenizationParameters)
    m.constraints.maxCompositionChange = 0.002

    m.setup()
    x = m.getCurrentX()
    dxdt = m.getdXdt(m.currentTime, x)
    dt = m.getDt(dxdt)

    # The dxdt values are changed due to a correction in how the mobility for each phase is computed
    # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    
    #Index 5
    ind5, vals5 = 5, np.array([-2.86475448e-8, -4.63706532e-9])

    #Index 10
    ind10, vals10 = 10, np.array([-1.70044057e-8, -3.8592184e-9])

    #Index 15
    ind15, vals15 = 15, np.array([-1.62187495e-8, -6.99884393e-9])
    
    print(dxdt[0][ind5], dxdt[0][ind10], dxdt[0][ind15], dt)
    assert_allclose(dxdt[0][ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 3343.03841738, rtol=1e-3)

def test_diffusionSavingLoading():
    '''
    Tests saving/loading behavior of diffusion model
    '''
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    temperature = TemperatureParameters(1200+273.15)
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh.setResponseProfile(profile)

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         thermodynamics=NiCrAlTherm,
                         temperature=temperature, record=True)

    m.solve(10*3600, verbose=True, vIt=1)
    m.save('kawin/tests/diff.npz')

    new_m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         thermodynamics=NiCrAlTherm,
                         temperature=temperature, record=True)
    new_m.load('kawin/tests/diff.npz')
    os.remove('kawin/tests/diff.npz')

    assert_allclose(m.mesh.y, new_m.mesh.y)
    assert_allclose(m.currentTime, new_m.currentTime)

def test_single_phase_2d():
    '''
    Test Cartesion2D mesh in a single phase model
    '''
    profile = ProfileBuilder([
        (BoundedRectangleProfile([1e-3, 1e-3], [2e-3, 2e-3], [0.077, 0.054], [0.359, 0.062]), ['CR', 'AL'])
    ])
    mesh = Cartesian2D(['CR', 'AL'], [0, 2e-3], 20, [0, 2e-3], 20)
    mesh.setResponseProfile(profile)

    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)

    print(dxdt[0][10,10])
    print(dt)
    assert_allclose(dxdt[0][10,10], [2.85828916e-6, 2.01865282e-6], rtol=1e-3)
    assert_allclose(dt, 12630.562798, rtol=1e-3)

def test_diffusion_profile():
    '''
    Test output of different profile build steps
    Constant, Gaussian, Dirac, Rectangle and Ellipse should support more than 1 dimension and response
    '''
    linear = LinearProfile1D(leftZ=0.25, leftValue=0.1, rightZ=0.75, rightValue=0.9, lowerLeftValue=0.2, upperRightValue=0.7)
    step = StepProfile1D(z=0.4, leftValue=0.2, rightValue=0.7)
    dirac = DiracDeltaProfile(z=0.61, value=0.8)
    constant = ConstantProfile(value=0.3)
    gaussian = GaussianProfile(z=0.6, sigma=0.2, maxValue=0.5)
    exp = ExperimentalProfile1D(z=[0, 0.1, 0.6, 1], values=[0, 0.5, 0.1, 0.3])
    rect = BoundedRectangleProfile(lowerZ=0.25, upperZ=0.75, innerValue=0.3, outerValue=0.6)

    indices = [5, 30, 60, 95]
    profiles = [
        (linear, [0.2, 0.188, 0.668, 0.7]),
        (step, [0.2, 0.2, 0.7, 0.7]),
        (dirac, [0, 0, 0.8, 0]),
        (constant, [0.3, 0.3, 0.3, 0.3]),
        (gaussian, [2.97894e-4, 5.67686e-2, 4.99688e-1, 2.14127e-2]),
        (exp, [0.275, 0.336, 0.1025, 0.2775]),
        (rect, [0.6, 0.3, 0.3, 0.6])
    ]
    for p in profiles:
        mesh = Cartesian1D(1, [0, 1], 100)
        profile = ProfileBuilder()
        profile.addBuildStep(p[0])
        mesh.setResponseProfile(profile)
        assert_allclose(mesh.y[indices,0], p[1], rtol=1e-3)

    constant = ConstantProfile([0.3, 0.5])
    gaussian = GaussianProfile(z=[0.3,0.4], sigma=[0.2,0.5], maxValue=[0.2,0.3])
    dirac = DiracDeltaProfile(z=[0.51, 0.51], value=[0.2, 0.1])
    rect = BoundedRectangleProfile(lowerZ=[0.5,0.5], upperZ=[0.6, 0.7], innerValue=[0.5, 0.1], outerValue=[0.2, 0.3])
    ell = BoundedEllipseProfile(z=[0.65, 0.65], r=[0.1, 0.1], innerValue=[0.5, 0.1], outerValue=[0.2, 0.3])

    indices = [[20,20], [50,50], [70,70]]
    profiles = [
        (constant, [[0.3, 0.5], [0.3, 0.5], [0.3, 0.5]]),
        (gaussian, [[1.37084e-1, 2.05626e-1], [6.69263e-2, 1.00389e-1], [2.28323e-3, 3.42485e-3]]),
        (dirac, [[0,0], [0.2,0.1], [0,0]]),
        (rect, [[0.2, 0.3], [0.5, 0.1], [0.2, 0.3]]),
        (ell, [[0.2, 0.3], [0.2, 0.3], [0.5, 0.1]]),
    ]
    for p in profiles:
        mesh = Cartesian2D(2, [0,1], 100, [0,1], 100)
        profile = ProfileBuilder()
        profile.addBuildStep(p[0], [0,1])
        mesh.setResponseProfile(profile)
        for i, ind in enumerate(indices):
            assert_allclose(mesh.y[ind[0], ind[1]], p[1][i], rtol=1e-3)

def test_diffusion_boundary_conditions():
    profile = ProfileBuilder()
    profile.addBuildStep(LinearProfile1D(0, [0.077, 0.054], 2e-3, [0.359, 0.062]), ['CR', 'AL'])

    bc = MixedBoundary1D(['CR', 'AL'])
    # We test the different input types (name and int)
    bc.setLBC('CR', 'dirichlet', 0.7)
    bc.setLBC('AL', MixedBoundary1D.DIRICHLET, 0.2)
    bc.setLBC('AL', 'composition', 0.2)
    bc.setRBC('CR', 'flux', 0)
    bc.setRBC('CR', 'neumann', 0)
    bc.setRBC('CR', MixedBoundary1D.NEUMANN, 1e-5)
    # Assert that a value error is raised for
    #  a. incorrect response variable
    #  b. incorrect int for boundary type
    #  c. incorrect str for boundary type
    with pytest.raises(ValueError):
        bc.setLBC('a', 'dirichlet', 0)
    with pytest.raises(ValueError):
        bc.setLBC('CR', 3, 0)
    with pytest.raises(ValueError):
        bc.setLBC('CR', 'comp', 0)

    mesh = Cartesian1D(['CR', 'AL'], [0, 2e-3], 20, computeMidpoint=True)
    mesh.setResponseProfile(profile, bc)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)
    print(dt)
    print(dxdt)

    assert_allclose(dt, 5031.54489, rtol=1e-3)
    assert_allclose(dxdt[0][0], [0, 0], atol=1e-3)
    assert_allclose(dxdt[0][-1], [-1e-1, -5.94812538e-8], rtol=1e-3)

    periodic = PeriodicBoundary1D()
    mesh = Cartesian1D(['CR', 'AL'], [0, 2e-3], 20, computeMidpoint=True)
    mesh.setResponseProfile(profile, periodic)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, 1473.15)
    m.setup()
    fluxes = m.getFluxes(m.currentTime, m.getCurrentX())
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)

    # First and last flux should be equal
    assert_allclose(dt, 26548.48400, rtol=1e-3)
    assert_allclose(fluxes[0], fluxes[-1], rtol=1e-3)

