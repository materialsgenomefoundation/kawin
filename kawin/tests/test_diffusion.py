import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from kawin.diffusion import SinglePhaseModel, HomogenizationModel, TemperatureParameters
from kawin.diffusion.mesh import Cartesian1D, ProfileBuilder, StepProfile1D, LinearProfile1D
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
    profile.addBuildStep(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])
    mesh = Cartesian1D(['CR', 'AL'], [-1e-3, 1e-3], 20)
    mesh.setResponseProfile(profile)
    m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], NiCrAlTherm, TemperatureParameters(1473.15))
    m.setup()
    dxdt = m.getdXdt(m.currentTime, m.getCurrentX())
    dt = m.getDt(dxdt)

    #Index 5
    ind5, vals5 = 5, np.array([1.63031418e-9, 5.87290513e-10])

    #Index 10
    ind10, vals10 = 10, np.array([1.54252455e-9, 1.08801591e-9])

    #Index 15
    ind15, vals15 = 15, np.array([1.59190900e-9, 1.79208543e-9])
    
    # assert_allclose(dxdt[0][:,ind5], vals5, atol=0, rtol=1e-3)
    # assert_allclose(dxdt[0][:,ind10], vals10, atol=0, rtol=1e-3)
    # assert_allclose(dxdt[0][:,ind15], vals15, atol=0, rtol=1e-3)
    # assert_allclose(dt, 28721.530474, rtol=1e-3)
    print(dxdt[0][ind5], dxdt[0][ind10], dxdt[0][ind15], dt)
    assert_allclose(dxdt[0][ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 25902.839039, rtol=1e-3)

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
                            thermodynamics=FeCrNiTherm, temperatureParameters=TemperatureParameters(1373.15),
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
                            thermodynamics=FeCrNiTherm, temperatureParameters=TemperatureParameters(1373.15),
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
                         temperatureParameters=temperature, record=True)

    m.solve(10*3600, verbose=True, vIt=1)
    m.save('kawin/tests/diff.npz')

    new_m = SinglePhaseModel(mesh, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         thermodynamics=NiCrAlTherm,
                         temperatureParameters=temperature, record=True)
    new_m.load('kawin/tests/diff.npz')
    os.remove('kawin/tests/diff.npz')

    assert_allclose(m.mesh.y, new_m.mesh.y)
    assert_allclose(m.currentTime, new_m.currentTime)


