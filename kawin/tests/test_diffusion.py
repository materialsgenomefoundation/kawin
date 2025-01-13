import os

import numpy as np
from numpy.testing import assert_allclose

from kawin.diffusion import SinglePhaseModel, HomogenizationModel
from kawin.diffusion.DiffusionParameters import computeMobility, CompositionProfile, BoundaryConditions, TemperatureParameters
from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
from kawin.thermo import GeneralThermodynamics
from kawin.tests.datasets import *

N = 100
singleModelBinary = SinglePhaseModel([-1e-3, 1e-3], N, ['NI', 'CR'], ['FCC_A1'])
singleModelTernary = SinglePhaseModel([-1e-3, 1e-3], N, ['NI', 'CR', 'AL'], ['FCC_A1'])
homogenizationBinary = HomogenizationModel([-1e-3, 1e-3], N, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
homogenizationTernary = HomogenizationModel([-1e-3, 1e-3], N, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])
NiCrTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
NiCrAlTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])
FeCrNiTherm = GeneralThermodynamics(FECRNI_DB, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])

def test_CompositionInput():
    '''
    Tests that after setting up a model, all components are greater than 0
    
    In practice, this greatly speeds up simulation time without sacrificing accuracy since it avoids 
    performing equilibrium calculations with a composition of 0 for any given component

    The composition and setup functions for both models inherit from the
    base diffusion model class, so any model can be used here
    '''
    compositionProfile = CompositionProfile()
    compositionProfile.addStepCompositionStep('CR', 0.2, 1, 0)
    compositionProfile.addStepCompositionStep('AL', 0.8, 0, 0)
    singleModelTernary.reset()
    singleModelTernary.compositionProfile = compositionProfile
    singleModelTernary.setTemperature(1200+273.15)
    singleModelTernary.setThermodynamics(NiCrAlTherm)

    singleModelTernary.setup()

    assert(singleModelTernary.x[0,25] + singleModelTernary.x[1,25] < 1)
    assert(singleModelTernary.x[0,75] + singleModelTernary.x[1,75] < 1)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.constraints.minComposition)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.constraints.minComposition)
    assert(singleModelTernary.x[1,75] >= singleModelTernary.constraints.minComposition)

def test_SinglePhaseFluxes_shape():
    '''
    Tests the dimensions of the single phase fluxes function

    Should be (E-1, N+1) where E is the number of elements and
    N is the number of points
    '''
    compositionProfile = CompositionProfile()
    compositionProfile.addStepCompositionStep('CR', 0.2, 0.8, 0)
    singleModelBinary.reset()
    singleModelBinary.compositionProfile = compositionProfile
    singleModelBinary.setTemperature(1073)
    singleModelBinary.setThermodynamics(NiCrTherm)
    singleModelBinary.setup()

    fBinary, _ = singleModelBinary.getFluxes()

    compositionProfile = CompositionProfile()
    compositionProfile.addStepCompositionStep('CR', 0.2, 0.4, 0)
    compositionProfile.addStepCompositionStep('AL', 0.4, 0.4, 0)
    singleModelTernary.reset()
    singleModelTernary.compositionProfile = compositionProfile
    singleModelTernary.setTemperature(1073)
    singleModelTernary.setThermodynamics(NiCrAlTherm)
    singleModelTernary.setup()

    fTernary, _ = singleModelTernary.getFluxes()

    assert(fBinary.shape == (1,N+1))
    assert(fTernary.shape == (2,N+1))

def test_HomogenizationMobility():
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
    compositionProfile = CompositionProfile()
    compositionProfile.addLinearCompositionStep('CR', 0.077, 0.359)
    compositionProfile.addLinearCompositionStep('AL', 0.054, 0.062)

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    #Since we defined L12, the disordered phase as DIS_ attached to the front
    m = SinglePhaseModel([-1e-3, 1e-3], 20, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         compositionProfile=compositionProfile)

    m.setThermodynamics(NiCrAlTherm)
    m.setTemperature(1200+273.15)

    m.setup()
    t, x = m.getCurrentX()
    dxdt = m.getdXdt(t, x)
    dt = m.getDt(dxdt)

    #Index 5
    ind5, vals5 = 5, np.array([1.640437e-9, 5.669268e-10])

    #Index 10
    ind10, vals10 = 10, np.array([1.542640e-9, 1.091229e-9])

    #Index 15
    ind15, vals15 = 15, np.array([1.596203e-9, 1.842238e-9])
    
    assert_allclose(dxdt[0][:,ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][:,ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][:,ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 28721.530474, rtol=1e-3)

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
    compositionProfile = CompositionProfile()
    compositionProfile.addLinearCompositionStep('CR', 0.077, 0.359)
    compositionProfile.addLinearCompositionStep('AL', 0.054, 0.062)

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    #Since we defined L12, the disordered phase as DIS_ attached to the front
    m = SinglePhaseModel([-1e-3, 1e-3], 20, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         compositionProfile=compositionProfile)

    m.setThermodynamics(NiCrAlTherm)
    m.setTemperature(1200 + 273.15)

    m.setup()
    t, x = m.getCurrentX()
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
    '''
    compositionProfile = CompositionProfile()
    compositionProfile.addLinearCompositionStep('CR', 0.257, 0.423)
    compositionProfile.addLinearCompositionStep('NI', 0.065, 0.276)

    homogenizationParameters = HomogenizationParameters(HomogenizationParameters.HASHIN_LOWER, eps=0.01)

    m = HomogenizationModel([-5e-4, 5e-4], 20, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'], 
                            compositionProfile=compositionProfile, 
                            homogenizationParameters=homogenizationParameters)
    m.setTemperature(1100+273.15)
    m.setThermodynamics(FeCrNiTherm)
    m.constraints.maxCompositionChange = 0.002

    m.setup()
    t, x = m.getCurrentX()
    dxdt = m.getdXdt(t, x)
    dt = m.getDt(dxdt)

    # The dxdt values are changed due to a correction in how the mobility for each phase is computed
    # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    
    #Index 5
    ind5, vals5 = 5, np.array([-1.480029e-9, 1.193852e-9])

    #Index 10
    ind10, vals10 = 10, np.array([-9.453766e-10, 1.681638e-9])

    #Index 15
    ind15, vals15 = 15, np.array([-3.441800e-10, 6.905748e-10])
    
    assert_allclose(dxdt[0][:,ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][:,ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][:,ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 65415.110254, rtol=1e-3)

def test_diffusionBackCompatibility():
    '''
    Tests that old API works for diffusion models
    '''
    compositionProfile = CompositionProfile()
    compositionProfile.addLinearCompositionStep('CR', 0.257, 0.423)
    compositionProfile.addLinearCompositionStep('NI', 0.065, 0.276)

    boundaryConditions = BoundaryConditions()
    boundaryConditions.setBoundaryCondition(BoundaryConditions.RIGHT, BoundaryConditions.FLUX_BC, 1, 'NI')
    boundaryConditions.setBoundaryCondition(BoundaryConditions.LEFT, BoundaryConditions.COMPOSITION_BC, 0.5, 'CR')

    temperature = TemperatureParameters(10)

    homogenizationParameters = HomogenizationParameters(HomogenizationParameters.HASHIN_LOWER, eps=0.01)

    m = HomogenizationModel([-5e-4, 5e-4], 20, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'], 
                            compositionProfile=compositionProfile, 
                            boundaryConditions=boundaryConditions,
                            temperatureParameters=temperature,
                            homogenizationParameters=homogenizationParameters)
    
    m2 = HomogenizationModel([-5e-4, 5e-4], 20, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
    m2.setCompositionLinear(0.257, 0.423, 'CR')
    m2.setCompositionLinear(0.065, 0.276, 'NI')
    m2.setBC(BoundaryConditions.COMPOSITION_BC, 0.5, BoundaryConditions.FLUX_BC, 0, element='CR')
    m2.setBC(BoundaryConditions.FLUX_BC, 0, BoundaryConditions.FLUX_BC, 1, element='NI')
    m2.setTemperature(10)
    m2.setMobilityFunction(HomogenizationParameters.HASHIN_LOWER)

    m.setup()
    m2.setup()

    assert_allclose(m.x, m2.x, rtol=1e-3)
    assert_allclose(m.temperatureParameters(m.z, 0), m2.temperatureParameters(m.z, 0), rtol=1e-3)
    assert m.boundaryConditions.leftBCtype == m2.boundaryConditions.leftBCtype
    assert m.boundaryConditions.leftBC == m2.boundaryConditions.leftBC
    assert m.boundaryConditions.rightBCtype == m2.boundaryConditions.rightBCtype
    assert m.boundaryConditions.rightBC == m2.boundaryConditions.rightBC

    assert m.homogenizationParameters.homogenizationFunction == m2.homogenizationParameters.homogenizationFunction

def test_diffusionSavingLoading():
    '''
    Tests saving/loading behavior of diffusion model
    '''
    compositionProfile = CompositionProfile()
    compositionProfile.addLinearCompositionStep('CR', 0.077, 0.359)
    compositionProfile.addLinearCompositionStep('AL', 0.054, 0.062)
    temperature = TemperatureParameters(1200+273.15)
    print([p for p in NiCrAlTherm.models])

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    #Since we defined L12, the disordered phase as DIS_ attached to the front
    m = SinglePhaseModel([-1e-3, 1e-3], 20, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         thermodynamics=NiCrAlTherm,
                         compositionProfile=compositionProfile,
                         temperatureParameters=temperature)

    m.solve(10*3600, verbose=True, vIt=1)
    m.save('kawin/tests/diff.npz')

    new_m = SinglePhaseModel([-1e-3, 1e-3], 20, ['NI', 'CR', 'AL'], ['FCC_A1'], 
                         thermodynamics=NiCrAlTherm,
                         compositionProfile=compositionProfile,
                         temperatureParameters=temperature)
    new_m.load('kawin/tests/diff.npz')
    os.remove('kawin/tests/diff.npz')

    assert_allclose(m.x, new_m.x)
    assert_allclose(m.t, new_m.t)


