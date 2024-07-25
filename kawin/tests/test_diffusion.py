from numpy.testing import assert_allclose
import numpy as np
from kawin.diffusion import SinglePhaseModel, HomogenizationModel
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
    singleModelTernary.reset()
    singleModelTernary.setCompositionStep(0.2, 1, 0, 'CR')
    singleModelTernary.setCompositionStep(0.8, 0, 0, 'AL')
    singleModelTernary.setThermodynamics(NiCrAlTherm)
    singleModelTernary.setTemperature(1200+273.15)
    singleModelTernary.setup()

    assert(singleModelTernary.x[0,25] + singleModelTernary.x[1,25] < 1)
    assert(singleModelTernary.x[0,75] + singleModelTernary.x[1,75] < 1)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.minComposition)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.minComposition)
    assert(singleModelTernary.x[1,75] >= singleModelTernary.minComposition)

def test_SinglePhaseFluxes_shape():
    '''
    Tests the dimensions of the single phase fluxes function

    Should be (E-1, N+1) where E is the number of elements and
    N is the number of points
    '''
    singleModelBinary.reset()
    singleModelBinary.setCompositionStep(0.2, 0.8, 0, 'CR')
    singleModelBinary.setThermodynamics(NiCrTherm)
    singleModelBinary.setTemperature(1073)
    singleModelBinary.setup()

    fBinary, _ = singleModelBinary.getFluxes()

    singleModelTernary.reset()
    singleModelTernary.setCompositionStep(0.2, 0.4, 0, 'CR')
    singleModelTernary.setCompositionStep(0.4, 0.4, 0, 'AL')
    singleModelTernary.setThermodynamics(NiCrAlTherm)
    singleModelTernary.setTemperature(1073)
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
    homogenizationBinary.reset()
    homogenizationBinary.setCompositionStep(0.2, 0.8, 0, 'CR')
    homogenizationBinary.setThermodynamics(NiCrTherm)
    homogenizationBinary.setTemperature(1073)
    homogenizationBinary.setup()

    mobBinary = homogenizationBinary.getMobility(homogenizationBinary.x)

    homogenizationTernary.reset()
    homogenizationTernary.setCompositionStep(0.2, 0.4, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.4, 0.4, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()

    mobTernary = homogenizationTernary.getMobility(homogenizationTernary.x)

    assert(mobBinary.shape == (len(homogenizationBinary.phases),2,N))
    assert(mobTernary.shape == (len(homogenizationTernary.phases),3,N))

def test_homogenizationSinglePhaseMobility():
    '''
    Tests that in a single phase region, any of the mobility functions will give 
    the same mobility of the single phase itself
    '''
    homogenizationTernary.reset()
    #Ni-5Cr-5Al should always be FCC_A1
    homogenizationTernary.setCompositionStep(0.05, 0.4, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.05, 0.4, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()

    mob = homogenizationTernary.getMobility(homogenizationTernary.x)

    mobFuncs = ['wiener upper', 'wiener lower', 'hashin upper', 'hashin lower', 'lab']
    mobs = []
    for f in mobFuncs:
        homogenizationTernary.clearCache()
        homogenizationTernary.setup()
        homogenizationTernary.setMobilityFunction(f)
        mobs.append(homogenizationTernary.mobilityFunction(homogenizationTernary.x))
        assert(np.allclose(mobs[-1][:,0], mob[0,:,0], atol=0, rtol=1e-3))

def test_homogenization_wiener_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    homogenizationTernary.clearCache()
    homogenizationTernary.reset()
    #Ni-5Cr-5Al should always be FCC_A1
    homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()
    homogenizationTernary.setMobilityFunction('wiener upper')

    mob = homogenizationTernary.mobilityFunction(homogenizationTernary.x)
    assert(np.allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3))
    assert(np.allclose(mob[:,-1], [2.025338e-22, 5.106062e-22, 8.524977e-23], atol=0, rtol=1e-3))

def test_homogenization_wiener_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    homogenizationTernary.clearCache()
    homogenizationTernary.reset()
    #Ni-5Cr-5Al should always be FCC_A1
    homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()
    homogenizationTernary.setMobilityFunction('wiener lower')

    mob = homogenizationTernary.mobilityFunction(homogenizationTernary.x)
    assert(np.allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3))
    assert(np.allclose(mob[:,-1], [1.527894e-21, 3.851959e-21, 6.431152e-22], atol=0, rtol=1e-3))

def test_homogenization_hashin_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    homogenizationTernary.clearCache()
    homogenizationTernary.reset()
    #Ni-5Cr-5Al should always be FCC_A1
    homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()
    homogenizationTernary.setMobilityFunction('hashin upper')

    mob = homogenizationTernary.mobilityFunction(homogenizationTernary.x)
    assert(np.allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3))
    assert(np.allclose(mob[:,-1], [1.536725e-22, 3.874223e-22, 6.468323e-23], atol=0, rtol=1e-3))

def test_homogenization_hashin_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    homogenizationTernary.clearCache()
    homogenizationTernary.reset()
    #Ni-5Cr-5Al should always be FCC_A1
    homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()
    homogenizationTernary.setMobilityFunction('hashin lower')

    mob = homogenizationTernary.mobilityFunction(homogenizationTernary.x)
    assert(np.allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3))
    assert(np.allclose(mob[:,-1], [3.471117e-21, 8.751001e-21, 1.461049e-21], atol=0, rtol=1e-3))

def test_homogenization_lab():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    homogenizationTernary.clearCache()
    homogenizationTernary.reset()
    #Ni-5Cr-5Al should always be FCC_A1
    homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(1073)
    homogenizationTernary.setup()
    homogenizationTernary.setMobilityFunction('lab')

    mob = homogenizationTernary.mobilityFunction(homogenizationTernary.x)
    assert(np.allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3))
    assert(np.allclose(mob[:,-1], [2.025338e-22, 5.106062e-22, 8.524977e-23], atol=0, rtol=1e-3))

def test_single_phase_dxdt():
    '''
    Check dxdt values of arbitrary single phase model problem

    We spot check a few points on dxdt rather than checking the entire array

    This uses the parameters from 06_Single_Phase_Diffusion example with the composition
    being linear rather then step functions
    '''
    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    #Since we defined L12, the disordered phase as DIS_ attached to the front
    m = SinglePhaseModel([-1e-3, 1e-3], 20, ['NI', 'CR', 'AL'], ['FCC_A1'])

    #Define Cr and Al composition, with step-wise change at z=0
    m.setCompositionLinear(0.077, 0.359, 'CR')
    m.setCompositionLinear(0.054, 0.062, 'AL')

    m.setThermodynamics(NiCrAlTherm)
    m.setTemperature(1200 + 273.15)

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
    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    #Since we defined L12, the disordered phase as DIS_ attached to the front
    m = SinglePhaseModel([-1e-3, 1e-3], 20, ['NI', 'CR', 'AL'], ['DIS_FCC_A1'])

    #Define Cr and Al composition, with step-wise change at z=0
    m.setCompositionLinear(0.077, 0.359, 'CR')
    m.setCompositionLinear(0.054, 0.062, 'AL')

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
    m = HomogenizationModel([-5e-4, 5e-4], 20, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
    m.setCompositionLinear(0.257, 0.423, 'CR')
    m.setCompositionLinear(0.065, 0.276, 'NI')
    m.setTemperature(1100+273.15)
    m.setThermodynamics(FeCrNiTherm)
    m.eps = 0.01

    m.setMobilityFunction('hashin lower')

    m.setup()
    t, x = m.getCurrentX()
    dxdt = m.getdXdt(t, x)
    dt = m.getDt(dxdt)
    
    #Index 5
    ind5, vals5 = 5, np.array([-1.581478e-9, 1.212876e-9])

    #Index 10
    ind10, vals10 = 10, np.array([-9.722631e-10, 1.703447e-9])

    #Index 15
    ind15, vals15 = 15, np.array([-4.720562e-10, 8.600518e-10])
    
    assert_allclose(dxdt[0][:,ind5], vals5, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][:,ind10], vals10, atol=0, rtol=1e-3)
    assert_allclose(dxdt[0][:,ind15], vals15, atol=0, rtol=1e-3)
    assert_allclose(dt, 62271.050081, rtol=1e-3)



