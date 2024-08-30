from numpy.testing import assert_allclose
import numpy as np
from kawin.diffusion import SinglePhaseModel, HomogenizationModel
from kawin.diffusion.DiffusionParameters import compute_homogenization_function, compute_mobility, DiffusionParameters
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
    singleModelTernary.parameters.composition_profile.add_step_composition_step('CR', 0.2, 1, 0)
    singleModelTernary.parameters.composition_profile.add_step_composition_step('AL', 0.8, 0, 0)
    singleModelTernary.parameters.temperature.set_isothermal_temperature(1200+273.15)
    singleModelTernary.setThermodynamics(NiCrAlTherm)

    singleModelTernary.setup()

    assert(singleModelTernary.x[0,25] + singleModelTernary.x[1,25] < 1)
    assert(singleModelTernary.x[0,75] + singleModelTernary.x[1,75] < 1)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.parameters.min_composition)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.parameters.min_composition)
    assert(singleModelTernary.x[1,75] >= singleModelTernary.parameters.min_composition)

def test_SinglePhaseFluxes_shape():
    '''
    Tests the dimensions of the single phase fluxes function

    Should be (E-1, N+1) where E is the number of elements and
    N is the number of points
    '''
    singleModelBinary.reset()
    singleModelBinary.parameters.composition_profile.add_step_composition_step('CR', 0.2, 0.8, 0)
    singleModelBinary.parameters.temperature.set_isothermal_temperature(1073)
    singleModelBinary.setThermodynamics(NiCrTherm)
    singleModelBinary.setup()

    fBinary, _ = singleModelBinary.getFluxes()

    singleModelTernary.reset()
    singleModelTernary.parameters.composition_profile.add_step_composition_step('CR', 0.2, 0.4, 0)
    singleModelTernary.parameters.composition_profile.add_step_composition_step('AL', 0.4, 0.4, 0)
    singleModelTernary.parameters.temperature.set_isothermal_temperature(1073)
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

    binaryParameters = DiffusionParameters(['CR'])
    x = np.linspace(0.2, 0.3, N)
    T = 1073*np.ones(N)
    mobBinary, _ = compute_homogenization_function(NiCrTherm, x, T, binaryParameters)
    assert(mobBinary.shape == (N,2))

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    x_cr = np.linspace(0.2, 0.3, N)
    x_al = np.linspace(0.3, 0.2, N)
    x = np.array([x_cr, x_al]).T
    T = 1073*np.ones(N)
    mobTernary, _ = compute_homogenization_function(NiCrAlTherm, x, T, ternaryParameters)
    assert(mobTernary.shape == (N,3))


    # homogenizationBinary.reset()
    # # homogenizationBinary.setCompositionStep(0.2, 0.8, 0, 'CR')
    # # homogenizationBinary.setThermodynamics(NiCrTherm)
    # # homogenizationBinary.setTemperature(1073)
    # homogenizationBinary.parameters.composition_profile.add_step_composition_step('CR', 0.2, 0.8, 0)
    # homogenizationBinary.parameters.temperature.set_isothermal_temperature(1073)
    # homogenizationBinary.setThermodynamics(NiCrTherm)
    # homogenizationBinary.setup()
    # T = homogenizationBinary.parameters.temperature(homogenizationBinary.z, 0)

    # mobBinary, _ = compute_homogenization_function(NiCrTherm, homogenizationBinary.x.T, T, homogenizationBinary.parameters)

    # homogenizationTernary.reset()
    # # homogenizationTernary.setCompositionStep(0.2, 0.4, 0, 'CR')
    # # homogenizationTernary.setCompositionStep(0.4, 0.4, 0, 'AL')
    # # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.parameters.composition_profile.add_step_composition_step('CR', 0.2, 0.4, 0)
    # homogenizationTernary.parameters.composition_profile.add_step_composition_step('AL', 0.4, 0.4, 0)
    # homogenizationTernary.parameters.temperature.set_isothermal_temperature(1073)
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setup()
    # T = homogenizationTernary.parameters.temperature(homogenizationTernary.z, 0)

    # mobTernary, _ = compute_homogenization_function(NiCrAlTherm, homogenizationTernary.x.T, T, homogenizationTernary.parameters)

    # #assert(mobBinary.shape == (N,len(homogenizationBinary.phases),2))
    # #assert(mobTernary.shape == (N,len(homogenizationTernary.phases),3))

    # assert(mobBinary.shape == (N,2))
    # assert(mobTernary.shape == (N,3))

def test_homogenizationSinglePhaseMobility():
    '''
    Tests that in a single phase region, any of the mobility functions will give 
    the same mobility of the single phase itself
    '''
    x = [0.05, 0.05]
    T = 1073

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    ternaryParameters.labyrinth_factor = 2
    mob_data = compute_mobility(NiCrAlTherm, x, T, ternaryParameters)

    mob_funcs = ['wiener upper', 'wiener lower', 'hashin upper', 'lab']
    for f in mob_funcs:
        ternaryParameters.hash_table.clearCache()
        ternaryParameters.setHomogenizationFunction(mob_funcs)
        mob, _ = compute_homogenization_function(NiCrAlTherm, x, T, ternaryParameters)
        assert(np.allclose(np.squeeze(mob), np.squeeze(mob_data.mobility[0]), atol=0, rtol=1e-3))

    # homogenizationTernary.reset()
    # #Ni-5Cr-5Al should always be FCC_A1
    # # homogenizationTernary.setCompositionStep(0.05, 0.4, 0, 'CR')
    # # homogenizationTernary.setCompositionStep(0.05, 0.4, 0, 'AL')
    # # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.parameters.composition_profile.add_step_composition_step('CR', 0.05, 0.4, 0)
    # homogenizationTernary.parameters.composition_profile.add_step_composition_step('AL', 0.05, 0.4, 0)
    # homogenizationTernary.parameters.temperature.set_isothermal_temperature(1073)
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setup()
    # T = homogenizationTernary.parameters.temperature(homogenizationTernary.z, 0)

    # #mob, phaseFracs, chemPot = compute_mobility(NiCrAlTherm, homogenizationTernary.x.T, homogenizationTernary.T)
    # #mob = mob.transpose(1,2,0)
    # #phaseFracs = phaseFracs.T

    # mobFuncs = ['wiener upper', 'wiener lower', 'hashin upper', 'hashin lower', 'lab']
    # mobs = []
    # for f in mobFuncs:
    #     homogenizationTernary.parameters.hash_table.clearCache()
    #     #homogenizationTernary.clearCache()
    #     homogenizationTernary.setup()
    #     homogenizationTernary.parameters.setHomogenizationFunction(f)
    #     mobs.append(compute_homogenization_function(homogenizationTernary.therm, homogenizationTernary.x.T, homogenizationTernary.T, homogenizationTernary.parameters)[0])
    #     #mobs.append(homogenizationTernary.mobilityFunction(mob.transpose(2,0,1), phaseFracs.T, labyrinth_factor=homogenizationTernary.labFactor).T)
    #     assert(np.allclose(mobs[-1][:,0], mob[0,:,0], atol=0, rtol=1e-3))

def test_homogenization_wiener_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    ternaryParameters.setHomogenizationFunction('wiener upper')

    mob, _ = compute_homogenization_function(NiCrAlTherm, x1, T, ternaryParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = compute_homogenization_function(NiCrAlTherm, x2, T, ternaryParameters)
    assert_allclose(mob, [5.422604e-22, 1.416420e-22, 2.327880e-22], atol=0, rtol=1e-3)

    # homogenizationTernary.clearCache()
    # homogenizationTernary.reset()
    # #Ni-5Cr-5Al should always be FCC_A1
    # homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    # homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.setup()
    # homogenizationTernary.setMobilityFunction('wiener upper')

    # mobArray, phaseFracs, chemPot = compute_mobility(NiCrAlTherm, homogenizationTernary.x.T, homogenizationTernary.T)
    # mobArray = mobArray.transpose(1,2,0)
    # phaseFracs = phaseFracs.T

    # mob = homogenizationTernary.mobilityFunction(mobArray.transpose(2,0,1), phaseFracs.T, labyrinth_factor=homogenizationTernary.labFactor).T
    # assert_allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)
    # # These values are changed due to a correction in how the mobility for each phase is computed
    # # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    # assert_allclose(mob[:,-1], [5.422604e-22, 1.416420e-22, 2.327880e-22], atol=0, rtol=1e-3)

def test_homogenization_wiener_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    ternaryParameters.setHomogenizationFunction('wiener lower')
    
    mob, _ = compute_homogenization_function(NiCrAlTherm, x1, T, ternaryParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = compute_homogenization_function(NiCrAlTherm, x2, T, ternaryParameters)
    assert_allclose(mob, [4.090531e-21, 1.068474e-21, 1.756032e-21], atol=0, rtol=1e-3)

    # homogenizationTernary.clearCache()
    # homogenizationTernary.reset()
    # #Ni-5Cr-5Al should always be FCC_A1
    # homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    # homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.setup()
    # homogenizationTernary.setMobilityFunction('wiener lower')

    # mobArray, phaseFracs, chemPot = compute_mobility(NiCrAlTherm, homogenizationTernary.x.T, homogenizationTernary.T)
    # mobArray = mobArray.transpose(1,2,0)
    # phaseFracs = phaseFracs.T

    # mob = homogenizationTernary.mobilityFunction(mobArray.transpose(2,0,1), phaseFracs.T, labyrinth_factor=homogenizationTernary.labFactor).T
    # assert_allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)
    # # These values are changed due to a correction in how the mobility for each phase is computed
    # # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    # assert_allclose(mob[:,-1], [4.090531e-21, 1.068474e-21, 1.756032e-21], atol=0, rtol=1e-3)

def test_homogenization_hashin_upper():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    ternaryParameters.setHomogenizationFunction('hashin upper')
    
    mob, _ = compute_homogenization_function(NiCrAlTherm, x1, T, ternaryParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = compute_homogenization_function(NiCrAlTherm, x2, T, ternaryParameters)
    assert_allclose(mob, [4.114414e-22, 1.074712e-22, 1.766285e-22], atol=0, rtol=1e-3)

    # homogenizationTernary.clearCache()
    # homogenizationTernary.reset()
    # #Ni-5Cr-5Al should always be FCC_A1
    # homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    # homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.setup()
    # homogenizationTernary.setMobilityFunction('hashin upper')

    # mobArray, phaseFracs, chemPot = compute_mobility(NiCrAlTherm, homogenizationTernary.x.T, homogenizationTernary.T)
    # mobArray = mobArray.transpose(1,2,0)
    # phaseFracs = phaseFracs.T

    # mob = homogenizationTernary.mobilityFunction(mobArray.transpose(2,0,1), phaseFracs.T, labyrinth_factor=homogenizationTernary.labFactor).T
    # assert_allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)
    # # These values are changed due to a correction in how the mobility for each phase is computed
    # # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    # assert_allclose(mob[:,-1], [4.114414e-22, 1.074712e-22, 1.766285e-22], atol=0, rtol=1e-3)

def test_homogenization_hashin_lower():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    ternaryParameters.setHomogenizationFunction('hashin lower')
    
    mob, _ = compute_homogenization_function(NiCrAlTherm, x1, T, ternaryParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = compute_homogenization_function(NiCrAlTherm, x2, T, ternaryParameters)
    assert_allclose(mob, [9.292913e-21, 2.427370e-21, 3.989373e-21], atol=0, rtol=1e-3)

    # homogenizationTernary.clearCache()
    # homogenizationTernary.reset()
    # #Ni-5Cr-5Al should always be FCC_A1
    # homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    # homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.setup()
    # homogenizationTernary.setMobilityFunction('hashin lower')

    # mobArray, phaseFracs, chemPot = compute_mobility(NiCrAlTherm, homogenizationTernary.x.T, homogenizationTernary.T)
    # mobArray = mobArray.transpose(1,2,0)
    # phaseFracs = phaseFracs.T

    # mob = homogenizationTernary.mobilityFunction(mobArray.transpose(2,0,1), phaseFracs.T, labyrinth_factor=homogenizationTernary.labFactor).T
    # assert_allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)
    # # These values are changed due to a correction in how the mobility for each phase is computed
    # # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    # assert_allclose(mob[:,-1], [9.292913e-21, 2.427370e-21, 3.989373e-21], atol=0, rtol=1e-3)

def test_homogenization_lab():
    '''
    Tests output of wiener upper bounds in single and two-phase regions
    '''
    x1 = [0.05, 0.05]
    x2 = [0.7, 0.05]
    T = 1073

    ternaryParameters = DiffusionParameters(['CR', 'AL'])
    ternaryParameters.setHomogenizationFunction('lab')
    
    mob, _ = compute_homogenization_function(NiCrAlTherm, x1, T, ternaryParameters)
    assert_allclose(mob, [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)

    mob, _ = compute_homogenization_function(NiCrAlTherm, x2, T, ternaryParameters)
    assert_allclose(mob, [5.422604e-22, 1.416420e-22, 2.327880e-22], atol=0, rtol=1e-3)

    # homogenizationTernary.clearCache()
    # homogenizationTernary.reset()
    # #Ni-5Cr-5Al should always be FCC_A1
    # homogenizationTernary.setCompositionStep(0.05, 0.7, 0, 'CR')
    # homogenizationTernary.setCompositionStep(0.05, 0.05, 0, 'AL')
    # homogenizationTernary.setThermodynamics(NiCrAlTherm)
    # homogenizationTernary.setTemperature(1073)
    # homogenizationTernary.setup()
    # homogenizationTernary.setMobilityFunction('lab')

    # mobArray, phaseFracs, chemPot = compute_mobility(NiCrAlTherm, homogenizationTernary.x.T, homogenizationTernary.T)
    # mobArray = mobArray.transpose(1,2,0)
    # phaseFracs = phaseFracs.T

    # mob = homogenizationTernary.mobilityFunction(mobArray.transpose(2,0,1), phaseFracs.T, labyrinth_factor=homogenizationTernary.labFactor).T
    # assert_allclose(mob[:,0], [3.927302e-22, 2.323337e-23, 6.206029e-23], atol=0, rtol=1e-3)
    # # These values are changed due to a correction in how the mobility for each phase is computed
    # # Before, the mobilities were multiplied by the overall composition rather than the phase composition
    # assert_allclose(mob[:,-1], [5.422604e-22, 1.416420e-22, 2.327880e-22], atol=0, rtol=1e-3)

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
    #m.setCompositionLinear(0.077, 0.359, 'CR')
    #m.setCompositionLinear(0.054, 0.062, 'AL')
    m.parameters.composition_profile.add_linear_composition_step('CR', 0.077, 0.359)
    m.parameters.composition_profile.add_linear_composition_step('AL', 0.054, 0.062)

    m.setThermodynamics(NiCrAlTherm)
    #m.setTemperature(1200 + 273.15)
    m.parameters.temperature.set_isothermal_temperature(1200+273.15)

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
    #m.setCompositionLinear(0.077, 0.359, 'CR')
    #m.setCompositionLinear(0.054, 0.062, 'AL')
    m.parameters.composition_profile.add_linear_composition_step('CR', 0.077, 0.359)
    m.parameters.composition_profile.add_linear_composition_step('AL', 0.054, 0.062)

    m.setThermodynamics(NiCrAlTherm)
    #m.setTemperature(1200 + 273.15)
    m.parameters.temperature.set_isothermal_temperature(1200+273.15)

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
    #m.setCompositionLinear(0.257, 0.423, 'CR')
    #m.setCompositionLinear(0.065, 0.276, 'NI')
    #m.setTemperature(1100+273.15)
    m.parameters.composition_profile.add_linear_composition_step('CR', 0.257, 0.423)
    m.parameters.composition_profile.add_linear_composition_step('NI', 0.065, 0.276)
    m.parameters.temperature.set_isothermal_temperature(1100+273.15)
    m.setThermodynamics(FeCrNiTherm)
    #m.eps = 0.01
    m.parameters.eps = 0.01
    m.parameters.max_composition_change = 0.002

    #m.setMobilityFunction('hashin lower')
    m.parameters.setHomogenizationFunction('hashin lower')

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



