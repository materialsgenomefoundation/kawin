from numpy.testing import assert_allclose
import numpy as np
from kawin.Diffusion import SinglePhaseModel, HomogenizationModel
from kawin.Thermodynamics import GeneralThermodynamics
from kawin.tests.datasets import *

N = 100
singleModelBinary = SinglePhaseModel([-1e-3, 1e-3], N, ['NI', 'CR'], ['FCC_A1'])
singleModelTernary = SinglePhaseModel([-1e-3, 1e-3], N, ['NI', 'CR', 'AL'], ['FCC_A1'])
homogenizationBinary = HomogenizationModel([-1e-3, 1e-3], N, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
homogenizationTernary = HomogenizationModel([-1e-3, 1e-3], N, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])
NiCrTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
NiCrAlTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])

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
    singleModelTernary.setup()

    assert(singleModelTernary.x[0,25] + singleModelTernary.x[1,25] < 1)
    assert(singleModelTernary.x[0,75] + singleModelTernary.x[1,75] < 1)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.minComposition)
    assert(1 - (singleModelTernary.x[0,75] + singleModelTernary.x[1,75]) >= singleModelTernary.minComposition)
    assert(singleModelTernary.x[1,75] >= singleModelTernary.minComposition)

def test_SinglePhaseFluxes():
    '''
    Tests the dimensions of the single phase fluxes function

    Should be (E-1, N+1) where E is the number of elements and
    N is the number of points
    '''
    singleModelBinary.reset()
    singleModelBinary.setCompositionStep(0.2, 0.8, 0, 'CR')
    singleModelBinary.setThermodynamics(NiCrTherm)
    singleModelBinary.setTemperature(800)
    singleModelBinary.setup()

    fBinary, _ = singleModelBinary.getFluxes()

    singleModelTernary.reset()
    singleModelTernary.setCompositionStep(0.2, 0.4, 0, 'CR')
    singleModelTernary.setCompositionStep(0.4, 0.4, 0, 'AL')
    singleModelTernary.setThermodynamics(NiCrAlTherm)
    singleModelTernary.setTemperature(800)
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
    homogenizationBinary.setTemperature(800)
    homogenizationBinary.setup()

    mobBinary = homogenizationBinary.getMobility(homogenizationBinary.x)

    homogenizationTernary.reset()
    homogenizationTernary.setCompositionStep(0.2, 0.4, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.4, 0.4, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(800)
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
    homogenizationTernary.setCompositionStep(0.1, 0.4, 0, 'CR')
    homogenizationTernary.setCompositionStep(0.1, 0.4, 0, 'AL')
    homogenizationTernary.setThermodynamics(NiCrAlTherm)
    homogenizationTernary.setTemperature(800)
    homogenizationTernary.setup()

    mob = homogenizationTernary.getMobility(homogenizationTernary.x)

    mobFuncs = ['wiener upper', 'wiener lower', 'hashin upper', 'hashin lower', 'lab']
    mobs = []
    for f in mobFuncs:
        homogenizationTernary.clearCache()
        homogenizationTernary.setup()
        homogenizationTernary.setMobilityFunction(f)
        mobs.append(homogenizationTernary.mobilityFunction(homogenizationTernary.x))
        assert(np.allclose(mobs[-1][:,0], mob[0,:,0], rtol=1e-3))

