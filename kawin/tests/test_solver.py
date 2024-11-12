from kawin.precipitation import PrecipitateModel
from kawin.precipitation.PrecipitationParameters import VolumeParameter
from kawin.diffusion import SinglePhaseModel
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics
from kawin.GenericModel import GenericModel, Coupler
from kawin.solver import SolverType
import numpy as np
from numpy.testing import assert_allclose
from kawin.tests.datasets import *

AlZrTherm = BinaryThermodynamics(ALZR_TDB, ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='tangent')
NiAlCrTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'AL', 'CR'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')

AlZrTherm.setDFSamplingDensity(2000)
AlZrTherm.setEQSamplingDensity(500)
NiAlCrTherm.setDFSamplingDensity(2000)
NiAlCrTherm.setEQSamplingDensity(500)

def test_iterators():
    '''
    Tests explicit euler and RK4 iterators
    '''
    class TestModel(GenericModel):
        def __init__(self):
            self.reset()

        def reset(self):
            self.x = np.array([0])
            self.time = np.zeros(1)

        def getCurrentX(self):
            return self.time[-1], [self.x[-1]]
        
        def getdXdt(self, t, x):
            return [np.cos(t)]
        
        def getDt(self, dXdt):
            return 0.001
        
        def postProcess(self, time, x):
            self.time = np.append(self.time, time)
            self.x = np.append(self.x, x[0])
            return x, False
        
    m = TestModel()
    m.solve(10, solverType=SolverType.EXPLICITEULER)
    eulerX = m.x[-1]

    m.reset()
    m.solve(10, solverType=SolverType.RK4)
    rkX = m.x[-1]

    assert_allclose(eulerX, np.sin(10), rtol=1e-2)
    assert_allclose(rkX, np.sin(10), rtol=1e-2)

def test_coupler_shape():
    '''
    Test that coupler returns correct shape when flattening and unflattening arrays

    Here we use a precipitate model and diffusion model where the shape of x is:
        Precipitate model: [(bins,)]
        Diffusion model: [(elements,cells,)]
    Flattening the arrays will result in a 1D array of [bins + elements*cells]
    '''
    #Create model
    p_model = PrecipitateModel(phases=['AL3ZR'])
    bins = 75
    minBins = 50
    maxBins = 100
    p_model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    xInit = 4e-3        #Initial composition (mole fraction)
    p_model.setInitialComposition(xInit)

    T = 450 + 273.15    #Temperature (K)
    p_model.setTemperature(T)

    gamma = 0.1         #Interfacial energy (J/m2)
    p_model.setInterfacialEnergy(gamma)

    D0 = 0.0768         #Diffusivity pre-factor (m2/s)
    Q = 242000          #Activation energy (J/mol)
    Diff = lambda T: D0 * np.exp(-Q / (8.314 * T))
    AlZrTherm.setDiffusivity(Diff, 'FCC_A1')
    #p_model.setDiffusivity(Diff)

    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al
    atomsPerCell = 4    #Atoms in an FCC unit cell
    p_model.setVolumeAlpha(Va, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    p_model.setVolumeBeta(Vb, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)

    #Average grain size (um) and dislocation density (1e15)
    p_model.setNucleationDensity(grainSize = 1, dislocationDensity = 1e15)
    p_model.setNucleationSite('dislocations')

    #Set thermodynamic functions
    #p_model.setThermodynamics(AlZrTherm, addDiffusivity=False)
    p_model.setThermodynamics(AlZrTherm)

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    #Since we defined L12, the disordered phase as DIS_ attached to the front
    N = 20
    d_model = SinglePhaseModel([-1e-3, 1e-3], N, ['NI', 'AL', 'CR'], ['DIS_FCC_A1'])

    #Define Cr and Al composition, with step-wise change at z=0
    #d_model.setCompositionLinear(0.077, 0.359, 'CR')
    #d_model.setCompositionLinear(0.054, 0.062, 'AL')
    d_model.parameters.compositionProfile.addLinearCompositionStep('CR', 0.077, 0.359)
    d_model.parameters.compositionProfile.addLinearCompositionStep('AL', 0.054, 0.062)

    d_model.setThermodynamics(NiAlCrTherm)
    #d_model.setTemperature(1200 + 273.15)
    d_model.parameters.temperature.setIsothermalTemperature(1200+273.15)

    coupled_model = Coupler([p_model, d_model])
    coupled_model.setup()

    t, x = coupled_model.getCurrentX()
    x_flat = coupled_model.flattenX(x)
    x_restore = coupled_model.unflattenX(x_flat, x)

    assert(len(x) == 2)
    assert(len(x[0]) == 1 and x[0][0].shape == (bins,))
    assert(len(x[1]) == 1 and x[1][0].shape == (2,N))
    assert(x_flat.shape == (bins+2*N,))
    assert(len(x_restore) == len(x))
    assert(len(x_restore[0]) == len(x[0]) and x_restore[0][0].shape == x[0][0].shape)
    assert(len(x_restore[1]) == 1 and x_restore[1][0].shape == x[1][0].shape)