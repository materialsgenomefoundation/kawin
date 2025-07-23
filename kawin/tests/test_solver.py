import numpy as np
from numpy.testing import assert_allclose

from kawin.precipitation import PrecipitateModel, PrecipitateParameters, MatrixParameters, TemperatureParameters as PrecTemp
from kawin.diffusion import SinglePhaseModel, TemperatureParameters as DiffTemp
from kawin.diffusion.mesh import ProfileBuilder, Cartesian1D, LinearProfile1D
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics
from kawin.GenericModel import GenericModel, Coupler
from kawin.solver import explicitEulerIterator, rk4Iterator
from kawin.tests.databases import *

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
            super().__init__()
            self.reset()

        def reset(self):
            super().reset()
            self.x = np.array([0])

        def getCurrentX(self):
            return [self.x[-1]]
        
        def getdXdt(self, t, x):
            return [np.cos(t)]
        
        def getDt(self, dXdt):
            return 0.001
        
        def postProcess(self, time, x):
            super().postProcess(time, x)
            self.x = np.append(self.x, x[0])
            return x, False
        
    m = TestModel()
    m.solve(10, iterator=explicitEulerIterator)
    eulerX = m.x[-1]

    m.reset()
    m.solve(10, iterator=rk4Iterator)
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
    D0 = 0.0768         #Diffusivity pre-factor (m2/s)
    Q = 242000          #Activation energy (J/mol)
    Diff = lambda T: D0 * np.exp(-Q / (8.314 * T))
    AlZrTherm.setDiffusivity(Diff, 'FCC_A1')

    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al
    atomsPerCell = 4    #Atoms in an FCC unit cell
    
    matrix = MatrixParameters(['ZR'])
    matrix.initComposition = 4e-3
    matrix.volume.setVolume(Va, 'VA', atomsPerCell)
    matrix.nucleationSites.setNucleationDensity(grainSize=1, dislocationDensity=1e15)

    precipitate = PrecipitateParameters('AL3ZR')
    precipitate.gamma = 0.1
    precipitate.volume.setVolume(Vb, 'VA', atomsPerCell)
    precipitate.nucleation.setNucleationType('dislocations')

    #Create model
    p_model = PrecipitateModel(matrix, [precipitate], AlZrTherm, PrecTemp(450+273.15))
    bins = 75
    minBins = 50
    maxBins = 100
    p_model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    #Define mesh spanning between -1mm to 1mm with 50 volume elements
    N = 20
    profile = ProfileBuilder([(LinearProfile1D(-1e-3, [0.077, 0.054], 1e-3, [0.359, 0.062]), ['CR', 'AL'])])
    mesh = Cartesian1D(['AL', 'CR'], [-1e-3, 1e-3], N)
    mesh.setResponseProfile(profile)
    d_model = SinglePhaseModel(mesh, ['NI', 'AL', 'CR'], ['FCC_A1'], NiAlCrTherm, DiffTemp(1200+273.15))

    coupled_model = Coupler([p_model, d_model])
    coupled_model.setup()

    x = coupled_model.getCurrentX()
    x_flat = coupled_model.flattenX(x)
    x_restore = coupled_model.unflattenX(x_flat, x)

    assert(len(x) == 2)
    assert(len(x[0]) == 1 and x[0][0].shape == (bins,))
    #assert(len(x[1]) == 1 and x[1][0].shape == (2,N))
    assert(len(x[1]) == 1 and x[1][0].shape == (N,2))
    assert(x_flat.shape == (bins+2*N,))
    assert(len(x_restore) == len(x))
    assert(len(x_restore[0]) == len(x[0]) and x_restore[0][0].shape == x[0][0].shape)
    assert(len(x_restore[1]) == 1 and x_restore[1][0].shape == x[1][0].shape)