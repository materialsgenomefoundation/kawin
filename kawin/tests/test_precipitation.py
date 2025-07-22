import os

import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

from kawin.tests.databases import ALZR_TDB, NICRAL_TDB, ALMGSI_DB
from kawin.precipitation import PrecipitateModel
from kawin.precipitation import VolumeParameter, PrecipitateParameters, MatrixParameters, TemperatureParameters
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics

from kawin.precipitation.coupling import GrainGrowthModel, StrengthModel, CoherencyContribution, DislocationParameters, SolidSolutionStrength
from kawin.precipitation.coupling import plotGrainCDF, plotGrainPDF, plotGrainPSD, plotRadiusvsTime, plotPrecipitateStrengthOverTime, plotContributionOverTime, plotAlloyStrength

from kawin.precipitation.StoppingConditions import Inequality, VolumeFractionCondition, AverageRadiusCondition, DrivingForceCondition, NucleationRateCondition, PrecipitateDensityCondition, CompositionCondition
from kawin.precipitation.TimeTemperaturePrecipitation import TTPCalculator, plotTTP

AlZrTherm = BinaryThermodynamics(ALZR_TDB, ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='tangent')
D0 = 0.0768         #Diffusivity pre-factor (m2/s)
Q = 242000          #Activation energy (J/mol)
Diff = lambda T: D0 * np.exp(-Q / (8.314 * T))
AlZrTherm.setDiffusivity(Diff, 'FCC_A1')

NiAlCrTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'AL', 'CR'], ['FCC_A1', 'FCC_L12'], drivingForceMethod='tangent')
AlMgSitherm = MulticomponentThermodynamics(ALMGSI_DB, ['AL', 'MG', 'SI'], ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE'], drivingForceMethod='tangent')

AlZrTherm.setDFSamplingDensity(2000)
AlZrTherm.setEQSamplingDensity(500)
NiAlCrTherm.setDFSamplingDensity(2000)
NiAlCrTherm.setEQSamplingDensity(500)
AlMgSitherm.setDFSamplingDensity(2000)
AlMgSitherm.setEQSamplingDensity(500)

def test_binary_precipitation_dxdt():
    '''
    Check flux values of arbitrary binary precipitation problem

    We spot check a few points on dxdt rather than checking the entire array

    This uses the parameters from 01_Binary_Precipitation example
    '''
    T = 450 + 273.15    #Temperature (K)
    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al

    matrix = MatrixParameters(['ZR'])
    matrix.initComposition = 4e-3       # initial composition
    matrix.volume.setVolume(Va, VolumeParameter.ATOMIC_VOLUME, 4)
    matrix.nucleationSites.setNucleationDensity(grainSize=1, dislocationDensity=1e15)
    precipitate = PrecipitateParameters('AL3ZR')
    precipitate.gamma = 0.1
    precipitate.volume.setVolume(Vb, VolumeParameter.ATOMIC_VOLUME, 4)
    precipitate.nucleation.setNucleationType('dislocations')

    #Create model
    model = PrecipitateModel(matrix, precipitate, AlZrTherm, T)
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=75, minBins=50, maxBins=100, phase='AL3ZR')

    #This roughly follows the steps in model.solve so we can get dxdt
    model.setup()

    #Replace x (which is just all 0 right now) with an arbitrary lognormal distribution
    r = model.PBM[0].PSDsize
    sigma = 0.25
    r0 = 0.1e-8
    n = 1/(r*sigma*np.sqrt(2*np.pi)) * np.exp(-np.log(r/r0)**2/(2*sigma**2))
    model.PBM[0].PSD = n

    x = model.getCurrentX()
    #Call calculateDependentTerms so it can recognize that we changed PSD, otherwise, it'll use the initial values
    model._calculateDependentTerms(model.currentTime, x)
    dxdt = model.getdXdt(model.currentTime, x)

    #Set arbitrary final time, this is done during the solve function, but we do it here since we're not using the solve function
    #  the initial guess for the time step will be 0.01*(1.001) regardless of finalTime
    model.finalTime = 1 
    dt = model.getDt(dxdt)

    indices = [10, 20, 30]
    vals = [6773393.32259, 1919.5404124, 0.4106318]
    assert_allclose(vals, [dxdt[0][i] for i in indices], rtol=1e-3)
    assert_allclose(dt, 0.01001, rtol=1e-3)

def test_multi_precipitation_dxdt():
    '''
    Check flux values of arbitrary binary precipitation problem

    We spot check a few points on dxdt rather than checking the entire array

    This uses the parameters from 02_Multicomponent_Precipitation example
    '''
    a = 0.352e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Ni
    Vb = Va             #Assume Ni3Al has same unit volume as FCC-Ni
    atomsPerCell = 4    #Atoms in an FCC unit cell

    matrix = MatrixParameters(['AL', 'CR'])
    matrix.initComposition = [0.098, 0.083]
    matrix.volume.setVolume(Va, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    matrix.nucleationSites.setBulkDensity(1e30)

    precipitate = PrecipitateParameters('FCC_L12')
    precipitate.gamma = 0.023
    precipitate.volume.setVolume(Vb, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    precipitate.nucleation.setNucleationType('bulk')

    model = PrecipitateModel(matrix, [precipitate], NiAlCrTherm, TemperatureParameters(1073))
    bins = 75
    minBins = 50
    maxBins = 100
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    #This roughly follows the steps in model.solve so we can get dxdt
    model.setup()

    #Replace x (which is just all 0 right now) with an arbitrary lognormal distribution
    r = model.PBM[0].PSDsize
    sigma = 0.25
    r0 = 0.1e-8
    n = 1/(r*sigma*np.sqrt(2*np.pi)) * np.exp(-np.log(r/r0)**2/(2*sigma**2))
    model.PBM[0].PSD = n

    x = model.getCurrentX()
    #Call calculateDependentTerms so it can recognize that we changed PSD, otherwise, it'll use the initial values
    model._calculateDependentTerms(model.currentTime, x)
    dxdt = model.getdXdt(model.currentTime, x)

    #Set arbitrary final time, this is done during the solve function, but we do it here since we're not using the solve function
    #  the initial guess for the time steo will be 0.01*(1.001) regardless of finalTime
    model.finalTime = 1 
    dt = model.getDt(dxdt)

    indices = [10, 20, 30]
    vals = [2.837811e+08, 8.424854e+05, 2.312587e+02]
    assert_allclose(vals, [dxdt[0][i] for i in indices], rtol=1e-3)
    assert_allclose(dt, 0.01001, rtol=1e-3)

def test_multiphase_precipitation_x_shape():
    '''
    Check the flatten and unflatten behavior for Precipitate model

    For this setup:
        getCurrentX will return a array of length p with each element being an array of length bins
        flattenX will return a 1D array of length p*bins
        unflattenX should take the output of flattenX and getCurrentX to bring the (p*bins,) to [(bins,), (bins,), ...]

    This uses the parameters from 07_Homogenization_Model example
    '''
    phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']
    precParams = []
    gamma = {
        'MGSI_B_P': 0.18,
        'MG5SI6_B_DP': 0.084,
        'B_PRIME_L': 0.18,
        'U1_PHASE': 0.18,
        'U2_PHASE': 0.18
            }

    for p in phases[1:]:
        prec = PrecipitateParameters(p)
        prec.gamma = gamma[p]
        prec.volume.setVolume(1e-5, VolumeParameter.MOLAR_VOLUME, 4)
        precParams.append(prec)

    matrix = MatrixParameters(['MG', 'SI'])
    matrix.initComposition = [0.0072, 0.0057]
    matrix.volume.setVolume(1e-5, VolumeParameter.MOLAR_VOLUME, 4)

    lowTemp = 175+273.15
    highTemp = 250+273.15
    temperature = TemperatureParameters([0, 16, 17], [lowTemp, lowTemp, highTemp])

    model = PrecipitateModel(matrix, precParams, AlMgSitherm, temperature)
    bins = 75
    minBins = 50
    maxBins = 100
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    model.setup()
    x = model.getCurrentX()
    origLen = 5

    x_flat = model.flattenX(x)
    flatShape = x_flat.shape

    x_restore = model.unflattenX(x_flat, x)

    assert(len(x) == origLen)
    assert(np.all(psd.shape == (bins,) for psd in x))
    assert(flatShape == (origLen*bins,))
    assert(len(x_restore) == origLen)
    assert(np.all(psd.shape == (bins,) for psd in x_restore))

def test_precipitationSavingLoading():
    '''
    Test saving loading behavior
    '''
    phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']

    matrix = MatrixParameters(['MG', 'SI'])
    matrix.initComposition = [0.0072, 0.0057]
    matrix.volume.setVolume(1e-5, 'VM', 4)

    lowTemp = 175+273.15
    highTemp = 250+273.15
    temperature = TemperatureParameters([0, 16, 17], [lowTemp, lowTemp, highTemp])

    gamma = {
        'MGSI_B_P': 0.18,
        'MG5SI6_B_DP': 0.084,
        'B_PRIME_L': 0.18,
        'U1_PHASE': 0.18,
        'U2_PHASE': 0.18
            }

    precipitates = []
    for p in phases[1:]:
        params = PrecipitateParameters(p)
        params.gamma = gamma[p]
        params.volume.setVolume(1e-5, 'VM', 4)
        precipitates.append(params)

    model = PrecipitateModel(matrix=matrix, precipitates=precipitates, thermodynamics=AlMgSitherm, temperature=temperature)
    model.solve(0.1, verbose=True, vIt=1)

    model.save('kawin/tests/prec.npz')

    new_model = PrecipitateModel(matrix=matrix, precipitates=precipitates, thermodynamics=AlMgSitherm, temperature=temperature)
    new_model.load('kawin/tests/prec.npz')
    os.remove('kawin/tests/prec.npz')

    assert_allclose(model.data.Ravg, new_model.data.Ravg)
    assert_allclose(model.data.time, new_model.data.time)
    assert_allclose(model.data.precipitateDensity, new_model.data.precipitateDensity)

def test_precipitationCoupling():
    '''
    Test that we can couple the grain growth and strength model and plot without error
    '''
    T = 450 + 273.15    #Temperature (K)
    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al

    matrix = MatrixParameters(['ZR'])
    matrix.initComposition = 4e-3       # initial composition
    matrix.volume.setVolume(Va, VolumeParameter.ATOMIC_VOLUME, 4)
    matrix.nucleationSites.setNucleationDensity(grainSize=1, dislocationDensity=1e15)
    precipitate = PrecipitateParameters('AL3ZR')
    precipitate.gamma = 0.1
    precipitate.volume.setVolume(Vb, VolumeParameter.ATOMIC_VOLUME, 4)
    precipitate.nucleation.setNucleationType('dislocations')

    #Create model
    model = PrecipitateModel(matrix, precipitate, AlZrTherm, T)
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=75, minBins=50, maxBins=100)

    grainModel = GrainGrowthModel(gbe=0.5, M=1e-14)
    
    coh = CoherencyContribution(0.01, 'AL3ZR')
    dislocations = DislocationParameters(G=50e9, b=1e-9, nu=1/3)
    ss = SolidSolutionStrength({'ZR': 1e9})
    strengthModel = StrengthModel(precipitate, coh, dislocations, ss)

    model.addCouplingModel(grainModel)
    model.addCouplingModel(strengthModel)

    model.solve(1)

    grainPlottingFunctions = [plotGrainPSD, plotGrainPDF, plotGrainCDF, plotRadiusvsTime]
    for func in grainPlottingFunctions:
        fig, ax = plt.subplots()
        func(grainModel, ax=ax)
        assert len(ax.lines) == 1
        plt.close(fig)

    fig, ax = plt.subplots()
    plotContributionOverTime(model, strengthModel, coh, ax=ax)
    assert len(ax.lines) == 2
    plt.close(fig)

    fig, ax = plt.subplots()
    plotPrecipitateStrengthOverTime(model, strengthModel, plotContributions=False, ax=ax)
    assert len(ax.lines) == 1
    plt.close(fig)

    fig, ax = plt.subplots()
    plotPrecipitateStrengthOverTime(model, strengthModel, plotContributions=True, ax=ax)
    assert len(ax.lines) == 4
    plt.close(fig)

    fig, ax = plt.subplots()
    plotAlloyStrength(model, strengthModel, plotContributions=False, ax=ax)
    assert len(ax.lines) == 1
    plt.close(fig)

    fig, ax = plt.subplots()
    plotAlloyStrength(model, strengthModel, plotContributions=True, ax=ax)
    assert len(ax.lines) == 4
    plt.close(fig)

def test_precipitationStopping():
    '''
    Test stopping conditions

    For each condition, we test whether it returns the correct satisfaction value (True if condition was satisfied)
    If condition is satisfied, we also test the interpolate time

    Also test usage in precipitate model with 'or'or 'and' conditions
    '''
    phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']
    precParams = []
    gamma = {
        'MGSI_B_P': 0.18,
        'MG5SI6_B_DP': 0.084,
        'B_PRIME_L': 0.18,
        'U1_PHASE': 0.18,
        'U2_PHASE': 0.18
            }

    for p in phases[1:]:
        prec = PrecipitateParameters(p)
        prec.gamma = gamma[p]
        prec.volume.setVolume(1e-5, VolumeParameter.MOLAR_VOLUME, 4)
        precParams.append(prec)

    matrix = MatrixParameters(['MG', 'SI'])
    matrix.initComposition = [0.0072, 0.0057]
    matrix.volume.setVolume(1e-5, VolumeParameter.MOLAR_VOLUME, 4)

    temperature = 175+273.15
    model = PrecipitateModel(matrix, precParams, AlMgSitherm, temperature)
    model.setup()

    # Create artificial data for precipitate model
    N = 10
    model.data.reset(N)
    model.data.time = np.linspace(0, 1, N)
    model.data.volFrac[:,0] = np.linspace(0, 0.1, N)                # MGSI_B_P
    model.data.Ravg[:,1] = np.linspace(0, 1e-7, N)                  # MG5SI6_B_DP
    model.data.drivingForce[:,2] = np.linspace(1000, 0, N)          # B_PRIME_L
    model.data.nucRate[:,3] = np.linspace(0, 1e10, N)               # U1_PHASE
    model.data.precipitateDensity[:,4] = np.linspace(0, 1e10, N)    # U2_PHASE
    model.data.composition[:,0] = 0.0072*np.ones(N)                 # MG
    model.data.composition[:,1] = np.linspace(0.006, 0.005, N)      # SI

    volCond = VolumeFractionCondition(Inequality.GREATER_THAN, 0.05, phase='MGSI_B_P')
    rCond = AverageRadiusCondition(Inequality.GREATER_THAN, 1e-6, phase='MG5SI6_B_DP')
    dgCond = DrivingForceCondition(Inequality.LESSER_THAN, 500, phase='B_PRIME_L')
    nucCond = NucleationRateCondition(Inequality.LESSER_THAN, 0.5e10, phase='U1_PHASE')
    precCond = PrecipitateDensityCondition(Inequality.GREATER_THAN, 0.5e10, phase='U2_PHASE')
    compCond = CompositionCondition(Inequality.LESSER_THAN, 0.0055, element='SI')

    volCond.testCondition(model)
    rCond.testCondition(model)
    dgCond.testCondition(model)
    nucCond.testCondition(model)
    precCond.testCondition(model)
    compCond.testCondition(model)

    assert volCond.isSatisfied()
    assert_allclose(volCond.satisfiedTime(), 0.5, rtol=1e-3)
    assert not rCond.isSatisfied()
    assert_allclose(rCond.satisfiedTime(), -1, rtol=1e-3)
    assert dgCond.isSatisfied()
    assert not nucCond.isSatisfied()
    assert precCond.isSatisfied()
    assert compCond.isSatisfied()
    assert_allclose(compCond.satisfiedTime(), 0.5, rtol=1e-3)

    # 'or' only needs 1 condition to be true to stop
    model.reset()
    volCond_0 = VolumeFractionCondition(Inequality.GREATER_THAN, 0.1, phase='MGSI_B_P')
    compCond_0 = CompositionCondition(Inequality.LESSER_THAN, 0.0075, element='MG')
    model.addStoppingCondition(volCond_0, 'or')
    model.addStoppingCondition(compCond_0, 'or')
    model.setup()
    _, stop = model.postProcess(0.1, [p.PSD for p in model.PBM])
    assert stop
    
    # 'and' needs all conditions to be true to stop
    model.reset()
    model.clearStoppingConditions()
    model.addStoppingCondition(volCond_0, 'and')
    model.addStoppingCondition(compCond_0, 'and')
    model.setup()
    _, stop = model.postProcess(0.1, [p.PSD for p in model.PBM])
    assert not stop

def test_ttp():
    T = 450 + 273.15    #Temperature (K)
    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al

    matrix = MatrixParameters(['ZR'])
    matrix.initComposition = 4e-3       # initial composition
    matrix.volume.setVolume(Va, VolumeParameter.ATOMIC_VOLUME, 4)
    matrix.nucleationSites.setNucleationDensity(grainSize=1, dislocationDensity=1e15)
    precipitate = PrecipitateParameters('AL3ZR')
    precipitate.gamma = 0.1
    precipitate.volume.setVolume(Vb, VolumeParameter.ATOMIC_VOLUME, 4)
    precipitate.nucleation.setNucleationType('dislocations')

    #Create model
    model = PrecipitateModel(matrix, precipitate, AlZrTherm, T)
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=75, minBins=50, maxBins=100)
    volCond = VolumeFractionCondition(Inequality.GREATER_THAN, -1)
    
    # Tests that the TTP can be ran and plotted
    ttp = TTPCalculator(model, volCond)
    ttp.calculateTTP(500, 600, 3, 10)
    plotTTP(ttp)


