import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from kawin.tests.datasets import ALZR_TDB, NICRAL_TDB, ALMGSI_DB
from kawin.precipitation import PrecipitateModel, StrainEnergy
from kawin.precipitation import VolumeParameter, PrecipitateParameters, MatrixParameters, TemperatureParameters
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics

AlZrTherm = BinaryThermodynamics(ALZR_TDB, ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'], drivingForceMethod='tangent')
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
    D0 = 0.0768         #Diffusivity pre-factor (m2/s)
    Q = 242000          #Activation energy (J/mol)
    Diff = lambda T: D0 * np.exp(-Q / (8.314 * T))
    AlZrTherm.setDiffusivity(Diff, 'FCC_A1')

    xInit = 4e-3        #Initial composition (mole fraction)
    T = 450 + 273.15    #Temperature (K)
    gamma = 0.1         #Interfacial energy (J/m2)
    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al
    atomsPerCell = 4    #Atoms in an FCC unit cell

    matrix = MatrixParameters(['ZR'])
    matrix.initComposition = xInit
    matrix.volume.setVolume(Va, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    matrix.nucleationSites.setNucleationDensity(grainSize=1, dislocationDensity=1e15)
    temperature = TemperatureParameters(T)
    precipitate = PrecipitateParameters('AL3ZR')
    precipitate.gamma = gamma
    precipitate.volume.setVolume(Vb, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    precipitate.nucleation.setNucleationType('dislocations')

    #Create model
    model = PrecipitateModel(matrix, [precipitate], AlZrTherm, temperature)
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