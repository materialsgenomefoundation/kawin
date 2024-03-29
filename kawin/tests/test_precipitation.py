from kawin.tests.datasets import ALZR_TDB, NICRAL_TDB, ALMGSI_DB
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics
import numpy as np
from numpy.testing import assert_allclose

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
    #Create model
    model = PrecipitateModel()
    bins = 75
    minBins = 50
    maxBins = 100
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    xInit = 4e-3        #Initial composition (mole fraction)
    model.setInitialComposition(xInit)

    T = 450 + 273.15    #Temperature (K)
    model.setTemperature(T)

    gamma = 0.1         #Interfacial energy (J/m2)
    model.setInterfacialEnergy(gamma)

    D0 = 0.0768         #Diffusivity pre-factor (m2/s)
    Q = 242000          #Activation energy (J/mol)
    Diff = lambda x, T: D0 * np.exp(-Q / (8.314 * T))
    model.setDiffusivity(Diff)

    a = 0.405e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Al
    Vb = a**3           #Assume Al3Zr has same unit volume as FCC-Al
    atomsPerCell = 4    #Atoms in an FCC unit cell
    model.setVolumeAlpha(Va, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    model.setVolumeBeta(Vb, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)

    #Average grain size (um) and dislocation density (1e15)
    model.setNucleationDensity(grainSize = 1, dislocationDensity = 1e15)
    model.setNucleationSite('dislocations')

    #Set thermodynamic functions
    model.setThermodynamics(AlZrTherm, addDiffusivity=False)

    #This roughly follows the steps in model.solve so we can get dxdt
    model.setup()

    #Replace x (which is just all 0 right now) with an arbitrary lognormal distribution
    r = model.PBM[0].PSDsize
    sigma = 0.25
    r0 = 0.1e-8
    n = 1/(r*sigma*np.sqrt(2*np.pi)) * np.exp(-np.log(r/r0)**2/(2*sigma**2))
    model.PBM[0].PSD = n

    t, x = model.getCurrentX()
    #Call calculateDependentTerms so it can recognize that we changed PSD, otherwise, it'll use the initial values
    model._calculateDependentTerms(t, x)
    dxdt = model.getdXdt(t, x)

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
    model = PrecipitateModel(elements=['Al', 'Cr'])
    bins = 75
    minBins = 50
    maxBins = 100
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    model.setInitialComposition([0.098, 0.083])
    model.setInterfacialEnergy(0.023)

    T = 1073
    model.setTemperature(T)

    a = 0.352e-9        #Lattice parameter
    Va = a**3           #Atomic volume of FCC-Ni
    Vb = Va             #Assume Ni3Al has same unit volume as FCC-Ni
    atomsPerCell = 4    #Atoms in an FCC unit cell
    model.setVolumeAlpha(Va, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    model.setVolumeBeta(Vb, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)

    #Set nucleation sites to dislocations and use defualt value of 5e12 m/m3
    #model.setNucleationSite('dislocations')
    #model.setNucleationDensity(dislocationDensity=5e12)
    model.setNucleationSite('bulk')
    model.setNucleationDensity(bulkN0=1e30)

    model.setThermodynamics(NiAlCrTherm)

    #This roughly follows the steps in model.solve so we can get dxdt
    model.setup()

    #Replace x (which is just all 0 right now) with an arbitrary lognormal distribution
    r = model.PBM[0].PSDsize
    sigma = 0.25
    r0 = 0.1e-8
    n = 1/(r*sigma*np.sqrt(2*np.pi)) * np.exp(-np.log(r/r0)**2/(2*sigma**2))
    model.PBM[0].PSD = n

    t, x = model.getCurrentX()
    #Call calculateDependentTerms so it can recognize that we changed PSD, otherwise, it'll use the initial values
    model._calculateDependentTerms(t, x)
    dxdt = model.getdXdt(t, x)

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
    model = PrecipitateModel(phases=phases[1:], elements=['MG', 'SI'])
    bins = 75
    minBins = 50
    maxBins = 100
    model.setPBMParameters(cMin=1e-10, cMax=1e-8, bins=bins, minBins=minBins, maxBins=maxBins)

    model.setInitialComposition([0.0072, 0.0057])
    model.setVolumeAlpha(1e-5, VolumeParameter.MOLAR_VOLUME, 4)

    lowTemp = 175+273.15
    highTemp = 250+273.15
    model.setTemperature(([0, 16, 17], [lowTemp, lowTemp, highTemp]))

    gamma = {
        'MGSI_B_P': 0.18,
        'MG5SI6_B_DP': 0.084,
        'B_PRIME_L': 0.18,
        'U1_PHASE': 0.18,
        'U2_PHASE': 0.18
            }

    for i in range(len(phases)-1):
        model.setInterfacialEnergy(gamma[phases[i+1]], phase=phases[i+1])
        model.setVolumeBeta(1e-5, VolumeParameter.MOLAR_VOLUME, 4, phase=phases[i+1])
        model.setThermodynamics(AlMgSitherm, phase=phases[i+1])

    model.setup()
    t, x = model.getCurrentX()
    origLen = 5

    x_flat = model.flattenX(x)
    flatShape = x_flat.shape

    x_restore = model.unflattenX(x_flat, x)

    assert(len(x) == origLen)
    assert(np.all(psd.shape == (bins,) for psd in x))
    assert(flatShape == (origLen*bins,))
    assert(len(x_restore) == origLen)
    assert(np.all(psd.shape == (bins,) for psd in x_restore))