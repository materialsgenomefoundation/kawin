from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.coupling.Strength import StrengthModel
import matplotlib.pyplot as plt

solveModel = False

if solveModel:
    phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']
    therm = MulticomponentThermodynamics('database//AlMgSi.tdb', ['AL', 'MG', 'SI'], phases, drivingForceMethod='approximate')

    model = PrecipitateModel(0, 25*3600, 1e4, phases=phases[1:], elements=['MG', 'SI'], linearTimeSpacing=True)

    model.setInitialComposition([0.0072, 0.0057])
    model.setVmAlpha(1e-5, 4)

    lowTemp = 175+273.15
    highTemp = 250+273.15
    model.setTemperatureArray([0, 16, 17], [lowTemp, lowTemp, highTemp])

    gamma = {
        'MGSI_B_P': 0.18,
        'MG5SI6_B_DP': 0.084,
        'B_PRIME_L': 0.18,
        'U1_PHASE': 0.18,
        'U2_PHASE': 0.18
            }

    for i in range(len(phases)-1):
        model.setInterfacialEnergy(gamma[phases[i+1]], phase=phases[i+1])
        model.setVmBeta(1e-5, 4, phase=phases[i+1])
        model.setThermodynamics(therm, phase=phases[i+1])

    sm = StrengthModel()
    sm.insertStrength(model)

    model.solve(verbose=True, vIt=5000)

    model.save('misc_scripts//outputs//AlMgSiStrength')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    model.plot(axes[0,0], 'Total Precipitate Density', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
    model.plot(axes[0,0], 'Precipitate Density', timeUnits='h')
    axes[0,0].set_ylim([1e5, 1e25])
    axes[0,0].set_xscale('linear')
    axes[0,0].set_yscale('log')

    model.plot(axes[0,1], 'Total Volume Fraction', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
    model.plot(axes[0,1], 'Volume Fraction', timeUnits='h')
    axes[0,1].set_xscale('linear')

    model.plot(axes[1,0], 'Average Radius', timeUnits='h')
    axes[1,0].set_xscale('linear')

    model.plot(axes[1,1], 'Composition', timeUnits='h')
    axes[1,1].set_xscale('linear')

    fig.tight_layout()

    plt.show()
else:
    sm = StrengthModel()
    sm.setDislocationParameters(G=25.4e9, b=0.286e-9, nu=0.34)
    sm.setCoherencyParameters(eps=2/3*0.0125, phase = 'all')
    sm.setModulusParameters(Gp=67.9e9, phase='MG5SI6_B_DP')
    sm.setAPBParameters(yAPB=0.5, phase='all')
    sm.setInterfacialParameters(gamma=0.1, phase='all')
    sm.setSolidSolutionStrength({'MG': 300e6, 'SI': 200e6}, exp=0.5)

    #model = PrecipitateModel.load('misc_scripts//outputs//AlMgSiStrength.npz')
    model = PrecipitateModel.load('misc_scripts//outputs//AlMgSiStrength.csv')
    #model.save('misc_scripts//outputs//AlMgSiStrength', toCSV=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    model.plot(axes[0,0], 'Total Precipitate Density', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
    model.plot(axes[0,0], 'Precipitate Density', timeUnits='h')
    axes[0,0].set_ylim([1e5, 1e25])
    axes[0,0].set_xscale('linear')
    axes[0,0].set_yscale('log')

    model.plot(axes[0,1], 'Total Volume Fraction', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
    model.plot(axes[0,1], 'Volume Fraction', timeUnits='h')
    axes[0,1].set_xscale('linear')

    model.plot(axes[1,0], 'Average Radius', timeUnits='h')
    axes[1,0].set_xscale('linear')
    sm.plotStrength(axes[1,1], model, plotContributions=True)

    fig.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(3,2,figsize=(10,10))
    sm.plotPrecipitateStrengthOverTime(ax, model, phase='U2_PHASE', plotContributions=True)
    for i in range(3):
        for j in range(2):
            ax[i,j].set_xscale('linear')
    fig.tight_layout()

    plt.show()