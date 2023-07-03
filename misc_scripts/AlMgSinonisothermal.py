from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.PopulationBalance import PopulationBalanceModel
import matplotlib.pyplot as plt
import numpy as np

solve = False

if solve:
    phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']
    phases = ['FCC_A1', 'MGSI_B_P']
    therm = MulticomponentThermodynamics('database//AlMgSi.tdb', ['AL', 'MG', 'SI'], phases, drivingForceMethod='approximate')

    model = PrecipitateModel(0, 25*3600, 1e4, phases=phases[1:], elements=['MG', 'SI'], linearTimeSpacing=True)

    model.setInitialComposition([0.0072, 0.0057])
    model.setVmAlpha(1e-5, 4)

    lowTemp = 200+273.15
    highTemp = 300+273.15
    model.setTemperatureArray([0, 12, 25], [highTemp, lowTemp, lowTemp])

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

    model.setPSDrecording()
    model.solve(verbose=True, vIt=100)
    model.save('misc_scripts//outputs//AlMgSinonisothermal', compressed=True)
    model.saveRecordedPSD('misc_scripts//outputs//AlMgSiPSD//AlMgSint')

else:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    model = PrecipitateModel.load('misc_scripts//outputs//AlMgSinonisothermal.npz')
    
    model.plot(axes[0,0], 'Total Precipitate Density', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
    model.plot(axes[0,0], 'Precipitate Density', timeUnits='h')
    axes[0,0].set_ylim([1e5, 1e25])
    axes[0,0].set_xscale('linear')
    axes[0,0].set_yscale('log')

    model.plot(axes[0,1], 'Total Volume Fraction', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
    model.plot(axes[0,1], 'Volume Fraction', timeUnits='h')
    #model.plot(axes[0,1], 'Driving Force', timeUnits='h')
    axes[0,1].set_xscale('linear')

    model.plot(axes[1,0], 'Average Radius', timeUnits='h')
    axes[1,0].set_xscale('linear')

    fig2, ax2 = plt.subplots(1,1)
    pbm = PopulationBalanceModel()
    pbm.loadRecordedPSD('misc_scripts//outputs//AlMgSiPSD//AlMgSint_MGSI_B_P.npz')

    times = np.linspace(0, 9e4, 10)
    for t in times:
        pbm.setPSDtoRecordedTime(t)
        pbm.PlotDistributionDensity(ax2, label=t)
    ax2.legend()

    fig.tight_layout()
    plt.show()