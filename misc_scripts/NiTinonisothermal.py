import numpy as np
import matplotlib.pyplot as plt
from kawin.thermo.Thermodynamics import BinaryThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.ElasticFactors import StrainEnergy
from kawin.ShapeFactors import ShapeFactor
from kawin.PopulationBalance import PopulationBalanceModel

saveName = 'NiTi_strain_cooldown'
solve = True

if solve:
    #Set up thermodynamics
    phases = ['BCC_B2', 'TI3NI4']
    therm = BinaryThermodynamics('database//NiTi_SMA.tdb', ['TI', 'NI'], phases)

    #Override guess composition(s) to reduce number of calculations
    # when finding interfacial composition
    therm.setGuessComposition(0.56)

    #Model parameters
    xinit = 0.508
    gamma = 0.053
    T = 450+273.15
    Dni = lambda x, T: 1.8e-8 * np.exp(-155000/(8.314*T))
    vaBCC, nBCC = 0.0268114e-27, 2
    vaNI3TI4, nNI3TI4 = 0.184615e-27, 14

    se = StrainEnergy()
    B2e = np.asarray([175,45,35]) * 1e9
    eigenstrain = [-0.00417, -0.00417, -0.0257]
    rotate = [[-4/np.sqrt(42), 5/np.sqrt(42), -1/np.sqrt(42)], 
            [-2/np.sqrt(14), -1/np.sqrt(14), 3/np.sqrt(14)], 
            [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]]
    se.setEigenstrain(eigenstrain)
    se.setElasticConstants(B2e[0],B2e[1],B2e[2])
    se.setRotationMatrix(rotate)

    #Initialize model
    model = PrecipitateModel(0, 1e6, 1e4, linearTimeSpacing=True)
    model.setInitialComposition(xinit)
    model.setInterfacialEnergy(gamma)
    #model.setTemperature(T)
    low, high = 250+273.15, 650+273.15
    model.setTemperatureArray([0, 0.25e6/3600, 0.5e6/3600, 0.6e6/3600, 0.75e6/3600, 1e6/3600], [low, high, low, low, high, low])
    model.setDiffusivity(Dni)
    model.setVaAlpha(vaBCC, nBCC)
    model.setVaBeta(vaNI3TI4, nNI3TI4)
    model.setThermodynamics(therm)

    fig, ax = plt.subplots(1,1)
    model.plot(ax, 'Temperature')
    ax.set_xscale('linear')
    plt.show()

    #model.setStrainEnergy(se, calculateAspectRatio=True)
    #model.setAspectRatioPlate()

    #model.setPSDrecording(True, np.float32)

    #model.solve(verbose=True, vIt=100)
    #model.save('misc_scripts//outputs//' + saveName)

    #model.saveRecordedPSD('misc_scripts//outputs//NiTiPSD')

else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    model = PrecipitateModel.load('misc_scripts//outputs//' + saveName + '.npz')

    model.plot(axes[0,0], 'Precipitate Density', linewidth=2, timeUnits='h')
    model.plot(axes[0,1], 'Driving Force', linewidth=2, timeUnits='h')
    model.plot(axes[1,0], 'Average Radius', linewidth=2, timeUnits='h')
    model.plot(axes[1,1], 'Volume Fraction', linewidth=2, timeUnits='h')
    #model.plot(axes[1,1], 'Size Distribution Density', linewidth=2)

    pbm = PopulationBalanceModel()
    pbm.loadRecordedPSD('misc_scripts//outputs//NiTiPSD_beta.npz')

    fig, ax2 = plt.subplots(1, 1)

    times = np.linspace(0e5, 10e5, 21)
    for t in times:
        pbm.setPSDtoRecordedTime(t)
        if t > 5e5:
            pbm.PlotDistributionDensity(ax2, label=t, linestyle='--')
        else:
            pbm.PlotDistributionDensity(ax2, label=t)

    ax2.legend()


    axes[0,0].set(ylim=[1e15, 1e25], yscale='log', xscale='linear')
    axes[0,1].set(xscale='linear')
    axes[1,0].set(xscale='linear')
    axes[1,1].set(xscale='linear')
    ax2.set(ylim=[1e15, 1e35], xlim=[1e-10, 1e-6], yscale='log', xscale='log')

    plt.tight_layout()
    plt.show()