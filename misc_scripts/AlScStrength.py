from kawin.KWNEuler import PrecipitateModel
from kawin.thermo.Thermodynamics import BinaryThermodynamics
from kawin.coupling.Strength import StrengthModel
import matplotlib.pyplot as plt
import numpy as np

solve = False

sm = StrengthModel()
sm.setDislocationParameters(G=25.4e9, b=0.286e-9, nu=0.34)
sm.setCoherencyParameters(eps=2/3*0.0125)
sm.setModulusParameters(Gp=67.9e9)
sm.setAPBParameters(yAPB=0.5)
sm.setInterfacialParameters(gamma=0.1)
sm.singlePhaseExp = 2

if solve:

    therm = BinaryThermodynamics('database//AlScZr.tdb', ['AL', 'SC'], ['FCC_A1', 'AL3SC'])
    therm.setGuessComposition(0.24)
    model = PrecipitateModel(0, 250*3600, 1e4, linearTimeSpacing=False)

    model.setInitialComposition(0.002)
    model.setTemperature(400+273.15)
    model.setInterfacialEnergy(0.1)

    Va = (0.405e-9)**3
    Vb = (0.4196e-9)**3
    model.setVaAlpha(Va, 4)
    model.setVaBeta(Vb, 4)

    diff = lambda x, T: 1.9e-4 * np.exp(-164000 / (8.314*T)) 
    model.setDiffusivity(diff)

    model.setThermodynamics(therm, addDiffusivity=False)

    #sm.insertStrength(model)

    model.solve(verbose=True, vIt=5000)
    model.save('misc_scripts//outputs//AlSc_nostrength', toCSV=True)
    model.save('misc_scripts//outputs//AlSc_nostrength')

else:
    #model = PrecipitateModel.load('misc_scripts//outputs//AlSc.npz')
    model = PrecipitateModel.load('misc_scripts//outputs//AlSc.csv')
    #model.save('misc_scripts//outputs//AlSc', toCSV=True)
    fig, ax = plt.subplots(3,2,figsize=(10,10))
    #fig, ax = plt.subplots(1, 1)
    rs = np.linspace(0, 14e-9, 1000)
    ls = rs * (np.sqrt(3*np.pi/4/0.0075) - np.pi/2)
    sm.plotPrecipitateStrengthOverR(ax, rs, ls, strengthUnits='MPa', plotContributions=True)
    fig.tight_layout()

    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    model.plot(axes[0,0], 'Precipitate Density')
    model.plot(axes[0,1], 'Volume Fraction')
    model.plot(axes[1,0], 'Average Radius', label='Average Radius')
    model.plot(axes[1,0], 'Critical Radius', label='Critical Radius')
    axes[1,0].legend()
    #sm.plotStrength(axes[1,1], model, plotContributions=True)

    fig.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(3,2,figsize=(10,10))
    sm.plotPrecipitateStrengthOverTime(ax, model, plotContributions=True)
    fig.tight_layout()

    plt.show()