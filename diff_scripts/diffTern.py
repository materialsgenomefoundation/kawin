from kawin.diffusion.Diffusion import HomogenizationModel
from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
import matplotlib.pyplot as plt
from pycalphad.plot import triangular

plotResults = True

if plotResults:
    fig = plt.figure()
    ax = fig.add_subplot(projection='triangular')
    m = HomogenizationModel.load('diff_scripts//diffoutput//difftern_750_grad1.csv')
    ax.plot(m.x[0,:], m.x[1,:])

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    ax[0].plot([0, 0.5e-5, 1e-5], [0, 0.05, 0.8])
    ax[0].plot([0, 0.5e-5, 1e-5], [0, 0.9, 0.2])
    ax[0].plot([0, 0.5e-5, 1e-5], [1, 0.05, 0])
    m.plotPhases(ax[1])
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m = HomogenizationModel.load('diff_scripts//diffoutput//difftern_450_wu.csv')
    m.plot(ax[0], True, linewidth=1, linestyle='--')
    m.plotPhases(ax[1], linewidth=1, linestyle='--')

    m = HomogenizationModel.load('diff_scripts//diffoutput//difftern_450_wl.csv')
    m.plot(ax[0], True, linewidth=1, linestyle='--')
    m.plotPhases(ax[1], linewidth=1, linestyle='--')

    m = HomogenizationModel.load('diff_scripts//diffoutput//difftern_450_hu.csv')
    m.plot(ax[0], True, linewidth=1, linestyle='-')
    m.plotPhases(ax[1], linewidth=1, linestyle='-')

    m = HomogenizationModel.load('diff_scripts//diffoutput//difftern_450_hl.csv')
    m.plot(ax[0], True, linewidth=1, linestyle='-')
    m.plotPhases(ax[1], linewidth=1, linestyle='-')
    plt.show()

if not plotResults:

    therm = MulticomponentThermodynamics('database//CoCrFeNiV_MOB_V2.TDB', ['NI', 'CR', 'V'], ['FCC_A1', 'BCC_A2'])

    m = HomogenizationModel([0, 1e-5], 150, ['NI', 'CR', 'V'], ['FCC_A1', 'BCC_A2'])
    m.setTemperature(900+273.15)
    m.setThermodynamics(therm)

    #m.setCompositionStep(0.5, 0.5, 0.5e-5, 'CR')
    #m.setCompositionStep(0.05, 0.45, 0.5e-5, 'V')
    #m.eps = 0.05

    m.setCompositionInBounds(0.4, 0, 1e-5/3, 'CR')
    m.setCompositionInBounds(0.9, 1e-5/3, 2e-5/3, 'CR')
    m.setCompositionInBounds(m.minComposition, 2e-5/3, 1e-5, 'CR')

    m.setCompositionInBounds(m.minComposition, 0, 1e-5/3, 'V')
    m.setCompositionInBounds(0.05, 1e-5/3, 2e-5/3, 'V')
    m.setCompositionInBounds(1 - 2*m.minComposition, 2e-5/3, 1e-5, 'V')

    #m.setCompositionProfile([0, 0.5e-5, 1e-5], [m.minComposition, 0.9, 0.2], 'CR')
    #m.setCompositionProfile([0, 0.5e-5, 1e-5], [1-2*m.minComposition, 0.05, m.minComposition], 'V')

    #m.setMobilityFunction('labrynth')
    #m.setLabyrinthFactor(1)

    m.setMobilityFunction('hashin upper')

    m.setHashSensitivity(3)
    
    #m.setup()
    #m.getFluxes()
    
    m.solve(3600, True, 100)
    #m.save('diff_scripts//diffoutput//difftern_450_hl', toCSV=True)

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    m.plotPhases(ax[1])
    plt.show()
    
    