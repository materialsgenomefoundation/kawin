from kawin.diffusion.Diffusion import HomogenizationModel
from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
import matplotlib.pyplot as plt

plotResults = False

if plotResults:
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='triangular')
    m = HomogenizationModel.load('scripts//diffbin_200_lab1.csv')

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    m.plotPhases(ax[1])

    m = HomogenizationModel.load('scripts//diffbin_200_lab2.csv')

    #fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    m.plotPhases(ax[1])
    plt.show()

if not plotResults:

    therm = MulticomponentThermodynamics('database//CoCrFeNiV_MOB_V2.TDB', ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])

    m = HomogenizationModel([0, 5e-6], 500, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
    m.setTemperature(900+273.15)
    m.setThermodynamics(therm)
    m.setHashSensitivity(3)
    m.setCompositionStep(0.2, 0.6, 2.5e-6, 'CR')

    m.setLabyrinthFactor(2)
    m.setMobilityFunction('lab')

    m.setMobilityFunction('hashin upper')

    m.solve(3600, True, 100)
    m.save('diff_scripts//diffoutput//diffbin_500_hu', toCSV=True)

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    m.plotPhases(ax[1])
    plt.show()