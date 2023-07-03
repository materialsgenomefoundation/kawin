from kawin.diffusion.Diffusion import HomogenizationModel
from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
import matplotlib.pyplot as plt
from pycalphad.plot import triangular
from pycalphad.plot.utils import phase_legend
from pycalphad import Database, ternplot, variables as v

plotResults = False

if plotResults:
    ms = []
    ms.append(HomogenizationModel.load('diff_scripts//VCrtoNiCr//V-Ni60Crlong.csv'))
    ms.append(HomogenizationModel.load('diff_scripts//VCrtoNiCr//V-Ni70Crlong.csv'))
    ms.append(HomogenizationModel.load('diff_scripts//VCrtoNiCr//V-Ni80Crlong.csv'))
    ms.append(HomogenizationModel.load('diff_scripts//VCrtoNiCr//V25Cr-Ni60Crlong.csv'))
    ms.append(HomogenizationModel.load('diff_scripts//VCrtoNiCr//V25Cr-Ni70Crlong.csv'))
    ms.append(HomogenizationModel.load('diff_scripts//VCrtoNiCr//V25Cr-Ni80Crlong.csv'))
    labels = ['V|Ni60Cr', 'V|Ni70Cr', 'V|Ni80Cr', 'V25Cr|Ni60Cr', 'V25Cr|Ni70Cr', 'V25Cr|Ni80Cr']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='triangular')

    db = Database('database//CoCrFeNiV_MOB_V2.TDB')
    phases = ['FCC_A1', 'BCC_A2', 'SIGMA']
    conds = {v.T: 900+273.15, v.P:101325, v.X('NI'): (0,1,0.015), v.X('V'): (0,1,0.015)}
    ternplot(db, ['NI', 'CR', 'V', 'VA'], phases, conds, x=v.X('NI'), y=v.X('V'), ax = ax)

    handles, _ = phase_legend(phases)

    lines = []
    for i in range(len(ms)):
        ln, = ax.plot(ms[i].getX('NI'), ms[i].getX('V'), label = labels[i], linewidth=2, color=colors[i])
        lines.append(ln)

    ax.legend(handles = handles + lines)
    #plt.show()
    
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    for i in range(len(ms)):
        #ax[0].plot(ms[i].z, ms[i].getP('FCC_A1'), color=colors[i], linestyle='-')
        #ax[0].plot(ms[i].z, ms[i].getP('BCC_A2'), color=colors[i], linestyle='-')
        ax[0].plot(ms[i].z, ms[i].getP('SIGMA'), color=colors[i], linestyle='-')

        #ax[1].plot(ms[i].z, ms[i].getX('NI'), color=colors[i], linestyle='-')
        ax[1].plot(ms[i].z, ms[i].getX('CR'), color=colors[i], linestyle='-')
        #ax[1].plot(ms[i].z, ms[i].getX('V'), color=colors[i], linestyle='-')
    '''
    plt.show()
    

if not plotResults:

    therm = MulticomponentThermodynamics('database//CoCrFeNiV_MOB_V2.TDB', ['NI', 'CR', 'V'], ['FCC_A1', 'BCC_A2', 'SIGMA'])
    
    d = 2e-5
    m = HomogenizationModel([0, d], 500, ['NI', 'CR', 'V'], ['FCC_A1', 'BCC_A2', 'SIGMA'])
    m.setTemperature(900+273.15)
    m.setThermodynamics(therm)

    m.setCompositionStep(0.25, 0.6, d/2, 'CR')
    m.setCompositionStep(0.75, 0, d/2, 'V')

    m.setMobilityFunction('hashin upper')

    m.setHashSensitivity(3)
    
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    m.plotPhases(ax[1])
    plt.show()
    
    m.solve(3600, True, 100)
    #m.save('diff_scripts//VCrtoNiCr//V25Cr-Ni60Crlong', toCSV=True)

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    m.plot(ax[0], True)
    m.plotPhases(ax[1])
    plt.show()
    
    
    