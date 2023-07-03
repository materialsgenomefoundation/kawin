from kawin.thermo import GeneralThermodynamics
from kawin.diffusion import HomogenizationModel
from kawin.solver import SolverType
import matplotlib.pyplot as plt
import numpy as np

solve = True

elements = ['FE', 'CR', 'NI']
phases = ['FCC_A1', 'BCC_A2']
ml = HomogenizationModel([-5e-4, 5e-4], 200, elements, phases)
therm = GeneralThermodynamics('examples//FeCrNi.tdb', elements, phases)
ml.setThermodynamics(therm)
ml.setTemperature(1100+273.15)
#tfunc = lambda x, t: np.interp(x, [-5e-4, 5e-4], [900+273.15, 1100+273.15])
#tfunc = lambda x, t: (1 / (1+np.exp(-x/5e-5)))*400+800+273.15
#ml.setTemperatureFunction(tfunc)

#ml.setTemperatureArray([0, 100], [900+273.15, 1100+273.15])

if solve:
    ml.setCompositionStep(0.257, 0.423, 0, 'CR')
    ml.setCompositionStep(0.065, 0.276, 0, 'NI')
    ml.eps = 0.01

    ml.enableRecording()

    ml.setMobilityFunction('hashin lower')
    #ml.solve2(100*3600, verbose=True, vIt=100)
    ml.solve(100*3600, solverType = SolverType.EXPLICITEULER, verbose=True, vIt=100)
    #ml.saveRecordedMesh('misc_scripts//outputs//FeCrNi_homogenization')

else:
    ml.loadRecordedMesh('misc_scripts//outputs//FeCrNi_homogenization.npz')
    #ml.setMeshtoRecordedTime(100*3600)

fig, ax = plt.subplots(1,3, figsize=(15,4))
t = np.linspace(0, 100*3600, 2)
for i in range(len(t)):
    ml.setMeshtoRecordedTime(t[i])
    ml.plot(ax[0], plotElement='CR', label=str(t[i]))
    ml.plot(ax[1], plotElement='NI', label=str(t[i]))
    ml.plotPhases(ax[2], plotPhase='BCC_A2', label=str(t[i]))

ax[0].set_ylabel('Composition CR (%at)')
ax[0].set_ylim([0.2, 0.45])
ax[1].set_ylabel('Composition NI (%at)')
ax[1].set_ylim([0, 0.35])
ax[2].set_ylabel(r'Fraction $\alpha$')
ax[2].set_ylim([0, 0.8])
plt.tight_layout()
plt.show()