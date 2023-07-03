from kawin.thermo import GeneralThermodynamics
from kawin.diffusion import SinglePhaseModel
from kawin.solver import SolverType
import matplotlib.pyplot as plt
import numpy as np

therm = GeneralThermodynamics('examples//NiCrAl.tdb', ['NI', 'CR', 'AL'], ['FCC_A1'])

#Define mesh spanning between -1mm to 1mm with 50 volume elements
m = SinglePhaseModel([-1e-3, 1e-3], 100, ['NI', 'CR', 'AL'], ['FCC_A1'])

#Define Cr and Al composition, with step-wise change at z=0
m.setCompositionStep(0.077, 0.359, 0, 'CR')
m.setCompositionStep(0.054, 0.062, 0, 'AL')

m.enableRecording()

#m.useCache(False)

m.setThermodynamics(therm)
m.setTemperature(1200 + 273.15)
#tfunc = lambda x, t: np.interp(x, [-10e-4, 0, 10e-4], [1000+273.15, 1200+273.15, 800+273.15])
#m.setTemperatureFunction(tfunc)
#m.setTemperatureArray([0, 100], [1000+273.15, 1200+273.15])

#solver = DESolver(SolverType.EXPLICITEULER)
#solver.setDtFunc(m.getDt)
#solver.preProcess = m.preProcess
#solver.postProcess = m.postProcess

#m.setup()
#solver.solve(m.getdXdt, 0, [m.x], 100*3600)

m.solve(100*3600, solverType=SolverType.RK4, verbose=True, vIt=100)

#m.saveRecordedMesh('misc_scripts//outputs//NiCrAldiff')

#m2 = SinglePhaseModel([-1e-3, 1e-3], 100, ['NI', 'CR', 'AL'], ['FCC_A1'])
#m2.loadRecordedMesh('misc_scripts//outputs//NiCrAldiff.npz')

fig, axL = plt.subplots(1, 1)
axR = axL.twinx()

t = np.linspace(0, 100*3600, 2)
for i in range(len(t)):
    m.setMeshtoRecordedTime(t[i])
    m.plot(axL, plotElement='AL', zScale=1/1000)
    m.plot(axR, plotElement='CR', zScale=1/1000)
    #axL, axR = m.plotTwoAxis(axL, ['AL'], ['CR'], zScale = 1/1000, axR = axR)
axL.set_xlim([-1, 1])
axL.set_xlabel('Distance (mm)')
axL.set_ylim([0, 0.1])
axR.set_ylim([0, 0.4])
plt.show()