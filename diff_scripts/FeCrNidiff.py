from kawin.diffusion.Diffusion import HomogenizationModel
from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
import numpy as np
import matplotlib.pyplot as plt

from pycalphad import equilibrium, Database, variables as v
from pycalphad import ternplot

therm = MulticomponentThermodynamics('database//FeCrNi.tdb', ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])

conds = {v.T: 1100+273.15, v.P:101325, v.X('CR'): (0,1,0.015), v.X('NI'): (0,1,0.015)}

ternplot(therm.db, ['FE', 'CR', 'NI', 'VA'], ['FCC_A1', 'BCC_A2'], conds, x=v.X('CR'), y=v.X('NI'))
plt.show()

m = HomogenizationModel([-5e-4, 5e-4], 200, ['FE', 'CR', 'NI'], ['FCC_A1', 'BCC_A2'])
m.setCompositionStep(.257, .423, 0, 'CR')
m.setCompositionStep(.065, .276, 0, 'NI')

m.setTemperature(1100+273.15)
m.setThermodynamics(therm)
m.eps = 0.01

m.setMobilityFunction('hashin lower')

m.solve(100*3600, True, 100)

fig, ax = plt.subplots(1,2, figsize=(10,4))
m.plot(ax[0], True)
m.plotPhases(ax[1])

m.reset()

m.setCompositionStep(.257, .423, 0, 'CR')
m.setCompositionStep(.065, .276, 0, 'NI')
m.setMobilityFunction('hashin upper')
m.solve(100*3600, True, 100)

m.plot(ax[0], True, linestyle='--')
m.plotPhases(ax[1], linestyle='--')

plt.show()