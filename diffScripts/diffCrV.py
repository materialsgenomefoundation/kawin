from kawin.Thermodynamics import MulticomponentThermodynamics
from kawin.Diffusion import DiffusionModel
from kawin.Mesh import Mesh1D
import numpy as np
import matplotlib.pyplot as plt
import time

el = ['TI', 'CR', 'V']
el = ['CR', 'V']
therm = MulticomponentThermodynamics('database//CrTiV.tdb', el, ['BCC_A2'])

m = Mesh1D([0, 1e-3], 50, el, ['BCC_A2'])
#m.setCompositionStep(0.0473, 0, 0.5e-3, 'CR')
m.setCompositionStep(1, 0, 0.5e-3, 'V')

d = DiffusionModel(therm, m)
d.setTemperature(1373)

d.solve(10000*3600, True)

fig, ax = plt.subplots(1, 1)
m.plot(ax, True)
ax.set_title(str(d.t/3600))
plt.show()
