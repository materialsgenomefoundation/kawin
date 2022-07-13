from kawin.Thermodynamics import MulticomponentThermodynamics
from kawin.Diffusion import DiffusionModel
from kawin.Mesh import Mesh1D
import numpy as np
import matplotlib.pyplot as plt
import time

def Alcomp(z):
    al = np.zeros(len(z))
    al[z > 0] = np.linspace(0.03, 0.1, len(z[z>0]))
    al[z <= 0] = np.linspace(0.15, 0.03, len(z[z<=0]))
    return al

def Crcomp(z):
    al = np.zeros(len(z))
    al[z > 0] = np.linspace(0.3, 0.1, len(z[z>0]))
    al[z <= 0] = np.linspace(0.2, 0.3, len(z[z<=0]))
    return al

el = ['NI', 'AL', 'CR']
therm = MulticomponentThermodynamics('database//NiCrAlW.tdb', el, ['FCC_A1'])

m = Mesh1D([-1e-3, 1e-3], 50, el, ['FCC_A1'])
#m.setCompositionFunction(Alcomp, 'AL')
#m.setCompositionFunction(Crcomp, 'CR')
m.setCompositionStep(0.067, 0.048, 0, 'AL')
m.setCompositionStep(0, 0.192, 0, 'CR')

d = DiffusionModel(therm, m)
d.setTemperature(1473)
d.solve(100*3600, True, 10)

fig, ax = plt.subplots(1, 1)
m.plotTwoAxis(ax, ['CR', 'AL'], ['NI'])
ax.set_title(str(d.t/3600))
plt.show()