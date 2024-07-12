from kawin.thermo import MulticomponentThermodynamics
import numpy as np
import matplotlib.pyplot as plt

elements = ['NI', 'FE', 'CR']
phases = ['FCC_A1', 'BCC_A2']
therm = MulticomponentThermodynamics('examples//FeCrNi.tdb', elements, phases)
'''
N = 100
x0 = [0.1, 0.4]
xs = np.array([x0 for _ in range(N)])
ts = np.linspace(600, 1100, N) + 273.15

methods = ['tangent', 'sampling', 'approximate']
fig, ax = plt.subplots(1,3)
linestyles = ['-', '--', ':']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
for m in methods:
    print(m)
    therm.clearCache()
    therm.setDrivingForceMethod(m)
    for p in therm.phases[1:]:
        dgs, xbs = therm.getDrivingForce(xs, ts, p, returnComp=True)

        mIndex = methods.index(m)
        pIndex = phases.index(p)-1
        ax[0].plot(ts, dgs, color=colors[mIndex], linestyle = linestyles[mIndex], label = m)
        ax[1].plot(ts, xbs[:,0], color=colors[mIndex], linestyle = linestyles[mIndex])
        ax[2].plot(ts, xbs[:,1], color=colors[mIndex], linestyle = linestyles[mIndex])

ax[0].legend()
ax[0].set_xlabel('Temperature (K)')
ax[1].set_xlabel('Temperature (K)')
ax[2].set_xlabel('Temperature (K)')
ax[0].set_ylabel('Driving Force (J/mol)')
ax[1].set_ylabel('x (Fe, BCC_A2)')
ax[2].set_ylabel('x (Cr, BCC_A2)')
plt.show()
'''
N = 1000
xfe = np.ones(N)*0.1
xcr = np.linspace(0.05, 0.4, N)
xs = np.array([xfe, xcr]).T
ts = np.ones(N) * 800

methods = ['tangent', 'sampling', 'approximate']
fig, ax = plt.subplots(1,3)
linestyles = ['-', '--', ':']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
for m in methods:
    print(m)
    therm.clearCache()
    therm.setDrivingForceMethod(m)
    for p in therm.phases[1:]:
        dgs, xbs = therm.getDrivingForce(xs, ts, p, returnComp=True)

        mIndex = methods.index(m)
        pIndex = phases.index(p)-1
        ax[0].plot(xs[:,1], dgs, linestyle=linestyles[mIndex], label=m)
        ax[1].plot(xs[:,1], xbs[:,0], linestyle=linestyles[mIndex])
        ax[2].plot(xs[:,1], xbs[:,1], linestyle=linestyles[mIndex])

ax[0].legend()
ax[0].set_xlabel('x (Cr)')
ax[1].set_xlabel('x (Cr)')
ax[2].set_xlabel('x (Cr)')
ax[0].set_ylabel('Driving Force (J/mol)')
ax[1].set_ylabel('x (Fe, BCC_A2)')
ax[2].set_ylabel('x (Cr, BCC_A2)')
plt.show()