from kawin.thermo import MulticomponentThermodynamics
import numpy as np
import matplotlib.pyplot as plt

elements = ['AL', 'MG', 'SI']
phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']
therm = MulticomponentThermodynamics('examples//AlMgSi.tdb', elements, phases)

N = 100
xs = [[0.0072, 0.0057], [0.002929, 0.000874]]
ts = np.linspace(175, 250, N) + 273.15

fig, ax = plt.subplots(1,len(xs))
linestyles = ['-', '--', ':']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

for i in range(len(xs)):
    therm.clearCache()
    x = np.array([xs[i] for _ in range(N)])
    for p in therm.phases[1:]:
        dgs, xbs = therm.getDrivingForce(x, ts, p, returnComp=True)
        ax[i].plot(ts, dgs, label=p)
ax[0].legend()

plt.show()