from kawin.coupling.Strength import StrengthModel
import numpy as np
import matplotlib.pyplot as plt

sm = StrengthModel()
G, b, nu = 79.3e9, 0.25e-9, 1/3
bp, ri = b, 2*b
eps, Gp = 0.001, 70e9
yAPB, ySFM, ySFP, gamma = 0.04, 0.1, 0.05, 0.5
sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
sm.setCoherencyParameters(eps)
sm.setModulusParameters(Gp, phase='all')
sm.setAPBParameters(yAPB, phase='all')
sm.setSFEParameters(ySFM, ySFP, bp, phase='all')
sm.setInterfacialParameters(gamma)
sm.setTaylorFactor(1)

rs = np.linspace(0, 100e-9, 100)
Ls = 300e-9 - 2*rs

fig, ax = plt.subplots(3, 2)
sm.plotPrecipitateStrengthOverR(ax, rs, Ls, plotContributions=True)
ax[0,0].set_ylim([0, 100])
ax[0,1].set_ylim([0, 100])
ax[1,0].set_ylim([0, 100])
ax[1,1].set_ylim([0, 100])
ax[2,0].set_ylim([0, 12])
ax[2,1].set_ylim([0, 250])
plt.show()