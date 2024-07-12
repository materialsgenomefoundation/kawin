from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.thermo import BinaryThermodynamics
import numpy as np
from kawin.precipitation.coupling import StrengthModel
from kawin.solver import SolverType

import matplotlib.pyplot as plt
import numpy as np

load = False

therm = BinaryThermodynamics('examples//AlScZr.tdb', ['AL', 'SC'], ['FCC_A1', 'AL3SC'])
therm.setGuessComposition(0.24)

sm = StrengthModel()
sm.setDislocationParameters(G=25.4e9, b=0.286e-9, nu=0.34)
sm.setCoherencyParameters(eps=2/3*0.0125)
sm.setModulusParameters(Gp=67.9e9)
sm.setAPBParameters(yAPB=0.5)
sm.setInterfacialParameters(gamma=0.1)

if load:
    model = PrecipitateModel.load('scripts/outputs/AlSc_prec.npz')
    sm.load('scripts/outputs/AlSc_strength.npz')
else:
    model = PrecipitateModel()

    model.setInitialComposition(0.002)
    model.setTemperature(400+273.15)
    model.setInterfacialEnergy(0.1)

    model.PBM[0].enableRecording()

    Va = (0.405e-9)**3
    Vb = (0.4196e-9)**3
    model.setVolumeAlpha(Va, VolumeParameter.ATOMIC_VOLUME, 4)
    model.setVolumeBeta(Vb, VolumeParameter.ATOMIC_VOLUME, 4)

    diff = lambda x, T: 1.9e-4 * np.exp(-164000 / (8.314*T)) 
    model.setDiffusivity(diff)

    model.setThermodynamics(therm, addDiffusivity=False)

    model.addCouplingModel(sm)

    model.solve(9e5, solverType=SolverType.RK4, verbose=True, vIt=500)

    model.save('scripts/outputs/AlSc_prec')
    sm.save('scripts/outputs/AlSc_strength')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

model.plot(axes[0,0], 'Precipitate Density')
axL = axes[0,0].twinx()
model.plot(axL, 'Composition', color='C1')
model.plot(axes[0,1], 'Driving Force')
model.plot(axes[1,0], 'Average Radius', label='Average Radius')
sm.plotStrength(axes[1,1], model, True)

fig, axes = plt.subplots(2,3)
contributions = [['Coherency', 'modulus', 'SFE'], ['interfacial', 'Orowan', 'All']]
for i in range(2):
    for j in range(3):
        sm.plotPrecipitateStrengthOverTime(axes[i,j], model, contribution=contributions[i][j])

fig.tight_layout()
plt.show()