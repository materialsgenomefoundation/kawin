from kawin.thermo import MulticomponentThermodynamics
from kawin.precipitation import PrecipitateModel, VolumeParameter
import matplotlib.pyplot as plt
from kawin.solver import SolverType
import numpy as np

phases = ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE']
therm = MulticomponentThermodynamics('examples//AlMgSi.tdb', ['AL', 'MG', 'SI'], phases, drivingForceMethod='approximate')

model = PrecipitateModel(phases=phases[1:], elements=['MG', 'SI'])

model.setInitialComposition([0.0072, 0.0057])
model.setVolumeAlpha(1e-5, VolumeParameter.MOLAR_VOLUME, 4)

lowTemp = 175+273.15
highTemp = 250+273.15
model.setTemperature(([0, 16, 17], [lowTemp, lowTemp, highTemp]))

print(therm.getEq([0.0072, 0.0057], lowTemp))
print(therm.getEq([0.0072, 0.0057], highTemp))


gamma = {
    'MGSI_B_P': 0.18,
    'MG5SI6_B_DP': 0.084,
    'B_PRIME_L': 0.18,
    'U1_PHASE': 0.18,
    'U2_PHASE': 0.18
        }

for i in range(len(phases)-1):
    model.setInterfacialEnergy(gamma[phases[i+1]], phase=phases[i+1])
    model.setVolumeBeta(1e-5, VolumeParameter.MOLAR_VOLUME, 4, phase=phases[i+1])
    model.setThermodynamics(therm, phase=phases[i+1])
    model.PBM[i].setRecording(True)

model.solve(25*3600, solverType=SolverType.RK4, verbose=True, vIt=500)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

model.plot(axes[0,0], 'Total Precipitate Density', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
model.plot(axes[0,0], 'Precipitate Density', timeUnits='h')
axes[0,0].set_ylim([1e5, 1e25])
axes[0,0].set_xscale('linear')
axes[0,0].set_yscale('log')

model.plot(axes[0,1], 'Total Volume Fraction', timeUnits='h', label='Total', color='k', linestyle=':', zorder=6)
model.plot(axes[0,1], 'Volume Fraction', timeUnits='h')
axes[0,1].set_xscale('linear')

model.plot(axes[1,0], 'Average Radius', timeUnits='h')
axes[1,0].set_xscale('linear')

model.plot(axes[1,1], 'Composition', timeUnits='h')
axes[1,1].set_xscale('linear')

fig.tight_layout()

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.log10(np.abs(model.PBM[0]._recordedPSD) + 1))
ax[1].imshow(np.log10(np.abs(model.PBM[1]._recordedPSD) + 1))

plt.show()
