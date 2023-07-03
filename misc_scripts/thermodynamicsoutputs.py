import numpy as np
import matplotlib.pyplot as plt

from kawin.thermo.Thermodynamics import MulticomponentThermodynamics

therm = MulticomponentThermodynamics('database//alcrni-dupin_diff.tdb', ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'])

print(therm.mobModels['DIS_FCC_A1'].diffusivity)

print(therm.getInterdiffusivity([0.2, 0.2], 1073))
print(therm.getTracerDiffusivity([0.2, 0.2], 1073))