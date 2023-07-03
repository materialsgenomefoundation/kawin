import numpy as np
import matplotlib.pyplot as plt
from kawin.diffusion.Diffusion import SinglePhaseModel
from kawin.thermo.Thermodynamics import GeneralThermodynamics

therm = GeneralThermodynamics('database//alcrni-dupin_mob.tdb', ['NI', 'CR', 'AL'], ['FCC_A1'])

m = SinglePhaseModel([-1e-3, 1e-3], 100, ['NI', 'CR', 'AL'], ['FCC_A1'])

m.setCompositionStep(0.8, 0.2, 0.5e-3, 'CR')
m.setCompositionStep(0.1, 0.5, 0, 'AL')

m.setup()