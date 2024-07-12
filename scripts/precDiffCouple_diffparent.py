import numpy as np
from kawin.thermo import MulticomponentThermodynamics
from kawin.diffusion import SinglePhaseModel
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.solver import SolverType
import matplotlib.pyplot as plt
from kawin.GenericModel import Coupler
from typing import List

class PrecipitateModelChild(PrecipitateModel):
    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meshIndex = index

    def updateCoupledModel(self, model):
        self.xComp[self.n] = model.x[:,self._meshIndex]
        dt = model._recordedTime[-1] - model._recordedTime[-2]
        print(self._meshIndex, dt)

        self.solve(dt, SolverType.EXPLICITEULER, False)

        model.x[:,self._meshIndex] = self.xComp[self.n]

therm = MulticomponentThermodynamics('examples//NiCrAl.tdb', ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'])


T = 1073
N = 20
diffModel = SinglePhaseModel([-1e-4, 1e-4], N, ['NI', 'CR', 'AL'], ['DIS_FCC_A1'])
therm.setMobilityCorrection('all', 1000)

diffModel.setCompositionStep(0.077, 0.359, 0, 'CR')
diffModel.setCompositionStep(0.054, 0.083, 0, 'AL')

diffModel.setThermodynamics(therm)
diffModel.setTemperature(T)
diffModel.enableRecording()

dgs = []
for i in range(N):
    dg, _ = therm.getDrivingForce(diffModel.x[:,i], T)
    dgs.append(dg)
plt.plot(diffModel.z, dgs)
plt.show()

precModels = []
for i in range(N):
    model = PrecipitateModelChild(i, elements=['CR', 'AL'])
    model.setPBMParameters(bins=75, minBins=50, maxBins=100)
    model.setInitialComposition(diffModel.x[:,i])
    model.setInterfacialEnergy(0.023)
    model.setTemperature(1073)

    a = 0.352e-9        #Lattice parameter
    model.setVolumeAlpha(a**3, VolumeParameter.ATOMIC_VOLUME, 4)
    model.setVolumeBeta(a**3, VolumeParameter.ATOMIC_VOLUME, 4)
    model.setNucleationSite('bulk')
    model.setNucleationDensity(bulkN0=1e30)
    therm = MulticomponentThermodynamics('examples//NiCrAl.tdb', ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'])
    model.setThermodynamics(therm)
    precModels.append(model)
    diffModel.addCouplingModel(model)

#coupled = CustomCoupler(diffModel, precModels)
#coupled.solve(3600, solverType=SolverType.EXPLICITEULER, verbose=True, vIt = 10)
diffModel.solve(100*3600, SolverType.EXPLICITEULER, verbose=True, vIt=10)

diffModel.save('diffprec_output/diff_model')
for i in range(len(precModels)):
    precModels[i].save('diffprec_output/prec_model_'+str(i))

fig, axL = plt.subplots(1, 1)
axL, axR = diffModel.plotTwoAxis(['AL'], ['CR'], zScale = 1/1000, axL = axL)
plt.show()

rs = []
for i in range(N):
    rs.append(precModels[i].avgR[-1,0])
plt.plot(diffModel.z, rs)
plt.show()




