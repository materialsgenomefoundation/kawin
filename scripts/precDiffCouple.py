import numpy as np
from kawin.thermo import MulticomponentThermodynamics
from kawin.diffusion import SinglePhaseModel
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.solver import SolverType
import matplotlib.pyplot as plt
from kawin.GenericModel import Coupler
from typing import List

class CustomCoupler(Coupler):
    def __init__(self, diffModel : SinglePhaseModel, precModels : List[PrecipitateModel]):
        self.diffModel = diffModel
        self.precModels = precModels
        super().__init__([diffModel] + precModels)

    def couplePostProcess(self):
        #Compostion at a node is affected by both fluxes in/out of the node and precipitation
        #We want to get the change predicted by both the diffusion and precipitate models and
        #   apply it to the other model
        if len(self.time) >= 2:
            #Get change in composition from diffusion model
            diffX = self.diffModel._recordedX[-1] - self.diffModel._recordedX[-2]
            for i in range(len(self.precModels)):
                #Get change in composition from precipitate model
                precX = self.precModels[i].xComp[-1] - self.precModels[i].xComp[-2]
                self.precModels[i].xComp[-1] += diffX[:,i]
                self.diffModel.x[:,i] += precX

    def printStatus(self, iteration, modelTime, simTimeElapsed):
        super().printStatus(iteration, modelTime, simTimeElapsed)
        xstr1 = ''
        xstr2 = ''
        rstr = ''
        for i in range(len(self.precModels)):
            xstr1 += '{:.2e}\t'.format(self.diffModel.x[0,i])
            xstr2 += '{:.2e}\t'.format(self.diffModel.x[1,i])
            rstr += '{:.2e}\t'.format(self.precModels[i].avgR[-1,0])
        print(xstr1)
        print(xstr2)
        print(rstr)

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
    model = PrecipitateModel(elements=['CR', 'AL'])
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

coupled = CustomCoupler(diffModel, precModels)
coupled.solve(3600, solverType=SolverType.EXPLICITEULER, verbose=True, vIt = 10)

fig, axL = plt.subplots(1, 1)
axL, axR = diffModel.plotTwoAxis(['AL'], ['CR'], zScale = 1/1000, axL = axL)
plt.show()

rs = []
for i in range(N):
    rs.append(precModels[i].avgR[-1,0])
plt.plot(diffModel.z, rs)
plt.show()




