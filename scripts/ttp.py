import numpy as np
from kawin.precipitation import PrecipitateModel, VolumeParameter
from kawin.precipitation.StoppingConditions import VolumeFractionCondition, Inequality
from kawin.thermo import BinaryThermodynamics
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import time
from kawin.precipitation import TTPCalculator

if __name__ == '__main__':
    #Set up precipitation model
    therm = BinaryThermodynamics('examples//AlScZr.tdb', ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'])
    therm.setGuessComposition(0.24)

    model = PrecipitateModel()

    model.setInitialComposition(4e-3)
    model.setTemperature(400 + 273.15)
    model.setInterfacialEnergy(0.1)

    Diff = lambda x, T: 0.0768 * np.exp(-242000 / (8.314 * T))
    model.setDiffusivity(Diff)

    a = 0.405e-9        #Lattice parameter
    atomsPerCell = 4    #Atoms in an FCC unit cell
    model.setVolumeAlpha(a**3, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)
    model.setVolumeBeta(a**3, VolumeParameter.ATOMIC_VOLUME, atomsPerCell)

    model.setNucleationDensity(grainSize = 1, dislocationDensity = 1e15)
    model.setNucleationSite('dislocations')

    model.setThermodynamics(therm, addDiffusivity=False)

    #Set up stopping conditions -> when volume fraction is greater than 0.5%, 1.0% and 1.5%
    vfLow = VolumeFractionCondition(Inequality.GREATER_THAN, 0.005)
    vfMid = VolumeFractionCondition(Inequality.GREATER_THAN, 0.01)
    vfHigh = VolumeFractionCondition(Inequality.GREATER_THAN, 0.015)
    stopConds = [vfLow, vfMid, vfHigh]

    #Calculate TTP diagram with 4 cores
    t0 = time.time()
    ttp = TTPCalculator(model, stopConds)
    pool = Pool(8)
    ttp.calculateTTP(Tlow=450+273.15, Thigh=550+273.15, Tsteps=20, maxTime=100*3600, pool=pool)
    tf = time.time()
    print('Time: {:.3f}'.format(tf-t0))

    #Plot TTP diagram
    fig, ax = plt.subplots(1,1)
    ttp.plot(ax, labels=['0.5 %', '1.0 %', '1.5 %'], xlim=[1, 1e6])
    plt.show()