from kawin.thermo.Thermodynamics import MulticomponentThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.thermo.Surrogate import MulticomponentSurrogate, generateTrainingPoints
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 12})

plotResults = False
folder = 'paper_scripts//adaptiveBinsOutput//'

bins = ['0.1', '0.25', '0.5', '0.75', '1', '2', '3', '4', '5', '6', '7', '8', 'adap']
binnames = ['0-1', '0-25', '0-5', '0-75', '1', '2', '3', '4', '5', '6', '7', '8', 'adap']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'r', 'g', 'k']
colors = ['mediumblue', 'deepskyblue', 'lightseagreen', 'lime', 'darkgreen', 'olive', 'goldenrod', 'orange', 'red', 'maroon', 'magenta', 'blueviolet', 'black']


if plotResults:
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    for i in range(len(bins)):
        if bins[i] == 'adap':
            ls = '--'
            label = 'Adaptive Bins'
        else:
            ls = '-'
            label = bins[i] + ' nm'
        modelLoad = PrecipitateModel.load(folder + 'NiCrAl_' + binnames[i] + '.npz')

        #modelLoad.plot(axes[0,0], 'Precipitate Density', linewidth=1, label=binnames[i])
        modelLoad.plot(axes[0], 'Precipitate Density', linewidth=1, label=label, color=colors[i], linestyle=ls)

        #modelLoad.plot(axes[0,1], 'Volume Fraction', linewidth=1)
        modelLoad.plot(axes[1], 'Volume Fraction', linewidth=1, label=label, color=colors[i], linestyle=ls)
        
        #modelLoad.plot(axes[1,0], 'Average Radius', linewidth=1, label='Avg. R')
        #modelLoad.plot(axes[1], 'Average Radius', linewidth=1, label='Avg. R')
        
        #modelLoad.plot(axes[1,1], 'Size Distribution Density', linewidth=1, marker='.')

    axes[0].set_ylim([1e16, 1e26])
    axes[0].set_yscale('log')
    axes[1].set_ylim([0, 0.12])
    #axes[1].set_yscale('log')
    
    #axes[0,0].set_ylim([1e10, 1e27])
    #axes[0,0].set_yscale('log')
    #axes[0,1].set_ylim([6e-3, 2e0])
    #axes[0,1].set_yscale('log')
    #axes[1,0].set_ylim([8e-11, 2e-6])
    #axes[1,0].set_yscale('log')
    #axes[0,0].set_xlim([4e-3, 2e6])
    #axes[1,0].set_xlim([4e-3, 2e6])
    #axes[0,1].set_xlim([4e-3, 2e6])
    

    #axes[0,0].legend()
    axes[1].legend(ncol=2)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    x, y = [], []
    with open(folder + 'times.txt', 'r') as file:
        lines = file.readlines()
        for l in range(len(lines)-1):
            data = lines[l].split()
            x.append(float(data[0]))
            y.append(float(data[1]))
        fdata = lines[-1].split()
        adap = float(fdata[1])
    plt.plot(x, y, marker='o', label='Fixed Bin Size')
    plt.xlim([0.1, 10])
    plt.xscale('log')
    plt.plot([0.1, 10], [adap, adap], label='Adaptive Bin Size')
    plt.ylabel('Time (s)')
    plt.xlabel('Bin size (nm)')
    plt.ylim([0, 400])
    plt.legend()
    plt.show()

else:
    binIndex = 0
    elements = ['NI', 'AL', 'CR']
    phases = ['FCC_A1', 'FCC_L12']
    therm = MulticomponentThermodynamics('paper_scripts//NiCrAl.tdb', elements, phases)

    t0, tf, steps = 1e-2, 1e6, 1e3
    model = PrecipitateModel(t0, tf, steps, elements=['AL', 'CR'])

    model.setInitialComposition([0.098, 0.083])
    model.setInterfacialEnergy(0.012)

    T = 1073
    model.setTemperature(T)

    a = 0.352e-9
    Va = a**3
    Vb = Va
    atomsPerCell = 4
    model.setVaAlpha(Va, atomsPerCell)
    model.setVaBeta(Vb, atomsPerCell)
    print(model.VmAlpha)

    #model.setNucleationSite('dislocations')
    #model.setNucleationDensity(dislocationDensity=5e12)
    model.setNucleationSite('bulk')
    model.setNucleationDensity(bulkN0 = 1e30)

    model.setThermodynamics(therm, removeCache=False)

    if bins[binIndex] == '0.1':
        model.setPBMParameters(cMin = 1e-10, cMax = 1e-8, bins = 100, adaptive = False)
    elif bins[binIndex] == '0.25':
        model.setPBMParameters(cMin = 1e-10, cMax = 2.5e-8, bins = 100, adaptive = False)
    elif bins[binIndex] == '0.5':
        model.setPBMParameters(cMin = 1e-10, cMax = 5e-8, bins = 100, adaptive = False)
    elif bins[binIndex] == '0.75':
        model.setPBMParameters(cMin = 1e-10, cMax = 7.5-8, bins = 100, adaptive = False)
    elif bins[binIndex] == '1':
        model.setPBMParameters(cMin = 1e-10, cMax = 1e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '2':
        model.setPBMParameters(cMin = 1e-10, cMax = 2e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '3':
        model.setPBMParameters(cMin = 1e-10, cMax = 3e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '4':
        model.setPBMParameters(cMin = 1e-10, cMax = 4e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '5':
        model.setPBMParameters(cMin = 1e-10, cMax = 5e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '6':
        model.setPBMParameters(cMin = 1e-10, cMax = 6e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '7':
        model.setPBMParameters(cMin = 1e-10, cMax = 7e-7, bins = 100, adaptive = False)
    elif bins[binIndex] == '8':
        model.setPBMParameters(cMin = 1e-10, cMax = 8e-7, bins = 100, adaptive = False)

    t0 = time.time()
    model.solve(verbose=True, vIt=100)
    tf = time.time()
    model.save(folder + 'NiCrAl_' + binnames[binIndex] + '_2')

    fig, axes = plt.subplots(2, 2, figsize=(8,8))

    modelLoad = PrecipitateModel.load(folder + 'NiCrAl_' + binnames[binIndex] + '_2.npz')
    with open(folder + 'times_2.txt', 'a') as file:
        file.write(bins[binIndex] + '\t' + '{:.5f}'.format(tf-t0) + '\n')

    modelLoad.plot(axes[0,0], 'Precipitate Density', linewidth=2)
    axes[0,0].set_ylim([1e10, 1e27])
    axes[0,0].set_yscale('log')

    #modelLoad.plot(axes[0,1], 'Composition', linewidth=2)
    modelLoad.plot(axes[0,1], 'Volume Fraction', linewidth=2)
    axes[0,1].set_ylim([6e-3, 2e0])
    axes[0,1].set_yscale('log')
    modelLoad.plot(axes[1,0], 'Average Radius', color='C0', linewidth=2, label='Avg. R')
    modelLoad.plot(axes[1,0], 'Critical Radius', color='C1', linewidth=2, linestyle='--', label='R*')
    axes[1,0].legend(loc='upper left')
    modelLoad.plot(axes[1,1], 'Size Distribution Density', linewidth=2, color='C0')
    axes[1,0].set_ylim([8e-11, 2e-6])
    axes[1,0].set_yscale('log')

    axes[0,0].set_xlim([4e-3, 2e6])
    axes[1,0].set_xlim([4e-3, 2e6])
    axes[0,1].set_xlim([4e-3, 2e6])

    fig.tight_layout()
    plt.show()
