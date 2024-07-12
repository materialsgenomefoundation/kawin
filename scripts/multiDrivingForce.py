from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics
from pycalphad import equilibrium, variables as v
from kawin.thermo.LocalEquilibrium import local_equilibrium
import numpy as np
import matplotlib.pyplot as plt
import time

def timeDG(therm, method, x, ts):
    therm.clearCache()
    therm.setDrivingForceMethod(method)
    t0 = time.time()
    dgs, xbs = therm.getDrivingForce(x, ts, returnComp=True, training=False)
    tf = time.time()
    print(method, '{:.5f}'.format(tf - t0))

    return dgs, xbs

binary = False

if binary:
    el = ['AL', 'ZR']
    therm = BinaryThermodynamics('examples//AlScZr.tdb', ['AL', 'ZR'], ['FCC_A1', 'AL3ZR'])
    x = [4e-3]
    T = 700

    ts = np.linspace(400, 700, 100)
    fig, ax = plt.subplots(1,2)

    methods = ['tangent', 'approximate', 'sampling']
    for m in methods:
        dgs, xbs = timeDG(therm, m, np.ones(100)*x, ts)
        ax[0].plot(ts, dgs)
        ax[1].plot(ts, xbs)
    plt.show()
else:
    el = ['NI', 'CR', 'AL']
    therm = MulticomponentThermodynamics('examples//NiCrAl.tdb', ['NI', 'CR', 'AL'], ['FCC_A1', 'FCC_L12'])

    '''
    x = [0.08, 0.1]
    T = 700

    ts = np.linspace(700, 1300, 100)
    #ts = np.linspace(1300, 700, 100)
    #print(ts)
    xs = np.array([x for _ in range(len(ts))])
    
    fig, ax = plt.subplots(1,3)
    methods = ['tangent', 'approximate', 'curvature', 'sampling']
    linestyles = ['-', '--', ':', '-.']
    for m in methods:
        dgs, xbs = timeDG(therm, m, xs, ts)
        mIndex = methods.index(m)
        ax[0].plot(ts, dgs, linestyle=linestyles[mIndex], label=m)
        ax[1].plot(ts, xbs[:,0], linestyle=linestyles[mIndex])
        ax[2].plot(ts, xbs[:,1], linestyle=linestyles[mIndex])

    ax[0].legend()
    ax[0].set_xlabel('Temperature (K)')
    ax[1].set_xlabel('Temperature (K)')
    ax[2].set_xlabel('Temperature (K)')
    ax[0].set_ylabel('Driving Force (J/mol)')
    ax[1].set_ylabel('x (Cr, BCC_A2)')
    ax[2].set_ylabel('x (Al, BCC_A2)')

    plt.show()
    '''
    T = 700
    ts = np.ones(100)*T
    xAl = np.linspace(0.02, 0.15, 100)
    xCr = np.ones(100)*0.08
    xs = np.array([xCr, xAl]).T

    fig, ax = plt.subplots(1,3)
    methods = ['tangent', 'approximate', 'curvature', 'sampling']
    linestyles = ['-', '--', ':', '-.']
    for m in methods:
        dgs, xbs = timeDG(therm, m, xs, ts)
        mIndex = methods.index(m)
        ax[0].plot(xs[:,1], dgs, linestyle=linestyles[mIndex], label=m)
        ax[1].plot(xs[:,1], xbs[:,0], linestyle=linestyles[mIndex])
        ax[2].plot(xs[:,1], xbs[:,1], linestyle=linestyles[mIndex])

    ax[0].legend()
    ax[0].set_xlabel('x (Al)')
    ax[1].set_xlabel('x (Al)')
    ax[2].set_xlabel('x (Al)')
    ax[0].set_ylabel('Driving Force (J/mol)')
    ax[1].set_ylabel('x (Cr, BCC_A2)')
    ax[2].set_ylabel('x (Al, BCC_A2)')

    plt.show()
    
                
    
    