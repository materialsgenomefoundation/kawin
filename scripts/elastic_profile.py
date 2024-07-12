import matplotlib.pyplot as plt
import numpy as np
from kawin.precipitation import StrainEnergy, ShapeFactor
import cProfile


def func():
    se = StrainEnergy()
    #se.setOhmInverseFunction('numpy')

    #By default, StrainEnergy outputs 0
    #This is changed within the KWN model before the model is solved for
    #However, we can manually change it. For this example, we need to set it to the calculate for ellipsoidal shapes
    se.setEllipsoidal()

    #Set elastic tensor by c11, c12 and c44 values
    se.setElasticConstants(168.4e9, 121.4e9, 75.4e9)

    #Set eigenstrains
    se.setEigenstrain([0.022, 0.022, 0.003])

    #Setup strain energy parameters
    se.setup()

    #Aspect ratio
    aspect = np.linspace(1,2,100)
    #aspect = 1.5

    #Equivalent spherical radius of 4 nm
    rSph = 4e-9 / np.cbrt(aspect)
    r = np.array([rSph, rSph, aspect*rSph]).T

    E = se.strainEnergy(r)

    return aspect, E

cProfile.run('func()')

#r, E = func()
#plt.plot(r, E)
#plt.show()
