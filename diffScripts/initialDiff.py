from kawin.Mesh import Mesh1D
import matplotlib.pyplot as plt
import numpy as np

el = ['NI', 'AL', 'CR']
phases = ['FCC']
z = [0, 1]
N = 100

m = Mesh1D(z, N, el, phases)
m.setCompositionStep(0.3, 0.6, 0.5, 'CR')
m.setCompositionSingle(0.5, 0.3, 'AL')

D = 0.1
dt = 0.5 * m.dz**2 / D
for i in range(10):
    fig, ax = plt.subplots(1, 1)
    m.plot(ax, True)
    plt.show()

    for j in range(100):
        f = np.zeros((len(el[1:]), N-1))
        f[0,:] = -D * (m.x[0,1:] - m.x[0,:-1]) / m.dz
        f[1,:] = -D * (m.x[1,1:] - m.x[1,:-1]) / m.dz
        m.update(f, dt)