import numpy as np
import matplotlib.pyplot as plt

v = 1.3467

dt = 0.01

N = 100
p = np.zeros(N)
x = np.linspace(0, 10, N)
dx = x[1] - x[0]
p[:int(N/2)] = 0.5
p[int(N/2):] = -0.5

i0 = len(p[p>0])
z0 = (x[i0] - x[i0-1]) / (p[i0] - p[i0-1]) * (0 - p[i0-1]) + x[i0-1]

for i in range(100):

    grad = np.zeros(N)
    grad[:-1] = (p[1:]-p[:-1]) / dx
    
    p = p + grad*v*dt

    if i % 10 == 0:
        #Find 0
        i0 = len(p[p>0])
        z = (x[i0] - x[i0-1]) / (p[i0] - p[i0-1]) * (0 - p[i0-1]) + x[i0-1]
        print(i*dt, i*dt*v, z, z-z0)
        plt.plot(x, p)
        plt.show()
