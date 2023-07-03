from kawin.solver.Solver import DESolver
import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.reset()
        self.solver = DESolver(defaultDt = 0.05)
        self.solver.postProcess = self.postProcess

    def reset(self):
        self.xs = np.zeros((1,2))
        self.y = np.array([0])
        self.t = np.array([0])

    def postProcess(self, t, X):
        self.xs = np.concatenate((self.xs, [X[0]]))
        self.y = np.concatenate((self.y, [X[1]]))
        self.t = np.concatenate((self.t, [t]))

    def f(self, t, X):
        dy1dt = -np.sin(t)*t + np.cos(t)
        dy2dt = -np.sin(t)*np.sin(X[0][0])*dy1dt + np.cos(t)*np.cos(X[0][0])
        dy3dt = (dy1dt + dy2dt)*np.exp(-X[0][1]) - (X[0][0] + X[0][1])*np.exp(-X[0][1]) * dy2dt
        return np.array([dy1dt, dy2dt]), dy3dt
    
    def solve(self, tf):
        self.solver.solve(self.f, self.t[-1], [self.xs[-1], self.y[-1]], tf)
    
t = np.linspace(0, 20, 1000)
y1func = lambda t: np.cos(t)*t
y2func = lambda t, y1: np.sin(t)*np.cos(y1)
y3func = lambda t, y1, y2: (y1+y2)*np.exp(-y2)

y1 = y1func(t)
y2 = y2func(t, y1)
y3 = y3func(t, y1, y2)

fig, ax = plt.subplots(1, 2)
ax[0].plot(t, y1, t, y2, t, y3, color='k')


m = Model()
for i in range(2):
    m.solve(20)

    
    ax[0].plot(m.t, m.xs[:,0], linestyle='--')
    ax[0].plot(m.t, m.xs[:,1], linestyle='--')
    ax[0].plot(m.t, m.y, linestyle='--')

    y1 = y1func(m.t)
    y2 = y2func(m.t, y1)
    y3 = y3func(m.t, y1, y2)
    ax[1].plot(m.t, y1 - m.xs[:,0])
    ax[1].plot(m.t, y2 - m.xs[:,1])
    ax[1].plot(m.t, y3 - m.y)

    m.reset()
    m.solver.setIterator('explicit euler')
plt.show()



