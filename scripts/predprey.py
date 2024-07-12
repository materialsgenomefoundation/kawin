from kawin.GenericModel import GenericModel, Coupler
import numpy as np
import matplotlib.pyplot as plt

class ExpModel(GenericModel):
    def __init__(self, alpha, X0):
        self.alpha = alpha
        self.X0 = X0
        self.reset()

    def reset(self):
        self.time = np.zeros(1)
        self.x = np.array([self.X0])

    def getCurrentX(self):
        return self.time[-1], [self.x[-1]]
    
    def getdXdt(self, t, x):
        return [self.alpha * x[0]]
    
    def getDt(self, dXdt):
        return 0.03
    
    def postProcess(self, time, x):
        self.time = np.append(self.time, time)
        self.x = np.append(self.x, x[0])
        return x, False
    
class PredatorPreyModel(Coupler):
    def __init__(self, preyModel, predatorModel, beta, delta):
        self.beta = beta
        self.delta = delta
        self.prey = preyModel
        self.pred = predatorModel
        super().__init__([self.prey, self.pred])

    def coupledXdt(self, t, x, dXdt):
        dXdt[0] -= self.beta * x[0] * x[1]
        dXdt[1] += self.delta * x[0] * x[1]


    
prey = ExpModel(1.1, 2)
pred = ExpModel(-0.4, 10)

prey.solve(5)
pred.solve(5)

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].plot(prey.time, prey.x)
ax[1].plot(pred.time, pred.x)
ax[0].set_xlim([0, 5])
ax[1].set_xlim([0, 5])
ax[0].set_ylabel('Prey population')
ax[1].set_ylabel('Predator population')
plt.show()

prey.reset()
pred.reset()

ppm = PredatorPreyModel(prey, pred, 0.4, 0.1)
ppm.solve(40)
fig, ax = plt.subplots()
ax.plot(prey.time, prey.x, label='Prey')
ax.plot(pred.time, pred.x, label='Predator')
ax.set_xlim([0, 40])
ax.set_ylabel('Population')
ax.legend()
plt.show()

plt.plot(prey.x, pred.x)
plt.show()