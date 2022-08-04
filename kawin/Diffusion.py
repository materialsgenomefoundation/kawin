from turtle import update
import numpy as np
import time

class DiffusionModel:
    def __init__(self, thermodynamics, mesh, removeCache = True):
        '''
        Class for defining generic diffusion model

        Parameters
        ----------
        thermodynamics : kawin.Thermodynamics object
        mesh : kawin.Mesh1D object or other
        '''
        self.thermodynamics = thermodynamics
        self.mesh = mesh

        self.interdiffusivity = self.thermodynamics.getInterdiffusivity

        self.t = 0
        self.tf = 0
        self.T = 0

    def setTemperature(self, T):
        self.T = T

    def createCache(self):
        self.composition_sets = []
        for i in range(self.mesh.N):
            results, compset = self.thermodynamics.getLocalEq(self.mesh.x[:,i], self.T, gExtra = 0, precPhase = -1)
            self.composition_sets.append(compset)

    def update(self):
        #Get fluxes
        fluxes, dt = self.mesh.getFluxes(self.T*np.ones(self.mesh.N-1))

        if self.t + dt > self.tf:
            dt = self.tf - self.t

        #Update mesh
        self.mesh.update(fluxes, dt)
        self.t += dt

    def solve(self, simTime, verbose=False, vIt=10):
        self.mesh.setup(self.thermodynamics, self.T)

        self.t = 0
        self.tf = simTime
        i = 0
        t0 = time.time()
        if verbose:
            print('Iteration\tSim Time (h)\tRun time (s)')
        while self.t < self.tf:
            if verbose and i % vIt == 0:
                tf = time.time()
                print(str(i) + '\t\t{:.3f}\t\t{:.3f}'.format(self.t/3600, tf-t0))
            self.update()
            i += 1

        tf = time.time()
        print(str(i) + '\t\t{:.3f}\t\t{:.3f}'.format(self.t/3600, tf-t0))
