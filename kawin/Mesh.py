import numpy as np
import matplotlib.pyplot as plt
from sympy import Rel

class Mesh1D:
    #Boundary conditions
    FLUX = 0
    COMPOSITION = 1

    def __init__(self, zlim, N, elements = ['A', 'B'], phases = ['alpha']):
        '''
        Class for defining a 1-dimensional mesh

        Parameters
        ----------
        zlim : tuple
            Z-bounds of mesh (lower, upper)
        N : int
            Number of nodes
        elements : list of str
            Elements in system (first element will be assumed as the reference element)
        phases : list of str
            Number of phases in the system
        '''
        self.zlim, self.N = zlim, N
        self.allElements, self.elements = elements, elements[1:]
        self.phases = phases

        self.z = np.linspace(zlim[0], zlim[1], N)
        self.dz = self.z[1] - self.z[0]

        self.x = np.zeros((len(self.elements), N))
        self.p = np.ones((1,N)) if len(phases) == 1 else np.zeros((len(phases), N))

        self.LBC, self.RBC = self.FLUX*np.ones(len(self.elements)), self.FLUX*np.ones(len(self.elements))
        self.LBCvalue, self.RBCvalue = np.zeros(len(self.elements)), np.zeros(len(self.elements))

    def _getElementIndex(self, element = None):
        '''
        Gets index of element in self.elements

        Parameters
        ----------
        elements : str
            Specified element, will return first element if None
        '''
        if element is None:
            return 0
        else:
            return self.elements.index(element)

    def setBC(self, LBCtype = 0, LBCvalue = 0, RBCtype = 0, RBCvalue = 0, element = None):
        '''
        Set boundary conditions

        Parameters
        ----------
        LBCtype : int
            Left boundary condition type
                Mesh1D.FLUX - constant flux
                Mesh1D.COMPOSITION - constant composition
        LBCvalue : float
            Value of left boundary condition
        RBCtype : int
            Right boundary condition type
                Mesh1D.FLUX - constant flux
                Mesh1D.COMPOSITION - constant composition
        RBCvalue : float
            Value of right boundary condition
        element : str
            Specified element to apply boundary conditions on
        '''
        eIndex = self._getElementIndex(element)
        self.LBC[eIndex] = LBCtype
        self.LBCvalue[eIndex] = LBCvalue
        if LBCtype == self.COMPOSITION:
            self.x[eIndex,0] = LBCvalue

        self.RBC[eIndex] = RBCtype
        self.RBCvalue[eIndex] = RBCvalue
        if RBCtype == self.COMPOSITION:
            self.x[eIndex,-1] = RBCvalue

    def setCompositionLinear(self, Lvalue, Rvalue, element = None):
        '''
        Sets composition as a linear function between ends of the mesh

        Parameters
        ----------
        Lvalue : float
            Value at left boundary
        Rvalue : float
            Value at right boundary
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        self.x[eIndex] = np.linspace(Lvalue, Rvalue, self.N)

    def setCompositionStep(self, Lvalue, Rvalue, z, element = None):
        '''
        Sets composition as a step-wise function

        Parameters
        ----------
        Lvalue : float
            Value on left side of mesh
        Rvalue : float
            Value on right side of mesh
        z : float
            Position on mesh where composition switches from Lvalue to Rvalue
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        Lindices = self.z <= z
        self.x[eIndex,Lindices] = Lvalue
        self.x[eIndex,~Lindices] = Rvalue

    def setCompositionSingle(self, value, z, element = None):
        '''
        Sets single node to specified composition

        Parameters
        ----------
        value : float
            Composition
        z : float
            Position to set value to (will use closest node to z)
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        zIndex = np.argmin(np.abs(self.z-z))
        self.x[eIndex,zIndex] = value

    def setCompositionFunction(self, func, element = None):
        '''
        Sets composition as a function of z

        Parameters
        ----------
        func : function
            Function taking in z and returning composition
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        self.x[eIndex,:] = func(self.z)

    def getFluxes(self, dFunc, **kwargs):
        xMid = (self.x[:,1:] + self.x[:,:-1]) / 2
        d = dFunc(xMid.T, **kwargs)

        dxdz = (self.x[:,1:] - self.x[:,:-1]) / self.dz
        fluxes = np.zeros((len(self.elements), self.N+1))
        if len(self.elements) == 1:
            fluxes[0,1:-1] = -d * dxdz
        else:
            dxdz = np.expand_dims(dxdz, axis=0)
            fluxes[:,1:-1] = -np.matmul(d, np.transpose(dxdz, (2,1,0)))[:,:,0].T
        for e in range(len(self.elements)):
            fluxes[e,0] = self.LBCvalue[e] if self.LBC[e] == self.FLUX else fluxes[e,1]
            fluxes[e,-1] = self.RBCvalue[e] if self.RBC[e] == self.FLUX else fluxes[e,-2]

        dt = 0.4 * self.dz**2 / np.amax(d)

        return fluxes, dt

    def update(self, fluxes, dt):
        '''
        Updates mesh using fluxes by time increment dt

        Parameters
        ----------
        fluxes : 2D array
            Fluxes for each element between each node. Size must be (E, N-1)
                E - number of elements (NOT including reference element)
                N - number of nodes
            Boundary conditions will automatically be applied
        dt : float
            Time increment
        '''
        for e in range(len(self.elements)):
            self.x[e] += -(fluxes[e,1:] - fluxes[e,:-1]) * dt / self.dz

    def plot(self, ax, plotReference = True, zScale = 1):
        '''
        Plots composition profile

        Parameters
        ----------
        ax : matplotlib Axes object
            Axis to plot on
        plotReference : bool
            Whether to plot reference element (composition = 1 - sum(composition of rest of elements))
        '''
        if plotReference:
            refE = 1 - np.sum(self.x, axis=0)
            ax.plot(self.z/zScale, refE, label=self.allElements[0])
        for e in range(len(self.elements)):
            ax.plot(self.z/zScale, self.x[e], label=self.elements[e])
            
        ax.set_xlim([self.zlim[0], self.zlim[1]])
        ax.legend()
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Composition (at.%)')

    def plotTwoAxis(self, axL, Lelements, Relements, zScale = 1):
        if type(Lelements) is str:
            Lelements = [Lelements]
        if type(Relements) is str:
            Relements = [Relements]

        ci = 0
        refE = 1 - np.sum(self.x, axis=0)
        axR = axL.twinx()
        for e in range(len(Lelements)):
            if Lelements[e] in self.elements:
                eIndex = self._getElementIndex(Lelements[e])
                axL.plot(self.z/zScale, self.x[eIndex], label=self.elements[eIndex], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0
            elif Lelements[e] in self.allElements:
                axL.plot(self.z/zScale, refE, label=self.allElements[0], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0
        for e in range(len(Relements)):
            if Relements[e] in self.elements:
                eIndex = self._getElementIndex(Relements[e])
                axR.plot(self.z/zScale, self.x[eIndex], label=self.elements[eIndex], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0
            elif Relements[e] in self.allElements:
                axR.plot(self.z/zScale, refE, label=self.allElements[0], color = 'C' + str(ci))
                ci = ci+1 if ci <= 9 else 0

        
        axL.set_xlim([self.zlim[0], self.zlim[1]])
        axL.set_xlabel('Distance (m)')
        axL.set_ylabel('Composition (at.%) ' + str(Lelements))
        axR.set_ylabel('Composition (at.%) ' + str(Relements))
        
        lines, labels = axL.get_legend_handles_labels()
        lines2, labels2 = axR.get_legend_handles_labels()
        axR.legend(lines+lines2, labels+labels2, framealpha=1)

        return axL, axR