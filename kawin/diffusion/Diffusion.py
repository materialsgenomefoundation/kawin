import numpy as np
import time
import csv
from itertools import zip_longest
from kawin.solver.Solver import DESolver, SolverType
from kawin.GenericModel import GenericModel
import kawin.diffusion.Plot as diffPlot
from kawin.diffusion.DiffusionParameters import TemperatureParameters, BoundaryConditions, CompositionProfile, DiffusionConstraints, HashTable

class DiffusionModel(GenericModel):
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
    #Boundary conditions
    FLUX = 0
    COMPOSITION = 1

    def __init__(self, zlim, N, elements, phases, 
                 thermodynamics = None,
                 temperatureParameters = None, 
                 boundaryConditions = None,
                 compositionProfile = None,
                 constraints = None,
                 record = True):
        super().__init__()
        if isinstance(phases, str):
            phases = [phases]
        self.zlim, self.N = zlim, N
        self.allElements, self.elements = elements, elements[1:]
        self.phases = phases
        self.therm = None

        self.z = np.linspace(zlim[0], zlim[1], N)
        self.dz = self.z[1] - self.z[0]
        self.t = 0

        self.temperatureParameters = temperatureParameters if temperatureParameters is not None else TemperatureParameters()
        self.boundaryConditions = boundaryConditions if boundaryConditions is not None else BoundaryConditions()
        self.compositionProfile = compositionProfile if compositionProfile is not None else CompositionProfile()
        self.constraints = constraints if constraints is not None else DiffusionConstraints()
        self.setThermodynamics(thermodynamics)

        self.hashTable = HashTable()

        self.reset()

        if record:
            self.enableRecording()
        else:
            self.disableRecording()
            self._recordedX = None
            self._recordedTime = None

    def reset(self):
        '''
        Resets model

        This involves clearing any caches in the Thermodynamics object and this model
        as well as resetting the composition and phase profiles
        '''
        if self.therm is not None:
            self.therm.clearCache()
        
        self.x = np.zeros((len(self.elements), self.N))
        self.isSetup = False
        self.t = 0

    def setThermodynamics(self, thermodynamics):
        '''
        Defines thermodynamics object for the diffusion model

        Parameters
        ----------
        thermodynamics : Thermodynamics object
            Requires the elements in the Thermodynamics and DiffusionModel objects to have the same order
        '''
        self.therm = thermodynamics

    def setTemperature(self, T):
        '''
        Sets isothermal temperature

        Parameters
        ----------
        T : float
        '''
        self.temperatureParameters.setIsothermalTemperature(T)

    def setTemperatureArray(self, times, temperatures):
        '''
        Sets array of times/temperatures

        Example:
            time = [0, 1, 2]
            temperature = [100, 200, 300]
            This will set temperature to 100 at t = 0 hours, then at 1 hour, temperature = 200, then after 2 hours, temperature = 300

        Parameters
        ----------
        times : list[float]
        temperatures : list[float]
        '''
        self.temperatureParameters.setTemperatureArray(times, temperatures)

    def setTemperatureFunction(self, func):
        '''
        Sets temperature function

        Parameters
        ----------
        func : Callable
            Function is in the form f(z,t) = T, where z is spatial coordinate and t is time
        '''
        self.temperatureParameters.setTemperatureFunction(func)

    def toDict(self):
        '''
        Converts diffusion data to dictionary
        '''
        data = {
            'finalTime': self.t,
            'finalX': self.x,
            'recordX': self._recordedX,
            'recordTime': self._recordedTime
        }
        return data

    def fromDict(self, data):
        '''
        Converts dictionary of data to diffusion data
        '''
        self.t = data['finalTime']
        self.x = data['finalX']
        self._recordedX = data['recordX']
        self._recordedTime = data['recordTime']
    
    def setHashSensitivity(self, s):
        '''
        If composition caching is used, this sets the composition precision (in sig figs) when creating a hash

        Ex. if s = 2, then a composition of [0.3334, 0.3333, 0.3333] will be converted to [0.33, 0.33, 0.33] when hashing
            While this can lead to faster compute times if new compositions has the same hash, this can also lower fidelity of
            the calculations since a range of compositions could correspond to the same moblity values

        Parameters
        ----------
        s : int
            Number of sig figs when creating composition hash
        '''
        self.hashTable.setHashSensitivity(s)

    def useCache(self, use):
        '''
        Whether to cache compositions

        Parameters
        ----------
        use : bool
        '''
        self.hashTable.enableCaching(use)

    def clearCache(self):
        '''
        Clears the composition cache
        '''
        self.hashTable.clearCache()

    def enableRecording(self):
        '''
        Enables recording of composition and phase
        '''
        self._record = True
        self._recordedX = np.zeros((1, len(self.elements), self.N))
        self._recordedTime = np.zeros(1)

    def disableRecording(self):
        '''
        Disables recording
        '''
        self._record = False

    def removeRecordedData(self):
        '''
        Removes recorded data
        '''
        self._recordedX = None
        self._recordedTime = None

    def record(self, time):
        '''
        Adds current mesh data to recorded arrays
        '''
        if self._record:
            if time > 0:
                self._recordedX = np.pad(self._recordedX, ((0, 1), (0, 0), (0, 0)))
                self._recordedTime = np.pad(self._recordedTime, (0, 1))

            self._recordedX[-1] = self.x
            self._recordedTime[-1] = time

    def setMeshtoRecordedTime(self, time):
        '''
        From recorded values, interpolated at time to get composition and phase fraction
        '''
        if self._record:
            if time < self._recordedTime[0]:
                print('Input time is lower than smallest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[0]))
                self.x = self._recordedX[0]
            elif time > self._recordedTime[-1]:
                print('Input time is larger than longest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[-1]))
                self.x = self._recordedX[-1]
            else:
                uind = np.argmax(self._recordedTime > time)
                lind = uind - 1

                ux, utime = self._recordedX[uind], self._recordedTime[uind]
                lx, ltime = self._recordedX[lind], self._recordedTime[lind]

                self.x = (ux - lx) * (time - ltime) / (utime - ltime) + lx

    def _getElementIndex(self, element = None):
        '''
        Gets index of element in self.elements

        Parameters
        ----------
        element : str
            Specified element, will return first element if None
        '''
        if element is None:
            return 0
        else:
            return self.elements.index(element)

    def _getPhaseIndex(self, phase = None):
        '''
        Gets index of phase in self.phases

        Parameters
        ----------
        phase : str
            Specified phase, will return first phase if None
        '''
        if phase is None:
            return 0
        else:
            return self.phases.index(phase)
        
    def setBC(self, LBCtype = BoundaryConditions.FLUX_BC, LBCValue = 0, RBCType = BoundaryConditions.FLUX_BC, RBCValue = 0, element = None):
        '''
        Sets boundary conditions

        Parameters
        ----------
        LBCtype : int
            Type of boundary condition on left side. Either BoundaryConditions.FLUX_BC or BoundaryConditions.COMPOSITION_BC
        LBCvalue : float
            Value of left boundary condition
        RBCtype : int
            Type of boundary condition on right side. Either BoundaryConditions.FLUX_BC or BoundaryConditions.COMPOSITION_BC
        RBCvalue : float
            Value of right boundary condition
        '''
        self.boundaryConditions.setBoundaryCondition(BoundaryConditions.LEFT, 
                                                                LBCtype, LBCValue, element)
        self.boundaryConditions.setBoundaryCondition(BoundaryConditions.RIGHT,
                                                                RBCType, RBCValue, element)
        
    def setCompositionLinear(self, Lvalue, Rvalue, element = None):
        '''
        Creates linear composition profile for element

        Parameters
        ----------
        Lvalue : float
            Composition of element on left
        Rvalue : float
            Composition of element on right
        element : str
            Element to apply linear profile. If None, will use first independent element
        '''
        element = self.elements[0] if element is None else element
        self.compositionProfile.clearCompositionBuildSteps(element)
        self.compositionProfile.addLinearCompositionStep(element, Lvalue, Rvalue)

    def setCompositionStep(self, Lvalue, Rvalue, z, element = None):
        '''
        Creates step composition profile for element

        Parameters
        ----------
        Lvalue : float
            Composition of element on left
        Rvalue : float
            Composition of element on right
        z : float
            Z coordinate where composition switches from left to right value
        element : str
            Element to apply step profile. If None, will use first independent element
        '''
        element = self.elements[0] if element is None else element
        self.compositionProfile.clearCompositionBuildSteps(element)
        self.compositionProfile.addStepCompositionStep(element, Lvalue, Rvalue, z)

    def setCompositionSingle(self, value, z, element = None):
        '''
        Creates composition of element at single node

        Parameters
        ----------
        value : float
            Composition of element at node
        z : float
            Z coordinate where the closest node will be used
        element : str
            Element to apply delta profile. If None, will use first independent element
        '''
        element = self.elements[0] if element is None else element
        self.compositionProfile.clearCompositionBuildSteps(element)
        self.compositionProfile.addSingleCompositionStep(element, value, z)

    def setCompositionInBounds(self, value, Lbound, Rbound, element = None):
        '''
        Set nodes between boundaries to composition for an element

        Parameters
        ----------
        value : float
            Composition of element
        Lbound : float
            Left coordinate of nearest node
        Rbound : float
            Right coordinate of nearest node
        element : str
            Element to apply profile. If None, will use first independent element
        '''
        element = self.elements[0] if element is None else element
        self.compositionProfile.clearCompositionBuildSteps(element)
        self.compositionProfile.addBoundedCompositionStep(element, value, Lbound, Rbound)

    def setCompositionFunction(self, func, element = None):
        '''
        Set composition of element according to function

        Parameters
        ----------
        func : Callable
            Function in form of f(z) = c where z is spatial coordinate and c is composition
        element : str
            Element to apply profile. If None, will use first independent element
        '''
        element = self.elements[0] if element is None else element
        self.compositionProfile.clearCompositionBuildSteps(element)
        self.compositionProfile.addFunctionCompositionStep(element, func)

    def setCompositionProfile(self, z, x, element = None):
        '''
        Set composition of element as interpolation of input profile

        Parameters
        ----------
        z : list[float]
            Spatial coordinates of dataset
        x : list[float]
            Composition of dataset (corresponds to z)
        element : str
            Element to apply profile. If None, will use first independent element
        '''
        element = self.elements[0] if element is None else element
        self.compositionProfile.clearCompositionBuildSteps(element)
        self.compositionProfile.addProfileCompositionStep(element, x, z)

    def setup(self):
        '''
        General setup function for all diffusio models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if not self.isSetup:
            if self.therm is not None:
                self.therm.clearCache()

            self.compositionProfile.buildProfile(self.elements, self.x, self.z)
            self.boundaryConditions.setupDefaults(self.elements)
            self.boundaryConditions.applyBoundaryConditionsToInitialProfile(self.elements, self.x, self.z)

        xsum = np.sum(self.x, axis=0)
        if any(xsum > 1):
            print('Compositions add up to above 1 between z = [{:.3e}, {:.3e}]'.format(np.amin(self.z[xsum>1]), np.amax(self.z[xsum>1])))
            raise Exception('Some compositions sum up to above 1')
        self.x[self.x > self.constraints.minComposition] = self.x[self.x > self.constraints.minComposition] - len(self.allElements)*self.constraints.minComposition
        self.x[self.x < self.constraints.minComposition] = self.constraints.minComposition
        self.isSetup = True
        self.record(self.t) #Record at t = 0

    def _getFluxes(self):
        '''
        "Virtual" function to be implemented by child objects

        Should return (fluxes (list), dt (float))
        '''
        raise NotImplementedError()
    
    def printHeader(self):
        print('Iteration\tSim Time (h)\tRun time (s)')

    def printStatus(self, iteration, modelTime, simTimeElapsed):
        # Convert time to hours
        super().printStatus(iteration, modelTime/3600, simTimeElapsed)

    def getCurrentX(self):
        return self.t, [self.x]
          
    def getdXdt(self, t, x):
        '''
        dXdt is defined as -dJ/dz
        '''
        fluxes = self._getFluxes(t, x)
        return [-(fluxes[:,1:] - fluxes[:,:-1])/self.dz]
    
    def preProcess(self):
        return
    
    def postProcess(self, time, x):
        '''
        Stores new x and t
        Records new values if recording is enabled
        '''
        self.t = time
        self.x = x[0]
        self.x = np.clip(self.x, self.constraints.minComposition, 1-self.constraints.minComposition)
        self.record(self.t)
        self.updateCoupledModels()
        return self.getCurrentX()[1], False
    
    def flattenX(self, X):
        '''
        np.hstack does not flatten a 2D array, so we have to overload this function
            By itself, this doesn't actually affect the solver/iterator, but when coupled with other models,
            it becomes an issue

        This will convert the 2D array X to a 1D array by reshaping to 1D array of len(# elements * # nodes)
        '''
        return np.reshape(X[0], (np.prod(X[0].shape)))
    
    def unflattenX(self, X_flat, X_ref):
        '''
        Reshape X_flat to original shape
        '''
        return [np.reshape(X_flat, X_ref[0].shape)]

    def getX(self, element):
        '''
        Gets composition profile of element
        
        Parameters
        ----------
        element : str
            Element to get profile of
        '''
        if element in self.allElements and element not in self.elements:
            return 1 - np.sum(self.x, axis=0)
        else:
            e = self._getElementIndex(element)
            return self.x[e]

    def plot(self, ax = None, plotReference = True, plotElement = None, zScale = 1, zOffset = 0, *args, **kwargs):
        '''
        Plots composition profile

        Parameters
        ----------
        ax : matplotlib Axes object
            Axis to plot on
        plotReference : bool
            Whether to plot reference element (composition = 1 - sum(composition of rest of elements))
        plotElement : None or str
            Plots single element if it is defined, otherwise, all elements are plotted
        zScale : float
            Scale factor for z-coordinates
        '''
        return diffPlot.plot(diffModel=self, 
                             ax=ax, 
                             plotReference=plotReference, 
                             plotElement=plotElement, 
                             zScale=zScale,
                             zOffset=zOffset, 
                             *args, **kwargs)

    def plotTwoAxis(self, Lelements, Relements, zScale = 1, zOffset = 0, axL = None, axR = None, *args, **kwargs):
        '''
        Plots composition profile with two y-axes

        Parameters
        ----------
        axL : matplotlib Axes object
            Left axis to plot on
        Lelements : list of str
            Elements to plot on left axis
        Relements : list of str
            Elements to plot on right axis
        axR : matplotlib Axes object (optional)
            Right axis to plot on
            If None, then the right axis will be created
        zScale : float
            Scale factor for z-coordinates
        '''
        return diffPlot.plotTwoAxis(diffModel=self, 
                                    Lelements=Lelements, 
                                    Relements=Relements, 
                                    zScale=zScale, 
                                    zOffset=zOffset, 
                                    axL=axL, 
                                    axR=axR, 
                                    *args, **kwargs)

    def plotPhases(self, ax = None, plotPhase = None, zScale = 1, zOffset = 0, *args, **kwargs):
        '''
        Plots phase fractions over z

        Parameters
        ----------
        ax : matplotlib Axes object
            Axis to plot on
        plotPhase : None or str
            Plots single phase if it is defined, otherwise, all phases are plotted
        zScale : float
            Scale factor for z-coordinates
        '''
        return diffPlot.plotPhases(diffModel=self, 
                                   ax=ax, 
                                   plotPhase=plotPhase, 
                                   zScale=zScale,
                                   zOffset=zOffset, 
                                   *args, **kwargs)
