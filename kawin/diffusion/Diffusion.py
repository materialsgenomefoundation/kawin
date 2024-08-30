import numpy as np
import time
import csv
from itertools import zip_longest
from kawin.solver.Solver import DESolver, SolverType
from kawin.GenericModel import GenericModel
import kawin.diffusion.Plot as diffPlot
from kawin.diffusion.DiffusionParameters import DiffusionParameters

class DiffusionModel(GenericModel):
    #Boundary conditions
    FLUX = 0
    COMPOSITION = 1

    def __init__(self, zlim, N, elements = ['A', 'B'], phases = ['alpha'], parameters = None, record = True):
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

        self.reset()
        self.parameters = parameters if parameters is not None else DiffusionParameters(self.elements)

        if record:
            self.enableRecording()
        else:
            self.disableRecording()
            self._recordedX = None
            self._recordedZ = None
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

    def _getVarDict(self):
        '''
        Returns mapping of { variable name : attribute name } for saving
        The variable name will be the name in the .npz file
        '''
        saveDict = {
            'elements': 'elements',
            'phases': 'phases',
            'z': 'z',
            'zLim': 'zLim',
            'N': 'N',
            'finalTime': 't',
            'finalX': 'x',
            'recordX': '_recordedX',
            'recordZ': '_recordedZ',
            'recordTime': '_recordedTime',
        }
        return saveDict

    def load(filename):
        '''
        Loads data from filename and returns a PrecipitateModel
        '''
        data = np.load(filename)
        model = DiffusionModel(data['zLim'], data['N'], data['elements'], data['phases'])
        model._loadData(data)
        model.isSetup = True
        return model

    def enableRecording(self, dtype = np.float32):
        '''
        Enables recording of composition and phase
        
        Parameters
        ----------
        dtype : numpy data type (optional)
            Data type to record particle size distribution in
            Defaults to np.float32
        '''
        self._record = True
        self._recordedX = np.zeros((1, len(self.elements), self.N))
        self._recordedZ = self.z
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
        self._recordedZ = None
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
            
            self.z = self._recordedZ

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

    def setup(self):
        '''
        General setup function for all diffusio models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if not self.isSetup:
            if self.therm is not None:
                self.therm.clearCache()

            self.parameters.composition_profile.build_profile(self.x, self.z)
            self.parameters.boundary_conditions.apply_boundary_conditions_to_initial_profile(self.x, self.z)

        xsum = np.sum(self.x, axis=0)
        if any(xsum > 1):
            print('Compositions add up to above 1 between z = [{:.3e}, {:.3e}]'.format(np.amin(self.z[xsum>1]), np.amax(self.z[xsum>1])))
            raise Exception('Some compositions sum up to above 1')
        self.x[self.x > self.parameters.min_composition] = self.x[self.x > self.parameters.min_composition] - len(self.allElements)*self.parameters.min_composition
        self.x[self.x < self.parameters.min_composition] = self.parameters.min_composition
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
        self.x = np.clip(self.x, self.parameters.min_composition, 1-self.parameters.min_composition)
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

    def plot(self, ax = None, plotReference = True, plotElement = None, zScale = 1, *args, **kwargs):
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
        return diffPlot.plot(self, ax, plotReference, plotElement, zScale, *args, **kwargs)

    def plotTwoAxis(self, Lelements, Relements, zScale = 1, axL = None, axR = None, *args, **kwargs):
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
        return diffPlot.plotTwoAxis(self, Lelements, Relements, zScale, axL, axR, *args, **kwargs)

    def plotPhases(self, ax = None, plotPhase = None, zScale = 1, *args, **kwargs):
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
        return diffPlot.plotPhases(self, ax, plotPhase, zScale, *args, **kwargs)
