import numpy as np
import time
import csv
from itertools import zip_longest
from kawin.solver.Solver import DESolver, SolverType
from kawin.GenericModel import GenericModel
import kawin.diffusion.Plot as diffPlot

class DiffusionModel(GenericModel):
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

        self.LBC, self.RBC = self.FLUX*np.ones(len(self.elements)), self.FLUX*np.ones(len(self.elements))
        self.LBCvalue, self.RBCvalue = np.zeros(len(self.elements)), np.zeros(len(self.elements))

        self.cache = True
        self.setHashSensitivity(4)
        self.minComposition = 1e-8

        self.maxCompositionChange = 0.002

        self._record = False
        self._recordedX = None
        self._recordedP = None
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
        self.p = np.ones((1,self.N)) if len(self.phases) == 1 else np.zeros((len(self.phases), self.N))
        self.hashTable = {}
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
        Sets iso-thermal temperature

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        '''
        self.Tparam = T
        self.T = T
        self.Tfunc = lambda z, t: self.Tparam * np.ones(len(z))

    def setTemperatureArray(self, times, temperatures):
        self.Tparam = (times, temperatures)
        self.T = temperatures[0]
        self.Tfunc = lambda z, t: np.interp(t/3600, self.Tparam[0], self.Tparam[1], self.Tparam[1][0], self.Tparam[1][-1]) * np.ones(len(z))

    def setTemperatureFunction(self, func):
        '''
        Function should be T = (x, t)
        '''
        self.Tparam = func
        self.Tfunc = lambda z, t: self.Tparam(z, t)

    def save(self, filename, compressed = False, toCSV = False):
        '''
        Saves mesh, composition and phases

        Parameters
        ----------
        filename : str
            File to save to
        compressed : bool
            Whether to compress data if saving to numpy binary format (toCSV = False)
        toCSV : bool
            Whether to output data to a .CSV file format
        '''
        if toCSV:
            headers = ['Distance(m)']
            arrays = [self.z]
            for i in range(len(self.allElements)):
                headers.append('x(' + self.allElements[i] + ')')
                if i == 0:
                    arrays.append(1 - np.sum(self.x, axis=0))
                else:
                    arrays.append(self.x[i-1,:])
            for i in range(len(self.phases)):
                headers.append('f(' + self.phases[i] + ')')
                arrays.append(self.p[i,:])
            rows = zip_longest(*arrays, fillvalue='')
            if '.csv' not in filename.lower():
                filename = filename + '.csv'
            with open(filename, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
                csv.writer(f).writerows(rows)
        else:
            variables = ['zlim', 'N', 'allElements', 'phases', 'z', 'x', 'p']
            vDict = {v: getattr(self, v) for v in variables}
            if compressed:
                np.savez_compressed(filename, **vDict, allow_pickle=True)
            else:
                np.savez(filename, **vDict, allow_pickle=True)

    def load(filename):
        '''
        Loads a previously saved model

        filename : str
            File name to load model from, must include file extension
        '''
        if '.np' in filename.lower():
            data = np.load(filename, allow_pickle=True)
            model = DiffusionModel(data['zlim'], data['N'], data['allElements'], data['phases'])
            model.z = data['z']
            model.x = data['x']
            model.p = data['p']
        else:
            with open(filename, 'r') as csvFile:
                data = csv.reader(csvFile, delimiter=',')
                i = 0
                headers = []
                columns = {}
                for row in data:
                    if i == 0:
                        headers = row
                        columns = {h: [] for h in headers}
                    else:
                        for j in range(len(row)):
                            if row[j] != '':
                                columns[headers[j]].append(float(row[j]))
                    i += 1
            
            elements, phases = [], []
            x, p = [], []
            for h in headers:
                if 'Distance' in h:
                    z = columns[h]
                elif 'x' in h:
                    elements.append(h[2:-1])
                    x.append(columns[h])
                elif 'f' in h:
                    phases.append(h[2:-1])
                    p.append(columns[h])
            model = DiffusionModel([z[0], z[-1]], len(z), elements, phases)
            model.z = np.array(z)
            model.x = np.array(x)[1:,:]
            model.p = np.array(p)
        return model   

    def setHashSensitivity(self, s):
        '''
        Sets sensitivity of the hash table by significant digits

        For example, if a composition set is (0.5693, 0.2937) and s = 3, then
        the hash will be stored as (0.569, 0.294)

        Lower s values will give faster simulation times at the expense of accuracy

        Parameters
        ----------
        s : int
            Number of significant digits to keep for the hash table
        '''
        self.hashSensitivity = np.power(10, int(s))

    def _getHash(self, x, T):
        '''
        Gets hash value for a composition set

        Parameters
        ----------
        x : list of floats
            Composition set to create hash
        '''
        return hash(tuple((np.concatenate((x, [T]))*self.hashSensitivity).astype(np.int32)))
        #return hash(tuple((x*self.hashSensitivity).astype(np.int32)))
        #return int(np.sum(np.power(self.hashSensitivity, 1+np.arange(len(x))) * x))

    def useCache(self, use):
        '''
        Whether to use the hash table

        Parameters
        ----------
        use : bool
            If True, then the hash table will be used
        '''
        self.cache = use

    def clearCache(self):
        '''
        Clears hash table
        '''
        self.hashTable = {}

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
        self._recordedP = np.zeros((1, 1,self.N)) if len(self.phases) == 1 else np.zeros((1, len(self.phases), self.N))
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
        self._recordedP = None
        self._recordedZ = None
        self._recordedTime = None

    def record(self, time):
        '''
        Adds current mesh data to recorded arrays
        '''
        if self._record:
            if time > 0:
                self._recordedX = np.pad(self._recordedX, ((0, 1), (0, 0), (0, 0)))
                self._recordedP = np.pad(self._recordedP, ((0, 1), (0, 0), (0, 0)))
                self._recordedTime = np.pad(self._recordedTime, (0, 1))

            self._recordedX[-1] = self.x
            self._recordedP[-1] = self.p
            self._recordedTime[-1] = time

    def saveRecordedMesh(self, filename, compressed = True):
        '''
        Saves recorded data into npz format

        Parameters
        ----------
        filename : str
            File name to save to
        compressed : bool (optional)
            Whether to save as in compressed format (defaults to True)
        '''
        if self._record:
            if compressed:
                np.savez_compressed(filename, time = self._recordedTime, x = self._recordedX, p = self._recordedP, z = self._recordedZ)
            else:
                np.savez(filename, time = self._recordedTime, x = self._recordedX, p = self._recordedP, z = self._recordedZ)

    def loadRecordedMesh(self, filename):
        '''
        Loads recorded mesh
        '''
        data = np.load(filename)
        self._record = True
        self._recordedTime = data['time']
        self._recordedX = data['x']
        self._recordedP = data['p']
        self._recordedZ = data['z']
        self.isSetup = True

    def setMeshtoRecordedTime(self, time):
        if self._record:
            if time < self._recordedTime[0]:
                print('Input time is lower than smallest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[0]))
                self.x, self.p = self._recordedX[0], self._recordedP[0]
            elif time > self._recordedTime[-1]:
                print('Input time is larger than longest recorded time, setting PSD to t = {:.3e}'.format(self._recordedTime[-1]))
                self.x, self.p = self._recordedX[-1], self._recordedP[-1]
            else:
                uind = np.argmax(self._recordedTime > time)
                lind = uind - 1

                ux, up, utime = self._recordedX[uind], self._recordedP[uind], self._recordedTime[uind]
                lx, lp, ltime = self._recordedX[lind], self._recordedP[lind], self._recordedTime[lind]

                self.x = (ux - lx) * (time - ltime) / (utime - ltime) + lx
                self.p = (up - lp) * (time - ltime) / (utime - ltime) + lp
            
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

    def setCompositionInBounds(self, value, Lbound, Rbound, element = None):
        '''
        Sets single node to specified composition

        Parameters
        ----------
        value : float
            Composition
        Lbound : float
            Position of left bound
        Rbound : float
            Position of right bound
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        indices = (self.z >= Lbound) & (self.z <= Rbound)
        self.x[eIndex,indices] = value

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

    def setCompositionProfile(self, z, x, element = None):
        '''
        Sets composition profile by linear interpolation

        Parameters
        ----------
        z : array
            z-coords of composition profile
        x : array
            Composition profile
        element : str
            Element to apply composition profile to
        '''
        eIndex = self._getElementIndex(element)
        z = np.array(z)
        x = np.array(x)
        sortIndices = np.argsort(z)
        z = z[sortIndices]
        x = x[sortIndices]
        self.x[eIndex,:] = np.interp(self.z, z, x)

    def setup(self):
        '''
        General setup function for all diffusio models

        This will clear any cached values in the thermodynamics function and check if all compositions add up to 1

        This will also make sure that all compositions are not 0 or 1 to speed up equilibrium calculations
        '''
        if self.therm is not None:
            self.therm.clearCache()
        xsum = np.sum(self.x, axis=0)
        if any(xsum > 1):
            print('Compositions add up to above 1 between z = [{:.3e}, {:.3e}]'.format(np.amin(self.z[xsum>1]), np.amax(self.z[xsum>1])))
            raise Exception('Some compositions sum up to above 1')
        self.x[self.x > self.minComposition] = self.x[self.x > self.minComposition] - len(self.allElements) * self.minComposition
        self.x[self.x < self.minComposition] = self.minComposition
        self.T = self.Tfunc(self.z, 0)
        self.isSetup = True
        self.record(self.t) #Record at t = 0

    def getFluxes(self):
        '''
        "Virtual" function to be implemented by child objects

        Should return (fluxes (list), dt (float))
        '''
        raise NotImplementedError()
    
    def headerStr(self):
        print('Iteration\tSim Time (h)\tRun time (s)')

    def printStatus(self, iteration, simTimeElapsed):
        print(str(iteration) + '\t\t{:.3f}\t\t{:.3f}'.format(self.t/3600, simTimeElapsed))

    def getCurrentX(self):
        return self.t, [self.x]

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

    def getP(self, phase):
        '''
        Gets phase profile

        Parameters
        ----------
        phase : str
            Phase to get profile of
        '''
        p = self._getPhaseIndex(phase)
        return self.p[p]

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
        diffPlot.plot(self, ax, plotReference, plotElement, zScale, *args, **kwargs)

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
        diffPlot.plotTwoAxis(self, Lelements, Relements, zScale, axL, axR, *args, **kwargs)

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
        diffPlot.plotPhases(self, ax, plotPhase, zScale, *args, **kwargs)
