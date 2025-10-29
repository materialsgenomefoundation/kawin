from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np
import pickle
from scipy.interpolate import Rbf, RBFInterpolator
import scipy.spatial.distance as spd

from kawin.thermo.utils import _process_TG_arrays, _process_xT_arrays, _getMatrixPhase, _getPrecipitatePhase, _process_x
from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.thermo.MultiTherm import CurvatureOutput, GrowthRateOutput, _growthRateOutputFromCurvature

def generateTrainingPoints(*arrays):
    '''
    Creates all combinations of inputted arrays
    Used for creating training points in composition space for
    MulticomponentSurrogate

    Parameters
    ----------
    arrays - arrays along each dimension
        Order of arrays should correspond to the order of elements when defining thermodynamic functions
    '''
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))

def _filter_points(inputs, outputs, tol = 1e-3):
    '''
    Filter set of input points such that the closest distance is above tolerance
    This is to avoid non-positive definite training matrices when creating surrogate models

    Parameters
    ----------
    inputs : m x n matrix of floats
        Input points of m observations in n-dimensional space
    outputs : list of array of floats
        Output points, each array must be m x n where
            m is number of observations and n is dimensions of output
    tol : float
        Tolerance for distance between two input points

    Outputs
    -------
    (filtered inputs, filtered outputs)
    filtered inputs - array of input points
    filtered outputs - array of output points
    '''
    #Make distance matrix
    distance = spd.squareform(spd.pdist(inputs))

    #Indices to remove
    indices = np.where((distance > 0) & (distance <= tol))
    indices = np.unique(indices[0][indices[0] < indices[1]])

    newInputs = np.delete(inputs, indices, axis=0)
    newOutputs = []
    for i in range(len(outputs)):
        newOutputs.append(np.delete(outputs[i], indices, axis=0))

    return newInputs, newOutputs

class NumpyEncoder(json.JSONEncoder):
    '''
    Converts all numpy arrays to list, to be used when serializing surrogate data
    We don't need custom decoding, since the surrogate model should convert lists to numpy arrays if needed
    '''
    def default(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, list):
            return np.array(data).tolist()
        return super().default(data)

class SurrogateKernel(ABC):
    '''
    Abstract class for kernel

    Attributes
      __init__ - builds the surrogate model from x and y, and may use *args and **kwargs for hyperparameters
      predict - returns a y-like output from an x-like input
    '''
    @abstractmethod
    def __init__(self, x, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

class RBFKernel(SurrogateKernel):
    '''
    Surrogate kernel using scipy radial basis function (RBFInterpolator)
    An additional parameter, 'normalize', can be used to normalize all dimensions of x from [0,1]
    '''
    def __init__(self, x, y, *args, **kwargs):
        if kwargs.get('normalize', False):
            self.xoffset = np.amin(x, axis=0)
            self.scale = (np.amax(x, axis=0) - np.amin(x, axis=0)) / len(x)
        else:
            self.xoffset = np.zeros(x.shape[1])
            self.scale = np.ones(x.shape[1])

        kwargs.pop('normalize', None)
        self.rbfModel = RBFInterpolator((x - self.xoffset[np.newaxis,:]) / self.scale[np.newaxis,:], y, *args, **kwargs)

    def predict(self, x):
        return self.rbfModel((x - self.xoffset[np.newaxis,:]) / self.scale[np.newaxis,:])

class GeneralSurrogate:
    '''
    By default, the untrained surrogate will use the thermodynamic functions

    As we train a model, it will be added to the model and replace the underlying thermodynamics
    Then any untrained model will still go back to the thermodynamic functions

    The intent of this is that the surrogate models API will be similar to the
    underlying thermodynamic modules
        Driving force
        Tracer diffusivity
        Interdiffusivity
        Interfacial composition - binary only
        Curvature factor - multicomponent only
        Impingement rate - multicomponent only

    Parameters
    ----------
    thermodynamics: GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
    kernel: SurrogateKernel (optional)
        Defaults to RBFKernel
    kernelKwargs: dict (optional)
        arguments for kernel
        Defaults to {'kernel': 'cubic', 'normalize': True}
    '''
    def __init__(self, thermodynamics: GeneralThermodynamics,
                 kernel: SurrogateKernel = RBFKernel,
                 kernelKwargs = {'kernel': 'cubic', 'normalize': True}):
        self.therm = thermodynamics
        self.numElements = self.therm.numElements
        self.elements = self.therm.elements
        self.phases = self.therm.phases

        self.drivingForceData = {}
        self.drivingForceModels = {}

        self.diffusivityData = {}
        self.diffusivityModels = {}

        self.kernel = kernel
        self.kernelKwargs = kernelKwargs

    def _processCompositionInput(self, x, T, broadcast = True):
        '''
        Makes sure x and T are np arrays
        If broadcasting, then x and T will be computed as a grid

        Parameters
        x : float, np.array
        T : float, np.array
        broadcast : bool (optional)
            Defaults to True
            If True, will create a grid over x and T
            If False, x and T represents all points and must have the same length
        '''
        x = np.atleast_2d(x)
        if self.numElements == 2 and x.shape[1] != 1:
            x = x.T
        T = np.atleast_1d(T)
        singleX, singleT = len(x) == 1, len(T) == 1
        if broadcast:
            xsize, Tsize = len(x), len(T)
            x = np.tile(x, (Tsize,1))
            T = np.repeat(T, xsize, axis=0)
        else:
            if len(x) != len(T):
                raise ValueError("If broadcast is False, x and T must have the same length")
        return x, T, singleX, singleT

    def _createInput(self, xs, singleXs):
        '''
        Create input data for the surrogate model by validating number of unique points in each dimension
        '''
        xIn = []
        for i in range(len(xs)):
            if not singleXs[i]:
                xIn.append(xs[i])
        if len(xIn) == 0:
            raise ValueError('Must have more than 1 datapoint for training')
        else:
            return np.concatenate(xIn, axis=1)

    def trainDrivingForce(self, x, T, precPhase=None, logX = False, broadcast=True):
        '''
        Creates surrogate model for driving force
        Model will be in the form of (dg, x) = f(x, T) or (dg, x) = f(ln(x), T)

        If only single x or T, then surrogate model will only be trained on
        non-scalar axis

        Parameters
        ----------
        x : float, np.array
            If binary, shape of x must be () or (N,)
            If multicomponent, shape of x must be (e,) or (N,e)
        T : float, np.array
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        logX : bool (optional)
            Defaults to False
            If True, fits composition on a log scale
        broadcast : bool (optional)
            Defaults to True
            If True, will create grid of points over x and T
        '''
        # Get x,T arrays and precipitate phase and compute driving force and precipitate composition
        x, T, singleX, singleT = self._processCompositionInput(x, T, broadcast=broadcast)
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        dg, xp = self.therm.getDrivingForce(x, T, precPhase=precPhase, removeCache=True)
        self.drivingForceData[precPhase] = {
            'x': x, 'T': T, 'dg': dg, 'xp': xp, 'logX': logX, 'singleX': singleX, 'singleT': singleT
        }
        self._fitDrivingForce(precPhase)

    def _fitDrivingForce(self, phase):
        '''
        Fits driving force data for phase
        '''
        data = self.drivingForceData.get(phase, None)
        if data is None:
            return
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        x, T = data['x'], data['T']
        xFit = np.atleast_2d(x)
        if data['logX']:
            xFit = np.log(xFit)
        TFit = np.atleast_2d(T).T
        xTrain = self._createInput([xFit, TFit], [data['singleX'], data['singleT']])

        # Format driving force and precipitate composition arrays
        dg, xp = data['dg'], data['xp']
        dgFit = np.atleast_2d(dg).T
        xpFit = np.atleast_2d(xp)
        if self.numElements == 2 and xpFit.shape[1] != 1:
            xpFit = xpFit.T
        yTrain = np.concatenate((dgFit, xpFit), axis=1)

        # Create surrogate model
        self.drivingForceModels[phase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def getDrivingForce(self, x, T, precPhase=None, *args, **kwargs):
        '''
        Computes driving force

        If surrogate model for driving force has not been trained, this will use the underlying thermodynamics function

        Parameters
        ----------
        x : float, np.array
            If binary, shape of x must be () or (N,)
            If multicomponent, shape of x must be (e,) or (N,e)
        T : float, np.array
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        '''
        # Check if precipitate phase has been trained on and use surrogate model if so
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.drivingForceModels:
            trainingData = self.drivingForceData[precPhase]

            # Format x, T arrays for model
            x, T = _process_xT_arrays(x, T, self.numElements == 2)
            T = np.atleast_2d(T).T
            if trainingData['logX']:
                x = np.log(x)

            xIn = self._createInput([x, T], [trainingData['singleX'], trainingData['singleT']])
            output = self.drivingForceModels[precPhase].predict(xIn)
            return np.squeeze(output[:,0]), np.squeeze(output[:,1:])

        # If precipitate phase has not been trained, used underlying thermodynamics function
        else:
            return self.therm.getDrivingForce(x, T, precPhase=precPhase, *args, **kwargs)

    def trainDiffusivity(self, x, T, phase=None, logX=False, broadcast=True):
        '''
        Creates surrogate model for diffusivity
        Model will be in the form of (Dnkj^1/3, D*^1/3) = f(x, 1/T) or (Dnkj^1/3, D*^1/3) = f(ln(x), 1/T)
        The cubic transformation is used since Dnkj may be negative due to chemical potential gradients

        If only single x or T, then surrogate model will only be trained on
        non-scalar axis

        Parameters
        ----------
        x : float, np.array
            If binary, shape of x must be () or (N,)
            If multicomponent, shape of x must be (e,) or (N,e)
        T : float, np.array
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        logX : bool (optional)
            Defaults to False
            If True, fits composition on a log scale
        broadcast : bool (optional)
            Defaults to True
            If True, will create grid of points over x and T
        '''
        # Get x,T arrays and precipitate phase and compute driving force and precipitate composition
        x, T, singleX, singleT = self._processCompositionInput(x, T, broadcast=broadcast)
        phase = _getMatrixPhase(self.phases, phase)
        dnkj = self.therm.getInterdiffusivity(x, T, phase=phase, removeCache=True)
        dtracer = self.therm.getTracerDiffusivity(x, T, phase=phase, removeCache=True)
        self.diffusivityData[phase] = {
            'x': x, 'T': T, 'dnkj': dnkj, 'dtracer': dtracer, 'logX': logX, 'singleX': singleX, 'singleT': singleT
        }
        self._fitDiffusivity(phase)

    def _fitDiffusivity(self, phase):
        '''
        Fits driving force data for phase
        '''
        data = self.diffusivityData.get(phase, None)
        if data is None:
            return
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        x, T = data['x'], data['T']
        xFit = np.atleast_2d(x)
        if data['logX']:
            xFit = np.log(xFit)
        TFit = np.atleast_2d(T).T
        xTrain = self._createInput([xFit, 1/TFit], [data['singleX'], data['singleT']])

        dnkj, dtracer = data['dnkj'], data['dtracer']
        if self.numElements == 2:
            dnkjFit = np.atleast_2d(dnkj).T
        else:
            dnkjFit = np.reshape(dnkj, (dnkj.shape[0], dnkj.shape[1]*dnkj.shape[2]))
        dtracerFit = np.atleast_2d(dtracer)
        yTrain = np.concatenate((dnkjFit, dtracerFit), axis=1)
        yTrain = np.sign(yTrain)*np.power(np.abs(yTrain),1/3)
        self.diffusivityModels[phase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def _getDiffusivity(self, x, T, phase):
        '''
        Evaluates diffusivity from surrogate model
        '''
        trainingData = self.diffusivityData[phase]

        # Format x, T arrays for model
        x, T = _process_xT_arrays(x, T, self.numElements == 2)
        T = np.atleast_2d(T).T
        if trainingData['logX']:
            x = np.log(x)

        xIn = self._createInput([x, 1/T], [trainingData['singleX'], trainingData['singleT']])
        output = self.diffusivityModels[phase].predict(xIn)
        return output

    def getInterdiffusivity(self, x, T, phase=None, *args, **kwargs):
        '''
        Computes interdiffusivity

        If surrogate model for driving force has not been trained, this will use the underlying thermodynamics function

        Parameters
        ----------
        x : float, np.array
            If binary, shape of x must be () or (N,)
            If multicomponent, shape of x must be (e,) or (N,e)
        T : float, np.array
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        '''
        phase = _getMatrixPhase(self.phases, phase)
        if phase in self.diffusivityModels:
            output = self._getDiffusivity(x, T, phase)
            if self.numElements == 2:
                return np.squeeze(np.power(output[:,0],3))
            else:
                d = np.power(output[:,:x.shape[1]*x.shape[1]], 3)
                d = np.reshape(d, (d.shape[0], x.shape[1], x.shape[1]))
                return np.squeeze(d)
        else:
            return self.therm.getInterdiffusivity(x, T, phase=phase, *args, **kwargs)

    def getTracerDiffusivity(self, x, T, phase=None, *args, **kwargs):
        '''
        Computes tracer diffusivity

        If surrogate model for driving force has not been trained, this will use the underlying thermodynamics function

        Parameters
        ----------
        x : float, np.array
            If binary, shape of x must be () or (N,)
            If multicomponent, shape of x must be (e,) or (N,e)
        T : float, np.array
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        '''
        phase = _getMatrixPhase(self.phases, phase)
        if phase in self.diffusivityModels:
            output = self._getDiffusivity(x, T, phase)
            d = np.power(output[:,x.shape[1]*x.shape[1]:],3)
            return np.squeeze(d)
        else:
            return self.therm.getInterdiffusivity(x, T, phase=phase, *args, **kwargs)

    def _collectSurrogateData(self):
        '''
        Creates dictionary of surrogate training data
        '''
        return {'drivingForce': self.drivingForceData, 'diffusivity': self.diffusivityData}

    def _processSurrogateData(self, data):
        '''
        Stores surrogate training data from dict and trains models
        '''
        self.drivingForceData = data['drivingForce']
        for ph in self.drivingForceData:
            self._fitDrivingForce(ph)

        self.diffusivityData = data['diffusivity']
        for ph in self.diffusivityData:
            self._fitDiffusivity(ph)

    def toJson(self, filename: str | Path):
        '''
        Saves surrogate data to json
        '''
        filename = str(filename)
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'w') as f:
            json.dump(self._collectSurrogateData(), f, cls=NumpyEncoder)

    def fromJson(self, filename: str | Path):
        '''
        Loads surrogate data from json
        '''
        filename = str(filename)
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'r') as f:
            data = json.load(f)
        self._processSurrogateData(data)

class BinarySurrogate(GeneralSurrogate):
    '''
    Same as GeneralSurrogate but implements models for interfacial composition
    '''
    def __init__(self, thermodynamics: BinaryThermodynamics,
                 kernel: SurrogateKernel = RBFKernel,
                 kernelKwargs = {'kernel': 'cubic', 'normalize': True}):
        super().__init__(thermodynamics, kernel, kernelKwargs)
        self.interfacialCompositionData = {}
        self.interfacialCompositionModels = {}

    def _processGibbsThompsonInput(self, T, gExtra, broadcast = False):
        '''
        Makes sure T and gExtra are np arrays
        If broadcasting, then T and gExtra will be computed as a grid
        '''
        T = np.atleast_1d(T)
        gExtra = np.atleast_1d(gExtra)
        singleT, singleG = len(T) == 1, len(gExtra) == 1
        if broadcast:
            Tsize, gsize = len(T), len(gExtra)
            T = np.tile(T, (gsize,1))
            gExtra = np.repeat(gExtra, Tsize, axis=0)
        return T, gExtra, singleT, singleG

    def trainInterfacialComposition(self, T, gExtra, precPhase=None, logY = False, broadcast=True):
        '''
        Creates surrogate model for interfacial composition
        Model will be in the form of (xa, xb) = f(T, 1/gExtra) or (ln(xa), xb) = f(T, 1/gExtra)

        If only single T or gExtra, then surrogate model will only be trained on
        non-scalar axis

        Parameters
        ----------
        T : float, np.array
        gExtra : float, np.array
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        logY : bool (optional)
            If True, will fit interfacial composition on a log scale
        broadcast : bool (optional)
            If True, will create grid of points from T and gExtra
        '''
        T, gExtra, singleT, singleG = self._processGibbsThompsonInput(T, gExtra, broadcast=broadcast)
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        xpalpha, xpbeta = self.therm.getInterfacialComposition(T, gExtra, precPhase=precPhase)
        indices = xpalpha > 0
        T = T[indices]
        gExtra = gExtra[indices]
        xpalpha = xpalpha[indices]
        xpbeta = xpbeta[indices]
        self.interfacialCompositionData[precPhase] = {
            'T': T, 'gExtra': gExtra, 'xpalpha': xpalpha, 'xpbeta': xpbeta, 'logY': logY, 'singleT': singleT, 'singleG': singleG,
        }
        self._fitInterfacialComposition(precPhase)

    def _fitInterfacialComposition(self, phase):
        '''
        Fits driving force data for phase
        '''
        data = self.interfacialCompositionData.get(phase, None)
        if data is None:
            return
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        T, gExtra = data['T'], data['gExtra']
        TFit = np.atleast_2d(T).T
        gFit = np.atleast_2d(gExtra).T
        xTrain = self._createInput([TFit, 1/gFit], [data['singleT'], data['singleG']])

        xpalpha, xpbeta = data['xpalpha'], data['xpbeta']
        xpalpha = np.atleast_2d(xpalpha).T
        if data['logY']:
            xpalpha = np.log(xpalpha)
        xpbeta = np.atleast_2d(xpbeta).T
        yTrain = np.concatenate((xpalpha, xpbeta), axis=1)
        self.interfacialCompositionModels[phase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def getInterfacialComposition(self, T, gExtra=0, precPhase=None):
        '''
        Computes interfacial composition

        Parameters
        ----------
        T : float, np.array
        gExtra : float, np.array (optional)
            Defaults to 0
        precPhase : str (optional)
            Defaults to None
            If None, defaults to first precipitate phase
        '''
        # Check if precipitate phase has been trained on and use surrogate model if so
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.interfacialCompositionModels:
            trainingData = self.interfacialCompositionData[precPhase]

            # Format T, g arrays for model
            T, gExtra = _process_TG_arrays(T, gExtra)
            T = np.atleast_2d(T).T
            gExtra = np.atleast_2d(gExtra).T

            xIn = self._createInput([T, 1/gExtra], [trainingData['singleT'], trainingData['singleG']])
            output = self.interfacialCompositionModels[precPhase].predict(xIn)
            if trainingData['logY']:
                output[:,0] = np.exp(output[:,0])
            return np.squeeze(output[:,0]), np.squeeze(output[:,1])

        # If precipitate phase has not been trained, used underlying thermodynamics function
        else:
            return self.therm.getInterfacialComposition(T, gExtra, precPhase=precPhase)

    def _collectSurrogateData(self):
        '''
        Adds interfacial composition data to GeneralSurrogate surrogate data
        '''
        data = super()._collectSurrogateData()
        data['interfacialComposition'] = self.interfacialCompositionData
        return data

    def _processSurrogateData(self, data):
        '''
        Stores interfacial composition data and fits models along with models from GeneralSurrogate
        '''
        super()._processSurrogateData(data)
        self.interfacialCompositionData = data['interfacialComposition']
        for ph in self.interfacialCompositionData:
            self._fitInterfacialComposition(ph)

class MulticomponentSurrogate(GeneralSurrogate):
    '''
    Same as GeneralSurrogate but implements models for curvature factors
        Curvature factor can then be used for growth rate, interfacial composition and impingement rate
    '''
    def __init__(self, thermodynamics: MulticomponentThermodynamics,
                 kernel: SurrogateKernel = RBFKernel,
                 kernelKwargs = {'kernel': 'cubic', 'normalize': True}):
        super().__init__(thermodynamics, kernel, kernelKwargs)
        self.curvatureData = {}
        self.curvatureModels = {}

    def trainCurvature(self, x, T, precPhase=None, logX = False, broadcast=True):
        '''
        Trains curvature factors

        Parameters
        ----------
        x : np.array
            Shape of (e,) of (N,e)
        T : float, np.array
        precPhase : str (optional)
            Defaults to None, which is first precipitate phase
        logX : bool (optional)
            Defaults to False
            If True, then x and matrix interfacial composition will be fitted on a log scale
        broadcast : bool (optional)
            Defaults to True
            If True, then a grid a points will be generated over x and T
        '''
        # Get x,T arrays and precipitate phase and compute driving force and precipitate composition
        x, T, singleX, singleT = self._processCompositionInput(x, T, broadcast=broadcast)
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        xSuccess, TSuccess = [], []
        dc, mc, gba, beta, xEqAlpha, xEqBeta = [], [], [], [], [], []
        for xi, Ti in zip(x, T):
            curvature = self.therm.curvatureFactor(xi, Ti, precPhase=precPhase, removeCache=True, computeSearchDir = True)
            if curvature is None:
                continue
            xSuccess.append(xi)
            TSuccess.append(Ti)
            dc.append(curvature.dc)
            mc.append(curvature.mc)
            gba.append(curvature.gba)
            beta.append(curvature.beta)
            xEqAlpha.append(curvature.c_eq_alpha)
            xEqBeta.append(curvature.c_eq_beta)

        self.curvatureData[precPhase] = {
            'x': xSuccess, 'T': TSuccess, 'dc': dc, 'mc': mc, 'gba': gba, 'beta': beta, 'xEqAlpha': xEqAlpha, 'xEqBeta': xEqBeta,
            'logX': logX, 'singleX': singleX, 'singleT': singleT
        }
        self._fitCurvature(precPhase)

    def _fitCurvature(self, phase):
        '''
        Fits cuvature data for phase
        '''
        data = self.curvatureData.get(phase, None)
        if data is None:
            return
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        x, T = data['x'], data['T']
        xFit = np.atleast_2d(x)
        if data['logX']:
            xFit = np.log(xFit)
        TFit = np.atleast_2d(T).T
        xTrain = self._createInput([xFit, TFit], [data['singleX'], data['singleT']])

        # Format curvature factors
        dc, mc, gba = data['dc'], data['mc'], data['gba']
        beta = data['beta']
        xEqAlpha, xEqBeta = data['xEqAlpha'], data['xEqBeta']

        dcFit = np.atleast_2d(dc)
        mcFit = np.atleast_2d(mc).T
        gba = np.array(gba)
        gbaFit = np.reshape(gba, (gba.shape[0], gba.shape[1]*gba.shape[2]))
        betaFit = np.atleast_2d(beta).T
        xEqAlphaFit = np.atleast_2d(xEqAlpha)
        xEqBetaFit = np.atleast_2d(xEqBeta)
        if data['logX']:
            xEqAlphaFit = np.log(xEqAlphaFit)
        yTrain = np.concatenate((dcFit, mcFit, gbaFit, betaFit, xEqAlphaFit, xEqBetaFit), axis=1)
        self.curvatureModels[phase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def _surrogateOutputToCurvature(self, output, numEle, logX):
        '''
        Converts surrogate output to CurvatureOutput

        '''
        # dc has shape (numEle,)
        idx = 0
        dc = np.squeeze(output[0,idx:idx+numEle])
        idx += numEle
        # mc is scalar
        mc = np.squeeze(output[0,idx:idx+1])
        idx += 1
        # gba has shape (numEle, numEle)
        gba = np.squeeze(output[0,idx:idx+numEle*numEle])
        gba = np.reshape(gba, (numEle,numEle))
        idx += numEle*numEle
        # beta is scalar
        beta = np.squeeze(output[0,idx:idx+1])
        idx += 1
        # xEqAlpha has shape (numEle,)
        xEqAlpha = np.squeeze(output[0,idx:idx+numEle])
        if logX:
            xEqAlpha = np.exp(xEqAlpha)
        idx += numEle
        # xEqBeta has shape (numEle,)
        xEqBeta = np.squeeze(output[0,idx:idx+numEle])
        curvature = CurvatureOutput(
            dc=dc,
            mc=mc,
            gba=gba,
            beta=beta,
            c_eq_alpha=xEqAlpha,
            c_eq_beta=xEqBeta
        )
        return curvature

    def curvatureFactor(self, x, T, precPhase = None, *args, **kwargs):
        '''
        Computes curvature factors

        Parameters
        ----------
        x : np.array
            Shape of (e,)
        T : float
        precPhase : str (optional)
            Defaults to None, which is first precipitate phase
        '''
        # Check if precipitate phase has been trained on and use surrogate model if so
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.curvatureModels:
            trainingData = self.curvatureData[precPhase]

            # Format x, T arrays for model
            x, T = _process_xT_arrays(x, T, self.numElements == 2)
            if len(x) > 1 or len(T) > 1:
                raise ValueError('Curvature factor only takes in a single x,T condition.')
            T = np.atleast_2d(T).T
            if trainingData['logX']:
                x = np.log(x)

            xIn = self._createInput([x, T], [trainingData['singleX'], trainingData['singleT']])
            output = self.curvatureModels[precPhase].predict(xIn)
            return self._surrogateOutputToCurvature(output, x.shape[1], trainingData['logX'])

        # If precipitate phase has not been trained, used underlying thermodynamics function
        else:
            return self.therm.curvatureFactor(x, T, precPhase=precPhase, *args, **kwargs)

    def getGrowthAndInterfacialComposition(self, x, T, dG, R, gExtra, precPhase = None, *args, **kwargs):
        '''
        Computes growth rate and interfacial composition

        Parameters
        ----------
        x : np.array
            Shape of (e,)
        T : float
        dG : float
            Driving force at x,T
        R : float, np.array
            Precipitate radius
        gExtra : float, np.array
            Gibbs-Thomson contribution corresponding to R
        precPhase : str (optional)
            Defaults to None, which is first precipitate phase
        '''
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.curvatureModels:
            curvature = self.curvatureFactor(x, T, precPhase)
            x = _process_x(x, self.numElements)
            return _growthRateOutputFromCurvature(x, dG, R, gExtra, curvature)
        else:
            return self.therm.getGrowthAndInterfacialComposition(x, T, dG, R, gExtra, precPhase, *args, **kwargs)

    def impingementFactor(self, x, T, precPhase = None, *args, **kwargs):
        '''
        Computes impingement factor

        Parameters
        ----------
        x : np.array
            Shape of (e,)
        T : float
        precPhase : str (optional)
            Defaults to None, which is first precipitate phase
        '''
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.curvatureModels:
            curvature = self.curvatureFactor(x, T, precPhase)
            return curvature.beta
        else:
            return self.therm.impingementFactor(x, T, precPhase, *args, **kwargs)

    def _collectSurrogateData(self):
        '''
        Adds interfacial composition data to GeneralSurrogate surrogate data
        '''
        data = super()._collectSurrogateData()
        data['curvature'] = self.curvatureData
        return data

    def _processSurrogateData(self, data):
        '''
        Stores interfacial composition data and fits models along with models from GeneralSurrogate
        '''
        super()._processSurrogateData(data)
        self.curvatureData = data['curvature']
        for ph in self.curvatureData:
            self._fitCurvature(ph)