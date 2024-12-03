from abc import ABC, abstractmethod

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

    def _processCompositionInput(self, x, T, broadcast = False):
        '''
        Makes sure x and T are np arrays
        If broadcasting, then x and T will be computed as a grid
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
        return x, T, singleX, singleT
    
    def _createInput(self, xs, singleXs):
        '''
        Create input data for the surrogate model
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
        '''
        # Get x,T arrays and precipitate phase and compute driving force and precipitate composition
        x, T, singleX, singleT = self._processCompositionInput(x, T, broadcast=broadcast)
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        dg, xp = self.therm.getDrivingForce(x, T, precPhase=precPhase, removeCache=True)
        self.drivingForceData[precPhase] = {
            'x': x, 'T': T, 'dg': dg, 'xp': xp, 'logX': logX, 'singleX': singleX, 'singleT': singleT
        }
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        xFit = np.atleast_2d(x)
        if logX:
            xFit = np.log(xFit)
        TFit = np.atleast_2d(T).T
        xTrain = self._createInput([xFit, TFit], [singleX, singleT])
        
        # Format driving force and precipitate composition arrays
        dgFit = np.atleast_2d(dg).T
        xpFit = np.atleast_2d(xp)
        if self.numElements == 2 and xpFit.shape[1] != 1:
            xpFit = xpFit.T
        yTrain = np.concatenate((dgFit, xpFit), axis=1)

        # Create surrogate model
        self.drivingForceModels[precPhase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def getDrivingForce(self, x, T, precPhase=None, *args, **kwargs):
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
        '''
        # Get x,T arrays and precipitate phase and compute driving force and precipitate composition
        x, T, singleX, singleT = self._processCompositionInput(x, T, broadcast=broadcast)
        phase = _getMatrixPhase(self.phases, phase)
        dnkj = self.therm.getInterdiffusivity(x, T, phase=phase, removeCache=True)
        dtracer = self.therm.getTracerDiffusivity(x, T, phase=phase, removeCache=True)
        self.diffusivityData[phase] = {
            'x': x, 'T': T, 'dnkj': dnkj, 'dtracer': dtracer, 'logX': logX, 'singleX': singleX, 'singleT': singleT
        }
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        xFit = np.atleast_2d(x)
        if logX:
            xFit = np.log(xFit)
        TFit = np.atleast_2d(T).T
        xTrain = self._createInput([xFit, 1/TFit], [singleX, singleT])

        if self.numElements == 2:
            dnkjFit = np.atleast_2d(dnkj).T
        else:
            dnkjFit = np.reshape(dnkj, (dnkj.shape[0], dnkj.shape[1]*dnkj.shape[2]))
        dtracerFit = np.atleast_2d(dtracer)
        yTrain = np.concatenate((dnkjFit, dtracerFit), axis=1)
        yTrain = np.sign(yTrain)*np.power(np.abs(yTrain),1/3)
        self.diffusivityModels[phase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def _getDiffusivity(self, x, T, phase):
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
        phase = _getMatrixPhase(self.phases, phase)
        if phase in self.diffusivityModels:
            output = self._getDiffusivity(x, T, phase)
            d = np.power(output[:,x.shape[1]*x.shape[1]:],3)
            return np.squeeze(d)
        else:
            return self.therm.getInterdiffusivity(x, T, phase=phase, *args, **kwargs)

class BinarySurrogate(GeneralSurrogate):
    """
    Same as GeneralSurrogate but implements models for interfacial composition
    """
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

        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        TFit = np.atleast_2d(T).T
        gFit = np.atleast_2d(gExtra).T
        xTrain = self._createInput([TFit, 1/gFit], [singleT, singleG])

        xpalpha = np.atleast_2d(xpalpha).T
        if logY:
            xpalpha = np.log(xpalpha)
        xpbeta = np.atleast_2d(xpbeta).T
        yTrain = np.concatenate((xpalpha, xpbeta), axis=1)
        self.interfacialCompositionModels[precPhase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def getInterfacialComposition(self, T, gExtra=0, precPhase=None):
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
        
class MulticomponentSurrogate(GeneralSurrogate):
    """
    Same as GeneralSurrogate but implements models for curvature factors
        Curvature factor can then be used for growth rate, interfacial composition and impingement rate
    """
    def __init__(self, thermodynamics: MulticomponentThermodynamics, 
                 kernel: SurrogateKernel = RBFKernel, 
                 kernelKwargs = {'kernel': 'cubic', 'normalize': True}):
        super().__init__(thermodynamics, kernel, kernelKwargs)
        self.curvatureData = {}
        self.curvatureModels = {}

    def trainCurvature(self, x, T, precPhase=None, logX = False, broadcast=True):
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
        xSuccess = np.array(xSuccess)   # (N,e)
        TSuccess = np.array(TSuccess)   # (N,)
        dc = np.array(dc)               # (N,e)
        mc = np.array(mc)               # (N,)
        gba = np.array(gba)             # (N,e,e)
        beta = np.array(beta)           # (N,)
        xEqAlpha = np.array(xEqAlpha)   # (N,e)
        xEqBeta = np.array(xEqBeta)     # (N,e)
        self.curvatureData[precPhase] = {
            'x': xSuccess, 'T': TSuccess, 'dc': dc, 'mc': mc, 'gba': gba, 'beta': beta, 'xEqAlpha': xEqAlpha, 'xEqBeta': xEqBeta,
            'logX': logX, 'singleX': singleX, 'singleT': singleT
        }
        # Format x and T and training data. If x or T is a single training point, then don't use for
        # training data as it will create a non-full rank matrix
        xFit = np.atleast_2d(xSuccess)
        if logX:
            xFit = np.log(xFit)
        TFit = np.atleast_2d(TSuccess).T
        xTrain = self._createInput([xFit, TFit], [singleX, singleT])

        # Format curvature factors
        dcFit = np.atleast_2d(dc)
        mcFit = np.atleast_2d(mc).T
        gbaFit = np.reshape(gba, (gba.shape[0], gba.shape[1]*gba.shape[2]))
        betaFit = np.atleast_2d(beta).T
        xEqAlphaFit = np.atleast_2d(xEqAlpha)
        xEqBetaFit = np.atleast_2d(xEqBeta)
        if logX:
            xEqAlphaFit = np.log(xEqAlphaFit)
        yTrain = np.concatenate((dcFit, mcFit, gbaFit, betaFit, xEqAlphaFit, xEqBetaFit), axis=1)
        self.curvatureModels[precPhase] = self.kernel(xTrain, yTrain, **self.kernelKwargs)

    def _surrogateOutputToCurvature(self, output, numEle, logX):
            idx = 0
            dc = np.squeeze(output[0,idx:idx+numEle])
            idx += numEle
            mc = np.squeeze(output[0,idx:idx+1])
            idx += 1
            gba = np.squeeze(output[0,idx:idx+numEle*numEle])
            gba = np.reshape(gba, (numEle,numEle))
            idx += numEle*numEle
            beta = np.squeeze(output[0,idx:idx+1])
            idx += 1
            xEqAlpha = np.squeeze(output[0,idx:idx+numEle])
            if logX:
                xEqAlpha = np.exp(xEqAlpha)
            idx += numEle
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
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.curvatureModels:
            curvature = self.curvatureFactor(x, T, precPhase)
            x = _process_x(x, self.numElements)
            return _growthRateOutputFromCurvature(x, dG, R, gExtra, curvature)
        else:
            return self.therm.getGrowthAndInterfacialComposition(x, T, dG, R, gExtra, precPhase, *args, **kwargs)
    
    def impingementFactor(self, x, T, precPhase = None, *args, **kwargs):
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if precPhase in self.curvatureModels:
            curvature = self.curvatureFactor(x, T, precPhase)
            return curvature.beta
        else:
            return self.therm.impingementFactor(x, T, precPhase, *args, **kwargs)

class BinarySurrogateOld:
    '''
    Handles surrogate models for driving force, interfacial composition and
    diffusivity in a binary system
    
    Parameters
    ----------
    binaryThermodynamics - BinaryThermodynamics (optional)
        Driving force and interfacial composition will be taken from this if not explicitly defined
        If None, then drivingForce and interfacialComposition must be defined
        
    drivingForce - function (optional)
        Function will take in (composition, temperature) and return driving force,
            where a positive value means that precipitation is favorable
        
    interfacialComposition - function (optional)
        Function will take in (temperature, excess gibbs free energy) and
            return tuple (parent composition, precipitate composition) or (None, None) if precipitate is unstable

    diffusivity - function (optional)
        Function will take in (composition, temperature) and return diffusivity

    precPhase : str (optional)
        Precipitate phase to consider if binaryThermodynamics is defined

    Note: if binaryThermodynamics is not defined, then drivingForce and
        interfacial composition needs to be defined
    '''
    def __init__(self, binaryThermodynamics = None, drivingForce = None, interfacialComposition = None, diffusivity = None, precPhase = None):
        self.binTherm = binaryThermodynamics
        self.precPhase = precPhase
        
        #If no driving force or interfacial composition function is supplied, then use function from thermodynamics class
        if drivingForce is None:
            self.drivingForceFunction = lambda x, T, removeCache = True: self.binTherm.getDrivingForce(x, T, self.precPhase, removeCache)
        else:
            self.drivingForceFunction = drivingForce
        
        if interfacialComposition is None:
            self.interfacialCompositionFunction = lambda T, ge: self.binTherm.getInterfacialComposition(T, ge, self.precPhase)
        else:
            self.interfacialCompositionFunction = interfacialComposition

        if binaryThermodynamics is not None:
            self.diffusivityFunction = self.binTherm.getInterdiffusivity
        elif diffusivity is not None:
            self.diffusivityFunction = diffusivity

        self.eps = 1e-3

        #Driving force variables -----------------------------------------
        #Training data points
        self.drivingForce = []
        self.precComp = []
        self.dGcoords = []
        self.precCompIndices = []
        self.precCompCoords = []

        #Scaling factor to normalize distance
        self.XDGscale = None
        self.TDGscale = None

        #Surrogates
        self.DGfunction = 'linear'
        self.DGepsilon = 1
        self.DGsmooth = 0
        self.linearDG = True
        self.LogSurrDG = None
        self.LogSurrPrecComp = None
        self.SurrogateDrivingForce = None
        self.SurrogatePrecComp = None

        #Interfacial composition -----------------------------------------
        #Training data
        self.xParent = []
        self.xPrec = []
        self.ICcoords = []
        self.uniqueXPrec = []
        self.Gcoords = []
        self.G = []
        
        #Scaling factor to normalize distance
        self.TICscale = None
        self.GICscale = None
        self.XGscale = None
        
        #Surrogates
        self.ICfunction = 'linear'
        self.ICepsilon = 1
        self.ICsmooth = 0
        self.linearIC = True
        self.SurrogateParent = None
        self.SurrogatePrec = None
        self.SurrogateG = None
        self.LogSurrParent = None
        self.LogSurrPrec = None

        #Diffusivity -----------------------------------------------------
        #Training data
        self.Diffcoords = []
        self.Diff = []

        #Scaling factor
        #Scale temperature and free energy range is proportional to amount of data along given axis
        self.XDiffscale = None
        self.singleXDiff = False
        self.TDiffscale = None

        #Surrogates
        self.linearDiff = False
        self.Difffunction = 'linear'
        self.Diffepsilon = 1
        self.Diffsmooth = 0
        self.SurrogateDiff = None
        self.LogSurrDiff = None
        
        
    def trainDrivingForce(self, comps, temperature, function='linear', epsilon=1, smooth=0, scale='linear'):
        '''
        Creates training points and sets up surrogate models for driving force calculations
        
        Parameters
        ----------
        comps : array of floats
            Range of compositions for training points
        temperature : float or array
            Temperature or range of temperatures for training points
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the composition training data should be in log or linear scale
            Note: 'log' is recommended for dilute solutions
        '''
        #Convert temperature to array if not so
        if not hasattr(temperature, '__len__'):
            temperature = [temperature]
        
        #Ensure that composition are defined values if in log scale
        if scale == 'log':
            comps[comps == 0] = 1e-9
        
        self.drivingForce = []
        self.precComp = []
        self.dGcoords = []
        self.precCompIndices = []
        
        #Create training data
        n = 0   #Index for precCompIndices (needs to correspond to indices self.drivingForce array)
        for t in temperature:
            for x in comps:
                dG, xP = self.drivingForceFunction(x, t)

                #If driving force can be obtained (generally True)
                if dG is not None:
                    self.drivingForce.append(dG)
                    self.dGcoords.append([x, t])
                    
                    #If driving force is positive, then store nucleate composition
                    if xP is not None:
                        self.precComp.append(xP)
                        self.precCompIndices.append(n)
                    
                    n += 1

        self.dGcoords = np.array(self.dGcoords)   
        self.drivingForce = np.array(self.drivingForce)
        self.precComp = np.array(self.precComp)
        self.precCompIndices = np.array(self.precCompIndices)
        
        #Log scale
        if scale == 'log':
            self.dGcoords[:,0] = np.log10(self.dGcoords[:,0])
        
        #Scale data so that it the range is proportional to the amount of data points along the given axis
        if len(comps) == 1:
            self.XDGscale = 1
        else:
            self.XDGscale = (np.amax(self.dGcoords[:,0]) - np.amin(self.dGcoords[:,0])) / len(comps)
            self.dGcoords[:,0] /= self.XDGscale
            
        if len(temperature) == 1:
            self.TDGscale = 1
        else:
            self.TDGscale = (np.amax(self.dGcoords[:,1]) - np.amin(self.dGcoords[:,1])) / len(temperature)
            self.dGcoords[:,1] /= self.TDGscale

        #Create new array of coordinates for precipitate composition (this is to allow for filtering)
        self.precCompCoords = self.dGcoords[self.precCompIndices]

        #Filter points such that all points are separated by a distance by at least self.eps
        self.dGcoords, outputs = _filter_points(self.dGcoords, [self.drivingForce], self.eps)
        self.drivingForce = outputs[0]
        
        self.precCompCoords, outputs = _filter_points(self.precCompCoords, [self.precComp], self.eps)
        self.precComp = outputs[0]

        if scale == 'log':
            self.linearDG = False
        else:
            self.linearDG = True
        self.DGfunction = function
        self.DGepsilon = epsilon
        self.DGsmooth = np.amax([smooth, self.eps])
        
        self._createDGSurrogate()

    def _createDGSurrogate(self):
        '''
        Build surrogates for driving force
        '''
        if self.linearDG:
            self.SurrogateDrivingForce = Rbf(self.dGcoords[:,0], self.dGcoords[:,1], self.drivingForce, function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth)
            self.SurrogatePrecComp = Rbf(self.precCompCoords[:,0], self.precCompCoords[:,1], self.precComp, function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth)
        else:
            self.linearDG = False
            self.LogSurrDG = Rbf(self.dGcoords[:,0], self.dGcoords[:,1], self.drivingForce, function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth)
            self.LogSurrPrecComp = Rbf(self.precCompCoords[:,0], self.precCompCoords[:,1], self.precComp, function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth)
            self.SurrogateDrivingForce = lambda x, T: self.LogSurrDG(np.log10(x) / self.XDGscale, T)
            self.SurrogatePrecComp = lambda x, T: self.LogSurrPrecComp(np.log10(x) / self.XDGscale, T)

    def changeDrivingForceHyperparameters(self, function = 'linear', epsilon = 1, smooth = 0):
        '''
        Re-create surrogate model for driving force with updated hyperparameters

        Parameters
        ----------
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        '''
        if self.TDGscale is None:
            raise Exception("Driving force has not been trained.")

        self.DGfunction = function
        self.DGepsilon = epsilon
        self.DGsmooth = np.amax([smooth, self.eps])
        
        self._createDGSurrogate()            
        
    def getDrivingForce(self, x, T):
        '''
        Gets driving force from surrogate models
        
        Parameters
        ----------
        x : float or array of floats
            Composition
        T : float or array of floats
            Temperature, must be same length as x
        
        Returns
        -------
        (driving force, precipitate composition)
        Both will be same shape as x and T
        Positive driving force means that precipitate will form
        precipitate composition will be None if dG is negative
        '''
        if self.TDGscale is None:
            raise Exception("Driving force has not been trained.")

        x = np.atleast_1d(x)
        T = np.atleast_1d(T)

        if self.linearDG:
            dG = self.SurrogateDrivingForce(x / self.XDGscale, T / self.TDGscale)
            xP = self.SurrogatePrecComp(x / self.XDGscale, T / self.TDGscale)
            return dG, xP
        else:
            dG = self.SurrogateDrivingForce(x, T / self.TDGscale)
            xP = self.SurrogatePrecComp(x, T / self.TDGscale)

        return np.squeeze(dG), np.squeeze(xP)
        
    def trainInterfacialComposition(self, temperature, freeEnergy, function='linear', epsilon=1, smooth=0, scale = 'linear'):
        '''
        Creates training points and sets up surrogate models for interfacial composition

        Parameters
        ----------
        temperature : float or array
            Temperature or range of temperatures for training points
        freeEnergy : array of floats
            range of free energy values from Gibbs-Thomson contribution
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the matrix composition output should be in log or linear scale
            Note: 'log' is recommended for dilute solutions
        '''
        #Make temperature an array if not so
        if not hasattr(temperature, '__len__'):
            temperature = [temperature]
        
        self.xParent = []
        self.xPrec = []
        self.ICcoords = []
        
        #Create training data
        for t in temperature:
            for g in freeEnergy:
                xM, xP = self.interfacialCompositionFunction(t, g)
                
                #If precipitate can be form at T,G, then add to training array
                if xM is not None and xM > 0:
                    self.xParent.append(xM)
                    self.xPrec.append(xP)
                    
                    self.ICcoords.append([t, g])
                    
        self._buildInterfacialCompositionModels(temperature, freeEnergy, function, epsilon, smooth, scale)
        
    def trainInterfacialCompositionFromDrivingForceData(self, function='linear', epsilon=1, smooth=0, scale='linear'):
        '''
        Converts driving force data ([x, T] -> G) to interfacial composition data ([T, G] -> x)
        This may lead to inaccuracies in the precipitate composition since driving force calculations are done by sampling the free energy curve

        Parameters
        ----------
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the matrix composition output should be in log or linear scale
            Note: 'log' is recommended for dilute solutions
        '''
        self.xPrec = [x for x in self.precComp]
        
        #Take driving force training data and convert to interfacial composition training data
        if self.linearDG:
            self.xParent = [self.dGcoords[i, 0] * self.XDGscale for i in self.precCompIndices]
        else:
            self.xParent = [10**self.dGcoords[i, 0] * self.XDGscale for i in self.precCompIndices]
            
        self.ICcoords = []
        for i in self.precCompIndices:
            self.ICcoords.append([self.dGcoords[i, 1] * self.TDGscale, self.drivingForce[i]])
        
        self._buildInterfacialCompositionModels(np.unique(self.dGcoords[:, 1]) * self.TDGscale, self.drivingForce, function, epsilon, smooth, scale)
                    
    def _buildInterfacialCompositionModels(self, temperature, freeEnergy, function, epsilon, smooth, scale):
        '''
        Builds interfacial composition model (this is separate to allow for both training from new data or driving force data)

        Parameters
        ----------
        temperature - range of temperatures
        freeEnergy - range of free energies
        function - radial basis function
        epsilon - scale factor
        smooth - smoothing factor
        scale - linear or log scale
        '''
        #Create new training points finding the driving force for precipitation at the precipitate composition
        self.uniqueXPrec = np.unique(self.xPrec)
        if np.amax(self.uniqueXPrec) - np.amin(self.uniqueXPrec) < 1e-4:
            self.uniqueXPrec = np.array([self.uniqueXPrec[0]])
        self.Gcoords = []
        self.G = []
        
        for t in temperature:
            for x in self.uniqueXPrec:
                #Driving force calcs can be interpreted as the maximum free energy that can be contributed by the Gibbs-Thomson effect
                dG, _ = self.drivingForceFunction(x, t)

                #if driving force can be obtained (generally True)
                if dG is not None:
                    self.G.append(dG)
                    self.Gcoords.append([x, t])
                    
                    #Add these points to the composition surrogate training data as an endpoint (these are for the minimum particle radius)
                    self.ICcoords.append([t, dG])
                    self.xParent.append(x)
                    self.xPrec.append(x)
        
        self.G = np.array(self.G)
        self.Gcoords = np.array(self.Gcoords)
        
        self.xParent = np.array(self.xParent)
        self.xPrec = np.array(self.xPrec)
        self.ICcoords = np.array(self.ICcoords)
        
        #Invert free energy coordinates - this will make the spacing more consistent
        self.ICcoords[:,1] =  1 / self.ICcoords[:,1]
        
        #Scale temperature and free energy range is proportional to amount of data along given axis
        if len(temperature) == 1:
            self.TICscale = 1
        else:
            self.TICscale = (np.amax(self.ICcoords[:,0]) - np.amin(self.ICcoords[:,0])) / len(temperature)
            self.ICcoords[:,0] /= self.TICscale
            
        if len(freeEnergy) == 1:
            self.GICscale = 1
        else:
            self.GICscale = (np.amax(self.ICcoords[:,1]) - np.amin(self.ICcoords[:,1])) / len(freeEnergy)
            self.ICcoords[:,1] /= self.GICscale
            
        if len(self.uniqueXPrec) == 1:
            self.XGscale = 1
        else:
            self.XGscale = (np.amax(self.Gcoords[:,0]) - np.amin(self.Gcoords[:,0])) / len(self.uniqueXPrec)
            self.Gcoords[:,0] /= self.XGscale
            
        self.Gcoords[:,1] /= self.TICscale

        #Filter points such that all points are separated by a distance by at least self.eps
        self.ICcoords, outputs = _filter_points(self.ICcoords, [self.xParent, self.xPrec], self.eps)
        self.xParent = outputs[0]
        self.xPrec = outputs[1]
        
        self.Gcoords, outputs = _filter_points(self.Gcoords, [self.G], self.eps)
        self.G = outputs[0]

        if scale == 'log':
            self.linearIC = False
        else:
            self.linearIC = True
        self.ICfunction = function
        self.ICepsilon = epsilon
        self.ICsmooth = np.amax([smooth, self.eps])

        self._createICSurrogate()

    def _createICSurrogate(self):
        '''
        Build surrogates for interfacial composition
        '''
        if self.linearIC:
            self.SurrogateParent = Rbf(self.ICcoords[:,0], self.ICcoords[:,1], self.xParent, function = self.ICfunction, epsilon = self.ICepsilon, smooth=self.ICsmooth)
            self.SurrogatePrec = Rbf(self.ICcoords[:,0], self.ICcoords[:,1], self.xPrec, function = self.ICfunction, epsilon = self.ICepsilon, smooth=self.ICsmooth)  
        else:
            self.LogSurrParent = Rbf(self.ICcoords[:,0], self.ICcoords[:,1], np.log10(self.xParent), function = self.ICfunction, epsilon = self.ICepsilon, smooth=self.ICsmooth)
            self.LogSurrPrec = Rbf(self.ICcoords[:,0], self.ICcoords[:,1], np.log10(self.xPrec), function = self.ICfunction, epsilon = self.ICepsilon, smooth=self.ICsmooth) 
            
            self.SurrogateParent = lambda T, gExtraInverse: 10**(self.LogSurrParent(T, gExtraInverse))
            self.SurrogatePrec = lambda T, gExtraInverse: 10**(self.LogSurrPrec(T, gExtraInverse))
           
        if len(self.G) == 1:
            self.SurrogateG = lambda x, T: self.G[0]
        else:
            self.SurrogateG = Rbf(self.Gcoords[:,0], self.Gcoords[:,1], self.G, function = 'linear', epsilon = 1)

    def changeInterfacialCompositionHyperparameters(self, function = 'linear', epsilon = 1, smooth = 0):
        '''
        Re-create surrogate model for interfacial composition with updated hyperparameters

        Parameters
        ----------
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        '''
        if self.TICscale is None:
            raise Exception("Interfacial composition has not been trained.")

        self.ICfunction = function
        self.ICepsilon = epsilon
        self.ICsmooth = np.amax([smooth, self.eps])

        self._createICSurrogate()
            
    def getInterfacialComposition(self, T, gExtra = 0):
        '''
        Gets Interfacial composition from surrogate models
        
        Parameters
        ----------
        T : float or array of floats
            Temperature
        gExtra : float or array of floats
            Free energy from Gibbs-Thomson contribution (must be same length as T)
        
        Returns
        -------
        (composition of parent phase, composition of precipitate phase)
        Composition will be in same shape as T and gExtra
        Will return (None, None) if gExtra is large enough that 
            precipitate becomes unstable
        '''
        if self.TICscale is None:
            raise Exception("Interfacial composition has not been trained.")

        #Convert arrays to Numpy arrays for math operations
        #Also raise gExtra to lowest training value to avoid erroneous values
        #   NOTE: Do not use this as a way to minimize the amount of training points!!
        #   While this is a safegaurd to avoid weird results, this creates a constant growth rate
        #   for any values of gExtra less than the lowest training point, which can lead to non-realistic
        #   values in the precipitate simulation (i.e. precipitates growing in an unsaturated matrix).
        #   Just train more points at lower gExtra values (larger radius sizes)
        gExtra = np.atleast_1d(gExtra)
        gExtra[(gExtra*self.GICscale) < 1/np.amax(self.ICcoords[:,1])] = 1 / (np.amax(self.ICcoords[:,1]) * self.GICscale)

        #If gExtra is array and T isn't, then convert T to array
        #This is to keep consistent with Thermodynamics counterpart
        T = np.atleast_1d(T)
        if len(gExtra) > 1 and len(T) == 1:
            T = T*np.ones(gExtra.shape)

        xM, xP = self.SurrogateParent(T / self.TICscale, 1 / (gExtra * self.GICscale)), self.SurrogatePrec(T / self.TICscale, 1 / (gExtra * self.GICscale))
        dG = self.SurrogateG(xP / self.XGscale, T / self.TICscale)
        noneVals = (dG < gExtra)
        xM[noneVals] = -1
        xP[noneVals] = -1
                    
        return np.squeeze(xM), np.squeeze(xP)

    def trainInterdiffusivity(self, comps, temperature, function='linear', epsilon=1, smooth=0, scale='linear'):
        '''
        Trains interdiffusivity from mobility parameters

        Parameters
        ----------
        comps : array of floats
            Range of compositions for training points
        temperature : float or array
            Temperature or range of temperatures for training points
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the diffusivity output should be in log or linear scale
        '''
        #Convert composition and temperature to array if not so
        if not hasattr(comps, '__len__'):
            comps = [comps]

        if not hasattr(temperature, '__len__'):
            temperature = [temperature]

        self.Diffcoords = []
        self.Diff = []

        for x in comps:
            for t in temperature:
                self.Diff.append(self.diffusivityFunction(x, t))
                self.Diffcoords.append([x, t])

        self.Diffcoords = np.array(self.Diffcoords)
        self.Diff = np.array(self.Diff)

        #Scale temperature and free energy range is proportional to amount of data along given axis
        if len(comps) == 1:
            self.XDiffscale = 1
            self.singleXDiff = True
        else:
            self.XDiffscale = (np.amax(self.Diffcoords[:,0]) - np.amin(self.Diffcoords[:,0])) / len(comps)
            self.Diffcoords[:,0] /= self.XDiffscale
            self.singleXDiff = False

        if len(temperature) == 1:
            self.TDiffscale = 1
        else:
            self.TDiffscale = (np.amax(self.Diffcoords[:,1]) - np.amin(self.Diffcoords[:,1])) / len(temperature)
            self.Diffcoords[:,1] /= self.TDiffscale

        #Filter diffusivity points
        self.Diffcoords, outputs = _filter_points(self.Diffcoords, [self.Diff], self.eps)
        self.Diff = outputs[0]

        if scale == 'log':
            self.linearDiff = False
        else:
            self.linearDiff = True
        self.Difffunction = function
        self.Diffepsilon = epsilon
        self.Diffsmooth = np.amax([smooth, self.eps])

        self._createDiffSurrogate()

    def _createDiffSurrogate(self):
        '''
        Builds surrogate for diffusivity
        '''
        #If only 1 data point, then create constant function
        if len(self.Diff) == 1:
            self.SurrogateDiff = lambda x, T: self.Diff[0]
            return

        #Build surrogates
        if self.linearDiff:
            if self.singleXDiff:
                self.SurrogateDiff = Rbf(self.Diffcoords[:,1], self.Diff, function = self.Difffunction, epsilon = self.Diffepsilon, smooth = self.Diffsmooth)
            else:
                self.SurrogateDiff = Rbf(self.Diffcoords[:,0], self.Diffcoords[:,1], self.Diff, function = self.Difffunction, epsilon = self.Diffepsilon, smooth = self.Diffsmooth) 
        else:
            if self.singleXDiff:
                self.LogSurrDiff = Rbf(self.Diffcoords[:,1], np.log10(self.Diff), function = self.Difffunction, epsilon = self.Diffepsilon, smooth = self.Diffsmooth)
                self.SurrogateDiff = lambda T: 10**(self.LogSurrDiff(T))
            else:
                self.LogSurrDiff = Rbf(self.Diffcoords[:,0], self.Diffcoords[:,1], np.log10(self.Diff), function = self.Difffunction, epsilon = self.Diffepsilon, smooth = self.Diffsmooth) 
                self.SurrogateDiff = lambda x, T: 10**(self.LogSurrDiff(x, T))

    def changeDiffusivityHyperparameters(self, function = 'linear', epsilon = 1, smooth = 0):
        '''
        Re-create surrogate model for diffusivity with updated hyperparameters

        Parameters
        ----------
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        '''
        if self.XDiffscale is None:
            raise Exception("Diffusivity has not been trained.")

        self.Difffunction = function
        self.Diffepsilon = epsilon
        self.Diffsmooth = np.amax([smooth, self.eps])

        self._createDiffSurrogate()

    def getInterdiffusivity(self, x, T):
        '''
        Returns interdiffusivity

        Parameters
        ----------
        x : float or array of floats
            Composition
        T : float or array of floats
            Temperature (must be same length as x)
        
        Returns
        -------
        diffusivity (same shape as x and T)
        '''
        x = np.atleast_1d(x)
        T = np.atleast_1d(T)

        if self.XDiffscale is None:
            raise Exception("Diffusivity has not been trained.")

        if self.singleXDiff:
            return np.squeeze(self.SurrogateDiff(T / self.TDiffscale))
        else:
            return np.squeeze(self.SurrogateDiff(x / self.XDiffscale, T / self.TDiffscale))
        
    def drivingForceTrainingTemperature(self):
        '''
        Returns the temperature coordinates of driving force training points
        '''
        return self.dGcoords[:,1] * self.TDGscale
        
    def drivingForceTrainingComposition(self):
        '''
        Returns the composition coordinates of driving force training points
        '''
        if self.linearDG:
            return self.dGcoords[:,0] * self.XDGscale
        else:
            return 10**(self.dGcoords[:,0] * self.XDGscale)
            
    def interfacialCompositionTrainingTemperature(self):
        '''
        Returns the temperature coordinates of interfacial composition training points
        '''
        return self.ICcoords[:,0] * self.TICscale
        
    def interfacialCompositionTrainingGibbsThomson(self):
        '''
        Returns the Gibbs-Thomson contribution coordinates of interfacial composition training points
        '''
        return 1 / (self.ICcoords[:,1] * self.GICscale)

    def save(self, fileName):
        '''
        Pickles surrogate data
        Note: this will remove the user-defined driving force and interfacial compositions
            This is not a problem; however, a loaded surrogate will not be
            able to be re-trained with different training points

        Parameters
        ----------
        fileName : str
        '''
        self.binTherm = None
        self.drivingForceFunction = None
        self.interfacialCompositionFunction = None
        self.diffusivityFunction = None

        self.LogSurrDG = None
        self.LogSurrPrecComp = None
        self.SurrogateDrivingForce = None
        self.SurrogatePrecComp = None

        self.SurrogateParent = None
        self.SurrogatePrec = None
        self.SurrogateG = None
        self.LogSurrParent = None
        self.LogSurrPrec = None

        self.SurrogateDiff = None
        self.LogSurrDiff = None

        with open(fileName, 'wb') as file:
            pickle.dump(self, file)

        #Re-create surrogates so that it could still be used after saving
        if self.XDGscale is not None:
            self._createDGSurrogate()
        if self.TICscale is not None:
            self._createICSurrogate()
        if self.XDiffscale is not None:
            self._createDiffSurrogate()

    def load(fileName):
        '''
        Loads data from a pickled surrogate and builds driving force and interfacial composition functions

        Parameters
        ----------
        fileName : str

        Returns
        -------
        BinarySurrogate object
        '''
        surr = None
        with open(fileName, 'rb') as file:
            surr = pickle.load(file)

        #Re-create surrogates so that it could still be used after saving
        if surr.XDGscale is not None:
            surr._createDGSurrogate()
        if surr.TICscale is not None:
            surr._createICSurrogate()
        if surr.XDiffscale is not None:
            surr._createDiffSurrogate()

        return surr
        
class MulticomponentSurrogateOld:
    '''
    Handles surrogate models for driving force, interfacial composition
        and growth rate in a multicomponent system
    
    Parameters
    ----------
    thermodynamics - MulticomponentThermodynamics (optional)
        Driving force, interfacial composition and 
            curvature functions will be taken from this
        If None, then drivingForce and curvature will need to be defined
        
    drivingForce - function (optional))
        Function will take in (composition, temperature) and return driving force
            where a positive value means that precipitation is favorable
            composition is an array for each element, excluding the reference element
        
    interfacialComposition - function (optional)
        Takes in (composition, temperature, gExtra) and returns matrix and precipitate composition
            composition is an array for each element, excluding the reference element
        Function should return (None, None) if precipitate is unstable

    curvature - function (optional)
        Function will take in (composition, temperature) and return the following:
            {D-1 dCbar / dCbar^T M-1 dCbar} - for calculating interfacial composition of matrix
            {1 / dCbar^T M-1 dCbar} - for calculating growth rate
            {Gb^-1 Ga} - for calculating precipitate composition
            Ca - interfacial composition of matrix phase
            Cb - interfacial composition of precipitate phase
        Function will return (None, None, None, None, None) if composition is outside two phase region
    
    precPhase : str (optional)
        Precipitate phase to consider if binaryThermodynamics is defined

    Note: if binaryThermodynamics is not defined, then drivingForce and
        interfacial composition needs to be defined
    '''
    def __init__(self, thermodynamics = None, drivingForce = None, interfacialComposition = None, curvature = None, precPhase = None):
        self.therm = thermodynamics
        self.precPhase = precPhase
        self.elements = self.therm.elements[1:-1]
        
        #Grab driving force and curvature function from thermodynamics class if not supplied
        if drivingForce is None:
            self.drivingForceFunction = lambda x, T, removeCache = True: self.therm.getDrivingForce(x, T, self.precPhase, removeCache)
        else:
            self.drivingForceFunction = drivingForce

        if interfacialComposition is None:
            self.interfacialCompositionFunction = lambda x, T, gExtra: self.therm.getInterfacialComposition(x, T, gExtra, self.precPhase)
        else:
            self.interfacialCompositionFunction = interfacialComposition

        #TODO: curvatureFactor should take in searchDir from drivingForceFunction
        #    but this needs to be compatible with the same parameters
        if curvature is None:
            #self.curvature = self.therm.curvatureFactor
            #self.curvature = lambda x, T, training = True: self.therm.curvatureFactor(x, T, self.precPhase, training)
            self.curvature = lambda x, T, removeCache = True, searchDir = None, computeSearchDir = True: self.therm.curvatureFactor(x, T, self.precPhase, removeCache=removeCache, searchDir=searchDir, computeSearchDir=computeSearchDir)
        else:
            self.curvature = curvature

        self.eps = 1e-3

        #Driving force ---------------------------------------------------
        #Training data
        self.drivingForce = []
        self.precComp = []
        self.dGcoords = []
        self.precCompIndices = []
        self.precCompCoords = []

        #Scaling factor
        self.XDGscale = []
        self.TDGscale = None
        
        #Surrogates
        self.linearDG = True
        self.DGfunction = 'linear'
        self.DGepsilon = 1
        self.DGsmooth = 0
        self.SurrogateDrivingForce = None
        self.SurrPrecComp = None
        self.SurrogatePrecComp = None
        self.LogSurrDG = None
        self.LogSurrPrecComp = None

        #Interfacial composition -----------------------------------------
        #Training data
        self.Dc = []
        self.Mc = []
        self.Gba = []
        self.beta = []
        self.Ca = []
        self.Cb = []
        self.ICcoords = []

        #Scaling factors
        self.XICscale = None
        self.TICscale = None

        #Interfacial composition surrogates
        self.linearIC = True
        self.ICfunction = 'linear'
        self.ICepsilon = 1
        self.ICsmooth = 0

        self.SurrDc = None
        self.SurrCa = None
        self.SurrCb = None
        self.SurrGba = None

        self.LogSurrDc = None
        self.LogSurrMc = None
        self.LogSurrBeta = None
        self.LogSurrCa = None
        self.LogSurrCb = None
        self.LogSurrGba = None
        
        self.SurrogateDc = None
        self.SurrogateMc = None
        self.SurrogateBeta = None
        self.SurrogateCa = None
        self.SurrogateCb = None
        self.SurrogateGba = None

        
    def trainDrivingForce(self, comps, temperature, function='linear', epsilon=1, smooth=0, scale='linear'):
        '''
        Creates training points and sets up surrogate models for driving force calculations
        
        Parameters
        ----------
        comps : 2-D array
            Range of compositions for training points
            0th axis represents an individual training point
            1st axis represents element composition
        temperature : float or array
            Range of temperatures for training points
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the composition training data should be in log or linear scale
            Note: 'log' is recommended for dilute solutions
        '''
        #Make temperature an array if not so
        if not hasattr(temperature, '__len__'):
            temperature = [temperature]

        #Ensure that composition are defined values when using log scale
        if scale == 'log':
            for x in comps:
                x[x == 0] = 1e-9
        
        #Create training data
        self.drivingForce = []
        self.precComp = []
        self.dGcoords = []
        self.precCompIndices = []
        
        n = 0   #Index for precCompIndices (needs to correspond to indices self.drivingForce array)
        for t in temperature:
            for x in comps:
                dG, xP = self.drivingForceFunction(x, t)
                
                #If driving force can be obtained (generally True)
                if dG is not None:
                    self.drivingForce.append(dG)
                    self.dGcoords.append(np.concatenate((x, [t])))
                    
                    #If driving force is positive, then add nucleate composition
                    #Also determine equilibrium matrix composition, then add training point where driving force is 0
                    
                    if xP is not None:
                        self.precComp.append(xP)
                        self.precCompIndices.append(n)
                        
                        xMeq, xPeq = self.interfacialCompositionFunction(x, t, 0)
                        if any(xMeq == -1):
                            self.drivingForce.append(0)
                            self.dGcoords.append(np.concatenate((xMeq[1:], [t])))
                            n += 1
                            self.precComp.append(xPeq[1:])
                            self.precCompIndices.append(n)
                        
                    n += 1
                
        self.drivingForce = np.array(self.drivingForce)
        self.precComp = np.array(self.precComp)
        self.dGcoords = np.array(self.dGcoords)
        self.precCompIndices = np.array(self.precCompIndices)
        
        #Log scale on only composition (good for dilute solutions)
        if scale == 'log':
            self.dGcoords[:,:-1] = np.log10(self.dGcoords[:,:-1])
        
        #Scale data so range is proportional to amount of data along given axis
        if len(comps) == 1:
            self.XDGscale = np.ones(len(self.elements))
        else:
            self.XDGscale = (np.amax(self.dGcoords[:,:-1]) - np.amin(self.dGcoords[:,:-1])) / len(comps)
            self.dGcoords[:,:-1] /= self.XDGscale
            
        if len(temperature) == 1:
            self.TDGscale = 1
        else:
            self.TDGscale = (np.amax(self.dGcoords[:,-1]) - np.amin(self.dGcoords[:,-1])) / len(temperature)
            self.dGcoords[:,-1] /= self.TDGscale

        #Create new array of coordinates for precipitate composition (this is to allow for filtering)
        self.precCompCoords = self.dGcoords[self.precCompIndices]

        #Filter points such that all points are separated by a distance by at least self.eps
        self.dGcoords, outputs = _filter_points(self.dGcoords, [self.drivingForce], self.eps)
        self.drivingForce = outputs[0]
        
        self.precCompCoords, outputs = _filter_points(self.precCompCoords, [self.precComp], self.eps)
        self.precComp = outputs[0]

        if scale == 'log':
            self.linearDG = False
        else:
            self.linearDG = True
        self.DGfunction = function
        self.DGepsilon = epsilon
        self.DGsmooth = np.amax([smooth, self.eps])

        self._createDGSurrogate()

    def _createDGSurrogate(self):
        '''
        Builds surrogate for driving force
        '''
        if self.linearDG:
            arguments = [self.dGcoords[:,i] for i in range(len(self.dGcoords[0]))]
            self.SurrogateDrivingForce = Rbf(*arguments, self.drivingForce, function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth)
            
            arguments = [self.precCompCoords[:,i] for i in range(len(self.dGcoords[0]))]
            self.SurrPrecComp = [Rbf(*arguments, self.precComp[:,i], function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth) for i in range(len(self.elements))]
            self.SurrogatePrecComp = lambda x, T: np.array([self.SurrPrecComp[i](*x, T) for i in range(len(self.elements))])
        else:
            arguments = [self.dGcoords[:,i] for i in range(len(self.dGcoords[0]))]
            self.LogSurrDG = Rbf(*arguments, self.drivingForce, function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth)
            self.LogSurrPrecComp = [Rbf(*arguments, self.precComp[:,i], function=self.DGfunction, epsilon=self.DGepsilon, smooth=self.DGsmooth) for i in range(len(self.elements))]
            self.SurrogateDrivingForce = lambda x, T: self.LogSurrDG(*(np.log10(x) / self.XDGscale), T)
            self.SurrogatePrecComp = lambda x, T: np.array([self.LogSurrPrecComp[i](*(np.log10(x) / self.XDGscale), T) for i in range(len(self.elements))])

    def changeDrivingForceHyperparameters(self, function = 'linear', epsilon = 1, smooth = 0):
        '''
        Re-create surrogate model for driving force with updated hyperparameters

        Parameters
        ----------
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        '''
        if self.TDGscale is None:
            raise Exception("Driving force has not been trained.")

        self.DGfunction = function
        self.DGepsilon = epsilon
        self.DGsmooth = np.amax([smooth, self.eps])
        
        self._createDGSurrogate()
        
    def getDrivingForce(self, x, T):
        '''
        Gets driving force from surrogate models
        
        Parameters
        ----------
        x : array or 2D array
            Composition (array of float for each minor element)
            2D arrays will have 0th axis for each set and 1st axis for composition
        T : float or array
            Temperature (must be float or same length as 0th axis of x if array)
        
        Returns
        -------
        driving force (positive value means that precipitate is stable)
        '''
        if self.TDGscale is None:
            raise Exception("Driving force has not been trained.")

        T = np.atleast_1d(T)
        x = np.atleast_2d(x).T

        if self.linearDG:
            dG = self.SurrogateDrivingForce(*(x / self.XDGscale), T / self.TDGscale)
            return np.squeeze(dG), np.squeeze(self.SurrogatePrecComp(x / self.XDGscale, T / self.TDGscale).T)
        else:
            dG = self.SurrogateDrivingForce(x, T / self.TDGscale)
            return np.squeeze(dG), np.squeeze(self.SurrogatePrecComp(x, T / self.TDGscale))

    def trainCurvature(self, comps, temperature, function='linear', epsilon=1, smooth=0, scale='linear'):
        '''
        Trains for curvature factor (from Phillipes and Voorhees - 2013) as a function of composition and temperature

        Creates 5 surrogates
        {D-1 dCbar / dCbar^T M-1 dCbar} - for calculating interfacial composition of matrix
        {1 / dCbar^T M-1 dCbar} - for calculating growth rate
        {Gb^-1 Ga} - for calculating precipitate composition
        Ca - interfacial composition of matrix phase
        Cb - interfacial composition of precipitate phase


        Parameters
        ----------
        comps : 2D array of floats
            Range of compositions for training points
            0th axis represents a training point
            1st axis represents element compositions
        temperature : float or array
            Range of temperatures for training points
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the matrix composition output should be in log or linear scale
            Note: 'log' is recommended for dilute solutions
        '''
        #Make temperature an array if not so
        if not hasattr(temperature, '__len__'):
            temperature = [temperature]

        #Ensure that composition are defined values when using log scale
        if scale == 'log':
            for x in comps:
                x[x == 0] = 1e-9
        
        #Create training data
        self.Dc = []
        self.Mc = []
        self.Gba = []
        self.beta = []
        self.Ca = []
        self.Cb = []
        self.ICcoords = []
        
        for t in temperature:
            for x in comps:
                results = self.curvature(x, t)

                if results is not None:
                    #Since Dc, Mc and Gba is constant for a given tie-line, add 3 training data points (at bulk compostion and phase boundaries)
                    #This should give more accurate values at very small or very large supersaturations without having to calculate a lot of training data
                    compCoords = [x, results.c_eq_alpha, results.c_eq_beta]
                    for i in range(3):
                        self.Dc.append(results.dc)
                        self.Mc.append(results.mc)
                        self.Gba.append(results.gba)
                        self.beta.append(results.beta)
                        self.Ca.append(results.c_eq_alpha)
                        self.Cb.append(results.c_eq_beta)
                        self.ICcoords.append(np.concatenate((compCoords[i], [t])))

        self.Dc = np.array(self.Dc)
        self.Mc = np.array(self.Mc)
        self.Gba = np.array(self.Gba)
        self.beta = np.array(self.beta)
        self.Ca = np.array(self.Ca)
        self.Cb = np.array(self.Cb)
        self.ICcoords = np.array(self.ICcoords)
        
        #Log scale only on compositions (good for low solubility)
        if scale == 'log':
            self.ICcoords[:,:-1] = np.log10(self.ICcoords[:,:-1])
        
        #Scale data so range is proportional to amount of data along given axis
        if len(comps) == 1:
            self.XICscale = np.ones(len(self.elements))
        else:
            self.XICscale = (np.amax(self.ICcoords[:,:-1]) - np.amin(self.ICcoords[:,:-1])) / len(comps)
            self.ICcoords[:,:-1] /= self.XICscale
            
        if len(temperature) == 1:
            self.TICscale = 1
        else:
            self.TICscale = (np.amax(self.ICcoords[:,-1]) - np.amin(self.ICcoords[:,-1])) / len(temperature)
            self.ICcoords[:,-1] /= self.TICscale

        #Filter points such that all points are separated by a distance by at least self.eps
        self.ICcoords, outputs = _filter_points(self.ICcoords, [self.Dc, self.Mc, self.Gba, self.beta, self.Ca, self.Cb], self.eps)
        self.Dc = outputs[0]
        self.Mc = outputs[1]
        self.Gba = outputs[2]
        self.beta = outputs[3]
        self.Ca = outputs[4]
        self.Cb = outputs[5]

        if scale == 'log':
            self.linearIC = False
        else:
            self.linearIC = True
        self.ICfunction = function
        self.ICepsilon = epsilon
        self.ICsmooth = np.amax([smooth, self.eps])

        self._createICSurrogate()

    def _createICSurrogate(self):
        '''
        Builds surrogate for interfacial composition and curvature
        '''
        if self.linearIC:
            arguments = [self.ICcoords[:,i] for i in range(len(self.ICcoords[0]))]
            self.SurrogateMc = Rbf(*arguments, self.Mc, function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth)
            self.SurrogateBeta = Rbf(*arguments, self.beta, function=self.ICfunction, epislon=self.ICepsilon, smooth=self.ICsmooth)

            self.SurrDc = [Rbf(*arguments, self.Dc[:,i], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for i in range(len(self.elements))]
            self.SurrCa = [Rbf(*arguments, self.Ca[:,i], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for i in range(len(self.elements))]
            self.SurrCb = [Rbf(*arguments, self.Cb[:,i], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for i in range(len(self.elements))]
            self.SurrGba = [[Rbf(*arguments, self.Gba[:,i,j], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for j in range(len(self.elements))] for i in range(len(self.elements))]
            
            self.SurrogateDc = lambda x, T: np.array([self.SurrDc[i](*x, T) for i in range(len(self.elements))])
            self.SurrogateCa = lambda x, T: np.array([self.SurrCa[i](*x, T) for i in range(len(self.elements))])
            self.SurrogateCb = lambda x, T: np.array([self.SurrCb[i](*x, T) for i in range(len(self.elements))])
            self.SurrogateGba = lambda x, T: np.array([[self.SurrGba[i][j](*x, T) for j in range(len(self.elements))] for i in range(len(self.elements))])
            
        else:
            arguments = [self.dGcoords[:,i] for i in range(len(self.dGcoords[0]))]
            self.LogSurrDc = [Rbf(*arguments, self.Dc[:,i], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for i in range(len(self.elements))]
            self.LogSurrMc = Rbf(*arguments, self.Mc, function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth)
            self.LogSurrBeta = Rbf(*arguments, self.beta, function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth)
            self.LogSurrCa = [Rbf(*arguments, self.Ca[:,i], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for i in range(len(self.elements))]
            self.LogSurrCb = [Rbf(*arguments, self.Cb[:,i], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for i in range(len(self.elements))]
            self.LogSurrGba = [[Rbf(*arguments, self.Gba[:,i,j], function=self.ICfunction, epsilon=self.ICepsilon, smooth=self.ICsmooth) for j in range(len(self.elements))] for i in range(len(self.elements))]
            
            self.SurrogateDc = lambda x, T: np.array([self.LogSurrDc[i](*(np.log10(x) / self.XICscale), T) for i in range(len(self.elements))])
            self.SurrogateMc = lambda x, T: self.LogSurrMc(*(np.log10(x) / self.XICscale), T)
            self.SurrogateBeta = lambda x, T: self.LogSurrBeta(*(np.log10(x) / self.XICscale), T)
            self.SurrogateCa = lambda x, T: np.array([self.LogSurrCa[i](*(np.log10(x) / self.XICscale), T) for i in range(len(self.elements))])
            self.SurrogateCb = lambda x, T: np.array([self.LogSurrCb[i](*(np.log10(x) / self.XICscale), T) for i in range(len(self.elements))])
            self.SurrogateGba = lambda x, T: np.array([[self.LogSurrGba[i][j](*(np.log10(x) / self.XICscale), T) for j in range(len(self.elements))] for i in range(len(self.elements))])

    def changeCurvatureHyperparameters(self, function = 'linear', epsilon = 1, smooth = 0):
        '''
        Re-create surrogate model for curvature factors with updated hyperparameters

        Parameters
        ----------
        function : str (optional)
            Radial basis function to use (defaults to 'linear')
            Other functions are 'multiquadric', 'inverse_multiquadric',
                'gaussian', 'cubic', 'quintic' and 'thin_plate'
        epsilon : float
            Scale for radial basis function (defaults to 1)
            Training data will be scaled automatically
                such that optimal scale is around 1
        smooth : float
            Smoothness of interpolation, (defaults to 0, where interpolation will go through all points)
        scale : float
            Whether the composition training data should be in log or linear scale
            Note: 'log' is recommended for dilute solutions
        '''
        if self.TICscale is None:
            raise Exception("Curvature has not been trained.")

        self.ICfunction = function
        self.ICepsilon = epsilon
        self.ICsmooth = np.amax([smooth, self.eps])

        self._createICSurrogate()

    def getCurvature(self, x, T):
        '''
        Gets driving force from surrogate models
        
        Parameters
        ----------
        x : array or 2D array
            Composition (array of float for each minor element)
            2D arrays will have 0th axis for each set and 1st axis for composition
        T : float or array
            Temperature (must be float or same length as 0th axis of x if array)
        
        Returns
        -------
        Curvature factors
            {D-1 dCbar / dCbar^T M-1 dCbar} - for calculating interfacial composition of matrix
            {1 / dCbar^T M-1 dCbar} - for calculating growth rate
            {Gb^-1 Ga} - for calculating precipitate composition
            Ca - interfacial composition of matrix phase
            Cb - interfacial composition of precipitate phase
        Note: this function currently does not return (None, None, None, None, None)
            if precipitate is unstable
        '''
        if self.TICscale is None:
            raise Exception("Curvature has not been trained.")

        if self.linearIC:
            dc = self.SurrogateDc(x / self.XICscale, T / self.TICscale)
            mc = self.SurrogateMc(*(x / self.XICscale), T / self.TICscale)
            gba = self.SurrogateGba(x / self.XICscale, T / self.TICscale)
            beta = self.SurrogateBeta(*(x / self.XICscale), T / self.TICscale)
            ca = self.SurrogateCa(x / self.XICscale, T / self.TICscale)
            cb = self.SurrogateCb(x / self.XICscale, T / self.TICscale)
            
        else:
            dc = self.SurrogateDc(x, T / self.TICscale)
            mc = self.SurrogateMc(x, T / self.TICscale)
            gba = self.SurrogateGba(x, T / self.TICscale)
            beta = self.SurrogateBeta(x, T / self.TICscale)
            ca = self.SurrogateCa(x, T / self.TICscale)
            cb = self.SurrogateCb(x, T / self.TICscale)
            
        return CurvatureOutput(dc=np.squeeze(dc), 
                                mc=np.squeeze(mc), 
                                gba=np.squeeze(gba), 
                                beta=np.squeeze(beta), 
                                c_eq_alpha=np.squeeze(ca), 
                                c_eq_beta=np.squeeze(cb))

    def getGrowthAndInterfacialComposition(self, x, T, dG, R, gExtra, searchDir = None):
        '''
        Returns growth rate and interfacial compostion given Gibbs-Thomson contribution

        Parameters
        ----------
        x : array of floats
            Matrix composition
        T : float
            Temperature
        dG : float
            Driving force
        R : float or array
            Precipitate radius
        gExtra : float or array
            Gibbs-Thomson contribution corresponding to R
            Must be same shape as R
        
        Returns
        -------
        (growth rate, matrix composition, precipitate composition)
        Growth rate will be float or array depending on R
        matrix and precipitate composition will be 1D or 2D array depending on R
            1D array will be length of composition
            2D array will have 0th axis be length of R and 1st axis be length of composition
        '''
        if self.TICscale is None:
            raise Exception("Curvature needs to be trained to calculated interfacial composition.")

        R = np.atleast_1d(R)
        gExtra = np.atleast_1d(gExtra)
        x = np.array(x)
        T = np.array(T)
        curv_results = self.getCurvature(x, T)

        Rdiff = (dG - gExtra)
        gr = (curv_results.mc / R) * Rdiff

        calpha = x[np.newaxis,:] - np.outer(Rdiff, curv_results.dc)
        dca = calpha - curv_results.c_eq_alpha[np.newaxis,:]
        dcb = np.matmul(curv_results.gba, dca.T).T
        cbeta = curv_results.c_eq_beta[np.newaxis,:] + dcb

        calpha = np.clip(calpha, 0, 1)
        cbeta = np.clip(cbeta, 0, 1)

        return GrowthRateOutput(growth_rate=np.squeeze(gr), 
                                c_alpha=np.squeeze(calpha), 
                                c_beta=np.squeeze(cbeta), 
                                c_eq_alpha=np.squeeze(curv_results.c_eq_alpha), 
                                c_eq_beta=np.squeeze(curv_results.c_eq_beta))

    def impingementFactor(self, x, T, searchDir = None):
        '''
        Calculates impingement factor for nucleation rate calculations

        Parameters
        ----------
        x : array of floats
            Matrix composition
        T : float
            Temperature
        '''
        if self.TICscale is None:
            raise Exception("Curvature needs to be trained to calculated impingement factor.")
        #return self.SurrogateBeta(x, T / self.TICscale)
        if self.linearIC:
            return self.SurrogateBeta(*(x / self.XICscale), T / self.TICscale)
        else:
            return self.SurrogateBeta(x, T / self.TICscale)

    def save(self, fileName):
        '''
        Pickles surrogate data
        Note: this will remove the user-defined driving force and curvature functions
            This is not a problem; however, a loaded surrogate will not be
            able to be re-trained with different training points

        Parameters
        ----------
        fileName : str
        '''
        self.therm = None
        self.drivingForceFunction = None
        self.interfacialCompositionFunction = None
        self.curvature = None

        self.SurrogateDrivingForce = None
        self.SurrPrecComp = None
        self.SurrogatePrecComp = None
        self.LogSurrDG = None
        self.LogSurrPrecComp = None

        self.SurrDc = None
        self.SurrCa = None
        self.SurrCb = None
        self.SurrGba = None

        self.LogSurrDc = None
        self.LogSurrMc = None
        self.LogSurrBeta = None
        self.LogSurrCa = None
        self.LogSurrCb = None
        self.LogSurrGba = None
        
        self.SurrogateDc = None
        self.SurrogateMc = None
        self.SurrogateBeta = None
        self.SurrogateCa = None
        self.SurrogateCb = None
        self.SurrogateGba = None

        with open(fileName, 'wb') as file:
            pickle.dump(self, file)

        if self.TDGscale is not None:
            self._createDGSurrogate()
        if self.TICscale is not None:
            self._createICSurrogate()

    def load(fileName):
        '''
        Loads data from a pickled surrogate and builds driving force and curvature functions

        Parameters
        ----------
        fileName : str

        Returns
        -------
        MulticomponentSurrogate object
        '''
        surr = None
        with open(fileName, 'rb') as file:
            surr = pickle.load(file)

        if surr.TDGscale is not None:
            surr._createDGSurrogate()
        if surr.TICscale is not None:
            surr._createICSurrogate()

        return surr
        