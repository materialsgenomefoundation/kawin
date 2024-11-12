from collections import namedtuple

import numpy as np

from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
from kawin.precipitation.PrecipitationParameters import MatrixParameters, PrecipitateParameters, AVOGADROS_NUMBER, BOLTZMANN_CONSTANT

NucleationData = namedtuple('NucleationData', [
    'x', 'T',
    'nucleation_rate', 'chemical_driving_force', 'volumetric_driving_force', 'Rcrit', 'Gcrit', 
    'precipitate_composition', 'Z', 'beta', 'tau', 'nucleation_radius'])

def volumetricDrivingForce(therm: GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, aspectRatio = 1, removeCache = False):
    '''
    Computes volumetric driving force (chemical DG / VM - strain energy)
        Strain energy will always reduce the driving force since the precipitate will add strain to the matrix
        In the case where the matrix is prestrained and the precipitate will relax the matrix, then the strain
        energy is negative
    '''
    x, T = therm.process_xT_arrays(x, T, squeeze_X=True)

    chemDGs, betaComp = therm.getDrivingForce(x, T, precPhase=precipitate.phase, removeCache=removeCache)
    volDGs = chemDGs / precipitate.volume.Vm
    volDGs -= precipitate.strainEnergy.strainEnergy(precipitate.shapeFactor.description.normalRadiiFromAR(aspectRatio))

    return np.squeeze(chemDGs), np.squeeze(volDGs), np.squeeze(betaComp)

def nucleationBarrier(volumeDrivingForce, precipitate : PrecipitateParameters, aspectRatio = 1):
    '''
    Critical Gibbs free energy and radius at the nucleation barrier
    For bulk and dislocation nucleation
        Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
        Gcrit = (4*pi/3)*gamma*Rcrit^2
    For grain boundary, edge or corner nucleation, critical G and R is computed to in GBFactors to account for grain boundary energy
    '''
    volumeDrivingForce = np.atleast_1d(volumeDrivingForce)
    indices = volumeDrivingForce > 0

    Rmin = precipitate.Rmin*np.ones(np.array(volumeDrivingForce).shape)
    Rcrit = np.zeros(volumeDrivingForce.shape)
    Gcrit = np.zeros(volumeDrivingForce.shape)

    if not precipitate.GBfactor.isGrainBoundaryNucleation:
        RcritProposal = 2*precipitate.shapeFactor.description.thermoFactorFromAR(aspectRatio) * precipitate.gamma / volumeDrivingForce[indices]
        Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
        Gcrit[indices] = (4*np.pi/3) * precipitate.gamma * Rcrit[indices]**2

    else:
        RcritProposal = precipitate.GBfactor.Rcrit(volumeDrivingForce[indices])
        Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
        Gcrit[indices] = precipitate.GBfactor.Gcrit(volumeDrivingForce[indices], Rcrit[indices])

    return np.squeeze(Rcrit), np.squeeze(Gcrit)

def zeldovich(T, Rcrit, precipitate : PrecipitateParameters):
    '''
    Zeldovich factor
    Z = sqrt(3*fv/4*pi) * Vm * sqrt(gamma/kB*T) / (2*pi*Nv*Rcrit^2)
    '''
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    Z = np.zeros(Rcrit.shape)
    Z[indices] = np.sqrt(3 * precipitate.GBfactor.volumeFactor / (4 * np.pi)) * precipitate.volume.Vm * np.sqrt(precipitate.gamma / (BOLTZMANN_CONSTANT * T[indices]))
    Z[indices] /= (2 * np.pi * AVOGADROS_NUMBER * Rcrit[indices]**2)
    return np.squeeze(Z)

def betaBinary1(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False):
    '''
    Impingement rate for binary systems

    beta = fa * Rcrit^2 * x * D / a**4
    '''
    x = np.atleast_1d(x)
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    beta = np.zeros(Rcrit.shape)
    D = np.atleast_2d(therm.getTracerDiffusivity(x[indices], T[indices], removeCache=removeCache))
    beta[indices] = precipitate.GBfactor.areaFactor * Rcrit[indices]**2 * x[indices] * D[:,1] / matrix.volume.a**4
    return np.squeeze(beta)

def betaBinary2(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, xEqAlpha = None, xEqBeta = None, removeCache = False):
    '''
    Impingement rate for binary systems similar to how multicomponent systems are computed

    beta = fa * Rcrit^2 * Dterm / a**4
    D = [(xB - xA)^2 / (xA*D) + (xB - xA)^2 / ((1-xA)*D)]^-1
    '''
    x = np.atleast_1d(x)
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0
    
    if xEqAlpha is None:
        xEqAlpha, xEqBeta = therm.getInterfacialComposition(T[indices], np.zeros(T[indices].shape), precipitate.phase)

    beta = np.zeros(Rcrit.shape)
    D = np.atleast_2d(therm.getTracerDiffusivity(x[indices], T[indices], removeCache=removeCache))
    Dfactor = (xEqBeta - xEqAlpha)**2 / (xEqAlpha*D[:,1]) + (xEqBeta - xEqAlpha)**2 / ((1 - xEqAlpha)*D[:,0])
    beta[indices] = precipitate.GBfactor.areaFactor * Rcrit[indices]**2 * (1/Dfactor) / matrix.volume.a**4
    return np.squeeze(beta)

def betaMulti(therm : MulticomponentThermodynamics, x, T, Rcrit,  matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False, searchDir = None):
    '''
    Impingement rate for multicomponent systems

    beta = fa * Rcrit^2 * Dterm / a**4
    Dterm = [sum_i((xB_i - xA_i)^2 / (xA_i*D_i))]^-1
    '''
    x = np.atleast_2d(x)
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    beta = np.zeros(Rcrit.shape)
    beta[indices] = np.array([therm.impingementFactor(xi, Ti, precPhase=precipitate.phase, removeCache=removeCache, searchDir=searchDir) for xi, Ti in zip(x[indices], T[indices])])
    beta[indices] *= (precipitate.GBfactor.areaFactor * Rcrit[indices]**2 / matrix.volume.a**4)
    return np.squeeze(beta)

def incubationTime(beta, Z, matrix : MatrixParameters):
    '''
    Returns incubation time (tau) and a time offset (this is to be compatible with the nonisothermal calculation)

    tau = 1 / (theta * beta * Z^2)
    '''
    beta = np.atleast_1d(beta)
    Z = np.atleast_1d(Z)
    indices = Z != 0

    tau = np.zeros(Z.shape)
    tau[indices] = 1 / (matrix.theta * beta[indices] * Z[indices]**2)
    return np.squeeze(tau)

def incubationTimeNonIsothermal(Z, currBeta, currTime, currTemp, betas, times, temperatures, matrix : MatrixParameters):
    '''
    Note: beta, times and temperature is a subslide from startIndex:n+1
        Start index is the incubationOffset term

    Solve tau for int_0^tau (beta(t)dt) = 1 / (theta*Z(tau)^2)
    '''
    #Assume that Z is constant except for temperature (this will assume that Rcrit is constant)
    #We can actually account for Rcrit since we record it in the KWN model
    LHS = 1 / (matrix.theta * Z**2 * (currTemp / temperatures))

    RHS = np.cumsum(betas[1:] * (times[1:] - times[:-1]))
    if len(RHS) == 0:
        RHS = currBeta * (times - times[0])
    else:
        RHS = np.concatenate((RHS, [RHS[-1] + currBeta * (currTime - times[0])]))

    #Test for intersection
    diff = RHS - LHS
    signChange = np.sign(diff[:-1]) != np.sign(diff[1:])

    #If no intersection
    if not any(signChange):
        #If RHS > LHS, then intersection is at t = 0
        if diff[0] > 0:
            tau = 0
        #Else, RHS intersects LHS beyond simulation time
        #Extrapolate integral of RHS from last point to intersect LHS
        #integral(beta(t-t0)) from t0 to ti + beta_i * (tau - (ti - t0)) = 1 / theta * Z(tau+t0)^2
        else:
            tau = LHS[-1] / currBeta - RHS[-1] / currBeta + (currTime - times[0])
    else:
        tau = times[:-1][signChange][0] - times[0]

    return tau

def nucleationRate(Z, beta, Gcrit, T, tau, time = np.inf):
    '''
    Nucleation rate

    d#/dt = Z * beta * exp(-Gcrit / kB*T) * exp(-tau / t)

    Units are 1/t
    '''
    Z = np.atleast_1d(Z)
    beta = np.atleast_1d(beta)
    Gcrit = np.atleast_1d(Gcrit)
    T = np.atleast_1d(T)
    tau = np.atleast_1d(tau)

    nucRate = np.zeros(Gcrit.shape)
    indices = Gcrit != 0
    incubationTime = np.amin([np.exp(-tau[indices] / time), np.ones(tau[indices].shape)], axis=0)
    nucRate[indices] = np.squeeze(Z[indices] * beta[indices] * np.exp(-Gcrit[indices] / (BOLTZMANN_CONSTANT * T[indices])) * incubationTime)
    return np.squeeze(nucRate)

def nucleationRadius(T, Rcrit, precipitateParameters: PrecipitateParameters):
    '''
    Adds 1/2 * sqrt(kb T / pi gamma) to critical radius to ensure they grow when growth rates are calculated
    '''
    T = np.squeeze(T)
    Rcrit = np.squeeze(Rcrit)
    Rad = Rcrit + 0.5*np.sqrt(BOLTZMANN_CONSTANT * T / (np.pi * precipitateParameters.gamma))
    return np.squeeze(Rad)

def computeSteadyStateNucleation(therm : GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, matrix : MatrixParameters, betaFunc = None, aspectRatio = 1, removeCache = False):
    x, T = therm.process_xT_arrays(x, T, squeeze_X=True)
    chemDGs, volDGs, betaComp = volumetricDrivingForce(therm, x, T, precipitate, aspectRatio = aspectRatio, removeCache = removeCache)
    Rcrit, Gcrit = nucleationBarrier(volDGs, precipitate, aspectRatio = aspectRatio)
    Z = zeldovich(T, Rcrit, precipitate)
    if betaFunc is None:
        betaFunc = betaBinary2 if therm._isBinary else betaMulti
    beta = betaFunc(therm, x, T, Rcrit, matrix, precipitate, removeCache = removeCache)
    tau = incubationTime(beta, Z, matrix)
    nucRate = nucleationRate(Z, beta, Gcrit, T, tau, time = np.inf)
    nucRadius = nucleationRadius(T, Rcrit, precipitate)

    return NucleationData(
        x = np.squeeze(x),
        T = np.squeeze(T),
        nucleation_rate = np.squeeze(nucRate),
        chemical_driving_force = np.squeeze(chemDGs),
        volumetric_driving_force = np.squeeze(volDGs),
        Rcrit = np.squeeze(Rcrit),
        Gcrit = np.squeeze(Gcrit),
        precipitate_composition = np.squeeze(betaComp),
        Z = np.squeeze(Z),
        beta = np.squeeze(beta),
        tau = np.squeeze(tau),
        nucleation_radius = np.squeeze(nucRadius)
    )