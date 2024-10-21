from collections import namedtuple

import numpy as np

from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
from kawin.precipitation.PrecipitationParameters import MatrixParameters, PrecipitateParameters, AVOGADROS_NUMBER, BOLTZMANN_CONSTANT

NucleationData = namedtuple('NucleationData', [
    'x', 'T',
    'nucleation_rate', 'chemical_driving_force', 'volumetric_driving_force', 'Rcrit', 'Gcrit', 
    'precipitate_composition', 'Z', 'beta', 'tau', ])

def volumetricDrivingForce(therm: GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, aspectRatio = 1, removeCache = False):
    '''
    Computes volumetric driving force (chemical DG / VM - strain energy)
        Strain energy will always reduce the driving force since the precipitate will add strain to the matrix
        In the case where the matrix is prestrained and the precipitate will relax the matrix, then the strain
        energy is negative
    '''
    x, T = therm._process_xT_arrays(x, T)
    if therm._isBinary:
        x = np.squeeze(x)

    chemdGs, betaComp = therm.getDrivingForce(x, T, precPhase=precipitate.phase, removeCache=removeCache)
    voldGs = chemdGs / precipitate.volume.Vm
    voldGs -= precipitate.strainEnergy.strainEnergy(precipitate.shapeFactor.description.normalRadiiFromAR(aspectRatio))

    nucData = NucleationData(
        x = np.squeeze(x),
        T = np.squeeze(T),
        nucleation_rate = None,
        chemical_driving_force = np.squeeze(chemdGs),
        volumetric_driving_force = np.squeeze(voldGs),
        Rcrit = None,
        Gcrit = None,
        precipitate_composition = np.squeeze(betaComp),
        Z = None,
        beta = None,
        tau = None
    )
    
    return nucData

def nucleationBarrier(nucleationData : NucleationData, precipitate : PrecipitateParameters, aspectRatio = 1, Rmin = 0):
    '''
    Critical Gibbs free energy and radius at the nucleation barrier
    For bulk and dislocation nucleation
        Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
        Gcrit = (4*pi/3)*gamma*Rcrit^2
    For grain boundary, edge or corner nucleation, critical G and R is computed to in GBFactors to account for grain boundary energy
    '''
    volumeDG = np.atleast_1d(nucleationData.volumetric_driving_force)
    indices = volumeDG > 0

    Rmin = Rmin*np.ones(np.array(volumeDG).shape)
    Rcrit = np.zeros(volumeDG.shape)
    Gcrit = np.zeros(volumeDG.shape)

    if precipitate.GBfactor.nucleationSiteType == GBFactors.BULK or precipitate.GBfactor.nucleationSiteType == GBFactors.DISLOCATION:
        RcritProposal = 2*precipitate.shapeFactor.description.thermoFactorFromAR(aspectRatio) * precipitate.gamma / volumeDG[indices]
        Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
        Gcrit[indices] = (4*np.pi/3) * precipitate.gamma * Rcrit[indices]**2

    else:
        RcritProposal = precipitate.GBfactor.Rcrit(volumeDG[indices])
        Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
        Gcrit[indices] = precipitate.GBfactor.Gcrit(volumeDG[indices], Rcrit[indices])
    
    nucleationData = nucleationData._replace(Rcrit=np.squeeze(Rcrit), Gcrit=np.squeeze(Gcrit))
    return nucleationData

def zeldovich(nucleationData : NucleationData, precipitate : PrecipitateParameters):
    '''
    Zeldovich factor
    Z = sqrt(3*fv/4*pi) * Vm * sqrt(gamma/kB*T) / (2*pi*Nv*Rcrit^2)
    '''
    T = np.atleast_1d(nucleationData.T)
    Rcrit = np.atleast_1d(nucleationData.Rcrit)
    indices = Rcrit != 0

    Z = np.zeros(Rcrit.shape)
    Z[indices] = np.sqrt(3 * precipitate.GBfactor.volumeFactor / (4 * np.pi)) * precipitate.volume.Vm * np.sqrt(precipitate.gamma / (BOLTZMANN_CONSTANT * T[indices]))
    Z[indices] /= (2 * np.pi * AVOGADROS_NUMBER * Rcrit[indices]**2)

    nucleationData = nucleationData._replace(Z=np.squeeze(Z))
    return nucleationData
        
def betaBinary1(therm : BinaryThermodynamics, nucleationData : NucleationData, matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False):
    '''
    Impingement rate for binary systems

    beta = fa * Rcrit^2 * x * D / a**4
    '''
    x = np.atleast_1d(nucleationData.x)
    T = np.atleast_1d(nucleationData.T)
    Rcrit = np.atleast_1d(nucleationData.Rcrit)
    indices = Rcrit != 0

    beta = np.zeros(Rcrit.shape)
    beta[indices] = precipitate.GBfactor.areaFactor * Rcrit[indices]**2 * x[indices] * therm.getInterdiffusivity(x[indices], T[indices], removeCache=removeCache) / matrix.volume.a**4

    nucleationData = nucleationData._replace(beta=np.squeeze(beta))
    return nucleationData

def betaBinary2(therm : BinaryThermodynamics, nucleationData : NucleationData, matrix : MatrixParameters, precipitate : PrecipitateParameters, xEqAlpha = None, xEqBeta = None, removeCache = False):
    '''
    Impingement rate for binary systems similar to how multicomponent systems are computed

    beta = fa * Rcrit^2 * Dterm / a**4
    D = [(xB - xA)^2 / (xA*D) + (xB - xA)^2 / ((1-xA)*D)]^-1
    '''
    x = np.atleast_1d(nucleationData.x)
    T = np.atleast_1d(nucleationData.T)
    Rcrit = np.atleast_1d(nucleationData.Rcrit)
    indices = Rcrit != 0
    
    if xEqAlpha is None:
        xEqAlpha, xEqBeta = therm.getInterfacialComposition(T[indices], np.zeros(T[indices].shape), precipitate.phase)

    beta = np.zeros(Rcrit.shape)
    
    D = therm.getInterdiffusivity(x[indices], T[indices], removeCache=removeCache)
    Dfactor = (xEqBeta - xEqAlpha)**2 / (xEqAlpha*D) + (xEqBeta - xEqAlpha)**2 / ((1 - xEqAlpha)*D)
    beta[indices] = precipitate.GBfactor.areaFactor * Rcrit[indices]**2 * (1/Dfactor) / matrix.volume.a**4

    nucleationData = nucleationData._replace(beta=np.squeeze(beta))
    return nucleationData


def betaMulti(therm : MulticomponentThermodynamics, nucleationData : NucleationData,  matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False):
    '''
    Impingement rate for multicomponent systems

    beta = fa * Rcrit^2 * Dterm / a**4
    Dterm = [sum_i((xB_i - xA_i)^2 / (xA_i*D_i))]^-1
    '''
    x = np.atleast_2d(nucleationData.x)
    T = np.atleast_1d(nucleationData.T)
    Rcrit = np.atleast_1d(nucleationData.Rcrit)
    indices = Rcrit != 0

    beta = np.zeros(Rcrit.shape)
    beta[indices] = np.array([therm.impingementFactor(xi, Ti, precPhase=precipitate.phase, removeCache=removeCache) for xi, Ti in zip(x[indices], T[indices])])
    beta[indices] *= (precipitate.GBfactor.areaFactor * Rcrit[indices]**2 / matrix.volume.a**4)

    nucleationData = nucleationData._replace(beta=np.squeeze(beta))
    return nucleationData

def incubationTime(nucleationData : NucleationData, matrix : MatrixParameters):
    '''
    Returns incubation time (tau) and a time offset (this is to be compatible with the nonisothermal calculation)

    tau = 1 / (theta * beta * Z^2)
    '''
    beta = np.atleast_1d(nucleationData.beta)
    Z = np.atleast_1d(nucleationData.Z)
    indices = Z != 0

    tau = np.zeros(Z.shape)
    tau[indices] = 1 / (matrix.theta * beta[indices] * Z[indices]**2)

    nucleationData = nucleationData._replace(tau=np.squeeze(tau))
    return nucleationData

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

def nucleationRate(nucleationData : NucleationData, time = np.inf):
    '''
    Nucleation rate

    d#/dt = Z * beta * exp(-Gcrit / kB*T) * exp(-tau / t)

    Units are 1/t
    '''
    Z = np.atleast_1d(nucleationData.Z)
    beta = np.atleast_1d(nucleationData.beta)
    Gcrit = np.atleast_1d(nucleationData.Gcrit)
    T = np.atleast_1d(nucleationData.T)
    tau = np.atleast_1d(nucleationData.tau)

    nucRate = np.zeros(Gcrit.shape)
    indices = Gcrit != 0
    nucRate[indices] = np.squeeze(Z[indices] * beta[indices] * np.exp(-Gcrit[indices] / (BOLTZMANN_CONSTANT * T[indices])) * np.exp(-tau[indices] / time))
    
    nucleationData = nucleationData._replace(nucleation_rate=np.squeeze(nucRate))
    return nucleationData
    