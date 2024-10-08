import numpy as np

from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.precipitation.non_ideal.GrainBoundaries import GBFactors
from kawin.precipitation.PrecipitationParameters import MatrixParameters, PrecipitateParameters, AVOGADROS_NUMBER, BOLTZMANN_CONSTANT

def volumetricDrivingForce(therm: GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, aspectRatio = 1):
    '''
    Computes volumetric driving force (chemical DG / VM - strain energy)
        Strain energy will always reduce the driving force since the precipitate will add strain to the matrix
        In the case where the matrix is prestrained and the precipitate will relax the matrix, then the strain
        energy is negative
    '''
    dGs, betaComp = therm.getDrivingForce(x, T, precPhase=precipitate.phase)
    dGs /= precipitate.volume.Vm
    dGs -= precipitate.strainEnergy.strainEnergy(precipitate.shapeFactor.description.normalRadiiFromAR(aspectRatio))
    
    return dGs, betaComp

def nucleationBarrier(volumeDG, precipitate : PrecipitateParameters, aspectRatio = 1, Rmin = 0):
    '''
    Critical Gibbs free energy and radius at the nucleation barrier
    For bulk and dislocation nucleation
        Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
        Gcrit = (4*pi/3)*gamma*Rcrit^2
    For grain boundary, edge or corner nucleation, critical G and R is computed to in GBFactors to account for grain boundary energy
    '''
    volumeDG = np.atleast_1d(volumeDG)
    Rmin = Rmin*np.ones(np.array(volumeDG).shape)

    if precipitate.GBfactor.nucleationSiteType == GBFactors.BULK or precipitate.GBfactor.nucleationSiteType == GBFactors.DISLOCATION:
        RcritProposal = 2*precipitate.shapeFactor.description.thermoFactorFromAR(aspectRatio) * precipitate.gamma / volumeDG
        Rcrit = np.amax([RcritProposal, Rmin], axis=0)
        Gcrit = (4*np.pi/3) * precipitate.gamma * Rcrit**2

    else:
        RcritProposal = precipitate.GBfactor.Rcrit(volumeDG)
        Rcrit = np.amax([RcritProposal, Rmin], axis=0)
        Gcrit = precipitate.GBfactor.Gcrit(volumeDG, Rcrit)
    
    return Gcrit, Rcrit

def zeldovich(T, Rcrit, precipitate : PrecipitateParameters):
    return np.sqrt(3 * precipitate.GBfactor.volumeFactor / (4 * np.pi)) * precipitate.volume.Vm * np.sqrt(precipitate.gamma / (BOLTZMANN_CONSTANT * T)) / (2 * np.pi * AVOGADROS_NUMBER * Rcrit**2)
        
def betaBinary1(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters):
    return precipitate.GBfactor.areaFactor * Rcrit**2 * x * therm.getInterdiffusivity(x, T) / matrix.volume.a**4

def betaBinary2(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, xEqAlpha = None, xEqBeta = None):
    if xEqAlpha is None:
        xEqAlpha, xEqBeta = therm.getInterfacialComposition(T, np.zeros(T.shape), precipitate.phase)
    
    D = therm.getInterdiffusivity(x, T)
    Dfactor = (xEqBeta - xEqAlpha)**2 / (xEqAlpha*D) + (xEqBeta - xEqAlpha)**2 / ((1 - xEqAlpha)*D)
    return precipitate.GBfactor.areaFactor * Rcrit**2 * (1/Dfactor) / matrix.volume.a**4

def betaMulti(therm : MulticomponentThermodynamics, x, T, precipitate : PrecipitateParameters):
    x = np.atleast_2d(x)
    T = np.atleast_1d(T)

    betas = np.array([therm.impingementFactor(xi, Ti, precPhase=precipitate.phase) for xi, Ti in zip(x, T)])
    return np.squeeze(betas)

def incubationTime(Z, beta, matrix : MatrixParameters):
    '''
    Returns incubation time (tau) and a time offset (this is to be compatible with the nonisothermal calculation)
    '''
    return 1 / (matrix.theta * beta * Z**2)

def incubationTimeNonIsothermal(Z, currBeta, currTime, currTemp, betas, times, temperatures, matrix : MatrixParameters):
    '''
    Note: beta, times and temperature is a subslide from startIndex:n+1
        Start index is the incubationOffset term
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

def nucleationRate(T, Gcrit, Z, beta, tau, time = np.inf):
    return Z * beta * np.exp(-Gcrit / (BOLTZMANN_CONSTANT * T)) * np.exp(-tau / time)
    