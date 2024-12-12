import itertools
import copy
from dataclasses import dataclass

import numpy as np
from kawin.precipitation.parameters.ShapeFactors import ShapeFactor
from kawin.precipitation.parameters.LebedevNodes import loadPoints


#Utility functions for tensors
def convert2To4rankTensor(c2):
    '''
    Converts 2nd rank elastic tensor to 4th rank

    Parameters
    ----------
    c : ndarray
        2nd rank elastic tensor

    Returns
    -------
    c4 : ndarray
        4th rank elastic tensor
    '''
    vMap = {
        frozenset({0}): 0, 
        frozenset({1}): 1, 
        frozenset({2}): 2,
        frozenset({1,2}): 3, 
        frozenset({0,2}): 4, 
        frozenset({0,1}): 5
        }

    c4 = np.zeros((3,3,3,3))
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        i2 = frozenset({i,j})
        j2 = frozenset({k,l})
        c4[i,j,k,l] = c2[vMap[i2], vMap[j2]]
    return c4

def convert4To2rankTensor(c4):
    '''
    Converts 4th rank elastic tensor to 4nd rank

    Parameters
    ----------
    c4 : ndarray
        4th rank elastic tensor

    Returns
    -------
    c : ndarray
        2nd rank elastic tensor
    '''
    vMap = [[0,0], [1,1], [2,2], [1,2], [0,2], [0,1]]
    c2 = np.zeros((6,6))
    for i, j in itertools.product(range(6), range(6)):
        c2[i,j] = c4[vMap[i][0], vMap[i][1], vMap[j][0], vMap[j][1]]
    return c2

def invert4rankTensor(c4):
    '''
    Inverts 4th rank tensor to give stiffness tensor

    This is done by converting to 2nd rank, inverting, then converting back to 4th rank
    '''
    c2 = convert4To2rankTensor(c4)
    return convert2To4rankTensor(np.linalg.inv(c2))

def convertVecTo2rankTensor(v):
    '''
    Converts strain/stress vector to 2nd rank tensor

    Parameters
    ----------
    v : 1d array
        Strain/stress vector

    Returns
    -------
    e : ndarray
        2nd rank elastic tensor
    '''
    return np.array([[v[0], v[5], v[4]], 
                     [v[5], v[1], v[3]], 
                     [v[3], v[4], v[2]]])

def convert2rankToVec(c):
    '''
    Converts 2nd rank tensor to vector

    Parameter
    ---------
    c : ndarray
        3x3 tensor

    Returns
    -------
    v : 1darray
        Strain/stress vector
    '''
    return np.array([c[0,0], c[1,1], c[2,2], c[1,2], c[0,2], c[0,1]])

def rotateRank2Tensor(rot, tensor):
    '''
    Rotates a 2nd rank tensor
    T_ij = r_il * r_jk * T_lk

    Parameters
    ----------
    tensor : ndarray
        2nd rank tensor to rotate (3x3 array)
    '''
    return np.tensordot(rot,
            np.tensordot(rot, tensor, axes=(1,1)), axes=(1,1))

def rotateRank4Tensor(rot, tensor):
    '''
    Rotates a 4th rank tensor
    T_ijkl = r_im * r_jn * r_ok * r_lp * T_mnop

    Parameters
    ----------
    tensor : ndarray
        4th rank tensor to rotate (3x3x3x3 array)
    '''
    return np.tensordot(rot, 
            np.tensordot(rot, 
            np.tensordot(rot, 
            np.tensordot(rot, tensor, axes=(1,3)), axes=(1,3)), axes=(1,3)), axes=(1,3))

def elasticConstantToC(c11, c12, c44):
    '''
    Creates elastic tensor from c11, c12 and c44 constants assuming isotropic system

    Parameters
    ----------
    c11 : float
    c12 : float
    c44 : float
    '''
    c = np.zeros((6, 6))
    c[0,0] = c[1,1] = c[2,2] = c11
    c[0,1] = c[0,2] = c[1,0] = c[1,2] = c[2,0] = c[2,1] = c12
    c[3,3] = c[4,4] = c[5,5] = c44
    return c

def moduliToC(E = None, nu = None, G = None, lam = None, K = None, M = None):
    '''
    Set elastic tensor by defining at least two of the moduli
    Everything will be converted to E, nu and G
    If more than two parameters are defined, then the following priority will be used:
    E - Youngs modulus
    nu - Poisson's ratio
    G - shear modulus
    lam - Lame's first parameter
    K - bulk modulus
    M - P-wave modulus

    NOTE: There's gotta be a better way to implement the conversions
    '''
    # Combinations
    # E-nu -> E,G,nu
    # E-G -> E,G,nu
    # E-lam -> E,nu -> E,G,nu
    # E-K -> E,nu -> E,G,nu
    # E-M -> E,nu -> E,G,nu
    # nu-G -> E,G,nu
    # nu-lam -> E,nu -> E,G,nu
    # nu-K -> E,nu -> E,G,nu
    # nu-M -> E,nu, -> E,G,nu
    # G-lam -> E,G -> E,G,nu
    # G-K -> E,G -> E,G,nu
    # G-M -> E,G, E,G,nu
    # lam-K -> E,nu -> E,G,nu
    # lam-M -> E,nu -> E,G,nu
    # K-M -> E,nu -> E,G,nu
    if E:
        if nu:
            G = E / (2 * (1 + nu))
        elif G:
            nu = E / (2 * G) - 1
        elif lam:
            R = np.sqrt(E**2 + 9*lam**2 + 2*E*lam)
            nu = 2*lam / (E + lam + R)
            G = (E - 3*lam + R) / 4
        elif K:
            nu = (3*K - E) / (6*K)
            G = 3*K*E / (9*K - E)
        elif M:
            S = np.sqrt(E**2 + 9*M**2 - 10*E*M)
            nu = (E - M + S) / (4*M)
            G = (3*M + E - S) / 8
    elif nu:
        if G:
            E = 2*G*(1 + nu)
        elif lam:
            E = lam * (1 + nu) * (1 - 2*nu) / nu
            G = lam * (1 - 2*nu) / (2*nu)
        elif K:
            E = 3*K*(1 - 2*nu)
            G = 3*K*(1 - 2*nu) / (2*(1 + nu))
        elif M:
            E = M * (1 + nu) * (1 - 2*nu) / (1 - nu)
            G = M * (1 - 2*nu) / (2 * (1 - nu))
    elif G:
        if lam:
            E = G * (3*lam + 2*G) / (lam + G)
            nu = lam / (2*(lam + G))
        elif K:
            E = 9*K*G / (3*K + G)
            nu = (3*K - 2*G) / (2*(3*K + G))
        elif M:
            E = G * (3*M - 4*G) / (M - G)
            nu = (M - 2*G) / (2*M - 2*G)
    elif lam:
        if K:
            E = 9*K*(K - lam) / (3*K - lam)
            G = 3*(K - lam) / 2
            nu = lam / (3*K - lam)
        elif M:
            E = (M - lam) * (M + 2*lam) / (M + lam)
            G = (M - lam) / 2
            nu = lam / (M + lam)
    elif K:
        if M:
            E = 9*K*(M - K) / (3*K + M)
            G = 3*(M - K) / 4
            nu = (3*K - M) / (3*K + M)

    s = np.zeros((6,6))
    s[0,0], s[1,1], s[2,2] = 1/E, 1/E, 1/E
    s[3,3], s[4,4], s[5,5] = 1/G, 1/G, 1/G
    s[0,1], s[0,2], s[1,0], s[1,2], s[2,0], s[2,1] = -nu/E, -nu/E, -nu/E, -nu/E, -nu/E, -nu/E
    return np.linalg.inv(s)

@dataclass
class StrainEnergyParameters:
    cMatrix_2nd = np.zeros((3,3))
    cMatrix_4th = np.zeros((3,3,3,3))
    cPrec_2nd = np.zeros((3,3))
    cPrec_4th = np.zeros((3,3,3,3))
    eigenstrain = np.zeros((3,3))
    appliedStress = np.zeros((3,3))
    appliedStrain = np.zeros((3,3))
    constantEnergy = 0

class StrainEnergyDescriptionBase:
    name = 'ABSTRACT STRAIN ENERGY DESCRIPTION'

    def __init__(self, params: StrainEnergyParameters = None):
        self.params = params

    def computeStrainEnergy(self, radius):
        raise NotImplementedError()
    
class ConstantEnergyDescription(StrainEnergyDescriptionBase):
    name = 'CONSTANT'

    def computeStrainEnergy(self, radius):
        return 4 * np.pi / 3 * np.prod(radius) * self.params.constantEnergy
    
class SphericalEnergyDescription(StrainEnergyDescriptionBase):
    name = 'SPHERE'

    def _Khachaturyan(self, I1, I2, radius):
        '''
        Khachaturyan's approximation for strain energy of spherical and cuboidal precipitates
        '''
        c = self.params.cMatrix_2nd
        eigenstrain = self.params.eigenstrain

        V = 4*np.pi/3 * np.prod(radius)
        A1 = 2 * (c[0,0] - c[0,1]) / c[0,0]
        A1 -= 12 * (c[0,0] + 2 * c[0,1]) * (c[0,0] - c[0,1] - 2 * c[3,3]) / (c[0,0] * (c[0,0] + c[0,1] + 2*c[3,3])) * I1
        A2 = -54 * (c[0,0] + 2 * c[0,1]) * (c[0,0] - c[0,1] - 2 * c[3,3])**2 / (c[0,0] * (c[0,0] + c[0,1] + 2 * c[3,3]) * (c[0,0] + 2 * c[0,1] + 4 * c[3,3])) * I2
        return 0.5 * (c[0,0] + 2 * c[0,1]) * (A1 + A2) * eigenstrain[0,0]**2 * V

    def computeStrainEnergy(self, radius):
        '''
        Strain energy of perfect sphere (cubic factor = 1)
        '''
        return self._Khachaturyan(1/15, 1/105, radius)

class CuboidalEnergyDescription(SphericalEnergyDescription):
    name = 'CUBIC'

    #Additional methods for verifying that the general methods reduce to these
    #specific cases for cubic crystal symmetry
    def _OhmCubic(self, n):
        '''
        Ohm term for cubic crystal symmetry
        '''
        c2 = self.params.cMatrix_2nd
        ohm = np.zeros((3,3))
        jk = [(1,2), (0,2), (0,1)]
        for i in range(3):
            for j in range(3):
                if i == j:
                    jn, k = jk[i]
                    ohm[i,i] = (c2[3,3] + (c2[0,0]-c2[3,3])*(n[jn]**2 + n[k]**2) + self._xi*(c2[0,0]+c2[0,1])*(n[jn]*n[k])**2) / (c2[3,3]*self._D(n))
                else:
                    k = 3 - (i + j)
                    ohm[i,j] = -(c2[0,1] + c2[3,3])*(1 + self._xi*n[k]**2)*n[i]*n[j] / (c2[3,3]*self._D(n))
        
        return ohm

    def _D(self, n):
        '''
        Needed for the Ohm term with cubic crystal symmetry
        '''
        c2 = self.params.cMatrix_2nd
        d = c2[0,0]
        d += self._xi*(c2[0,0] + c2[0,1])*((n[0]*n[1])**2 + (n[0]*n[2])**2 + (n[1]*n[2])**2)
        d += self._xi**2 * (c2[0,0] + 2*c2[0,1] + c2[3,3])*(n[0]*n[1]*n[2])**2
        return d

    @property
    def _xi(self):
        '''
        Needed for the Ohm term with cubic crystal symmetry
        '''
        c2 = self.params.cMatrix_2nd
        return (c2[0,0] - c2[0,1] - 2*c2[3,3]) / c2[3,3]

    def computeStrainEnergy(self, radius):
        '''
        Strain energy of perfect cube (cubic factor = sqrt(2))
        '''
        return self._Khachaturyan(0.006931, 0.000959, radius)

class EllipsoidalEnergyDescription(StrainEnergyDescriptionBase):
    name = 'ELLIPSOID'
    def __init__(self):
        self.setOhmInverseFunction()
        self.setLebedevIntegration()

    def setOhmInverseFunction(self, method = 'quick'):
        '''
        Sets method to invert the ohm term in calculating eshelby's tensor

        Parameters
        ----------
        method : str
            'numpy' - uses np.linalg.inv, which can be slower for batch, but runs through
                        multiple checks for whether values are real/complex or if inverse exists
            'quick' - quick inverse using Cramer's rule assuming that values are real and inverse exists
                    - this is the recommended method since strain energy will be computed many times
        '''
        if method == 'numpy':
            self._ohm_inverse = self._ohm_npinv
        elif method == 'quick':
            self._ohm_inverse = self._ohm_quickInverse
        else:
            raise ValueError(f"Unknown inverse function method: '{method}'. Function must be either 'quick' or 'numpy'")

    def setIntegrationIntervals(self, phiInt, thetaInt, assumeSymmetric=True):
        '''
        Number of intervals to split domain along phi and theta for integration

        Parameters
        ----------
        phiInt : int
            Number of intervals to divide along phi
        thetaInt : int
            Number of intervals to divide along theta
        assumeSymmetric : bool (optional)
            If True (default), will only integrate Eshelby's tensor on a single quadrant and
            multiply the results by 8
        '''
        #If assume symmetric, then only a single quadrant will be integrated over (then multipled by 8 for each quadrant)
        phiRange = np.pi/2 if assumeSymmetric else 2*np.pi
        thetaRange = np.pi/2 if assumeSymmetric else np.pi
        dphi = phiRange / phiInt
        dtheta = thetaRange / thetaInt
        midPhi = np.linspace(dphi/2, phiRange - dphi/2, phiInt)
        midTheta = np.linspace(dtheta/2, thetaRange - dtheta/2, thetaInt)

        #Cartesian product of phi and theta intervals
        self.midPhiGrid, self.midThetaGrid = np.meshgrid(midPhi, midTheta)
        self.midPhiGrid = self.midPhiGrid.ravel()
        self.midThetaGrid = self.midThetaGrid.ravel()
        self.midWeights = np.sin(self.midThetaGrid)
        self.dA = dtheta*dphi if assumeSymmetric else 1/8 * dtheta*dphi

    def setLebedevIntegration(self, order = 'high'):
        '''
        Creates Lebedev quadrature points and nodes for integrating Eshebly's tensor
        This is preferred over discretizing phi and theta

        Parameters
        ----------
        order : str
            'low' - uses quadrature order or 53 (974 points)
            'mid' - uses quadrature order or 83 (2354 points)
            'high' (default) - uses quadrature order or 131 (5810 points)
        '''
        if order == 'low':
            order = 53
        elif order == 'mid':
            order = 83
        elif order == 'high':
            order = 131
        else:
            raise ValueError(f"Unknown integration order {order}. Order must be ['low', 'mid', 'high']")

        self.midPhiGrid, self.midThetaGrid, self.midWeights = loadPoints(order)
        self.dA = np.pi/2

    def _n(self, phi, theta):
        '''
        Unit normal vector of sphere

        Parameters
        ----------
        phi - azimuthal angle
        theta - polar angle
        '''
        return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    
    def _beta(self, a, b, c, phi, theta):
        '''
        Distance from center to surface of ellipsoid

        Parameters
        ----------
        a, b, c - radii of ellipsoid along x,y,z axes
        phi, theta - azimuthal, polar angle
        '''
        return np.sqrt(((a*np.cos(phi))**2 + (b*np.sin(phi))**2)*np.sin(theta)**2 + (c*np.cos(theta))**2)

    def _OhmGeneral(self, n, c4):
        '''
        Ohm term for general system

        Ohm_ij = inverse(C_iklj * n_k * n_l)
        '''
        invOhm = np.tensordot(c4, np.tensordot(n, n, axes=0), axes=[[1,2], [0,1]])
        return np.linalg.inv(invOhm)
    
    def _ohm_quickInverse(self, m):
        '''
        Hard coded inverse of m which is of shape (3,3,n)

        numpy inv is more optimized for larger matrices, but can be slower for small
        matrices such as a 2x2 or 3x3. We can take advantage of 3x3 matrices having a computable
        inverse to make it faster

        NOTE: this only works since we know that m has a shape of (3,3,n) and is only composed of real numbers

        This function can probably be a bit more efficient, but quick 
        profiling on sphInt gives around a 35x speedup compared to doing 
        np.transpose(np.linalg.inv(np.transpose(m, (2,0,1))), (1,2,0)) 
        where the slowdown was in np.linalg.inv

        For matrix:
            |  a  b  c  |
            |  d  e  f  |
            |  g  h  i  |

        Inverse is defined as:
            |  ei-fh  fg-di  dh-eg  |          |  A  B  C  |
            |  ch-bi  ai-cg  bg-ah  | / det -> |  D  E  F  | / det
            |  bf-ce  cd-af  ae-bd  |          |  G  H  I  |
            Where det = aA + bB + cC
        '''
        a, b, c, d, e, f, g, h, i = m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2], m[2,0], m[2,1], m[2,2]
        A = e*i - f*h
        B = f*g - d*i
        C = d*h - e*g
        D = c*h - b*i
        E = a*i - c*g
        F = b*g - a*h
        G = b*f - c*e
        H = c*d - a*f
        I = a*e - b*d
        det = a*A + b*B + c*C
        return np.array([[A, B, C], [D, E, F], [G, H, I]]) / det
    
    def _ohm_npinv(self, m):
        '''
        Inverts ohm term using np.linalg.inv

        numpy inverse function takes in an array of shape (m,n,n) and inverts each nxn matrix
            So we have to transpose m from (3,3,n) -> (n,3,3), then invert, then transpose (n,3,3) ->(3,3,n)
        '''
        return np.transpose(np.linalg.inv(np.transpose(m, (2,0,1))), (1,2,0))

    def sphInt(self, radius, c4):
        '''
        Faster version of calculating the Dijkl, which avoids
        using a for loop to iterate across phi and theta
        '''
        #Normal vector (1 x 3 x n) where n is number of integration points
        n = self._n(self.midPhiGrid, self.midThetaGrid)
        nexp = np.expand_dims(n, axis=0)

        #n_i*n_j for each grid point (n x 3 x 1) x (n x 1 x 3) = (n x 3 x 3) -> (3 x 3 x n)
        nProd = np.matmul(np.transpose(nexp, (2,1,0)), np.transpose(nexp, (2,0,1))).transpose((1,2,0))

        #End term inside integral = sin(theta) / beta**3
        #endTerm = np.sin(self.midThetaGrid) / self._beta(self.r[0], self.r[1], self.r[2], self.midPhiGrid, self.midThetaGrid)**3
        endTerm = 1 / self._beta(radius[0], radius[1], radius[2], self.midPhiGrid, self.midThetaGrid)**3

        #Ohm term (Ohm_ij = inverse(C_iklj * n_k * n_l))
        #For all grid points (Ohm_ijn = inverse(C_iklj) * nProd_kln)
        invOhm = np.tensordot(c4, nProd, axes=[[1,2], [0,1]])

        ohm = self._ohm_inverse(invOhm)

        #Tensor product (D_ijkl = intergral(ohm_ij * n_k * n_l * endTerm))
        #For summing over grid points (D_ijkl = ohm_ij * nProd_kln * endTerm_n)
        d = np.tensordot(ohm, np.multiply(nProd, endTerm * self.midWeights), axes=[[2], [2]])

        #Multiply by differential area and across the 8 quadrants
        return 8*d*self.dA

    def Dijkl(self, radius, c4):
        '''
        Dijkl term for Eshelby's theory
        '''
        #return -np.prod(self.r)/(4*np.pi) * self.sphericalIntegral(self.Dfunc)
        return -np.prod(radius)/(4*np.pi) * self.sphInt(radius, c4)

    def Sijmn(self, D):
        '''
        S_ijmn = -0.5 * C_lkmn * (D_iklj + D_jkli)
        '''
        c4 = self.params.cMatrix_4th
        S = -0.5 * np.tensordot(c4, D + np.transpose(D, (3,1,2,0)), axes=[[0,1],[2,1]])
        #The tensor product gives S_mnij so we'll need to transpose it
        return np.transpose(S, (2,3,0,1))

    def _multiply(self, a, b):
        '''
        Multiplies 2 tensors
        4th x 2nd -> c_ij = a_ijkl * b_kl
        4th x 4th -> c_ijkl = a_ijmn * b_mnkl
        '''
        return np.tensordot(a, b, axes=[[2,3], [0,1]])

    def _strainEnergy(self, stress, strain, V):
        '''
        u = -0.5 * V * sigma_ij * strain_ij
        '''
        return -0.5 * V * np.sum(stress * strain)

    def strainEnergyEllipsoidWithStress(self, radius):
        '''
        Strain energy of ellipsoidal particle with applied stress
        '''
        c4 = self.params.cMatrix_4th
        eigenstrain = self.params.eigenstrain
        appliedStrain = self.params.appliedStrain
        
        V = 4*np.pi/3 * np.prod(radius)
        S = self.Sijmn(self.Dijkl(radius, c4))
        stress = self._multiply(c4, self._multiply(S, eigenstrain) - eigenstrain)
        stress0 = self._multiply(c4, self._multiply(S, appliedStrain) - appliedStrain)
        return self._strainEnergy(stress - stress0, eigenstrain - appliedStrain, V)

    def strainEnergyEllipsoid(self, radius):
        '''
        Strain energy of ellipsoidal particle
        '''
        c4 = self.params.cMatrix_4th
        eigenstrain = self.params.eigenstrain
        
        V = 4*np.pi/3 * np.prod(radius)
        S = self.Sijmn(self.Dijkl(radius, c4))
        stress = self._multiply(c4, self._multiply(S, eigenstrain) - eigenstrain)
        return self._strainEnergy(stress, eigenstrain, V)

    def strainEnergyEllipsoid2ndRank(self, radius):
        '''
        Alternative method of strain energy on ellipsoidal particle using 2nd rank tensors
        '''
        c4 = self.params.cMatrix_4th
        c2 = self.params.cMatrix_2nd
        eigenstrain = self.params.eigenstrain

        V = 4*np.pi/3 * np.prod(radius)
        S = convert4To2rankTensor(self.Sijmn(self.Dijkl(radius, c4)))
        eigFlat = convert2rankToVec(eigenstrain)
        multTerm = np.matmul(c2, S - np.eye(6))
        return -0.5 * V * np.matmul(eigFlat, np.matmul(multTerm, eigFlat))

    def strainEnergyBohm(self, radius):
        '''
        Strain energy of particle for when matrix and precipitate phases have different elastic tensors
        '''
        cM4 = self.params.cMatrix_4th
        eigenstrain = self.params.eigenstrain
        cP4 = self.params.cPrec_4th

        V = 4*np.pi/3 * np.prod(radius)
        S = self.Sijmn(self.Dijkl(radius, cM4))
        invTerm = invert4rankTensor(self._multiply(cP4 - cM4, S) + cM4)
        multTerm = self._multiply(invTerm, cP4)
        stressC = self._multiply(cM4, self._multiply(self._multiply(S, multTerm), eigenstrain))
        stress0 = self._multiply(cM4, self._multiply(multTerm, eigenstrain))
        return self._strainEnergy(stressC-stress0, eigenstrain, V)

    def strainEnergyBohm2ndRank(self, radius):
        '''
        Strain energy of particle for when matrix and precipitate phases have different elastic tensors using 2nd rank tensors
        '''
        cM4 = self.params.cMatrix_4th
        cM2 = self.params.cMatrix_2nd
        eigenstrain = self.params.eigenstrain
        cP2 = self.params.cPrec_2nd

        V = 4*np.pi/3 * np.prod(radius)
        S = convert4To2rankTensor(self.Sijmn(self.Dijkl(radius, cM4)))
        eigFlat = convert2rankToVec(eigenstrain)
        invTerm = np.linalg.inv(np.matmul(cP2 - cM2, S) + cM2)
        multTerm = np.matmul(invTerm, cP2)
        stressC = np.matmul(cM2, np.matmul(np.matmul(S, multTerm), eigFlat))
        stress0 = np.matmul(cM2, np.matmul(multTerm, eigFlat))
        return -0.5 * V * np.matmul(eigFlat, stressC - stress0)

    def computeStrainEnergy(self, radius):
        return self.strainEnergyBohm(radius)

class StrainEnergy:
    '''
    Defines class for calculating strain energy of a precipitate

    Ellipsoidal precipitates will use the Eshelby's tensor
    Spherical and Cuboidal precipitates will use the Khachaturyan's approximation
    '''
    def __init__(self, shape='constant'):
        self.params = StrainEnergyParameters()
        self._unrotated_cMatrix_4th = np.zeros((3,3,3,3))
        self._unrotated_cPrec_4th = np.zeros((3,3,3,3))
        self.rotation = np.eye(3)
        self.rotationPrec = np.eye(3)
        
        #Cached values for calculating equilibrium aspect ratio
        self.ifmethod = 1
        self._aspectRatios = None
        self._normEnergies = None
        self._cachedRange = 5
        self._cachedIntervals = 100

        self._description = ConstantEnergyDescription()
        self._updateCallbacks = []
        self.setShape(shape)

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, value):
        self._description = value
        for callback in self._updateCallbacks:
            callback()

    @property
    def unrotated_cMatrix_4th(self):
        return self._unrotated_cMatrix_4th
    
    @unrotated_cMatrix_4th.setter
    def unrotated_cMatrix_4th(self, value):
        if value.shape == (6,6):
            value = convert2To4rankTensor(value)
            self._unrotated_cMatrix_4th = value
        elif value.shape == (3,3,3,3):
            self._unrotated_cMatrix_4th = value
        else:
            raise ValueError("Matrix tensor must be 2nd rank (6x6) or 4th rank (3x3x3x3)")
        self.update()
        
    @property
    def unrotated_cPrec_4th(self):
        return self._unrotated_cPrec_4th
    
    @unrotated_cPrec_4th.setter
    def unrotated_cPrec_4th(self, value):
        if value.shape == (6,6):
            value = convert2To4rankTensor(value)
            self._unrotated_cPrec_4th = value
        elif value.shape == (3,3,3,3):
            self._unrotated_cPrec_4th = value
        else:
            raise ValueError("Precipitate tensor must be 2nd rank (6x6) or 4th rank (3x3x3x3)")
        self.update()

    def setShape(self, shape):
        # TODO: this creates an instance of the function, can we not do that?
        descriptionDict = {
            ConstantEnergyDescription.name.upper(): ConstantEnergyDescription(),
            SphericalEnergyDescription.name.upper(): SphericalEnergyDescription(),
            CuboidalEnergyDescription.name.upper(): CuboidalEnergyDescription(),
            EllipsoidalEnergyDescription.name.upper(): EllipsoidalEnergyDescription(),
            'PLATE': EllipsoidalEnergyDescription(),
            'NEEDLE': EllipsoidalEnergyDescription(),
        }
        if isinstance(shape, str):
            shape = shape.upper()
        newDescription = descriptionDict.get(shape, shape)
        if not isinstance(newDescription, StrainEnergyDescriptionBase):
            validValues = ', '.join(list(descriptionDict.keys()))
            raise ValueError(f"Unknown value '{shape}'. Value must be: {validValues} or an instance of ShapeDescriptionBase")
        self.description = newDescription
        self.description.params = self.params

    def setSpherical(self):
        '''
        Assumes spherical geometry for strain energy calculation
        Uses Khachaturyan's approximation
        '''
        self.setShape(SphericalEnergyDescription())
    
    def setCuboidal(self):
        '''
        Assumes cuboidal geometry for strain energy calculation
        Uses Khachaturyan's approximation
        '''
        self.setShape(CuboidalEnergyDescription())

    def setEllipsoidal(self):
        '''
        Assumes ellipsoidal geometry for strain energy calculation
        Uses Eshelby's tensor
        '''
        self.setShape(EllipsoidalEnergyDescription())

    def setConstantElasticEnergy(self, energy):
        '''
        If elastic strain energy is known to be a constant, this can be use to greatly
        simplify calculations

        Parameters
        ----------
        energy - strain energy in J/m3
        '''
        self.params.constantEnergy = energy
        self.setShape(ConstantEnergyDescription())

    def setElasticTensor(self, tensor):
        '''
        Sets elastic tensor of matrix using 2nd rank tensor

        Parameters
        ----------
        tensor : 6x6 array
            2nd rank elastic tensor
        '''
        self.unrotated_cMatrix_4th = np.array(tensor)

    def setElasticConstants(self, c11, c12, c44):
        '''
        Sets elastic tensor of matrix by elastic constants, assuming cubic symmetry

        Parameters
        ----------
        c11 : float
            Modulus for compression
            c11 = E(1-nu) / (1+nu)(1-2nu)
        c12 : float
            Modulus for dilation (accounts for compression and Poisson's ratio)
            c12 = E nu / (1+nu)(1-2nu)
        c44 : float
            Modulus for shear
            c44 = (c11-c12)/2
        '''
        self.unrotated_cMatrix_4th = elasticConstantToC(c11, c12, c44)

    def setModuli(self, E = None, nu = None, G = None, lam = None, K = None, M = None):
        '''
        Sets elastic tensor of matrix by 2 moduli

        Parameters (only 2 has to be defined)
        ----------
        E : float
            Elastic modulus
        nu : float
            Poisson's ratio
        G : float
            Shear modulus
        lam : float
            Lame's first parameter
        K : float
            Bulk modulus
        M : float
            P-wave modulus
        '''
        self.unrotated_cMatrix_4th = moduliToC(E, nu, G, lam, K, M)

    def setElasticTensorPrecipitate(self, tensor):
        '''
        Sets elastic tensor of precipitate using 2nd rank tensor

        Parameters
        ----------
        tensor : 6x6 array
            2nd rank elastic tensor
        '''
        self.unrotated_cPrec_4th = np.array(tensor)

    def setElasticConsantsPrecipitate(self, c11, c12, c44):
        '''
        Sets elastic tensor of precipitate by elastic constants, assuming cubic symmetry

        Parameters
        ----------
        c11 : float
            Modulus for compression
            c11 = E(1-nu) / (1+nu)(1-2nu)
        c12 : float
            Modulus for dilation (accounts for compression and Poisson's ratio)
            c12 = E nu / (1+nu)(1-2nu)
        c44 : float
            Modulus for shear
            c44 = (c11-c12)/2
        '''
        self.unrotated_cPrec_4th = elasticConstantToC(c11, c12, c44)
    
    def setModuliPrecipitate(self, E = None, nu = None, G = None, lam = None, K = None, M = None):
        '''
        Sets elastic tensor of precipitate by 2 moduli

        Parameters (only 2 has to be defined)
        ----------
        E : float
            Elastic modulus
        nu : float
            Poisson's ratio
        G : float
            Shear modulus
        lam : float
            Lame's first parameter
        K : float
            Bulk modulus
        M : float
            P-wave modulus
        '''
        self.unrotated_cPrec_4th = moduliToC(E, nu, G, lam, K, M)

    def setRotationMatrix(self, rot):
        '''
        Sets rotation matrix to be applied to the matrix

        This is for cases where the axes of the precipitate does not align with the axes of the matrix
        (e.g., the long/short axes of the precipitate is not parallel to the <100> directions of the matrix)

        Parameters
        ----------
        rot : matrix
            3x3 rotation matrix
        '''
        self.rotation = np.array(rot)

    def setRotationPrecipitate(self, rot):
        self.rotationPrec = np.array(rot)

    def setEigenstrain(self, strain):
        '''
        Sets eigenstrain of precipitate

        Parameters
        ----------
        strain : float, array or matrix
            float - assume strain is the same along all 3 axis
            array - each index corresponds to strain in a given axis
            matrix - full 2nd rank strain tensor

        NOTE: when using in conjunction with ShapeFactors, the axis are order specific
            For needle-like precipitates, x1, x2 and x3 correspond to (short axis, short axis, long axis)
            For plate-like precipitates, x1, x2 and x3 correspond to (long axis, long axis, short axis)
        '''
        strain = np.array(strain)
        #If scalar, then apply to all 3 axis
        if strain.ndim == 0:
            self.params.eigenstrain = strain * np.identity(3)
        #If array of length 3, then apply strain along each index to corresponding axis
        elif strain.shape == (3,):
            self.params.eigenstrain = np.array([[strain[0], 0, 0], 
                                                [0, strain[1], 0], 
                                                [0, 0, strain[2]]])
        #Else, assume it's a tensor
        elif strain.shape == (3,3):
            self.params.eigenstrain = strain
        else:
            raise ValueError("Eigenstrain must be scalar, 3-length vector of 3x3 matrix")

    def setAppliedStress(self, stress):
        '''
        Sets applied stress tensor
        Axes of stress tensor should be the same as the matrix

        Parameters
        ----------
        stress : float, array or matrix
            float - assume stress is the same along all 3 axis
            array - each index corresponds to stress in a given axis
            matrix - full 2nd rank stress tensor

        NOTE: The applied stress is in reference to the coordinate axes of the precipitates
        Thus, this is only valid if the following conditions are met:
            The matrix phase has a single orientation (either as a single crystal or highly textured)
            Precipitates will form only in a single orientation with respect to the matrix
        
        NOTE: this is only available in EllipsoidalEnergyDescription.strainEnergyEllipsoidWithStress and is currently not used in the KWNModel

        TODO: It will be nice to expand on this.
            For polycrystalline matrix and randomly oriented precipitates, it should be possible
            to average the strain energy contributions over all matrix/precipitate orientations
        '''
        stress = np.array(stress)
        #If scalar, then apply to all 3 axis
        if stress.ndim == 0:
            self.params.appliedStress = stress * np.identity(3)
        #If array of length 3, then apply stress along each index to corresponding axis
        elif stress.shape == (3,):
            self.params.appliedStress = np.array([[stress[0], 0, 0], 
                                                  [0, stress[1], 0], 
                                                  [0, 0, stress[2]]])
        #Else, assume it's a tensor
        elif stress.shape == (3,3):
            self.params.appliedStress = stress
        else:
            raise ValueError("Applied stress must be scalar, 3-length vector of 3x3 matrix")

    def _computeAppliedStrain(self, cM2, stress):
        '''
        Calculates applied strain tensor from applied stress tensor and elastic tensor
        '''
        if stress.any() and cM2.any():
            flatStress = convert2rankToVec(stress)
            flatStrain = np.matmul(np.linalg.inv(cM2), flatStress)
            return convertVecTo2rankTensor(flatStrain)
        else:
            return np.zeros((3,3))

    def update(self):
        # If matrix elastic constants are set, then fill parameters
        if self.unrotated_cMatrix_4th.any():
            # If we have elastic constants, then we default to spherical assumption
            # If the user had already set a precipitate shape before hand, then don't override unless it's constant
            if isinstance(self.description, ConstantEnergyDescription):
                self.setShape(SphericalEnergyDescription())

            self.params.cMatrix_4th = rotateRank4Tensor(self.rotation, self.unrotated_cMatrix_4th)
            self.params.cMatrix_2nd = convert4To2rankTensor(self.params.cMatrix_4th)

            # If precipitate elastic constants are set, then fill rotate and fill parameters
            # Otherwise, assumes precipitate constants are same as matrix constants
            if self.unrotated_cPrec_4th.any():
                self.params.cPrec_4th = rotateRank4Tensor(self.rotationPrec, self.unrotated_cPrec_4th)
                self.params.cPrec_2nd = convert4To2rankTensor(self.params.cPrec_4th)
            else:
                self.params.cPrec_4th = self.params.cMatrix_4th
                self.params.cPrec_2nd = self.params.cMatrix_2nd

            self.params.appliedStress = rotateRank2Tensor(self.rotation, self.params.appliedStress)
            self.params.appliedStrain = self._computeAppliedStrain(self.params.cMatrix_2nd, self.params.appliedStress)

        # If matrix elastic constants are not set, then default to constant strain energy
        else:
            self.setShape(ConstantEnergyDescription())

    def compute(self, r):
        r = np.atleast_2d(r)
        energies = [self.description.computeStrainEnergy(ri) for ri in r]
        return np.squeeze(energies)
    
    def setAspectRatioResolution(self, resolution = 0.01, cachedRange = 5):
        '''
        Sets resolution to which equilibrium aspect ratios are calculated

        Equilibrium aspect ratios are found by calculated strain energy for a range of aspect ratios
        and finding the aspect ratio giving the minimum energy (strain + interfacial energy)

        If aspect ratio does not vary much in a given system, then the default parameters may lead to poor
        prediction of the aspect ratios

        Parameters
        ----------
        resolution : float (optional)
            Minimum distance between aspect ratios when calculating cache (default at 0.01)
        cachedRange : float (optional)
            Range of aspect ratio to calculate strain energy when updated cache (default at 5)
        '''
        self._cachedRange = cachedRange
        self._cachedIntervals = int(1 / resolution)

    def setInterfacialEnergyMethod(self, method = 'thermo'):
        '''
        Sets method for calculating interfacial energy as a function of aspect ratio

        Parameters
        ----------
        method : str
            'eqradius' - interfacial energy is determined using the equivalent spherical radius
            'thermo' - interfacial energy is determined using dG/dSA (default)
        '''
        if method == 'eqradius':
            self.ifmethod = 0
        elif method == 'thermo':
            self.ifmethod = 1
        else:
            raise ValueError(f"Unknown interfacial energy method: '{method}'. Must be either 'thermo' or 'eqradius'")
        
    def updateCache(self, normR):
        '''
        Update cached calculations
        '''
        if self._aspectRatios is None:
            self._aspectRatios = np.linspace(1 + 1/self._cachedIntervals, 1+self._cachedRange, int(self._cachedRange*self._cachedIntervals))
            self._normEnergies = self.compute(normR(self._aspectRatios))
        else:
            addedAspectRatios = np.linspace(self._aspectRatios[-1] + 1/self._cachedIntervals, self._aspectRatios[-1] + self._cachedRange, int(self._cachedRange*self._cachedIntervals))
            addedNormEnergies = self.compute(normR(addedAspectRatios))
            self._aspectRatios = np.concatenate((self._aspectRatios, addedAspectRatios))
            self._normEnergies = np.concatenate((self._normEnergies, addedNormEnergies))

    def clearCache(self):
        '''
        Clear cached calculations
        '''
        self._aspectRatios = None
        self._normEnergies = None

    #Equilibrium aspect ratios
    #Determined by minimum of strain energy + interfacial energy
    def eqAR_byGR(self, Rsph, gamma, shpFactor : ShapeFactor, a=1.001, b=100):
        '''
        Equilibrium aspect ratio using golden ratio search

        Parameters
        ----------
        Rsph : float or array
            Equivalent spherical radius
        gamma : float
            Interfacial energy
        shpFactor : ShapeFactor object
        a, b : float
            Min and max bounds
            Default is 1.001 and 100
        '''
        normR = shpFactor.description.normalRadii
        interfacial = shpFactor.description.thermoFactor if self.ifmethod == 1 else shpFactor.description.eqRadiusFactor
        if hasattr(Rsph, '__len__'):
            eqAR = np.ones(len(Rsph))
            for i in range(len(Rsph)):
                eqAR[i] = self._GRsearch(Rsph[i], gamma, interfacial, normR, a, b)
        else:
            eqAR = self._GRsearch(Rsph, gamma, interfacial, normR, a, b)

        return eqAR

    def _GRsearch(self, Rsph, gamma, interfacial, normR, a, b):
        '''
        Golden ratio search for a single radius
        '''
        f = lambda ar: self.compute(normR(ar))*(4/3)*np.pi*Rsph**3 + gamma*interfacial(ar)*4*np.pi*Rsph**2
    
        tol = 1e-3
        invphi = (np.sqrt(5) - 1) / 2
        invphi2 = (3 - np.sqrt(5)) / 2
        h = b - a
        c, d = a+invphi2*h, a+invphi*h
        fc, fd = f(c), f(d)

        while abs(h) > tol:
            if fc < fd:
                b, d, fd = d, c, fc
                h *= invphi
                c = a + invphi2 * h
                fc = f(c)  
            else:
                a, c, fc = c, d, fd
                h *= invphi
                d = a + invphi * h
                fd = f(d)

        return (c+d)/2

    def eqAR_bySearch(self, Rsph, gamma, shpFactor : ShapeFactor):
        '''
        Equilibrium aspect ratio by cached search

        Parameters
        ----------
        Rsph : float or array
            Equivalent spherical radius
        gamma : float
            Interfacial energy
        shpFactor : ShapeFactor object
        '''
        normR = shpFactor.description.normalRadii
        interfacial = shpFactor.description.thermoFactor if self.ifmethod == 1 else shpFactor.description.eqRadiusFactor
        if hasattr(Rsph, '__len__'):
            eqAR = np.ones(len(Rsph))
            for i in range(len(Rsph)):
                eqAR[i] = self._cachedSearch(Rsph[i], gamma, interfacial, normR)
        else:
            eqAR = self._cachedSearch(Rsph, gamma, interfacial, normR)
        return eqAR

    def _cachedSearch(self, Rsph, gamma, interfacial, normR):
        '''
        Cached search for a single radius
        '''
        if self._aspectRatios is None:
            self.updateCache(normR)
        
        Vsph = 4/3*np.pi*Rsph**3
        Asph = 4*np.pi*Rsph**2
        eInter = Asph*interfacial(self._aspectRatios)*gamma
        eqAR = self._aspectRatios[np.argmin(self._normEnergies*Vsph+eInter)]

        #If eqAR is on the upper end (3/4) of the cached aspect ratios, then
        #added to the cached arrays until it's not
        while eqAR > self._aspectRatios[int(-self._cachedIntervals/4)] and self._aspectRatios[-1] < 100:
            self.updateCache(normR)
            eInter = Asph*interfacial(self._aspectRatios)*gamma
            eqAR = self._aspectRatios[np.argmin(self._normEnergies*Vsph+eInter)]

        return eqAR

