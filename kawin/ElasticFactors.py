import numpy as np
import itertools

class StrainEnergy:
    '''
    Defines class for calculating strain energy of a precipitate

    Ellipsoidal precipitates will use the Eshelby's tensor
    Spherical and Cuboidal precipitates will use the Khachaturyan's approximation
    '''

    SPHERE = 0
    ELLIPSE = 1
    CUBE = 2
    CONSTANT = 3

    def __init__(self):
        self.c = None
        self._c4 = np.zeros((3,3,3,3))
        self.eigstrain = np.zeros((3,3))
        self.appstress = np.zeros((3,3))
        self.appstrain = np.zeros((3,3))

        self.setIntegrationIntervals(8, 8)

        self._strainEnergyGeneric = self._strainEnergyConstant
        self.type = self.CONSTANT
        self.r = np.zeros(3)
        self.oldr = np.zeros(3)
        self._prevEnergy = 0

        self._gElasticConstant = 0

    def setSpherical(self):
        self._strainEnergyGeneric = self._strainEnergySphere
        self.type = self.SPHERE
    
    def setCuboidal(self):
        self._strainEnergyGeneric = self._strainEnergyCube
        self.type = self.CUBE

    def setEllipsoidal(self):
        self._strainEnergyGeneric = self._strainEnergyEllipsoid
        self.type = self.ELLIPSE

    def setConstantElasticEnergy(self, energy):
        '''
        If elastic strain energy is known to be a constant, this can be use to greatly
        simplify calculations

        Parameters
        ----------
        energy - strain energy in J/m3
        '''
        self._gElasticConstant = energy
        self._strainEnergyGeneric = self._strainEnergyConstant
        self.type = self.CONSTANT
        
    def setIntegrationIntervals(self, phiInt, thetaInt):
        '''
        Number of intervals to split domain along phi and theta for integration
        '''
        self.phiInt = phiInt
        self.thetaInt = thetaInt

        #Integral should be symmetric per quadrant (need to check)
        #Thus we only need to integrate along a single quadrant
        self.dphi = np.pi/2 / phiInt
        self.dtheta = np.pi/2 / thetaInt
        self.midPhi = np.linspace(self.dphi/2, np.pi/2 - self.dphi/2, phiInt)
        self.midTheta = np.linspace(self.dtheta/2, np.pi/2 - self.dtheta/2, thetaInt)

    def setElasticTensor(self, tensor):
        '''
        6x6 elastic matrix
        '''
        self.c = np.array(tensor)
        self._setupElasticTensor()

    def setElasticConstants(self, c11, c12, c44):
        self.c = np.zeros((6, 6))
        self.c[0,0], self.c[1,1], self.c[2,2] = c11, c11, c11
        self.c[0,1], self.c[0,2], self.c[1,0], self.c[1,2], self.c[2,0], self.c[2,1] = c12, c12, c12, c12, c12, c12
        self.c[3,3], self.c[4,4], self.c[5,5] = c44, c44, c44
        self._setupElasticTensor()

    def setModuli(self, E = None, nu = None, G = None, lam = None, K = None, M = None):
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

        There's gotta be a better way to do this
        '''
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
                S = -np.sqrt(E**2 + 9*M**2 - 10*E*M)
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
        self.c = np.linalg.inv(s)

        self._setupElasticTensor()

    def _setupElasticTensor(self):
        '''
        Creates the 4th rank elastic tensor
        This makes it easer to use np.tensordot for most calculations

        This will also automatically calculate applied strain, in case the applied stress is
        given before the elastic constants are
        '''
        vMap = {frozenset({0}): 0, frozenset({1}): 1, frozenset({2}): 2,
                frozenset({1,2}): 3, frozenset({0,2}): 4, frozenset({0,1}): 5}

        self._c4 = np.zeros((3,3,3,3))
        for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
            self._c4[i,j,k,l] = self.c[vMap[frozenset({i,j})], vMap[frozenset({k,l})]]

        self.getAppliedStrain()

        #Set type to something other than CONSTANT
        #Since CONSTANT is the only method not requiring the elastic tensor,
        #    we assume that the user is intending to calculate strain energy when inputting the tensor
        self.type = self.SPHERE

    def setEigenstrain(self, strain):
        strain = np.array(strain)
        #If scalar, then apply to all 3 axis
        if (type(strain) == np.ndarray and strain.ndim == 0):
            self.eigstrain = strain * np.identity(3)
        #If array of length 3, then apply strain along each index to corresponding axis
        elif strain.ndim == 1:
            self.eigstrain = np.array([[strain[0], 0, 0], [0, strain[1], 0], [0, 0, strain[2]]])
        #Else, assume it's a tensor
        else:
            self.eigstrain = strain

    def setAppliedStress(self, stress):
        stress = np.array(stress)
        #If scalar, then apply to all 3 axis
        if (type(stress) == np.ndarray and stress.ndim == 0):
            self.appstress = stress * np.identity(3)
        #If array of length 3, then apply stress along each index to corresponding axis
        elif stress.ndim == 1:
            self.appstress = np.array([[stress[0], 0, 0], [0, stress[1], 0], [0, 0, stress[2]]])
        #Else, assume it's a tensor
        else:
            self.appstress = stress

        if self.c is not None:
            self.getAppliedStrain()

    def getAppliedStrain(self):
        flatStress = np.array([self.appstress[0,0], self.appstress[1,1], self.appstress[2,2], \
                                self.appstress[1,2], self.appstress[0,2], self.appstress[0,1]])
        flatStrain = np.matmul(np.linalg.inv(self.c), flatStress)
        self.appstrain = np.array([[flatStrain[0], flatStrain[5], flatStrain[4]], \
                                    [flatStrain[5], flatStrain[1], flatStrain[3]], \
                                    [flatStrain[3], flatStrain[4], flatStrain[2]]])

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

    def _OhmGeneral(self, n):
        '''
        Ohm term for general system

        Ohm_ij = inverse(C_iklj * n_k * n_l)
        '''
        invOhm = np.tensordot(self._c4, np.tensordot(n, n, axes=0), axes=[[1,2], [0,1]])
        return np.linalg.inv(invOhm)

    def sphericalIntegral(self, func):
        '''
        Integrates over spherical surface given function that takes in (phi, theta)
        '''
        intSum = 0
        for i, j in itertools.product(range(self.phiInt), range(self.thetaInt)):
            intSum += func(self.midPhi[i], self.midTheta[j])
        intSum *= self.dtheta * self.dphi

        return 8*intSum

    def Dfunc(self, phi, theta):
        n = self._n(phi, theta)
        endTerm = np.sin(theta) / self._beta(self.r[0], self.r[1], self.r[2], phi, theta)**3
        d = np.tensordot(self._OhmGeneral(n), np.tensordot(n, n, axes=0), axes=0) * endTerm
        return d

    def Dijkl(self):
        return -np.product(self.r)/(4*np.pi) * self.sphericalIntegral(self.Dfunc)

    def Sijmn(self, D):
        '''
        S_ijmn = -0.5 * C_lkmn * (D_iklj + D_jkli)
        '''
        S = -0.5 * np.tensordot(self._c4, D + np.transpose(D, (3,1,2,0)), axes=[[0,1],[2,1]])
        #The tensor product gives S_mnij so we'll need to transpose it
        return np.transpose(S, (2,3,0,1))

    def _strainC(self, S, strain):
        '''
        ec_ij = S_ijkl * e_kl
        '''
        return np.tensordot(S, strain, axes=[[2,3],[0,1]])

    def _stress(self, strain):
        '''
        sigma_ij = C_ijkl * e_kl
        '''
        return np.tensordot(self._c4, strain, axes = [[2,3], [0,1]])

    def _strainEnergy(self, stress, strain, V):
        '''
        u = -0.5 * V * sigma_ij * strain_ij
        '''
        return -0.5 * V * np.sum(stress * strain)

    def _strainEnergyEllipsoid(self):
        V = 4*np.pi/3 * np.product(self.r)
        S = self.Sijmn(self.Dijkl())
        stress = self._stress(self._strainC(S, self.eigstrain) - self.eigstrain)
        stress0 = self._stress(self._strainC(S, self.appstrain) - self.appstrain)
        return self._strainEnergy(stress - stress0, self.eigstrain - self.appstrain, V)

    def _Khachaturyan(self, I1, I2):
        '''
        Khachaturyan's approximation for strain energy of spherical and cuboidal precipitates
        '''
        V = 4*np.pi/3 * np.product(self.r)
        A1 = 2 * (self.c[0,0] - self.c[0,1]) / self.c[0,0]
        A1 -= 12 * (self.c[0,0] + 2 * self.c[0,1]) * (self.c[0,0] - self.c[0,1] - 2 * self.c[3,3]) / (self.c[0,0] * (self.c[0,0] + self.c[0,1] + 2*self.c[3,3])) * I1
        A2 = -54 * (self.c[0,0] + 2 * self.c[0,1]) * (self.c[0,0] - self.c[0,1] - 2 * self.c[3,3])**2 / (self.c[0,0] * (self.c[0,0] + self.c[0,1] + 2 * self.c[3,3]) * (self.c[0,0] + 2 * self.c[0,1] + 4 * self.c[3,3])) * I2
        return 0.5 * (self.c[0,0] + 2 * self.c[0,1]) * (A1 + A2) * self.eigstrain[0,0]**2 * V

    def _strainEnergyCube(self):
        '''
        Strain energy of perfect cube (cubic factor = sqrt(2))
        '''
        return self._Khachaturyan(0.006931, 0.000959)
    
    def _strainEnergySphere(self):
        '''
        Strain energy of perfect sphere (cubic factor = 1)
        '''
        return self._Khachaturyan(1/15, 1/105)

    def _strainEnergyCuboidal(self, eta = 1):
        '''
        For cuboidal preciptitates, strain energy can be approximated
        as a linear interpolation between a perfect cube and sphere
        '''
        sC = self.strainEnergyCube()
        sS = self.strainEnergySphere()
        return (sC - sS) * (eta - 1) / (np.sqrt(2) - 1) + sS

    def _strainEnergyConstant(self):
        return 4 * np.pi / 3 * np.product(self.r) * self._gElasticConstant

    def _strainEnergySingle(self, rsingle):
        '''
        Generic strain energy function
        '''
        self.r = rsingle
        self._prevEnergy = self._prevEnergy if np.array_equal(self.r, self.oldr) else self._strainEnergyGeneric()
        self.oldr = self.r
        return self._prevEnergy

    def strainEnergy(self, r):
        r = np.array(r)
        #If single set of radii
        if r.ndim == 1:
            return self._strainEnergySingle(r)
        else:
            eng = np.zeros(len(r))
            for i in range(len(r)):
                eng[i] = self._strainEnergySingle(r[i])
            return eng

    #Additional methods for verifying that the general methods reduce to these
    #specific cases for cubic crystal symmetry
    def _OhmCubic(self, n):
        '''
        Ohm term for cubic crystal symmetry
        We can use this to check validity of _OhmGeneral
        '''
        ohm = np.zeros((3,3))
        jk = [(1,2), (0,2), (0,1)]
        for i in range(3):
            for j in range(3):
                if i == j:
                    jn, k = jk[i]
                    ohm[i,i] = (self.c[3,3] + (self.c[0,0]-self.c[3,3])*(n[jn]**2 + n[k]**2) + self._xi*(self.c[0,0]+self.c[0,1])*(n[jn]*n[k])**2) / (self.c[3,3]*self._D(n))
                else:
                    k = 3 - (i + j)
                    ohm[i,j] = -(self.c[0,1] + self.c[3,3])*(1 + self._xi*n[k]**2)*n[i]*n[j] / (self.c[3,3]*self._D(n))
        
        return ohm

    def _D(self, n):
        '''
        Needed for the Ohm term with cubic crystal symmetry
        '''
        d = self.c[0,0]
        d += self._xi*(self.c[0,0] + self.c[0,1])*((n[0]*n[1])**2 + (n[0]*n[2])**2 + (n[1]*n[2])**2)
        d += self._xi**2 * (self.c[0,0] + 2*self.c[0,1] + self.c[3,3])*(n[0]*n[1]*n[2])**2
        return d

    @property
    def _xi(self):
        return (self.c[0,0] - self.c[0,1] - 2*self.c[3,3]) / self.c[3,3]

