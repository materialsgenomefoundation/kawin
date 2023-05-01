from kawin.coupling.Strength import StrengthModel
import numpy as np

sm = StrengthModel()
G, b, nu = 79.3e9, 0.25e-9, 1/3
bp, ri = b, 2*b
eps, Gp = 0.001, 70e9
yAPB, ySFM, ySFP, gamma = 0.04, 0.1, 0.05, 0.5
sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
sm.setCoherencyParameters(eps)
sm.setModulusParameters(Gp, phase='all')
sm.setModulusParameters(2*Gp, phase='beta')
sm.setAPBParameters(yAPB, phase='alpha')
sm.setSFEParameters(ySFM, ySFP, bp, phase='beta')
sm.setInterfacialParameters(gamma)
sm.setTaylorFactor(1)

sm2 = StrengthModel()
sm2.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
sm2.setCoherencyParameters(eps, phase='alpha')

rs = 50e-9
Lss = 300e-9 - 2*rs

def test_coherency_edge():
    '''
    Tests weak and strong coherency contribution for edge dislocations
    '''
    weak = sm.coherencyWeak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)))
    strong = sm.coherencyStrong(rs, Lss, Lss)
    assert(np.allclose([weak, strong], [7.26102e7, 4.82891e7], atol=0, rtol=1e-3))

def test_coherency_screw():
    '''
    Tests weak and strong coherency contribution for screw dislocations
    '''
    sm.setDislocationParameters(G, b, nu, ri, theta=0, psi=120)
    weak = sm.coherencyWeak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)))
    strong = sm.coherencyStrong(rs, Lss, Lss)
    sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
    assert(np.allclose([weak, strong], [1.18431e7, 1.27934e8], atol=0, rtol=1e-3))

def test_modulus_edge():
    '''
    Tests weak and strong modulus contribution for edge dislocations
    '''
    weak = sm.modulusWeak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)))
    strong = sm.modulusStrong(rs, Lss, Lss)
    assert(np.allclose([weak, strong], [5.38138e7, 5.25096e7], atol=0, rtol=1e-3))

def test_modulus_screw():
    '''
    Tests weak and strong modulus contribution for screw dislocations
    '''
    sm.setDislocationParameters(G, b, nu, ri, theta=0, psi=120)
    weak = sm.modulusWeak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)))
    strong = sm.modulusStrong(rs, Lss, Lss)
    sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
    assert(np.allclose([weak, strong], [2.69069e7, 5.25096e7], atol=0, rtol=1e-3))

def test_APB_edge():
    '''
    Tests weak and strong anti-phase boundary contribution for edge dislocations
    '''
    weak = sm.APBweak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)), phase='alpha')
    strong = sm.APBstrong(rs, Lss, Lss, phase='alpha')
    assert(np.allclose([weak, strong], [8.42212e7, 5.79670e7], atol=0, rtol=1e-3))

def test_APB_screw():
    '''
    Tests weak and strong anti-phase boundary contribution for screw dislocations
    '''
    sm.setDislocationParameters(G, b, nu, ri, theta=0, psi=120)
    weak = sm.APBweak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)), phase='alpha')
    strong = sm.APBstrong(rs, Lss, Lss, phase='alpha')
    sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
    assert(np.allclose([weak, strong], [3.36224e7, 1.15934e8], atol=0, rtol=1e-3))

def test_SFE_edge():
    '''
    Tests weak and strong stacking fault energy contribution for edge dislocations
    '''
    weak = sm.SFEweak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)), phase='beta')
    strong = sm.SFEstrong(rs, Lss, Lss, phase='beta')
    assert(np.allclose([weak, strong], [3.83623e7, 4.19031e7], atol=0, rtol=1e-3))

def test_SFE_screw():
    '''
    Tests weak and strong stacking fault energy contribution for screw dislocations
    '''
    sm.setDislocationParameters(G, b, nu, ri, theta=0, psi=120)
    weak = sm.SFEweak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)), phase='beta')
    strong = sm.SFEstrong(rs, Lss, Lss, phase='beta')
    sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
    assert(np.allclose([weak, strong], [1.03693e7, 2.78075e7], atol=0, rtol=1e-3))

def test_IFE_edge():
    '''
    Tests weak and strong interfacial energy contribution for edge dislocations
    '''
    weak = sm.interfacialWeak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)))
    strong = sm.interfacialStrong(rs, Lss, Lss)
    assert(np.allclose([weak, strong], [1.58121e6, 5.00000e6], atol=0, rtol=1e-3))

def test_IFE_screw():
    '''
    Tests weak and strong interfacial energy contribution for screw dislocations
    '''
    sm.setDislocationParameters(G, b, nu, ri, theta=0, psi=120)
    weak = sm.interfacialWeak(rs, Lss, Lss / np.sqrt(np.cos(sm.psi / 2)))
    strong = sm.interfacialStrong(rs, Lss, Lss)
    sm.setDislocationParameters(G, b, nu, ri, theta=90, psi=120)
    assert(np.allclose([weak, strong], [7.90607e5, 5.00000e6], atol=0, rtol=1e-3))

def test_orowan():
    '''
    Tests orowan contribution
    '''
    oro = sm.orowan(rs, Lss)
    assert(np.allclose(oro, 1.02373e8, atol=0, rtol=1e-3))

def test_strength_output():
    '''
    Tests strength output values for when not all contributions are supplied
    No phase input or 'all' input should apply contributions to all phases
    Having a specific phase as an input should apply contributions to only that phase

    'all' should include only coherency, modulus and interfacial energy
    'alpha' should include coherency, modulus, anti-phase boundary and interfacial energy
    'beta' should include coherency, modulus, stacking fault energy and interfacial energy
    '''
    rs = 50e-9
    Ls = 300e-9 - 2*rs
    Leff = Ls / np.sqrt(np.cos(sm.psi / 2))

    cohWeak = sm.coherencyWeak(rs, Ls, Leff)
    cohStrong = sm.coherencyStrong(rs, Ls, Ls)

    #Values for 'all' and 'beta' will be defined separately to make sure getting strength
    #for the beta phase will override 'all'
    modWeak = sm.modulusWeak(rs, Ls, Leff)
    modStrong = sm.modulusStrong(rs, Ls, Ls)
    modWeakBeta = sm.modulusWeak(rs, Ls, Leff, 'beta')
    modStrongBeta = sm.modulusStrong(rs, Ls, Ls, 'beta')
    apbWeak = sm.APBweak(rs, Ls, Leff, 'alpha')
    apbStrong = sm.APBstrong(rs, Ls, Ls, 'alpha')
    sfeWeak = sm.SFEweak(rs, Ls, Leff, 'beta')
    sfeStrong = sm.SFEstrong(rs, Ls, Ls, 'beta')
    ifeWeak = sm.interfacialWeak(rs, Ls, Leff)
    ifeStrong = sm.interfacialStrong(rs, Ls, Ls)
    orowan = sm.orowan(rs, Ls)


    tauall = sm.getStrengthContributions(rs, Ls, )
    taualpha = sm.getStrengthContributions(rs, Ls, 'alpha')
    taubeta = sm.getStrengthContributions(rs, Ls, 'beta')

    assert(np.allclose(tauall[0], [cohWeak, modWeak, ifeWeak], atol=0, rtol=1e-3))
    assert(np.allclose(tauall[1], [cohStrong, modStrong, ifeStrong], atol=0, rtol=1e-3))
    assert(np.allclose(tauall[2], [orowan], atol=0, rtol=1e-3))

    assert(np.allclose(taualpha[0], [cohWeak, modWeak, apbWeak, ifeWeak], atol=0, rtol=1e-3))
    assert(np.allclose(taualpha[1], [cohStrong, modStrong, apbStrong, ifeStrong], atol=0, rtol=1e-3))
    assert(np.allclose(taualpha[2], [orowan], atol=0, rtol=1e-3))

    assert(np.allclose(taubeta[0], [cohWeak, modWeakBeta, sfeWeak, ifeWeak], atol=0, rtol=1e-3))
    assert(np.allclose(taubeta[1], [cohStrong, modStrongBeta, sfeStrong, ifeStrong], atol=0, rtol=1e-3))
    assert(np.allclose(taubeta[2], [orowan], atol=0, rtol=1e-3))

def test_no_strength_contribution():
    '''
    Tests that strength model will return 0 if there is no strength contribution

    Also test that the model will return an array of 0s if the input is an array
    '''
    rs = 50e-9
    Ls = 300e-9 - 2*rs
    Leff = Ls / np.sqrt(np.cos(sm.psi / 2))

    taugamma = sm2.getStrengthContributions(rs, Ls, 'gamma')
    strengthgamma = sm2.combineStrengthContributions(taugamma[0], taugamma[1], taugamma[2])

    N = 5
    rs = np.linspace(10e-9, 50e-9, N)
    Ls = 300e-9 - 2*rs
    Leff = Ls / np.sqrt(np.cos(sm.psi / 2))

    taugamma2 = sm2.getStrengthContributions(rs, Ls, 'gamma')
    strengthgamma2 = sm2.combineStrengthContributions(taugamma[0], taugamma[1], taugamma[2])


    assert(np.allclose(strengthgamma, 0, atol=0, rtol=1e-3))
    assert(np.allclose(strengthgamma2, np.zeros(N), atol=0, rtol=1e-3))