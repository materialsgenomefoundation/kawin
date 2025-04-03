import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from kawin.precipitation.coupling import StrengthModel
from kawin.precipitation.coupling import DislocationParameters, CoherencyContribution, ModulusContribution, APBContribution, SFEContribution, InterfacialContribution, OrowanContribution
from kawin.precipitation.coupling.Strength import computeCRSS, combineCRSS
from kawin.precipitation.coupling import plotContribution, plotPrecipitateStrength

G = 79.3e9
b = 0.25e-9
nu = 1/3
ri = 2*b

edge_dislocations = DislocationParameters(G, b, nu, ri, theta=90)
screw_dislocations = DislocationParameters(G, b, nu, ri, theta=0)

eps = 0.001
Gp = 70e9
Gp_beta = 2*Gp
yAPB = 0.04
ySFM, ySFP, bp = 0.1, 0.05, b
gamma = 0.5
coherency = CoherencyContribution(eps=eps)
modulus = ModulusContribution(Gp=Gp, phase='all')
modulus_beta = ModulusContribution(Gp=Gp_beta, phase='beta')
apb_alpha = APBContribution(yAPB=yAPB, phase='alpha')
sfe_beta = SFEContribution(ySFM=ySFM, ySFP=ySFP, bp=bp, phase='beta')
interfacial = InterfacialContribution(gamma=gamma)

rss = 50e-9
Ls = 300e-9 - 2*rss

def test_coherency():
    '''
    Tests weak and strong coherency contribution for edge and screw dislocations
    '''
    weak, strong = coherency.computeCRSS(rss, Ls, edge_dislocations)
    assert_allclose([weak, strong], [7.26102e7, 4.82891e7], rtol=1e-3)

    weak, strong = coherency.computeCRSS(rss, Ls, screw_dislocations)
    assert_allclose([weak, strong], [1.18431e7, 1.27934e8], rtol=1e-3)

def test_modulus():
    '''
    Tests weak and strong modulus contribution for edge and screw dislocations
    '''
    weak, strong = modulus.computeCRSS(rss, Ls, edge_dislocations)
    assert_allclose([weak, strong], [5.38138e7, 5.25096e7], rtol=1e-3)

    weak, strong = modulus.computeCRSS(rss, Ls, screw_dislocations)
    assert_allclose([weak, strong], [2.69069e7, 5.25096e7], rtol=1e-3)

def test_APB():
    '''
    Tests weak and strong anti-phase boundary contribution for edge and screw dislocations
    '''
    weak, strong = apb_alpha.computeCRSS(rss, Ls, edge_dislocations)
    assert_allclose([weak, strong], [8.42212e7, 5.79670e7], rtol=1e-3)

    weak, strong = apb_alpha.computeCRSS(rss, Ls, screw_dislocations)
    assert_allclose([weak, strong], [3.36224e7, 1.15934e8], rtol=1e-3)

def test_SFE():
    '''
    Tests weak and strong stacking fault energy contribution for edge and screw dislocations
    '''
    weak, strong = sfe_beta.computeCRSS(rss, Ls, edge_dislocations)
    assert_allclose([weak, strong], [3.83623e7, 4.19031e7], rtol=1e-3)

    weak, strong = sfe_beta.computeCRSS(rss, Ls, screw_dislocations)
    assert_allclose([weak, strong], [1.03693e7, 2.78075e7], rtol=1e-3)

def test_IFE():
    '''
    Tests weak and strong interfacial energy contribution for edge and screw dislocations
    '''
    weak, strong = interfacial.computeCRSS(rss, Ls, edge_dislocations)
    assert_allclose([weak, strong], [1.58121e6, 5.00000e6], rtol=1e-3)

    weak, strong = interfacial.computeCRSS(rss, Ls, screw_dislocations)
    assert_allclose([weak, strong], [7.90607e5, 5.00000e6], rtol=1e-3)

def test_orowan():
    '''
    Tests orowan contribution
    '''
    oro = OrowanContribution().computeCRSS(rss, Ls, edge_dislocations)
    assert_allclose(oro, 1.02373e8, rtol=1e-3)

def test_strength_output():
    '''
    Tests strength output values for when not all contributions are supplied
    No phase input or 'all' input should apply contributions to all phases
    Having a specific phase as an input should apply contributions to only that phase

    'all' should include only coherency, modulus and interfacial energy
    'alpha' should include coherency, modulus, anti-phase boundary and interfacial energy
    'beta' should include coherency, modulus, stacking fault energy and interfacial energy
    '''
    rss = 50e-9
    Ls = 300e-9 - 2*rss

    contributions = [coherency, modulus, modulus_beta, apb_alpha, sfe_beta, interfacial]
    coh_weak, coh_strong = coherency.computeCRSS(rss, Ls, edge_dislocations)
    mod_weak, mod_strong = modulus.computeCRSS(rss, Ls, edge_dislocations)
    mod_weak_beta, mod_strong_beta = modulus_beta.computeCRSS(rss, Ls, edge_dislocations)
    apb_weak_alpha, apb_strong_alpha = apb_alpha.computeCRSS(rss, Ls, edge_dislocations)
    sfe_weak_beta, sfe_strong_beta = sfe_beta.computeCRSS(rss, Ls, edge_dislocations)
    ife_weak, ife_strong = interfacial.computeCRSS(rss, Ls, edge_dislocations)
    orowan = OrowanContribution().computeCRSS(rss, Ls, edge_dislocations)

    weak_all, strong_all, orowan_all = computeCRSS(rss, Ls, contributions, edge_dislocations, 'all')
    weak_alpha, strong_alpha, orowan_alpha = computeCRSS(rss, Ls, contributions, edge_dislocations, 'alpha')
    weak_beta, strong_beta, orowan_beta = computeCRSS(rss, Ls, contributions, edge_dislocations, 'beta')

    assert_allclose(orowan_all, orowan, rtol=1e-3)
    assert_allclose(orowan_alpha, orowan, rtol=1e-3)
    assert_allclose(orowan_beta, orowan, rtol=1e-3)

    all_names = [coherency.name, interfacial.name, modulus.name]
    assert sorted(list(weak_all.keys())) == all_names
    assert sorted(list(strong_all.keys())) == all_names
    assert_allclose([weak_all[c] for c in all_names], [coh_weak, ife_weak, mod_weak])
    assert_allclose([strong_all[c] for c in all_names], [coh_strong, ife_strong, mod_strong])

    alpha_names = [apb_alpha.name, coherency.name, interfacial.name, modulus.name]
    assert sorted(list(weak_alpha.keys())) == alpha_names
    assert sorted(list(strong_alpha.keys())) == alpha_names
    assert_allclose([weak_alpha[c] for c in alpha_names], [apb_weak_alpha, coh_weak, ife_weak, mod_weak])
    assert_allclose([strong_alpha[c] for c in alpha_names], [apb_strong_alpha, coh_strong, ife_strong, mod_strong])

    beta_names = [coherency.name, interfacial.name, modulus.name, sfe_beta.name]
    assert sorted(list(weak_beta.keys())) == beta_names
    assert sorted(list(strong_beta.keys())) == beta_names
    assert_allclose([weak_beta[c] for c in beta_names], [coh_weak, ife_weak, mod_weak_beta, sfe_weak_beta])
    assert_allclose([strong_beta[c] for c in beta_names], [coh_strong, ife_strong, mod_strong_beta, sfe_strong_beta])

    exp = 1.8
    strength_all = combineCRSS(weak_all, strong_all, orowan_all, exp)
    strength_alpha = combineCRSS(weak_alpha, strong_alpha, orowan_alpha, exp, False)
    strength_beta, weak_total_beta, strong_total_beta, orowan_total_beta = combineCRSS(weak_beta, strong_beta, orowan_beta, exp, True)

    assert_allclose(strength_all, 7.44463e7, rtol=1e-3)
    assert_allclose(strength_alpha, 9.79077e7, rtol=1e-3)
    assert_allclose([strength_beta, weak_total_beta, strong_total_beta, orowan_total_beta], [1.02373e8, 9.04422e8, 3.52631e8, 1.02373e8], rtol=1e-3)

def test_no_strength_contribution():
    '''
    Tests that strength model will return 0 if there is no strength contribution

    Also test that the model will return an array of 0s if the input is an array
    '''
    weak_gamma, strong_gamma, orowan = computeCRSS(rss, Ls, [sfe_beta], edge_dislocations, 'gamma')
    strength_gamma = combineCRSS(weak_gamma, strong_gamma, orowan)

    N = 5
    weak_gamma, strong_gamma, orowan = computeCRSS(rss*np.ones(N), Ls*np.ones(N), [sfe_beta], edge_dislocations, 'gamma')
    strength_gamma_array = combineCRSS(weak_gamma, strong_gamma, orowan)

    assert(np.allclose(strength_gamma, 0, atol=0, rtol=1e-3))
    assert(np.allclose(strength_gamma_array, np.zeros(N), atol=0, rtol=1e-3))

def test_strength_plotting():
    rss_array = np.linspace(10e-9, 50e-9, 100)
    Ls_array = 300e-9 - 2*rss_array

    fig, ax = plt.subplots()
    # non-orowan contributions will plot weak and strong
    plotContribution(rss_array, Ls_array, sfe_beta, edge_dislocations, ax=ax)
    assert len(ax.lines) == 2
    plt.close(fig)

    fig, ax = plt.subplots()
    # orowan contribution only has 1 output
    plotContribution(rss_array, Ls_array, OrowanContribution(), edge_dislocations, strengthUnits='GPa', ax=ax)
    assert len(ax.lines) == 1
    plt.close(fig)

    sm = StrengthModel(['alpha', 'beta'], [coherency, modulus, modulus_beta, apb_alpha, sfe_beta, interfacial], edge_dislocations)

    fig, ax = plt.subplots()
    # if not plotting contributions, then only total strength is plotted
    plotPrecipitateStrength(rss_array, Ls_array, sm, 'alpha', False, strengthUnits='Pa', ax=ax)
    assert len(ax.lines) == 1
    plt.close(fig)

    fig, ax = plt.subplots()
    # if plotting contributions, then base, solid solution, precipitation and total strength is plotted
    plotPrecipitateStrength(rss_array, Ls_array, sm, 'beta', True, strengthUnits='Pa', ax=ax)
    assert len(ax.lines) == 4
    plt.close(fig)
