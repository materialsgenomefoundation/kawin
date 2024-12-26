import numpy as np
from numpy.testing import assert_allclose
import pytest

from kawin.precipitation import PrecipitateParameters, MatrixParameters
from kawin.precipitation.parameters.Nucleation import NucleationBarrierParameters, NucleationSiteParameters

def test_nucleation_barrier_updating():
    '''
    tests that the nucleation barrier factors in precipitate parameters
    will automatically update with updated values
    '''
    prec = PrecipitateParameters('phase')
    
    with pytest.raises(ValueError):
        value = prec.nucleation.GBk

    prec.gamma = 0.3
    assert_allclose(prec.nucleation.GBk, 0.5, rtol=1e-3)

    types = ['bulk', 'dislocations', 'grain boundaries', 'grain edges', 'grain corners']
    # Order is [area factor, volume factor, gb removal, area removal]
    test_values = {
        'bulk': [12.566370614359172, 4.1887902047863905, 0.0, 1.0],
        'dislocations': [12.566370614359172, 4.1887902047863905, 0.0, 1.0],
        'grain boundaries': [6.283185307179586, 1.308996938995747, 2.356194490192345, 0.8660254037844386],
        'grain edges': [4.078042913449462, 0.6718303352064217, 2.0625519078301955, 0.8102657977661342],
        'grain corners': [2.975471716584403, 0.42215773311582705, 1.7089985172369215, 0.7375575391181026],
    }
    for t in types:
        prec.nucleation.setNucleationType(t)
        assert_allclose([prec.nucleation.areaFactor, prec.nucleation.volumeFactor, prec.nucleation.gbRemoval, prec.nucleation.areaRemoval], 
                        test_values[t], rtol=1e-3)
        
    # test if gamma is too small, then ValueError is raised for grain boundaries, edges, corners
    prec.gamma = 0.1
    for t in ['grain boundaries', 'grain edges', 'grain corners']:
        prec.nucleation.setNucleationType(t)
        with pytest.raises(ValueError):
            value = prec.nucleation.areaFactor

        with pytest.raises(ValueError):
            value = prec.nucleation.volumeFactor

        with pytest.raises(ValueError):
            value = prec.nucleation.gbRemoval

        with pytest.raises(ValueError):
            value = prec.nucleation.areaRemoval

    # assert that if gb nucleation, then setting shape to non-spherical will raise ValueError
    with pytest.raises(ValueError):
        prec.nucleation.setNucleationType('grain boundaries')
        prec.shapeFactor.setPrecipitateShape('plate')

    # assert that if non-spherical shape, then setting gb nucleation will raise ValueError
    prec.nucleation.setNucleationType('dislocations')
    prec.shapeFactor.setPrecipitateShape('sphere')
    with pytest.raises(ValueError):
        prec.shapeFactor.setPrecipitateShape('plate')
        prec.nucleation.setNucleationType('grain boundaries')
        

def test_nucleation_barrier_shape():
    '''
    test that shape of nucleation barrier factors are same length as GBk
    '''
    prec = PrecipitateParameters('phase')
    
    gbk = prec.nucleation.description.gbRatio(0.3, np.linspace(0.1, 0.2, 10))
    values = prec.nucleation.description.areaFactor(gbk)
    assert gbk.shape == values.shape

    values = prec.nucleation.description.volumeFactor(gbk)
    assert gbk.shape == values.shape

    values = prec.nucleation.description.gbRemoval(gbk)
    assert gbk.shape == values.shape

    values = prec.nucleation.description.areaRemoval(gbk)
    assert gbk.shape == values.shape

def test_nucleation_barrier_rcrit():
    '''
    test value of Rcrit and Gcrit
    '''
    prec = PrecipitateParameters('phase')
    prec.gamma = 0.2
    prec.nucleation.setNucleationType('grain boundaries')

    rcrit = prec.nucleation.Rcrit(10000)
    gcrit = prec.nucleation.Gcrit(10000, 3e-9)

    assert_allclose(rcrit, 3.9999e-5, rtol=1e-3)
    assert_allclose(gcrit, 1.9437632e-18, rtol=1e-3)

def test_nucleation_site_updating():
    '''
    test that updating volume and composition will update nucleation site parameters
    '''
    matrix = MatrixParameters('A')

    with pytest.raises(ValueError):
        value = matrix.nucleationSites.GBareaN0

    with pytest.raises(ValueError):
        value = matrix.nucleationSites.GBedgeN0

    # GB corner N0 does not depend on molar volume
    value = matrix.nucleationSites.GBcornerN0

    matrix.volume.setVolume(1e-5, 'VM', 4)
    assert_allclose(matrix.nucleationSites.VmAlpha, 1e-5, rtol=1e-3)
    
    matrix.initComposition = 1e-3
    assert_allclose(matrix.nucleationSites.bulkN0, 6.022e25, rtol=1e-3)

    # If bulkN0 is set, then don't update
    matrix.nucleationSites.bulkN0 =1e30
    matrix.initComposition = 2e-3
    assert_allclose(matrix.nucleationSites.bulkN0, 1e30, rtol=1e-3)