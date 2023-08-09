import numpy as np
from numpy.testing import assert_allclose
from kawin.precipitation import PopulationBalanceModel

#Set parameters for pbm. Default bins are increased here so that added bins should be 50
bins = 200
qBins = int(200/4)
minBins = 100
maxBins = 300
pbm = PopulationBalanceModel(1e-10, 1e-8, 200, 100, 300)

def test_reset():
    '''
    Resetting the model should bring it to the initialized parameters
    '''
    #Add and adjust pbm arbitrarily, changing these should not affect this test function
    pbm.addSizeClasses(150)
    pbm.PSD[-1] = 2
    pbm.adjustSizeClassesEuler(False)

    pbm.reset()
    assert(len(pbm.PSD) == bins and pbm.bins == len(pbm.PSD))
    assert(len(pbm.PSDbounds) == bins+1)
    assert(len(pbm.PSDsize) == bins)
    assert(pbm.PSDbounds[0] == 1e-10)
    assert(pbm.PSDbounds[-1] == 1e-8)

def test_addBins():
    '''
    If last bin is filled, then the number of bins added should be default bins/2
    '''
    pbm.PSD[-1] = 2
    finalLength = pbm.PSDbounds[-1] + qBins * (pbm.PSDbounds[1] - pbm.PSDbounds[0])
    pbm.adjustSizeClassesEuler(False)
    assert(len(pbm.PSD) == bins+qBins and pbm.bins == len(pbm.PSD))
    assert_allclose(pbm.PSDbounds[-1], finalLength, rtol=1e-6)
    assert_allclose(pbm.max, finalLength, rtol=1e-6)
    assert(len(pbm.PSDbounds) == bins+qBins+1)
    assert(len(pbm.PSDsize) == bins+qBins)
    assert_allclose(pbm.PSDsize[0], 0.5*(pbm.PSDbounds[0] + pbm.PSDbounds[1]), atol=0, rtol=1e-6)
    pbm.reset()

def test_increaseBinSize():
    '''
    If number of bins > maxBins, then increase bin size to keep range with bins = minBins
    '''
    pbm.addSizeClasses(int(0.9*bins))
    pbm.PSD[-1] = 2
    finalLength = pbm.PSDbounds[-1] + qBins*(pbm.PSDbounds[1] - pbm.PSDbounds[0])
    pbm.adjustSizeClassesEuler(False)
    assert(len(pbm.PSD) == minBins and pbm.bins == len(pbm.PSD))
    assert_allclose(pbm.PSDbounds[-1], finalLength, rtol=1e-6)
    assert_allclose(pbm.max, finalLength, rtol=1e-6)
    assert(len(pbm.PSDbounds) == minBins+1)
    assert(len(pbm.PSDsize) == minBins)
    assert_allclose(pbm.PSDsize[0], 0.5*(pbm.PSDbounds[0] + pbm.PSDbounds[1]), atol=0, rtol=1e-6)
    pbm.reset()

def test_decreaseBinSize():
    '''
    If max filled bin < 1/2 minBins, then decrease bin size so the last filled bin is the max and bins = maxBins
    '''
    filledBin = int(1/4 * minBins)
    pbm.PSD[filledBin] = 2
    finalLength = pbm.PSDbounds[filledBin+1]
    pbm.adjustSizeClassesEuler(True)
    assert(len(pbm.PSD) == maxBins and pbm.bins == len(pbm.PSD))
    assert_allclose(pbm.PSDbounds[-1], finalLength, rtol=1e-6)
    assert_allclose(pbm.max, finalLength, rtol=1e-6)
    assert(len(pbm.PSDbounds) == maxBins+1)
    assert(len(pbm.PSDsize) == maxBins)
    assert_allclose(pbm.PSDsize[0], 0.5*(pbm.PSDbounds[0] + pbm.PSDbounds[1]), atol=0, rtol=1e-6)
    pbm.reset()

'''
Nucleate function is deprecated behavior for PBM in favor of the new solver system, so we'll take this out for now

def test_nucleateSmall():
    #If nucleate radius is smaller than PSD length, then no change
    pbm.Nucleate(10, 1e-9)
    assert(len(pbm.PSD) == bins and pbm.bins == len(pbm.PSD))
    assert(len(pbm.PSDbounds) == bins+1)
    assert(len(pbm.PSDsize) == bins)
    assert(pbm.Moment(0) == 10)
    assert(pbm.PSDbounds[-1] == 1e-8)

def test_nucleateBig():
    #If nucleate radius is larger than PSD length, then increase bin size
    #such that number of bins is the same, but max if 5*radius
    r = 1e-7
    pbm.Nucleate(10, r)
    assert(len(pbm.PSD) == bins and pbm.bins == len(pbm.PSD))
    assert(len(pbm.PSDbounds) == bins+1)
    assert(len(pbm.PSDsize) == bins)
    assert(pbm.Moment(0) == 10)
    assert_allclose(pbm.PSDbounds[-1], 5*r, rtol=1e-6)
    assert_allclose(pbm.PSDsize[0], 0.5*(pbm.PSDbounds[0] + pbm.PSDbounds[1]), atol=0, rtol=1e-6)
    pbm.reset()
'''

def test_DT():
    '''
    Calculated DT with constant growth rate
    DT = ratio * binSize / (max(growth rate))

    Previous version had ratio of 0.5, but this was decreased slightly to 0.4 for numerical stability
    '''
    growth = 5*np.ones(pbm.bins+1)
    pbm.PSD = 2*np.ones(pbm.bins)
    ratio = 0.4
    trueDT = ratio * (pbm.PSDbounds[1] - pbm.PSDbounds[0]) / (growth[0])
    dissIndex = pbm.getDissolutionIndex(1e-3, 0)
    calcDT = pbm.getDTEuler(5, growth, dissIndex)
    assert_allclose(trueDT, calcDT, rtol=1e-6)
