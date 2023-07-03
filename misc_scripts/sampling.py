import numpy as np
import matplotlib.pyplot as plt

N = 10000
m = 10
nBins = 50
pts = np.random.random((N, m))

sortedPts = np.sort(pts, axis=1)
sortedPts = np.concatenate((np.zeros((N,1)), sortedPts), axis=1)
segPts = sortedPts[:,1:] - sortedPts[:,:-1]

fig, ax = plt.subplots(2,2, figsize=(10,10))
for i in range(m):
    hist, bins = np.histogram(segPts[:,i], bins=nBins)
    ax[0,0].plot((bins[1:]+bins[:-1])/2, hist)

ax[0,0].set_xlim([0,1])
ax[0,0].set_ylim([0, 2*N/nBins])

ax[1,0].scatter(segPts[:,0], segPts[:,1], s=3)
ax[1,0].set_xlim([0,1])
ax[1,0].set_ylim([0,1])

pts2 = np.random.dirichlet(np.ones(m+1), N)
print(pts2.shape)

for i in range(m):
    hist, bins = np.histogram(pts2[:,i], bins=nBins)
    ax[0,1].plot((bins[1:]+bins[:-1])/2, hist)

ax[0,1].set_xlim([0,1])
ax[0,1].set_ylim([0, 2*N/nBins])

ax[1,1].scatter(pts2[:,0], pts2[:,1], s=3)
ax[1,1].set_xlim([0,1])
ax[1,1].set_ylim([0,1])

plt.show()