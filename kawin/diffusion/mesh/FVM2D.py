import numpy as np
import matplotlib.pyplot as plt
from kawin.diffusion.mesh.MeshBase import FiniteVolumeGrid, arithmeticMean

class Cartesian2D(FiniteVolumeGrid):
    '''
    2D finite volume mesh

    y - value at node center (Nx, Ny, dims)
    z - spatial coordinate at node center (Nx, Ny, 2)
    zCorner - spatial coordinate at node corners (Nx+1, Ny+1, 2)
    dz - thickness of node in both dimensions

    TODO: boundary conditions not supported yet, so only no flux conditions

    Parameters
    ----------
    zx: list[float]
        Left and right boundary position of mesh
    Nx: int
        Number of cells along x
    zy: list[float]
        Top and bottom boundary position of mesh
    Ny: int
        Number of cells along y
    responses: int | list[str]
        Response variables
    '''
    def __init__(self, responses, zx, Nx, zy, Ny):
        super().__init__(responses, [zx, zy], [Nx, Ny], 2)

    def defineZCoordinates(self):
        zxEdge = np.linspace(self.zlim[0][0], self.zlim[0][1], self.N[0]+1)
        zyEdge = np.linspace(self.zlim[1][0], self.zlim[1][1], self.N[1]+1)
        # meshgrid will return (2, Ny, Nx), but we want (Nx, Ny, 2)
        self.zCorner = np.transpose(np.meshgrid(zxEdge, zyEdge), axes=(2,1,0))

        # we get z by averaging the 4 corners of zCorner
        self.z = (self.zCorner[:-1,:-1] + self.zCorner[1:,:-1] + self.zCorner[:-1,1:] + self.zCorner[1:,1:]) / 4
        self.dz = [self.z[1,0,0]-self.z[0,0,0], self.z[0,1,1]-self.z[0,0,1]]

    def _computeFluxes(self, pairs):
        '''
        Compute fluxes from (diffusivity, response) pairs on a 2D FVM mesh
        '''
        fluxX = np.zeros((self.N[0]+1, self.N[1], self.numResponses))
        fluxY = np.zeros((self.N[0], self.N[1]+1, self.numResponses))
        for p in pairs:
            D, r = p[0], p[1]
            D = self.unflattenResponse(D)
            r = self.unflattenResponse(r)
            avgFunc = p[2] if len(p) == 3 else arithmeticMean

            # flux along x (neighboring cells to compute D and flux are along x direction, 1st index)
            DmidX = avgFunc([D[:-1,:], D[1:,:]])
            fluxX[1:-1,:] += self._diffusiveFlux(DmidX, r[1:,:], r[:-1,:], self.dz[0])

            # flux along y (neighboring cells to compute D and flux are along y direction, 2nd index)
            DmidY = avgFunc([D[:,:-1], D[:,1:]])
            fluxY[:,1:-1] += self._diffusiveFlux(DmidY, r[:,1:], r[:,:-1], self.dz[1])

        return fluxX, fluxY

    def computedXdt(self, pairs):
        '''
        dx/dt = -dJx/dz + -dJy/dz
        '''
        fluxX, fluxY = self._computeFluxes(pairs)
        dXdt = -(fluxX[1:,:] - fluxX[:-1,:]) / self.dz[0] + -(fluxY[:,1:] - fluxY[:,:-1]) / self.dz[1]
        return dXdt

