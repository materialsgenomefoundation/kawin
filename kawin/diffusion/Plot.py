from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.DiffusionParameters import computeMobility, HashTable
from kawin.diffusion.mesh import FiniteVolumeGrid, FiniteVolume1D, Cartesian2D

def _get_axis(ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        return ax
    else:
        return ax
    
def _adjust_kwargs(varName, defaultKwargs = {}, userKwargs = {}):
    '''
    Merges default kwargs with user input kwargs
    '''
    # Search through all user kwargs (this ensures any kwarg not in defaultKwargs will be added)
    for p in userKwargs:
        # If the kwarg is already defined in defaultKwargs, then override default with user
        if p in defaultKwargs:
            # If user specifies a dict for the kwarg (based off varName, then get the variable specific kwarg)
            if isinstance(userKwargs[p], dict) and varName in userKwargs[p]:
                defaultKwargs[p] = userKwargs[p][varName]
            else:
                defaultKwargs[p] = userKwargs[p]
        else:
            defaultKwargs[p] = userKwargs[p]
    return defaultKwargs

def _get_1D_mesh(model: DiffusionModel):
    mesh: FiniteVolume1D = model.mesh
    if not isinstance(mesh, FiniteVolume1D):
        raise ValueError('Diffusion mesh must be Cartesian1D, Cylindrical1D, Spherical1D or a subclass of FiniteVolume1D')
    return mesh

def _get_y(model: DiffusionModel, mesh_y: np.ndarray, element, sumValue = 1):
    # If element is the dependent/reference element, then compute the dependent composition
    # For composition, sum(y) = 1
    # For flux, sum(y) = 0
    if element in set(model.allElements) - set(model.elements):
        return sumValue - np.sum(mesh_y, axis=mesh_y.ndim-1)
    else:
        # TODO: while this is okay for FiniteVolume meshes, I might want a function in
        # the mesh as something like mesh.getResponse(varName) which will return a mesh with 1 response dimension
        return mesh_y[...,model.elements.index(element)]

def _set_1D_xlim(ax, mesh: FiniteVolume1D, zScale, zOffset):
    ax.set_xlim([(mesh.zEdge[0]+zOffset)/zScale, (mesh.zEdge[-1]+zOffset)/zScale])
    ax.set_xlabel(f'Distance*{zScale:.0e} (m)')

def plot1D(model: DiffusionModel, elements=None, zScale=1, zOffset=0, ax=None, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1D_mesh(model)
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    for e in elements:
        y = _get_y(model, mesh.y, e, 1)
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot((mesh.z+zOffset)/zScale, y, *args, **plot_kwargs)

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    if len(elements) > 1:
        ax.legend()
    ax.set_ylim([0,1])
    ax.set_ylabel(f'Composition (at.%)')
    return ax

def plot1DTwoAxis(model: DiffusionModel, elementsL, elementsR, zScale=1, zOffset=0, axL=None, axR=None, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh on left and right axes

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elementsL: list[str]
        List of elements to plot (can include the dependent/reference element)
    elementsR: list[str]
        List of elements to plot (can include the dependent/reference element)
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    axL: matplotlib axis (optional)
        Left axis
        Will be created if None
    axR: matplotlib axis (optional)
        Right axis
        Will be created if None

    Returns
    -------
    matplotlib Axis for left axis
    matplotlib Axis for right axis
    '''
    axL = _get_axis(axL)
    if axR is None:
        axR = axL.twinx()
    mesh = _get_1D_mesh(model)
    if isinstance(elementsL, str):
        elementsL = [elementsL]
    if isinstance(elementsR, str):
        elementsR = [elementsR]

    # Plotting is the same for each axis, so we can just loop across the two
    i = 0
    for ax, elements in zip([axL, axR], [elementsL, elementsR]):
        for e in elements:
            y = _get_y(model, mesh.y, e, 1)
            plot_kwargs = _adjust_kwargs(e, {'label': e, 'color': f'C{i}'}, kwargs)
            ax.plot((mesh.z+zOffset)/zScale, y, *args, **plot_kwargs)
            i += 1

        # If the list of elements is small, we can add them to the y label
        if len(elements) <= 3:
            elList = ', '.join(elements)
            elementLabel = f'[{elList}] '
        else:
            elementLabel = ''
        ax.set_ylabel(f'Composition {elementLabel}(at.%)') 
        ax.set_ylim([0,1])

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    linesL, labelsL = axL.get_legend_handles_labels()
    linesR, labelsR = axR.get_legend_handles_labels()
    axL.legend(linesL+linesR, labelsL+labelsR, framealpha=1)
    return axL, axR

def plot1DPhases(model: DiffusionModel, phases=None, zScale=1, zOffset=0, ax=None, *args, **kwargs):
    '''
    Plots phase fractions over z

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    phases : list[str]
        Plots phases. If None, all phases in model are plotted
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax : matplotlib Axes object
        Axis to plot on

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1D_mesh(model)
    
    # Compute phase fraction
    T = model.temperatureParameters(mesh.z, model.currentTime)
    # Temporary hash table, since we don't want to interfere with the internal model hash
    hashTable = HashTable()
    mob_data = computeMobility(model.therm, mesh.y, T, hashTable)
    phases = model.phases if phases is None else phases
    if isinstance(phases, str):
        phases = [phases]

    # plot phase fraction
    for p in phases:
        pf = []
        for p_labels, p_fracs in zip(mob_data.phases, mob_data.phase_fractions):
            pf.append(np.sum(p_fracs[p_labels==p]))
        plot_kwargs = _adjust_kwargs(p, {'label': p}, kwargs)
        ax.plot((mesh.z+zOffset)/zScale, pf, *args, **plot_kwargs)

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    if len(phases) > 1:
        ax.legend()
    ax.set_ylim([0,1])
    ax.set_ylabel(f'Phase Fraction')
    return ax

def plot1DFlux(model: DiffusionModel, elements=None, zScale=1, zOffset=0, ax=None, *args, **kwargs):
    '''
    Plots flux of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1D_mesh(model)
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())
    for e in elements:
        y = _get_y(model, fluxes, e, 0)
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot((mesh.zEdge+zOffset)/zScale, y, *args, **plot_kwargs)

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    if len(elements) > 1:
        ax.legend()
    ax.set_ylabel(f'$J/V_m$ ($m/s$)')
    return ax

def plot2D(model: DiffusionModel, element, zScale=1, ax=None, *args, **kwargs):
    '''
    Plots a composition profile on a 2D mesh

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    element: str
        Element to plot
    zScale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    
    # make sure zScale has 2 elements (for x and y axis)
    zScale = np.atleast_1d(zScale)
    if zScale.shape[0] == 1:
        zScale = zScale[0]*np.ones(2)

    # plot element
    # TODO: there should be a way to do contour and contourf (maybe separate plot functions?)
    y = _get_y(model, mesh.y, element, 1)
    plot_kwargs = _adjust_kwargs(element, {'vmin': 0, 'vmax': 1}, kwargs)
    cm = ax.pcolormesh(mesh.z[...,0]/zScale[0], mesh.z[...,1]/zScale[1], y, *args, **plot_kwargs)
    ax.set_title(element)
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm

def plot2DPhases(model: DiffusionModel, phase, zScale=1, ax=None, *args, **kwargs):
    '''
    Plots a composition profile on a 2D mesh

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    phase: str
        phase to plot
    zScale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    # make sure zScale has 2 elements (for x and y axis)
    zScale = np.atleast_1d(zScale)
    if zScale.shape[0] == 1:
        zScale = zScale[0]*np.ones(2)

    # We want z and y to be in [N,d] and [N,e] to be compatible with TemperatureParameters and computeMobility
    flatZ = mesh.flattenSpatial(mesh.z)
    flatY = mesh.flattenResponse(mesh.y)
    T = model.temperatureParameters(flatZ, model.currentTime)
    mob_data = computeMobility(model.therm, flatY, T, model.hashTable)
    pf = []
    for p_labels, p_fracs in zip(mob_data.phases, mob_data.phase_fractions):
        pf.append(np.sum(p_fracs[p_labels==phase]))

    # Reshape phase fraction to mesh shape (this will be (Nx, Ny, 1) for Cartesian2D)
    pf = mesh.unflattenResponse(pf, 1)
    plot_kwargs = _adjust_kwargs(phase, {'vmin': 0, 'vmax': 1}, kwargs)
    cm = ax.pcolormesh(mesh.z[...,0]/zScale[0], mesh.z[...,1]/zScale[1], pf, *args, **plot_kwargs)
    ax.set_title(phase)
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm

def plot2DFluxes(model: DiffusionModel, element, direction, zScale=1, ax=None, *args, **kwargs):
    '''
    Plots flux in x or y direction of element

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    element: str
        Element to plot
    direction: str
        'x' or 'y'
    zScale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    # make sure zScale has 2 elements (for x and y axis)
    zScale = np.atleast_1d(zScale)
    if zScale.shape[0] == 1:
        zScale = zScale[0]*np.ones(2)

    if direction != 'x' and direction != 'y':
        raise ValueError("direction must be \'x\' or \'y\'")

    fluxes = model.getFluxes(model.currentTime, model.getCurrentX())
    if direction == 'x':
        fluxes = fluxes[0]
        z = (mesh.zCorner[:,:-1] + mesh.zCorner[:,1:]) / 2
    elif direction == 'y':
        fluxes = fluxes[1]
        z = (mesh.zCorner[:-1,:] + mesh.zCorner[1:,:]) / 2

    y = _get_y(model, fluxes, element, 0)
    cm = ax.pcolormesh(z[...,0]/zScale[0], z[...,1]/zScale[1], y, *args, **kwargs)
    ax.set_title(f'$J_x/V_m$ {element} ($m/s$)')
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm


