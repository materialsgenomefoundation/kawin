import numpy as np
import matplotlib.pyplot as plt

from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.DiffusionParameters import computeMobility
from kawin.diffusion.mesh import FiniteVolume1D, Cartesian2D

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

def plot1D(model: DiffusionModel, elements=None, zScale=1, ax=None, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    zScale: float
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh: FiniteVolume1D = model.mesh
    if not isinstance(mesh, FiniteVolume1D):
        raise ValueError('Diffusion mesh must be Cartesian1D, Cylindrical1D, Spherical1D or a subclass of FiniteVolume1D')
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    for e in elements:
        # If element is the dependent/reference element, then compute the dependent composition yr=1-sum(yi)
        if e in set(model.allElements) - set(model.elements):
            y = 1 - np.sum(mesh.y, axis=1)
        else:
            # TODO: while this is okay for FiniteVolume1D meshes, I might want a function in
            # the mesh as something like mesh.getResponse(varName) which will return a mesh with 1 response dimension
            y = mesh.y[:,model.elements.index(e)]
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot(mesh.z/zScale, y, *args, **plot_kwargs)

    ax.set_xlim([mesh.zEdge[0]/zScale, mesh.zEdge[-1]/zScale])
    ax.set_ylim([0,1])
    if len(elements) > 1:
        ax.legend()
    ax.set_xlabel(f'Distance*{zScale:.0e} (m)')
    ax.set_ylabel(f'Composition (at.%)')
    return ax

def plot1DTwoAxis(model: DiffusionModel, elementsL, elementsR, zScale=1, axL=None, axR=None, *args, **kwargs):
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
    zScale: float
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
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

    mesh: FiniteVolume1D = model.mesh
    if not isinstance(mesh, FiniteVolume1D):
        raise ValueError('Diffusion mesh must be Cartesian1D, Cylindrical1D, Spherical1D or a subclass of FiniteVolume1D')
    if isinstance(elementsL, str):
        elementsL = [elementsL]
    if isinstance(elementsR, str):
        elementsR = [elementsR]

    i = 0
    # Plotting is the same for each axis, so we can just loop across the two
    for ax, elements in zip([axL, axR], [elementsL, elementsR]):
        for e in elements:
            if e in set(model.allElements) - set(model.elements):
                y = 1 - np.sum(mesh.y, axis=1)
            else:
                y = mesh.y[:,model.elements.index(e)]
            plot_kwargs = _adjust_kwargs(e, {'label': e, 'color': f'C{i}'}, kwargs)
            ax.plot(mesh.z/zScale, y, *args, **plot_kwargs)
            i += 1

        # If the list of elements is small, we can add them to the y label
        if len(elements) <= 3:
            elementLabel = f'[{', '.join(elements)}] '
        else:
            elementLabel = ''
        ax.set_ylabel(f'Composition {elementLabel}(at.%)')

    axL.set_xlim([mesh.zEdge[0]/zScale, mesh.zEdge[-1]/zScale])
    axL.set_ylim([0,1])
    axR.set_ylim([0,1])
    axL.set_xlabel(f'Distance*{zScale:.0e} (m)')
    linesL, labelsL = axL.get_legend_handles_labels()
    linesR, labelsR = axR.get_legend_handles_labels()
    axL.legend(linesL+linesR, labelsL+labelsR, framealpha=1)
    return axL, axR

def plot1DPhases(model: DiffusionModel, phases = None, zScale = 1, ax=None, *args, **kwargs):
    '''
    Plots phase fractions over z

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    phases : list[str]
        Plots phases. If None, all phases in model are plotted
    zScale : float
        Scale factor for z-coordinates
    ax : matplotlib Axes object
        Axis to plot on

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh: FiniteVolume1D = model.mesh
    if not isinstance(mesh, FiniteVolume1D):
        raise ValueError('Diffusion mesh must be Cartesian1D, Cylindrical1D, Spherical1D or a subclass of FiniteVolume1D')
    
    # Compute phase fraction
    T = model.temperatureParameters(mesh.z, model.currentTime)
    mob_data = computeMobility(model.therm, mesh.y, T, model.hashTable)
    phases = model.phases if phases is None else phases
    if isinstance(phases, str):
        phases = [phases]

    # plot phase fraction
    for p in phases:
        pf = []
        for p_labels, p_fracs in zip(mob_data.phases, mob_data.phase_fractions):
            pf.append(np.sum(p_fracs[p_labels==p]))
        plot_kwargs = _adjust_kwargs(p, {'label': p}, kwargs)
        ax.plot(mesh.z/zScale, pf, *args, **plot_kwargs)

    ax.set_xlim([mesh.zEdge[0]/zScale, mesh.zEdge[-1]/zScale])
    ax.set_ylim([0,1])
    if len(phases) > 1:
        ax.legend()
    ax.set_xlabel(f'Distance*{zScale:.0e} (m)')
    ax.set_ylabel(f'Phase Fraction')
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
    if element in set(model.allElements) - set(model.elements):
        y = 1-np.sum(mesh.y, axis=2)
    else:
        y = mesh.y[...,model.elements.index(element)]
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
    cm = ax.plot(mesh.z[...,0]/zScale[0], mesh.z[...,1]/zScale[1], pf, *args, **plot_kwargs)
    ax.set_title(phase)
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm
