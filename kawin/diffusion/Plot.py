import matplotlib.pyplot as plt
import numpy as np

def plot(diffModel, ax = None, plotReference = True, plotElement = None, zScale = 1, *args, **kwargs):
    '''
    Plots composition profile

    Parameters
    ----------
    ax : matplotlib Axes object
        Axis to plot on
    plotReference : bool
        Whether to plot reference element (composition = 1 - sum(composition of rest of elements))
    plotElement : None or str
        Plots single element if it is defined, otherwise, all elements are plotted
    zScale : float
        Scale factor for z-coordinates
    '''
    if ax is None:
        fig, ax = plt.subplots(1,1)

    if not diffModel.isSetup:
        diffModel.setup()

    if plotElement is not None:
        if plotElement not in diffModel.elements and plotElement in diffModel.allElements:
            x = 1 - np.sum(diffModel.x, axis=0)
        else:
            e = diffModel._getElementIndex(plotElement)
            x = diffModel.x[e]
        ax.plot(diffModel.z/zScale, x, *args, **kwargs)
    else:
        if plotReference:
            refE = 1 - np.sum(diffModel.x, axis=0)
            ax.plot(diffModel.z/zScale, refE, label=diffModel.allElements[0], *args, **kwargs)
        for e in range(len(diffModel.elements)):
            ax.plot(diffModel.z/zScale, diffModel.x[e], label=diffModel.elements[e], *args, **kwargs)
        
    ax.set_xlim([diffModel.zlim[0]/zScale, diffModel.zlim[1]/zScale])
    if plotElement is None:
        ax.legend()
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Composition (at.%)')

    return ax

def plotTwoAxis(diffModel, Lelements, Relements, zScale = 1, axL = None, axR = None, *args, **kwargs):
    '''
    Plots composition profile with two y-axes

    Parameters
    ----------
    axL : matplotlib Axes object
        Left axis to plot on
    Lelements : list of str
        Elements to plot on left axis
    Relements : list of str
        Elements to plot on right axis
    axR : matplotlib Axes object (optional)
        Right axis to plot on
        If None, then the right axis will be created
    zScale : float
        Scale factor for z-coordinates
    '''
    if axL is None:
        fig, axL = plt.subplots(1,1)

    if not diffModel.isSetup:
        diffModel.setup()

    if type(Lelements) is str:
        Lelements = [Lelements]
    if type(Relements) is str:
        Relements = [Relements]

    ci = 0
    refE = 1 - np.sum(diffModel.x, axis=0)
    if axR is None:
        axR = axL.twinx()
    for e in range(len(Lelements)):
        if Lelements[e] in diffModel.elements:
            eIndex = diffModel._getElementIndex(Lelements[e])
            axL.plot(diffModel.z/zScale, diffModel.x[eIndex], label=diffModel.elements[eIndex], color = 'C' + str(ci), *args, **kwargs)
            ci = ci+1 if ci <= 9 else 0
        elif Lelements[e] in diffModel.allElements:
            axL.plot(diffModel.z/zScale, refE, label=diffModel.allElements[0], color = 'C' + str(ci), *args, **kwargs)
            ci = ci+1 if ci <= 9 else 0
    for e in range(len(Relements)):
        if Relements[e] in diffModel.elements:
            eIndex = diffModel._getElementIndex(Relements[e])
            axR.plot(diffModel.z/zScale, diffModel.x[eIndex], label=diffModel.elements[eIndex], color = 'C' + str(ci), *args, **kwargs)
            ci = ci+1 if ci <= 9 else 0
        elif Relements[e] in diffModel.allElements:
            axR.plot(diffModel.z/zScale, refE, label=diffModel.allElements[0], color = 'C' + str(ci), *args, **kwargs)
            ci = ci+1 if ci <= 9 else 0

    
    axL.set_xlim([diffModel.zlim[0]/zScale, diffModel.zlim[1]/zScale])
    axL.set_xlabel('Distance (m)')
    axL.set_ylabel('Composition (at.%) ' + str(Lelements))
    axR.set_ylabel('Composition (at.%) ' + str(Relements))
    
    lines, labels = axL.get_legend_handles_labels()
    lines2, labels2 = axR.get_legend_handles_labels()
    axR.legend(lines+lines2, labels+labels2, framealpha=1)

    return axL, axR

def plotPhases(diffModel, ax = None, plotPhase = None, zScale = 1, *args, **kwargs):
    '''
    Plots phase fractions over z

    Parameters
    ----------
    ax : matplotlib Axes object
        Axis to plot on
    plotPhase : None or str
        Plots single phase if it is defined, otherwise, all phases are plotted
    zScale : float
        Scale factor for z-coordinates
    '''
    if ax is None:
        fig, ax = plt.subplots(1,1)

    if not diffModel.isSetup:
        diffModel.setup()

    if plotPhase is not None:
        p = diffModel._getPhaseIndex(plotPhase)
        ax.plot(diffModel.z/zScale, diffModel.p[p], *args, **kwargs)
    else:
        for p in range(len(diffModel.phases)):
            ax.plot(diffModel.z/zScale, diffModel.p[p], label=diffModel.phases[p], *args, **kwargs)
    ax.set_xlim([diffModel.zlim[0]/zScale, diffModel.zlim[1]/zScale])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Phase Fraction')

    if plotPhase is None:
        ax.legend()

    return ax