import numpy as np
import matplotlib.pyplot as plt

from kawin.PlotUtils import _get_axis, _adjust_kwargs
from kawin.precipitation.PrecipitationParameters import PrecipitateParameters
from kawin.precipitation.PopulationBalance import PopulationBalanceModel, plotPDF, plotPSD, plotCDF
from kawin.precipitation import PrecipitateBase, PrecipitateModel

def _get_time_axis(time, timeUnits='s', bounds=None):
        '''
        Returns scaling factor, label and x-limits depending on units of time

        Parameters
        ----------
        timeUnits : str
            's' / 'sec' / 'seconds' - seconds
            'min' / 'minutes' - minutes
            'h' / 'hrs' / 'hours' - hours
        '''
        timeScale = 1
        timeLabel = 'Time (s)'
        if 'min' in timeUnits:
            timeScale = 1/60
            timeLabel = 'Time (min)'
        if 'h' in timeUnits:
            timeScale = 1/3600
            timeLabel = 'Time (hrs)'

        if bounds is None:
            bounds = [timeScale*1e-5*time[-1], timeScale * time[-1]]

        return timeScale, timeLabel, bounds

def _total_sum(model: PrecipitateBase, plotVar):
    '''
    Total var is sum along phases
    '''
    return np.sum(plotVar, axis=1)

def _total_average(model: PrecipitateBase, plotVar):
    '''
    Total var is average along phases
    '''
    totalN = np.sum(model.data.precipitateDensity, axis=1)
    totalN[totalN==0] = 1
    totalV = np.sum(plotVar*model.data.precipitateDensity, axis=1)
    return totalV/totalN

def _total_vol_average(model: PrecipitateBase, plotVar):
    '''
    This is specific for volume average radius, where R^3 = fv / (4/3*pi*N)
    '''
    totalN = np.sum(model.data.precipitateDensity, axis=1)
    totalV = np.sum(model.data.volFrac, axis=1)
    indices = totalN > 0
    volAvg = np.zeros(totalV.shape)
    volAvg[indices] = np.cbrt(totalV[indices] / totalN[indices] / (4/3*np.pi))
    return volAvg

def _total_none(model: PrecipitateBase, plotVar):
    '''
    For variables that done have a defined total (ex. driving force, supersaturation, eq volume fraction)
    '''
    return None

def _get_plot_list(currList, defaultList):
    if currList is None:
        currList = defaultList
    if isinstance(currList, str):
        currList = [currList]
    return currList

def _get_ys_phases(model: PrecipitateBase, plotVar, phases=None, totalFunc=_total_sum):
    '''
    Given the plot variable and list of phases (which may include 'total')
    return a list of y's and corresponding labels to plot
    '''
    phases = _get_plot_list(phases, model.phases)
    ys, labels = [], []
    for p in phases:
        if p.upper() == 'TOTAL':
            totalVar = totalFunc(model, plotVar)
            if totalVar is not None:
                ys.append(totalVar)
                labels.append(p)
        else:
            ys.append(plotVar[:,list(model.phases).index(p)])
            labels.append(p)
    return ys, labels

def _get_ys_elements(model: PrecipitateBase, plotVar, elements=None):
    '''
    Given the plot variable and list of elements
    return a list of y's and corresponding labels to plot
    '''
    elements = _get_plot_list(elements, model.elements)
    ys = []
    for e in elements:
        ys.append(plotVar[:,list(model.elements).index(e)])
    return ys, elements

def _radius_scale(precipitate: PrecipitateParameters, radius, r):
    radius = radius.lower()
    if radius not in ['spherical', 'short', 'long']:
        raise ValueError("Radius must be \'spherical\', \'short\' or \'long\'")
    
    scale = 1
    if precipitate.nucleation.isGrainBoundaryNucleation:
        scale = precipitate.nucleation.areaRemoval
    else:
        if radius != 'spherical':
            scale = 1 / precipitate.shapeFactor.eqRadiusFactor(r)
            if radius == 'long':
                scale *= precipitate.shapeFactor.aspectRatio(r)
    return scale

def _plot_term(model: PrecipitateBase, ys, labels, yLabel, timeUnits='s', ax=None, yBottom=0, *args, **kwargs):
    '''
    Plot a variable from a precipitate model
    This will scale the x-axis to the desired time unit and apply labels when possible
    '''
    ax = _get_axis(ax)
    time = model.data.time
    timeScale, timeLabel, bounds = _get_time_axis(time, timeUnits)
    for label, y in zip(labels, ys):
        plot_kwargs = _adjust_kwargs(label, {'label': label}, kwargs)
        ax.semilogx(timeScale*time, y, *args, **plot_kwargs)
    if len(labels) > 1:
        ax.legend()

    ax.set_xlabel(timeLabel)
    ax.set_xlim(bounds)
    ax.set_ylabel(yLabel)
    if yBottom is not None:
        ax.set_ylim(bottom=yBottom)
    return ax

def plotVolumeFraction(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots volume fraction vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then sum of f_v will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, phases = _get_ys_phases(model, model.data.volFrac, phases, totalFunc=_total_sum)
    return _plot_term(model, ys, phases, r'Volume Fraction', timeUnits, ax=ax, yBottom=0, *args, **kwargs)

def plotCriticalRadius(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots critical radius vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then average of R_crit will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, phases = _get_ys_phases(model, model.data.Rcrit, phases, totalFunc=_total_average)
    return _plot_term(model, ys, phases, r'Critical Radius ($m$)', timeUnits, ax=ax, yBottom=0, *args, **kwargs)

def plotAverageRadius(model: PrecipitateBase, timeUnits='s', phases=None, radius='spherical', ax = None, *args, **kwargs):
    '''
    Plots average radius vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then average of R_avg will be used
    radius: str (optional)
        For non-spherical precipitates, the average radius can be transformed to:
            'spherical' - equivalent spherical radius (default)
            'short' - short axis
            'long' - long axis
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    # Convert average radius to equivalent spherical, short axis or long axis
    plotVar = model.data.Ravg
    for p in range(len(model.phases)):
        plotVar[:,p] *= _radius_scale(model.precipitates[p], radius, plotVar[:,p])

    ys, phases = _get_ys_phases(model, plotVar, phases, totalFunc=_total_average)
    return _plot_term(model, ys, phases, r'Average Radius ($m$)', timeUnits, ax=ax, yBottom=0, *args, **kwargs)

def plotVolumeAverageRadius(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots volume averaged radius vs time on a semilog plot
        R_v = cbrt(f_v / (4/3 pi N_v)

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then average will be cbrt(sum(f_v) / sum(N_v) / (4/3 pi))
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    plotVariable = np.zeros(model.data.volFrac.shape)
    indices = model.data.precipitateDensity > 0
    plotVariable[indices] = np.cbrt(model.data.volFrac[indices] / model.data.precipitateDensity[indices] / (4/3*np.pi))
    ys, phases = _get_ys_phases(model, plotVariable, phases, totalFunc=_total_vol_average)
    return _plot_term(model, ys, phases, r'Volume Average Radius ($m$)', timeUnits, ax=ax, yBottom=0, *args, **kwargs)

def plotAspectRatio(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots precipitate aspect ratio vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then average of AR will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, phases = _get_ys_phases(model, model.data.ARavg, phases, totalFunc=_total_average)
    return _plot_term(model, ys, phases, r'Aspect Ratio', timeUnits, ax=ax, yBottom=1, *args, **kwargs)

def plotDrivingForce(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots driving force vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        'total' is not supported for driving forces
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, phases = _get_ys_phases(model, model.data.drivingForce, phases, totalFunc=_total_none)
    return _plot_term(model, ys, phases, r'Driving Force ($J/m^3$)', timeUnits, ax=ax, yBottom=None)

def plotNucleationRate(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots nucleation rate vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then sum of dN/dt will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, phases = _get_ys_phases(model, model.data.nucRate, phases, totalFunc=_total_sum)
    return _plot_term(model, ys, phases, r'Nucleation Rate (#$/m^3-s$)', timeUnits, ax=ax, yBottom=0, *args, **kwargs)

def plotPrecipitateDensity(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots precipitate density vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
        If 'total' is in phase list, then sum of N will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, phases = _get_ys_phases(model, model.data.precipitateDensity, phases, totalFunc=_total_sum)
    return _plot_term(model, ys, phases, r'Precipitate Density (#$/m^3$)', timeUnits, ax=ax, yBottom=0, *args, **kwargs)

def plotTemperature(model: PrecipitateBase, timeUnits='s', ax = None, *args, **kwargs):
    '''
    Plots temperature vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    return _plot_term(model, [model.data.temperature], ['Temperature'], 'Temperature (K)', timeUnits, ax=ax, *args, **kwargs)

def plotComposition(model: PrecipitateBase, timeUnits='s', elements=None, ax = None, *args, **kwargs):
    '''
    Plots composition vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    elements: str | list[str] (optional)
        If None, then model elements will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ys, elements = _get_ys_elements(model, model.data.composition, elements)
    return _plot_term(model, ys, elements, 'Composition (at.)', timeUnits, ax=ax, *args, **kwargs)

def plotEqMatrixComposition(model: PrecipitateBase, timeUnits='s', elements=None, phase=None, ax = None, *args, **kwargs):
    '''
    Plots equilibrium matrix composition vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    elements: str | list[str] (optional)
        If None, then model elements will be used
    phase: str (optional)
        Precipitate phase that matrix is in equilibrium with
        If None, first phase will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    index = model.phaseIndex(phase)
    ys, elements = _get_ys_elements(model, model.data.xEqAlpha[:,index,:], elements)
    return _plot_term(model, ys, elements, r'Eq. Composition $\alpha$ (at.)', timeUnits, ax=ax, *args, **kwargs)

def plotEqPrecipitateComposition(model: PrecipitateBase, timeUnits='s', elements=None, phase=None, ax = None, *args, **kwargs):
    '''
    Plots equilibrium precipitate composition vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    elements: str | list[str] (optional)
        If None, then model elements will be used
    phase: str (optional)
        Precipitate phase that matrix is in equilibrium with
        If None, first phase will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    index = model.phaseIndex(phase)
    ys, elements = _get_ys_elements(model, model.data.xEqBeta[:,index,:], elements)
    return _plot_term(model, ys, elements, r'Eq. Composition $\beta$ (at.)', timeUnits, ax=ax, *args, **kwargs)

def plotEqVolumeFraction(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots equilibrium volume fraction vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    #Since supersaturation is calculated in respect to the tie-line, it is the same for each element
    #Thus only a single element is needed
    num = model.data.composition[0,0] - model.data.xEqAlpha[:,:,0]
    den = model.data.xEqBeta[:,:,0] - model.data.xEqAlpha[:,:,0]
    num[den == 0] = 0
    den[den == 0] = 1
    plotVariable = num / den
    ys, phases = _get_ys_phases(model, plotVariable, phases, totalFunc=_total_none)
    return _plot_term(model, ys, phases, 'Eq. Volume Fraction', timeUnits, ax=ax, *args, **kwargs)

def plotSupersaturation(model: PrecipitateBase, timeUnits='s', phases=None, ax = None, *args, **kwargs):
    '''
    Plots supersaturation vs time on a semilog plot

    Parameters
    ----------
    model: PrecipitateBase
    timeUnits: str
        ['s', 'min', 'hr']
    phases: str | list[str] (optional)
        If None, then model phases will be used
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    #Since supersaturation is calculated in respect to the tie-line, it is the same for each element
    #Thus only a single element is needed
    num = model.data.composition[:,0][:,np.newaxis] - model.data.xEqAlpha[:,:,0]
    den = model.data.xEqBeta[:,:,0] - model.data.xEqAlpha[:,:,0]
    num[den == 0] = 0
    den[den == 0] = 1
    plotVariable = num / den
    ys, phases = _get_ys_phases(model, plotVariable, phases, totalFunc=_total_none)
    return _plot_term(model, ys, phases, 'Eq. Volume Fraction', timeUnits, ax=ax, *args, **kwargs)

def plotSizeDistribution(model: PrecipitateModel, phases=None, radius='spherical', fill=False, ax=None, *args, **kwargs):
    '''
    Plots final particle size distribution (N vs r)

    Parameters
    ----------
    model: PrecipitateBase
    phases: str | list[str] (optional)
        If None, then model phases will be used
        'total' not supported for equilibrium volume fraction
    radius: str (optional)
        For non-spherical precipitates, the average radius can be transformed to:
            'spherical' - equivalent spherical radius (default)
            'short' - short axis
            'long' - long axis
    fill: bool (optional)
        If True, will fill between the PSD curve and y=0
        Defaults to False
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ax = _get_axis(ax)
    phases = _get_plot_list(phases, model.phases)
    for p in phases:
        pbm = model.getPBM(p)
        scale = _radius_scale(model.precipitates[model.phaseIndex(p)], radius, pbm.PSDsize)
        plotPSD(pbm, scale=scale, fill=fill, ax=ax, *args, **kwargs)
    return ax

def plotDistributionDensity(model: PrecipitateModel, phases=None, radius='spherical', fill=False, ax=None, *args, **kwargs):
    '''
    Plots final particle size distribution density (N/R) vs R

    Parameters
    ----------
    model: PrecipitateBase
    phases: str | list[str] (optional)
        If None, then model phases will be used
        'total' not supported for equilibrium volume fraction
    radius: str (optional)
        For non-spherical precipitates, the average radius can be transformed to:
            'spherical' - equivalent spherical radius (default)
            'short' - short axis
            'long' - long axis
    fill: bool (optional)
        If True, will fill between the PSD curve and y=0
        Defaults to False
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ax = _get_axis(ax)
    phases = _get_plot_list(phases, model.phases)
    for p in phases:
        pbm = model.getPBM(p)
        scale = _radius_scale(model.precipitates[model.phaseIndex(p)], radius, pbm.PSDsize)
        plotPDF(pbm, scale=scale, fill=fill, ax=ax, *args, **kwargs)
    return ax

def plotCumulativeDistribution(model: PrecipitateModel, phases=None, radius='spherical', order=1, ax=None, *args, **kwargs):
    '''
    Plots cumulative density functio of particle size distribution

    Parameters
    ----------
    model: PrecipitateBase
    phases: str | list[str] (optional)
        If None, then model phases will be used
        'total' not supported for equilibrium volume fraction
    radius: str (optional)
        For non-spherical precipitates, the average radius can be transformed to:
            'spherical' - equivalent spherical radius (default)
            'short' - short axis
            'long' - long axis
    order: int (optional)
        Moment of PSD to plot CDF in
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    Returns
    -------
    Matplotlib Axis
    '''
    ax = _get_axis(ax)
    phases = _get_plot_list(phases, model.phases)
    for p in phases:
        pbm = model.getPBM(p)
        scale = _radius_scale(model.precipitates[model.phaseIndex(p)], radius, pbm.PSDsize)
        plotCDF(pbm, scale=scale, order=order, ax=ax, *args, **kwargs)
    return ax

def plotPrecipitateResults(model: PrecipitateModel, term, ax=None, *args, **kwargs):
    '''
    General function to plot precipitate results

    Parameters
    ----------
    model: PrecipitateBase
    term: str
        Term to plot. Options are:
            'volume fraction'
            'critical radius'
            'volume average radius' - average radius given volume fraction and precipitate density
            'aspect ratio' - average aspect ratio vs time
            'driving force'
            'nucleation rate'
            'precipitate density'
            'temperature'
            'composition'
            'eq comp alpha' - equilibrium composition of matrix phase
            'eq comp beta' - equilibrium composition of precipitate phase
            'supersaturation'
            'eq volume fraction' - equilibrium volume fraction of precipitate phase
            'psd' - particle size distribution
            'pdf' - particle size distribution density
            'cdf' - particle cumulative size distribution
    ax: matplotlib Axis (optional)
        If None, then ax will be generated

    '''
    plotFunctions = {
        'volume fraction': plotVolumeFraction,
        'critical radius': plotCriticalRadius,
        'average radius': plotAverageRadius,
        'volume average radius': plotVolumeAverageRadius,
        'aspect ratio': plotAspectRatio,
        'driving force': plotDrivingForce,
        'nucleation rate': plotNucleationRate,
        'precipitate density': plotPrecipitateDensity,
        'temperature': plotTemperature,
        'composition': plotComposition,
        'eq comp alpha': plotEqMatrixComposition,
        'eq comp beta': plotEqPrecipitateComposition,
        'supersaturation': plotSupersaturation,
        'eq volume fraction': plotEqVolumeFraction,
        'psd': plotSizeDistribution,
        'pdf': plotDistributionDensity,
        'cdf': plotCumulativeDistribution
    }
    if term.lower() not in plotFunctions:
        functionList = ', '.join(list(plotFunctions.keys()))
        raise ValueError(f'term must be one of the following: [{functionList}]')
    return plotFunctions[term.lower()](model, ax=ax, *args, **kwargs)