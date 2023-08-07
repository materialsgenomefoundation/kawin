from kawin.precipitation.KWNBase import PrecipitateBase
from kawin.precipitation.KWNEuler import PrecipitateModel
import numpy as np
import matplotlib.pyplot as plt

def plotModel(precModel, axes, variable, bounds = None, timeUnits = 's', radius='spherical', *args, **kwargs):
    if isinstance(precModel, PrecipitateBase):
        plotBase(precModel, axes, variable, bounds, timeUnits, radius, *args, **kwargs)
    elif isinstance(precModel, PrecipitateModel):
        plotEuler(precModel, axes, variable, bounds, timeUnits, radius, *args, **kwargs)

def plotBase(precModel: PrecipitateBase, axes, variable, bounds = None, timeUnits = 's', radius='spherical', *args, **kwargs):
    '''
    Plots model outputs
    
    Parameters
    ----------
    axes : Axis
    variable : str
        Specified variable to plot
        Options are 'Volume Fraction', 'Total Volume Fraction', 'Critical Radius',
            'Average Radius', 'Volume Average Radius', 'Total Average Radius', 
            'Total Volume Average Radius', 'Aspect Ratio', 'Total Aspect Ratio'
            'Driving Force', 'Nucleation Rate', 'Total Nucleation Rate',
            'Precipitate Density', 'Total Precipitate Density', 
            'Temperature' and 'Composition'

            Note: for multi-phase simulations, adding the word 'Total' will
                sum the variable for all phases. Without the word 'Total', the variable
                for each phase will be plotted separately
                
    bounds : tuple (optional)
        Limits on the x-axis (float, float) or None (default, this will set bounds to (initial time, final time))
    timeUnits : str (optional)
        Plot time dependent variables per seconds ('s'), minutes ('min') or hours ('h')
    radius : str (optional)
        For non-spherical precipitates, plot the Average Radius by the -
            Equivalent spherical radius ('spherical')
            Short axis ('short')
            Long axis ('long')
        Note: Total Average Radius and Volume Average Radius will still use the equivalent spherical radius
    *args, **kwargs - extra arguments for plotting
    '''
    timeScale, timeLabel, bounds = precModel.getTimeAxis(timeUnits, bounds)

    axes.set_xlabel(timeLabel)
    axes.set_xlim(bounds)

    labels = {
        'Volume Fraction': 'Volume Fraction',
        'Total Volume Fraction': 'Volume Fraction',
        'Critical Radius': 'Critical Radius (m)',
        'Average Radius': 'Average Radius (m)',
        'Volume Average Radius': 'Volume Average Radius (m)',
        'Total Average Radius': 'Average Radius (m)',
        'Total Volume Average Radius': 'Volume Average Radius (m)',
        'Aspect Ratio': 'Mean Aspect Ratio',
        'Total Aspect Ratio': 'Mean Aspect Ratio',
        'Driving Force': 'Driving Force (J/m$^3$)',
        'Nucleation Rate': 'Nucleation Rate (#/m$^3$-s)',
        'Total Nucleation Rate': 'Nucleation Rate (#/m$^3$-s)',
        'Precipitate Density': 'Precipitate Density (#/m$^3$)',
        'Total Precipitate Density': 'Precipitate Density (#/m$^3$)',
        'Temperature': 'Temperature (K)',
        'Composition': 'Matrix Composition (at.%)',
        'Eq Composition Alpha': 'Matrix Composition (at.%)',
        'Eq Composition Beta': 'Matrix Composition (at.%)',
        'Supersaturation': 'Supersaturation',
        'Eq Volume Fraction': 'Volume Fraction'
    }

    totalVariables = ['Total Volume Fraction', 'Total Average Radius', 'Total Aspect Ratio', \
                        'Total Nucleation Rate', 'Total Precipitate Density']
    singleVariables = ['Volume Fraction', 'Critical Radius', 'Average Radius', 'Aspect Ratio', \
                        'Driving Force', 'Nucleation Rate', 'Precipitate Density', 'Volume Average Radius']
    eqCompositions = ['Eq Composition Alpha', 'Eq Composition Beta']
    saturations = ['Supersaturation', 'Eq Volume Fraction']

    if variable == 'Temperature':
        plotTemperature(precModel, timeScale, labels, variable, axes, *args, **kwargs)
    elif variable == 'Composition':
        plotCompositions(precModel, timeScale, labels, variable, axes, *args, **kwargs)
    elif variable in eqCompositions:
        plotEqCompositions(precModel, timeScale, labels, variable, axes, *args, **kwargs)
    elif variable in saturations:
        plotSaurations(precModel, timeScale, labels, variable, axes, *args, **kwargs)
    elif variable in singleVariables:
        plotSingleVariables(precModel, timeScale, radius, labels, variable, axes, *args, **kwargs)
    elif variable in totalVariables:
        plotTotalVariables(precModel, timeScale, labels, variable, axes, *args, **kwargs)

def plotEuler(precModel: PrecipitateModel, axes, variable, bounds = None, timeUnits = 's', radius='spherical', *args, **kwargs):
    '''
    Plots model outputs
    
    Parameters
    ----------
    axes : Axis
    variable : str
        Specified variable to plot
        Options are 'Volume Fraction', 'Total Volume Fraction', 'Critical Radius',
            'Average Radius', 'Volume Average Radius', 'Total Average Radius', 
            'Total Volume Average Radius', 'Aspect Ratio', 'Total Aspect Ratio'
            'Driving Force', 'Nucleation Rate', 'Total Nucleation Rate',
            'Precipitate Density', 'Total Precipitate Density', 
            'Temperature', 'Composition',
            'Size Distribution', 'Size Distribution Curve',
            'Size Distribution KDE', 'Size Distribution Density
            'Interfacial Composition Alpha', 'Interfacial Composition Beta'

            Note: for multi-phase simulations, adding the word 'Total' will
                sum the variable for all phases. Without the word 'Total', the variable
                for each phase will be plotted separately

                Interfacial composition terms are more relavent for binary systems than
                for multicomponent systems
                
    bounds : tuple (optional)
        Limits on the x-axis (float, float) or None (default, this will set bounds to (initial time, final time))
    radius : str (optional)
        For non-spherical precipitates, plot the Average Radius by the -
            Equivalent spherical radius ('spherical')
            Short axis ('short')
            Long axis ('long')
        Note: Total Average Radius and Volume Average Radius will still use the equivalent spherical radius
    *args, **kwargs - extra arguments for plotting
    '''
    sizeDistributionVariables = ['Size Distribution', 'Size Distribution Curve', 'Size Distribution KDE', 'Size Distribution Density']
    compositionVariables = ['Interfacial Composition Alpha', 'Interfacial Composition Beta']

    scale = []
    for p in range(len(precModel.phases)):
        if precModel.GB[p].nucleationSiteType == precModel.GB[p].BULK or precModel.GB[p].nucleationSiteType == precModel.GB[p].DISLOCATION:
            if radius == 'spherical':
                scale.append(precModel._GBareaRemoval(p) * np.ones(len(precModel.PBM[p].PSDbounds)))
            else:
                scale.append(1/precModel.shapeFactors[p].eqRadiusFactor(precModel.PBM[p].PSDbounds))
                if radius == 'long':
                    scale.append(precModel.shapeFactors[p].aspectRatio(precModel.PBM[p].PSDbounds) / precModel.shapeFactors[p].eqRadiusFactor(precModel.PBM[p].PSDbounds))
        else:
            scale.append(precModel._GBareaRemoval(p) * np.ones(len(precModel.PBM[p].PSDbounds)))

    if variable in compositionVariables:
        plotEulerComposition(precModel, variable, axes, *args, **kwargs)
    elif variable in sizeDistributionVariables:
        plotEulerSizeDistribution(precModel, variable, axes, *args, **kwargs)
    elif variable == 'Cumulative Size Distribution':
        plotEulerCumulativeSizeDistribution(precModel, scale, variable, axes, *args, **kwargs)
    elif variable == 'Aspect Ratio Distribution':
        plotEulerAspectRatioDistribution(precModel, scale, variable, axes, *args, **kwargs) 
    else:
        plotBase(axes, variable, bounds, timeUnits, radius, *args, **kwargs)

def plotTemperature(precModel, timeScale, labels, variable, axes, *args, **kwargs):
    axes.semilogx(timeScale * precModel.time, precModel.T, *args, **kwargs)
    axes.set_ylabel(labels[variable])

def plotCompositions(precModel, timeScale, labels, variable, axes, *args, **kwargs):
    if precModel.numberOfElements == 1:
        axes.semilogx(timeScale * precModel.time, precModel.xComp, *args, **kwargs)
        axes.set_ylabel('Matrix Composition (at.% ' + precModel.elements[0] + ')')
    else:
        for i in range(precModel.numberOfElements):
            #Keep color consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
            if 'color' in kwargs:
                axes.semilogx(timeScale * precModel.time, precModel.xComp[:,i], label=precModel.elements[i], *args, **kwargs)
            else:
                axes.semilogx(timeScale * precModel.time, precModel.xComp[:,i], label=precModel.elements[i], color='C'+str(i), *args, **kwargs)
        axes.legend(precModel.elements)
        axes.set_ylabel(labels[variable])
    yRange = [np.amin(precModel.xComp), np.amax(precModel.xComp)]
    axes.set_ylim([yRange[0] - 0.1 * (yRange[1] - yRange[0]), yRange[1] + 0.1 * (yRange[1] - yRange[0])])


def plotEqCompositions(precModel, timeScale, labels, variable, axes, *args, **kwargs):
    if variable == 'Eq Composition Alpha':
        plotVariable = precModel.xEqAlpha
    elif variable == 'Eq Composition Beta':
        plotVariable = precModel.xEqBeta

    if len(precModel.phases) == 1:
        if precModel.numberOfElements == 1:
            axes.semilogx(timeScale * precModel.time, plotVariable[0], *args, **kwargs)
            axes.set_ylabel('Matrix Composition (at.% ' + precModel.elements[0] + ')')
        else:
            for i in range(precModel.numberOfElements):
                #Keep color consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                if 'color' in kwargs:
                    axes.semilogx(timeScale * precModel.time, plotVariable[0,:,i], label=precModel.elements[i]+'_Eq', *args, **kwargs)
                else:
                    axes.semilogx(timeScale * precModel.time, plotVariable[0,:,i], label=precModel.elements[i]+'_Eq', color='C'+str(i), *args, **kwargs)
            axes.legend()
            axes.set_ylabel(labels[variable])
    else:
        if precModel.numberOfElements == 1:
            for p in range(len(precModel.phases)):
                #Keep color somewhat consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                if 'color' in kwargs:
                    axes.semilogx(timeScale * precModel.time, plotVariable[p], label=precModel.phases[p]+'_Eq', *args, **kwargs)
                else:
                    axes.semilogx(timeScale * precModel.time, plotVariable[p], label=precModel.phases[p]+'_Eq', color='C'+str(p), *args, **kwargs)
            axes.legend()
            axes.set_ylabel('Matrix Composition (at.% ' + precModel.elements[0] + ')')
        else:
            cIndex = 0
            for p in range(len(precModel.phases)):
                for i in range(precModel.numberOfElements):
                    #Keep color somewhat consistent between Composition, Eq Composition Alpha and Eq Composition Beta if color isn't passed as an arguement
                    if 'color' in kwargs:
                        axes.semilogx(timeScale * precModel.time, plotVariable[p,:,i], label=precModel.phases[p]+'_'+precModel.elements[i]+'_Eq', *args, **kwargs)
                    else:
                        axes.semilogx(timeScale * precModel.time, plotVariable[p,:,i], label=precModel.phases[p]+'_'+precModel.elements[i]+'_Eq', color='C'+str(cIndex), *args, **kwargs)
                    cIndex += 1
            axes.legend()
            axes.set_ylabel(labels[variable])

def plotSaurations(precModel, timeScale, labels, variable, axes, *args, **kwargs):
    #Since supersaturation is calculated in respect to the tie-line, it is the same for each element
    #Thus only a single element is needed
    plotVariable = np.zeros(precModel.betaFrac.shape)
    for p in range(len(precModel.phases)):
        if precModel.numberOfElements == 1:
            if variable == 'Eq Volume Fraction':
                num = precModel.xComp[0] - precModel.xEqAlpha[p]
            else:
                num = precModel.xComp - precModel.xEqAlpha[p]
            den = precModel.xEqBeta[p] - precModel.xEqAlpha[p]
        else:
            if variable == 'Eq Volume Fraction':
                num = precModel.xComp[0,0] - precModel.xEqAlpha[p,:,0]
            else:
                num = precModel.xComp[:,0] - precModel.xEqAlpha[p,:,0]
            den = precModel.xEqBeta[p,:,0] - precModel.xEqAlpha[p,:,0]
        #If precipitate is unstable, both xEqAlpha and xEqBeta are set to 0
        #For these cases, change the values of numerator and denominator so that supersaturation is 0 instead of undefined
        num[den == 0] = 0
        den[den == 0] = 1
        plotVariable[p] = num / den
    
    if len(precModel.phases) == 1:
        axes.semilogx(timeScale * precModel.time, plotVariable[0], *args, **kwargs)
    else:
        for p in range(len(precModel.phases)):
            if 'color' in kwargs:
                axes.semilogx(timeScale * precModel.time, plotVariable[p], label=precModel.phases[p], *args, **kwargs)
            else:
                axes.semilogx(timeScale * precModel.time, plotVariable[p], label=precModel.phases[p], color='C'+str(p), *args, **kwargs)
        axes.legend()
    axes.set_ylabel(labels[variable])

def plotSingleVariables(precModel, timeScale, radius, labels, variable, axes, *args, **kwargs):
    if variable == 'Volume Fraction':
        plotVariable = precModel.betaFrac
    elif variable == 'Critical Radius':
        plotVariable = precModel.Rcrit
    elif variable == 'Average Radius':
        plotVariable = precModel.avgR
        for p in range(len(precModel.phases)):
            if precModel.GB[p].nucleationSiteType == precModel.GB[p].BULK or precModel.GB[p].nucleationSiteType == precModel.GB[p].DISLOCATION:
                if radius != 'spherical':
                    plotVariable[p] /= precModel.shapeFactors[p].eqRadiusFactor(precModel.avgR[p])
                if radius == 'long':
                    plotVariable[p] *= precModel.avgAR[p]
            else:
                plotVariable[p] *= precModel._GBareaRemoval(p)
    elif variable == 'Volume Average Radius':
        plotVariable = np.zeros(precModel.betaFrac.shape)
        indices = precModel.precipitateDensity > 0
        plotVariable[indices] = np.cbrt(precModel.betaFrac[indices] / precModel.precipitateDensity[indices] / (4/3*np.pi))
    elif variable == 'Aspect Ratio':
        plotVariable = precModel.avgAR
    elif variable == 'Driving Force':
        plotVariable = precModel.dGs
    elif variable == 'Nucleation Rate':
        plotVariable = precModel.nucRate
    elif variable == 'Precipitate Density':
        plotVariable = precModel.precipitateDensity

    if (len(precModel.phases)) == 1:
        axes.semilogx(timeScale * precModel.time, plotVariable[0], *args, **kwargs)
    else:
        for p in range(len(precModel.phases)):
            axes.semilogx(timeScale * precModel.time, plotVariable[p], label=precModel.phases[p], color='C'+str(p), *args, **kwargs)
        axes.legend()
    axes.set_ylabel(labels[variable])
    yb = 1 if variable == 'Aspect Ratio' else 0
    axes.set_ylim([yb, 1.1 * np.amax(plotVariable)])

def plotTotalVariables(precModel, timeScale, labels, variable, axes, *args, **kwargs):
    if variable == 'Total Volume Fraction':
        plotVariable = np.sum(precModel.betaFrac, axis=0)
    elif variable == 'Total Average Radius':
        totalN = np.sum(precModel.precipitateDensity, axis=0)
        totalN[totalN == 0] = 1
        totalR = np.sum(precModel.avgR * precModel.precipitateDensity, axis=0)
        plotVariable = totalR / totalN
    elif variable == 'Total Volume Average Radius':
        totalN = np.sum(precModel.precipitateDensity, axis=0)
        totalN[totalN == 0] = 1
        totalVol = np.sum(precModel.betaFrac, axis=0)
        plotVariable = np.cbrt(totalVol / totalN)
    elif variable == 'Total Aspect Ratio':
        totalN = np.sum(precModel.precipitateDensity, axis=0)
        totalN[totalN == 0] = 1
        totalAR = np.sum(precModel.avgAR * precModel.precipitateDensity, axis=0)
        plotVariable = totalAR / totalN
    elif variable == 'Total Nucleation Rate':
        plotVariable = np.sum(precModel.nucRate, axis=0)
    elif variable == 'Total Precipitate Density':
        plotVariable = np.sum(precModel.precipitateDensity, axis=0)

    axes.semilogx(timeScale * precModel.time, plotVariable, *args, **kwargs)
    axes.set_ylabel(labels[variable])
    yb = 1 if variable == 'Total Aspect Ratio' else 0
    axes.set_ylim(bottom=yb)

def plotEulerComposition(precModel, variable, axes, *args, **kwargs):
    if variable == 'Interfacial Composition Alpha':
        yVar = precModel.PSDXalpha
        ylabel = 'Composition in Alpha phase'
    else:
        yVar = precModel.PSDXbeta
        ylabel = 'Composition in Beta Phase'

    if (len(precModel.phases)) == 1:
        axes.semilogx(precModel.PBM[0].PSDbounds, yVar[0], *args, **kwargs)
    else:
        for p in range(len(precModel.phases)):
            axes.plot(precModel.PBM[p].PSDbounds, yVar[p], label=precModel.phases[p], *args, **kwargs)
        axes.legend()
    axes.set_xlim([precModel.PBM[0].PSDbounds[0], precModel.PBM[0].PSDbounds[-1]])
    axes.set_xlabel('Radius (m)')
    axes.set_ylabel(ylabel)

def plotEulerSizeDistribution(precModel, scale, variable, axes, *args, **kwargs):
    ylabel = 'Frequency (#/$m^3$)'
    if variable == 'Size Distribution':
        functionName = 'PlotHistogram'
    elif variable == 'Size Distribution KDE':
        functionName = 'PlotKDE'
    elif variable == 'Size Distribution Density':
        functionName = 'PlotDistributionDensity'
        ylabel = 'Distribution Density (#/$m^4$)'
    else:
        functionName = 'PlotCurve'

    if len(precModel.phases) == 1:
        getattr(precModel.PBM[0], functionName)(axes, scale=scale[0], *args, **kwargs)
    else:
        for p in range(len(precModel.phases)):
            getattr(precModel.PBM[p], functionName)(axes, label=precModel.phases[p], scale=scale[p], *args, **kwargs)
        axes.legend()
    axes.set_xlabel('Radius (m)')
    axes.set_ylabel(ylabel)
    axes.set_xlim([0, np.amax([pb.max for pb in precModel.PBM])])
    if variable == 'Size Distribution Density':
        axes.set_ylim([0, 1.1*np.amax(np.concatenate(([np.amax(pb.PSD/(pb.PSDbounds[1:] - pb.PSDbounds[:-1])) for pb in precModel.PBM], [1])))])
    else:
        axes.set_ylim([0, 1.1*np.amax(np.concatenate(([np.amax(pb.PSD) for pb in precModel.PBM], [1])))])

def plotEulerCumulativeSizeDistribution(precModel, scale, variable, axes, *args, **kwargs):
    ylabel = 'CDF'
    if len(precModel.phases) == 1:
        precModel.PBM[0].PlotCDF(axes, scale=scale[0], *args, **kwargs)
    else:
        for p in range(len(precModel.phases)):
            precModel.PBM[p].PlotCDF(axes, label=precModel.phases[p], scale=scale[p], *args, **kwargs)
        axes.legend()
    axes.set_xlabel('Radius (m)')
    axes.set_ylabel(ylabel)
    axes.set_xlim([0, np.amax([pb.max for pb in precModel.PBM])])

def plotEulerAspectRatioDistribution(precModel, scale, variable, axes, *args, **kwargs):
    if len(precModel.phases) == 1:
        axes.plot(precModel.PBM[0].PSDbounds * np.interp(precModel.PBM[p].PSDbounds, precModel.PBM[0].PSDbounds, scale[0]), self.eqAspectRatio[0], *args, **kwargs)
    else:
        for p in range(len(precModel.phases)):
            axes.plot(precModel.PBM[p].PSDbounds * np.interp(precModel.PBM[p].PSDbounds, precModel.PBM[p].PSDbounds, scale[p]), self.eqAspectRatio[p], label=self.phases[p], *args, **kwargs)
        axes.legend()
    axes.set_xlim([0, np.amax(precModel.PBM[p].PSDbounds * np.interp(precModel.PBM[p].PSDbounds, self.PBM[p].PSDbounds, scale[p]))])
    axes.set_ylim(bottom=1)
    axes.set_xlabel('Radius (m)')
    axes.set_ylabel('Aspect ratio distribution')

