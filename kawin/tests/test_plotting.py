from kawin.precipitation import PrecipitateModel
from kawin.diffusion.Diffusion import DiffusionModel
import matplotlib.pyplot as plt
import numpy as np

def test_precipitate_plotting():
    binary_single = PrecipitateModel(phases=['beta'], elements=['A'])
    binary_multi = PrecipitateModel(phases=['beta', 'gamma', 'zeta'], elements=['A'])
    ternary_single = PrecipitateModel(phases=['beta'], elements=['A', 'B'])
    ternary_multi = PrecipitateModel(phases=['beta', 'gamma', 'zeta'], elements=['A', 'B'])

    models = [
        (binary_single, 1, 1),
        (binary_multi, 1, 3),
        (ternary_single, 2, 1),
        (ternary_multi, 2, 3),
    ]

    varTypes = [
        ('Volume Fraction', [2]),
        ('Total Volume Fraction', None),
        ('Critical Radius', [2]),
        ('Average Radius', [2]),
        ('Volume Average Radius', [2]),
        ('Total Average Radius', None),
        ('Total Volume Average Radius', None),
        ('Aspect Ratio', [2]),
        ('Total Aspect Ratio', None),
        ('Driving Force', [2]),
        ('Nucleation Rate', [2]),
        ('Total Nucleation Rate', None),
        ('Precipitate Density', [2]),
        ('Total Precipitate Density', None),
        ('Temperature', None),
        ('Composition', [1]),
        ('Eq Composition Alpha', [1,2]),
        ('Eq Composition Beta', [1,2]),
        ('Supersaturation', [2]),
        ('Eq Volume Fraction', [2]),
        ('Size Distribution', [2]),
        ('Size Distribution Curve', [2]),
        ('Size Distribution KDE', [2]),
        ('Size Distribution Density', [2]),
    ]

    for m in models:
        for v in varTypes:
            fig, ax = plt.subplots(1,1)
            m[0].plot(ax, v[0])
            numLines = len(ax.lines)
            plt.close(fig)

            #Check that the number of lines on the plot correspond to the right amount
            #   Number of lines should either be 1, elements, phases or elements*phases depending on variable
            desiredNumber = 1
            if v[1] is not None:
                desiredNumber = np.prod([m[vi] for vi in v[1]], dtype=np.int32)
            assert numLines == desiredNumber

def test_diffusion_plotting():
    #Single phase and Homogenizaton model goes through the same path for plotting
    binary_single = DiffusionModel(zlim=[-1,1], N=100, elements=['A', 'B'], phases=['alpha'])
    binary_multi = DiffusionModel(zlim=[-1,1], N=100, elements=['A', 'B'], phases=['alpha', 'beta', 'gamma'])
    ternary_single = DiffusionModel(zlim=[-1,1], N=100, elements=['A', 'B', 'C'], phases=['alpha'])
    ternary_multi = DiffusionModel(zlim=[-1,1], N=100, elements=['A', 'B', 'C'], phases=['alpha', 'beta', 'gamma'])

    models = [
        (binary_single, 2, 1),
        (binary_multi, 2, 3),
        (ternary_single, 3, 1),
        (ternary_multi, 3, 3),
    ]

    for m in models:
        #m[0].setTemperature(900)
        m[0].parameters.temperature.set_isothermal_temperature(900)

        #For each plot, check that the number of lines correspond to number of elements or phases
        #For 'plot', number of lines should be elements (with or without reference) or a single element
        #For 'plotTwoAxis', number of lines for each axis should be length of input array
        #For 'plotPhases', number of lines is number of phases or single phase
        fig, ax = plt.subplots(1,1)
        m[0].plot(ax, plotReference = False)
        assert len(ax.lines) == m[1]-1
        plt.close(fig)

        fig, ax = plt.subplots(1,1)
        m[0].plot(ax, plotReference = True)
        assert len(ax.lines) == m[1]
        plt.close(fig)

        fig, ax = plt.subplots(1,1)
        m[0].plot(ax, plotElement = m[0].allElements[0])
        assert len(ax.lines) == 1
        plt.close(fig)

        fig, ax = plt.subplots(1,1)
        m[0].plot(ax, plotElement = m[0].allElements[1])
        assert len(ax.lines) == 1
        plt.close(fig)


        fig, axL = plt.subplots(1,1)
        axR = ax.twinx()
        m[0].plotTwoAxis(Lelements=[m[0].allElements[0]], Relements = m[0].allElements[1:], axL=axL, axR=axR)
        assert len(axL.lines) == 1
        assert len(axR.lines) == len(m[0].allElements)-1
        plt.close(fig)

        # This requires thermodynamics to compute phases, commenting out for now
        # fig, ax = plt.subplots(1,1)
        # m[0].plotPhases(ax)
        # assert len(ax.lines) == m[2]
        # plt.close(fig)

        # fig, ax = plt.subplots(1,1)
        # m[0].plotPhases(ax, plotPhase=m[0].phases[0])
        # assert len(ax.lines) == 1
        # plt.close(fig)