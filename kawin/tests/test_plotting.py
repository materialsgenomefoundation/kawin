import numpy as np
import matplotlib.pyplot as plt

from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics, GeneralThermodynamics
from kawin.precipitation import PrecipitateModel, MatrixParameters, PrecipitateParameters, TemperatureParameters as PrecTemp
from kawin.diffusion import SinglePhaseModel, TemperatureParameters as DiffTemp
from kawin.diffusion.mesh import Cartesian1D, StepProfile1D, ProfileBuilder

from kawin.precipitation.Plot import plotEuler
from kawin.diffusion.Plot import plot1D, plot1DFlux, plot1DPhases, plot1DTwoAxis

from kawin.tests.datasets import NICRAL_TDB

binPrecTherm = BinaryThermodynamics(NICRAL_TDB, ['NI', 'AL'], ['FCC_A1', 'FCC_L12', 'C14_LAVES', 'C15_LAVES'], drivingForceMethod='tangent')
ternPrecTherm = MulticomponentThermodynamics(NICRAL_TDB, ['NI', 'AL', 'CR'], ['FCC_A1', 'FCC_L12', 'C14_LAVES', 'C15_LAVES'], drivingForceMethod='tangent')

binDiffTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'])
ternDiffTherm = GeneralThermodynamics(NICRAL_TDB, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'])

def test_precipitate_plotting():
    binary_matrix = MatrixParameters(['AL'])
    ternary_matrix = MatrixParameters(['AL', 'CR'])

    fcc_prec = PrecipitateParameters('FCC_L12')
    fcc_prec.gamma = 0.1

    c14_prec = PrecipitateParameters('C14_LAVES')
    c14_prec.gamma = 0.1

    c15_prec = PrecipitateParameters('C15_LAVES')
    c15_prec.gamma = 0.1

    temperature = PrecTemp(500)
    binary_single = PrecipitateModel(binary_matrix, [fcc_prec], binPrecTherm, temperature)
    binary_multi = PrecipitateModel(binary_matrix, [fcc_prec, c14_prec, c15_prec], binPrecTherm, temperature)
    ternary_single = PrecipitateModel(ternary_matrix, [fcc_prec], ternPrecTherm, temperature)
    ternary_multi = PrecipitateModel(ternary_matrix, [fcc_prec, c14_prec, c15_prec], ternPrecTherm, temperature)

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
            plotEuler(m[0], ax, v[0])
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
    profile_binary = ProfileBuilder([(StepProfile1D(0.5, 0.1, 0.9), 'CR')])
    mesh_binary = Cartesian1D(['CR'], [-1,1], 100)
    mesh_binary.setResponseProfile(profile_binary)

    profile_ternary = ProfileBuilder([(StepProfile1D(0.5, [0.1,0.2], [0.9,0.01]), ['CR', 'AL'])])
    mesh_ternary = Cartesian1D(['CR', 'AL'], [-1,1], 100)
    mesh_ternary.setResponseProfile(profile_ternary)

    temperature = DiffTemp(1000)
    binary_single = SinglePhaseModel(mesh_binary, ['NI', 'CR'], ['FCC_A1'], binDiffTherm, temperature)
    binary_multi = SinglePhaseModel(mesh_binary, ['NI', 'CR'], ['FCC_A1', 'BCC_A2'], binDiffTherm, temperature)
    ternary_single = SinglePhaseModel(mesh_ternary, ['NI', 'CR', 'AL'], ['FCC_A1'], ternDiffTherm, temperature)
    ternary_multi = SinglePhaseModel(mesh_ternary, ['NI', 'CR', 'AL'], ['FCC_A1', 'BCC_A2'], ternDiffTherm, temperature)

    models = [
        (binary_single, 2, 1),
        (binary_multi, 2, 2),
        (ternary_single, 3, 1),
        (ternary_multi, 3, 2),
    ]

    for m in models:
        #For each plot, check that the number of lines correspond to number of elements or phases
        #For 'plot', number of lines should be elements (with or without reference) or a single element
        #For 'plotTwoAxis', number of lines for each axis should be length of input array
        #For 'plotPhases', number of lines is number of phases or single phase
        fig, ax = plt.subplots()
        plot1D(m[0], elements=m[0].allElements, ax=ax)
        assert len(ax.lines) == m[1]
        plt.close(fig)

        fig, ax = plt.subplots()
        plot1D(m[0], elements=None, ax=ax)
        assert len(ax.lines) == m[1]-1
        plt.close(fig)

        fig, ax = plt.subplots()
        plot1D(m[0], elements=m[0].allElements[0], ax=ax)
        assert len(ax.lines) == 1
        plt.close(fig)

        fig, axL = plt.subplots()
        axR = ax.twinx()
        plot1DTwoAxis(m[0], m[0].allElements[0], m[0].allElements[1:], axL=axL, axR=axR)
        assert len(axL.lines) == 1
        assert len(axR.lines) == len(m[0].allElements)-1
        plt.close(fig)

        # This requires thermodynamics to compute, commenting out for now
        fig, ax = plt.subplots()
        plot1DPhases(m[0], phases=None, ax=ax)
        assert len(ax.lines) == m[2]
        plt.close(fig)
        
        fig, ax = plt.subplots()
        plot1DFlux(m[0], elements=m[0].elements, ax=ax)
        assert len(ax.lines) == m[1]-1