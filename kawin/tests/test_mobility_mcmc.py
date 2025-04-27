import numpy as np

from pycalphad import Database
from espei.utils import PickleableTinyDB, MemoryStorage
from espei.phase_models import PhaseModelSpecification
from espei.error_functions.context import setup_context

from kawin.mobility_fitting.error_functions import NonEquilibriumMobilityResidual, EquilibriumMobilityResidual
from kawin.tests.databases import *
from kawin.tests.datasets import *

def test_non_eq_tracer_D0():
    phase_models = {
        "components": ["AL", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_tracer_D0_Al)
    dbf = Database(ALMGSI_DB)
    residual_func = NonEquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [-0.1], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 0.88341, rtol=1e-3)

def test_non_eq_tracer_Q():
    phase_models = {
        "components": ["AL", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_tracer_Q_Si)
    dbf = Database(ALMGSI_DB)
    residual_func = NonEquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [7600], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -10.41807, rtol=1e-3)

def test_non_eq_tracer_diff():
    phase_models = {
        "components": ["AL", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_tracer_diff_Mg)
    dbf = Database(ALMGSI_DB)
    residual_func = NonEquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [0.00436798], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 1.3826926, rtol=1e-3)

def test_eq_tracer_D0():
    phase_models = {
        "components": ["AL", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_eq_tracer_D0_Al)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [-0.1], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 0.88341, rtol=1e-3)

def test_eq_tracer_Q():
    phase_models = {
        "components": ["AL", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_eq_tracer_Q_Si)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [7600], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -10.41807, rtol=1e-3)

def test_eq_tracer_diff():
    phase_models = {
        "components": ["AL", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_eq_diff_Mg)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [0.00436798], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 1.3826926, rtol=1e-3)

def test_eq_tracer_interdiff():
    phase_models = {
        "components": ["AL", "MG", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "MG"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }
    phase_models = PhaseModelSpecification(**phase_models)

    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(AlSi_eq_interdiff)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [0.00409745], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 1.382807, rtol=1e-3)

def  test_espei_compatibility():
    '''
    Tests that mobility residuals are recognized by espei and datasets are properly loaded in

    Main things to test:
    - mobility residuals can compute without error even if no datasets exist
    - mobility data is picked up by mobility residuals and not by built-in espei residuals
    '''
    dbf = Database(ALMGSI_DB)
    # add symbol to create context
    dbf.symbols["VV0000"] = 0

    phase_models = {
        "components": ["AL", "MG", "SI", "VA"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["AL", "MG", "SI"], ["VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }

    datasets_db = PickleableTinyDB(storage=MemoryStorage)

    # First test that everything can run without datasets
    ctx = setup_context(dbf, datasets_db, None, phase_models=phase_models, make_callables=False)
    for residual_obj in ctx['residual_objs']:
        likelihood = residual_obj.get_likelihood(np.asarray([0], dtype=np.float64))

    datasets_db.insert(AlSi_tracer_D0_Al)
    datasets_db.insert(AlSi_tracer_Q_Si)
    datasets_db.insert(AlSi_tracer_diff_Mg)
    datasets_db.insert(AlSi_eq_tracer_D0_Al)
    datasets_db.insert(AlSi_eq_tracer_Q_Si)
    datasets_db.insert(AlSi_eq_diff_Mg)
    datasets_db.insert(AlSi_eq_interdiff)

    # Test that everything can run with datasets and that NonEquilibriumMobilityResidual and EquilibriumMobilityResidual give non-zero values
    # and that every other residual gives 0 for likelihood
    ctx = setup_context(dbf, datasets_db, None, phase_models=phase_models, make_callables=False)
    for residual_obj in ctx['residual_objs']:
        likelihood = residual_obj.get_likelihood(np.asarray([0], dtype=np.float64))
        if isinstance(residual_obj, NonEquilibriumMobilityResidual) or isinstance(residual_obj, EquilibriumMobilityResidual):
            assert likelihood != 0
        else:
            assert likelihood == 0

