import numpy as np

from pycalphad import Database
from espei.utils import PickleableTinyDB, MemoryStorage
from espei.phase_models import PhaseModelSpecification

from kawin.mobility_fitting.error_functions import NonEquilibriumMobilityResidual, EquilibriumMobilityResidual
from kawin.tests.databases import *

def test_non_eq_tracer_D0():
    dataset = {
        "components": ["AL", "SI", "VA"], 
        "phases": ["FCC_A1"], 
        "conditions": {
            "P": 101325, 
            "T": 298.15
            }, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [[["AL", "SI"], "VA"]], 
            "sublattice_occupancies": [[[0.9, 0.1], 1]]
            }, 
        "output": "TRACER_D0_AL", 
        "values": [[[1.75e-5]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = NonEquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [-0.1], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 0.88341, rtol=1e-3)

def test_non_eq_tracer_Q():
    dataset = {
        "components": ["AL", "SI", "VA"], 
        "phases": ["FCC_A1"], 
        "conditions": {
            "P": 101325, 
            "T": 298.15
            }, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [[["AL", "SI"], "VA"]], 
            "sublattice_occupancies": [[[0.9, 0.1], 1]]
            }, 
        "output": "TRACER_Q_SI", 
        "values": [[[160000]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = NonEquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [7600], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -10.41807, rtol=1e-3)

def test_non_eq_tracer_diff():
    dataset = {
        "components": ["AL", "SI", "VA"], 
        "phases": ["FCC_A1"], 
        "conditions": {
            "P": 101325, 
            "T": 298.15
            }, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [[["AL", "SI"], "VA"]], 
            "sublattice_occupancies": [[[0.9, 0.1], 1]]
            }, 
        "output": "TRACER_DIFF_MG", 
        "values": [[[5.2e-26]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = NonEquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [0.00436798], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 1.3826926, rtol=1e-3)

def test_eq_tracer_D0():
    dataset = {
        "components": ["AL", "SI", "VA"], 
        "phases": ["FCC_A1"], 
        "conditions": {
            "P": 101325, 
            "T": 298.15,
            "X_SI": 0.1
            }, 
        "output": "TRACER_D0_AL", 
        "values": [[[1.75e-5]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [-0.1], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 0.88341, rtol=1e-3)

def test_eq_tracer_Q():
    dataset = {
        "components": ["AL", "SI", "VA"], 
        "phases": ["FCC_A1"], 
        "conditions": {
            "P": 101325, 
            "T": 298.15,
            "X_SI": 0.1
            }, 
        "output": "TRACER_Q_SI", 
        "values": [[[160000]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [7600], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -10.41807, rtol=1e-3)

def test_eq_tracer_diff():
    dataset = {
        "components": ["AL", "SI", "VA"], 
        "phases": ["FCC_A1"], 
        "conditions": {
            "P": 101325, 
            "T": 298.15,
            "X_SI": 0.1
            }, 
        "output": "TRACER_DIFF_MG", 
        "values": [[[5.2e-26]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [0.00436798], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 1.3826926, rtol=1e-3)

def test_eq_tracer_interdiff():
    dataset = {
        "components": ["AL", "MG", "VA"], 
        "phases": ["FCC_A1"], 
        "ref_el": ["AL"],
        "dependent_el": ["MG", "MG"],
        "conditions": {
            "P": 101325, 
            "T": 298.15,
            "X_MG": 0.1
            }, 
        "output": "INTER_DIFF", 
        "values": [[[4.6e-25]]]
    }

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
    datasets_db.insert(dataset)
    dbf = Database(ALMGSI_DB)
    residual_func = EquilibriumMobilityResidual(dbf, datasets_db, phase_models=phase_models, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert np.isclose(residuals, [0.00409745], rtol=1e-3)

    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, 1.382807, rtol=1e-3)

