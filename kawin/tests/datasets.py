'''
Espei datasets for unit testing
'''

CuNi_datasets = [
    # D0 for Cu in Cu
    {
        "components": ["CU", "VA"],  "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["CU", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_D0_CU", 
        "values": [[[4.89e-5]]], 
        "reference": "Ghosh2001"
    },
    # Q for Cu in Cu
    {
        "components": ["CU", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["CU", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_Q_CU", 
        "values": [[[205872]]], 
        "reference": "Ghosh2001"
    },
    # D0 for Cu in Ni
    {
        "components": ["NI", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["NI", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_D0_CU", 
        "values": [[[7.24e-5]]], 
        "reference": "Anand1965"
    },
    # Q for Cu in Ni
    {
        "components": ["NI", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["NI", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_Q_CU", 
        "values": [[[255224]]], 
        "reference": "Anand1965"
    },
    # D0 for Cu in Cu-Ni
    {
        "components": ["CU", "NI", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [ [["CU", "NI"], "VA"], [["CU", "NI"], "VA"], [["CU", "NI"], "VA"]], 
            "sublattice_occupancies": [ [[0.7173, 0.2827], 1], [[0.8028, 0.1972], 1], [[0.9008, 0.0992], 1]]
    }, 
        "output": "TRACER_D0_CU", 
        "values": [[[2.67e-5, 1.63e-5, 4.14e-5]]], 
        "reference": "Anusavice1972"
    },
    # Q for Cu in Cu-Ni
    {
        "components": ["CU", "NI", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [[["CU", "NI"], "VA"], [["CU", "NI"], "VA"], [["CU", "NI"], "VA"]], 
            "sublattice_occupancies": [ [[0.7173, 0.2827], 1], [[0.8028, 0.1972], 1], [[0.9008, 0.0992], 1]]
        }, 
        "output": "TRACER_Q_CU", 
        "values": [[[215847, 205847, 210162]]], 
        "reference": "Anusavice1972"
    },
    # D0 for Ni in Cu
    {
        "components": ["CU", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["CU", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_D0_NI", 
        "values": [[[1.94e-4]]], 
        "reference": "Anusavice1972"
    },
    # Q for Ni in Cu
    {
        "components": ["CU", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["CU", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_Q_NI", 
        "values": [[[232826]]], 
        "reference": "Anusavice1972"
    },
    # Q for Ni in Ni
    {
        "components": ["NI", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": 1273}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["NI", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_Q_NI", 
        "values": [[[285142]]], 
        "reference": "Bakker1968"
    }
]

Cu_tracer_diff_datasets = [
    {
        "components": ["CU", "VA"], "phases": ["FCC_A1"], 
        "conditions": {"P": 101325, "T": [1013, 1080, 1103, 1185, 1210, 1248, 1283, 1318]}, 
        "solver": {
            "mode": "manual", 
            "sublattice_site_ratios": [1, 3], 
            "sublattice_configurations": [["CU", "VA"]], 
            "sublattice_occupancies": [[1, 1]]
        }, 
        "output": "TRACER_DIFF_CU", 
        "values": [[[0.115e-14], [0.501e-14], [0.858e-14], [3.55e-14], [5.74e-14], [11e-14], [17.5e-14], [30.2e-14]]], 
        "reference": "Ghosh2001"
    }
]