from numpy.testing import assert_allclose
from tinydb import where
from tinydb.storages import MemoryStorage

from pycalphad import Database, variables as v
from espei.utils import PickleableTinyDB

from kawin.mobility_fitting import generate_mobility, MobilityTemplate, EquilibriumSiteFractionGenerator
from kawin.mobility_fitting import generate_liquid_mobility_liu, LiuSpecies, generate_liquid_mobility_su, SuSpecies
from kawin.mobility_fitting.manual_fitting import grab_tracer_datasets, fit_prefactor, fit_activation_energy, select_best_model
from kawin.mobility_fitting.utils import find_last_variable, get_used_database_symbols

from kawin.tests.databases import CUNI_TDB
from kawin.tests.datasets import CuNi_datasets, Cu_tracer_diff_datasets

def test_espei_mobility_generation():
    '''
    Test espei mobility generation from activation energy and pre-factor data
    '''
    phase_models = {
        "components": ["CU", "NI", "VA"],
        "phases": {
            "FCC_A1": {"sublattice_model": [["CU", "NI"], ["VA"]], "sublattice_site_ratios": [1, 1]},
        }
    }

    with PickleableTinyDB(storage=MemoryStorage) as datasets_db:
        for ds in CuNi_datasets:
            datasets_db.insert(ds)

        aicc = {'FCC_A1': {'TRACER_Q_CU': 2, 'TRACER_D0_CU': 2}}
        dbf = generate_mobility(phase_models, datasets_db, aicc_penalty_factor=aicc)
        #print(dbf.to_string(fmt='tdb'))
        vv_symbols = [s for s in dbf.symbols if s.startswith('VV00')]
        vv_vals = [dbf.symbols[s] for s in vv_symbols]
        # dbf should have 9 symbols (2 for Cu in Cu, 2 for Cu in Ni, 2 for Ni in Cu, 1 for Ni in Ni, and 2 for Cu in Cu-Ni)
        assert len(vv_symbols) == 9
        sym_vals = [-82.5275, -205872.0, -71.0695, -232826.0, -79.2647, -255224.0, -285142.0, -39.2064, 32429.60]
        assert_allclose(vv_vals, sym_vals, rtol=1e-3)

        dbf_ni = generate_mobility(phase_models, datasets_db, diffusing_species=['NI'])
        #print(dbf_ni.to_string(fmt='tdb'))
        vv_symbols = [s for s in dbf_ni.symbols if s.startswith('VV00')]
        vv_vals = [dbf_ni.symbols[s] for s in vv_symbols]
        # dbf should have 3 symbols (2 for Ni in Cu and 1 for Ni in Ni)
        assert len(vv_symbols) == 3
        sym_vals = [-71.0695, -232826.0, -285142.0]
        assert_allclose(vv_vals, sym_vals, rtol=1e-3)

def test_espei_mobility_generation_from_tracer_diff():
    '''
    Test espei mobility generation from tracer diffusivity data
    '''
    phase_models = {
        "components": ["CU", "VA"],
        "phases": {
            "FCC_A1": {"sublattice_model": [["CU"], ["VA"]], "sublattice_site_ratios": [1, 1]},
        }
    }

    with PickleableTinyDB(storage=MemoryStorage) as datasets_db:
        for ds in Cu_tracer_diff_datasets:
            datasets_db.insert(ds)

        dbf = generate_mobility(phase_models, datasets_db, fit_to_tracer=True)
        #print(dbf.to_string(fmt='tdb'))
        vv_symbols = [s for s in dbf.symbols if s.startswith('VV00')]
        vv_vals = [dbf.symbols[s] for s in vv_symbols]
        # dbf should have 2 symbols for Cu in Cu
        assert len(vv_symbols) == 2
        sym_vals = [-202181.0, -86.4134]
        assert_allclose(vv_vals, sym_vals, rtol=1e-3)

def test_manual_mobility_generation():
    '''
    Tests manual mobility generation from activation and prefactor data and
    that selected model templates can be added to database
    '''
    dbf = Database(CUNI_TDB)
    eqgen = EquilibriumSiteFractionGenerator(dbf)

    phase = 'FCC_A1'
    sublattice_model = [1, 1]
    constituents = [['CU', 'NI'], ['VA']]
    diffusing_species = 'CU'

    # Create mobility templates. We'll do three options here: 
    # a) no mixing term, b) 1st order mixing, c) 1st and 2nd order mixing
    t0 = MobilityTemplate(phase, diffusing_species, sublattice_model, constituents)

    t1 = MobilityTemplate(phase, diffusing_species, sublattice_model, constituents)
    t1.add_activation_energy([['CU', 'NI'], ['VA']], 0)
    t1.add_prefactor([['CU', 'NI'], ['VA']], 0)

    t2 = MobilityTemplate(phase, diffusing_species, sublattice_model, constituents)
    t2.add_activation_energy([['CU', 'NI'], ['VA']], [0, 1])
    t2.add_prefactor([['CU', 'NI'], ['VA']], [0, 1])

    with PickleableTinyDB(storage=MemoryStorage) as datasets_db:
        for ds in CuNi_datasets:
            datasets_db.insert(ds)

        # Fit prefactor parameters to data
        d_data = grab_tracer_datasets(datasets_db, 'D0', phase, ['CU', 'NI', 'VA'], diffusing_species)
        d_model, d_results = select_best_model(d_data, [t0, t1, t2], fit_prefactor, eqgen, p=0.1, return_all_models = True)
        assert_allclose(d_model.aicc, -13.633828, rtol=1e-3)

        # Fit activation energy parameters to data
        q_data = grab_tracer_datasets(datasets_db, 'Q', phase, ['CU', 'NI', 'VA'], diffusing_species)
        q_model, q_results = select_best_model(q_data, [t0, t1, t2], fit_activation_energy, eqgen, return_all_models = True)

        assert_allclose(q_model.aicc, 92.05808)

        d_model.template.add_to_database(dbf, d_model.parameters)
        q_model.template.add_to_database(dbf, q_model.parameters)
        #print(dbf.to_string(fmt='tdb'))
        vv_symbols = [s for s in dbf.symbols if s.startswith('VV00')]
        vv_vals = [dbf.symbols[s] for s in vv_symbols]
        # dbf should have 6 symbols (2 for Cu in Cu, 2 for Cu in Ni, 2 for D0 mixing)
        assert len(vv_symbols) == 6
        sym_vals = [-81.78946, -79.25679, -18.55759, -45.44596, -202291.969, -253898.678]
        assert_allclose(vv_vals, sym_vals, rtol=1e-3)

def test_liquid_mobility():
    '''
    Test Liu and Su liquid mobility models and that they can be added to the database
    '''
    dbf = Database(CUNI_TDB)
    eqgen = EquilibriumSiteFractionGenerator(dbf)

    # Create mobility parameters using the Liu model
    liu_species = [
            LiuSpecies('CU', diameter=2.55, atomic_number=29, Tm=1358, melting_phase='FCC_A1'),
            LiuSpecies('NI', diameter=2.49, atomic_number=28, Tm=1728, melting_phase='FCC_A1'),
        ]
    liu_templates = generate_liquid_mobility_liu(dbf, liu_species)
    
    liq_mob_params = dbf.search((where('parameter_type') == 'MQ') & (where('phase_name') == 'LIQUID'))
    # 4 liquid parameters for Cu in Cu, Cu in Ni, Ni in Cu and Ni in Ni
    assert len(liq_mob_params) == 4

    # Create mobility parameters using the Su model
    su_species = [
        SuSpecies('CU', atomic_mass=63.55, density=8.96, Tm=1358),
        SuSpecies('NI', atomic_mass=58.69, density=8.90, Tm=1728),
    ]
    # We could disable adding the liquid mobility models to the database
    su_templates = generate_liquid_mobility_su(dbf, su_species, add_to_database=False)

    # Assert correctness of liquid mobility models
    conditions = {v.T: 2000, v.P: 101325, v.X('CU'): 0.3}
    vals = {'CU': 2.988006e-13, 'NI': 3.060001e-13}
    for s in liu_templates:
        y = liu_templates[s].evaluate(liu_templates[s].mobility_function, {}, eqgen, conditions)
        assert_allclose(y, vals[s], rtol=1e-3)

        mq = liu_templates[s].evaluate(liu_templates[s].MQ, {}, eqgen, conditions)
        d0 = liu_templates[s].evaluate(liu_templates[s].prefactor, {}, eqgen, conditions)
        q = liu_templates[s].evaluate(liu_templates[s].activation_energy, {}, eqgen, conditions)
        print(mq, d0, q)

    vals = {'CU': 3.331584e-13, 'NI': 3.413458e-13}
    for s in su_templates:
        y = su_templates[s].evaluate(su_templates[s].mobility_function, {}, eqgen, conditions)
        assert_allclose(y, vals[s], rtol=1e-3)

def test_template_evaluation():
    '''
    Test that template can evaluate mobility, MQ, prefactor and activation energy
    Test that evaluation can also work when 1 dependent variable is present
    '''
    dbf = Database(CUNI_TDB)
    eqgen = EquilibriumSiteFractionGenerator(dbf)

    species = LiuSpecies('CU', diameter=2.55, atomic_number=29, Tm=1358, melting_phase='FCC_A1')
    templates = generate_liquid_mobility_liu(dbf, [species])

    conditions = {v.T: 2000, v.P: 101325}
    d = templates['CU'].evaluate(templates['CU'].mobility_function, {}, eqgen, conditions)
    mq = templates['CU'].evaluate(templates['CU'].MQ, {}, eqgen, conditions)
    d0 = templates['CU'].evaluate(templates['CU'].prefactor, {}, eqgen, conditions)
    q = templates['CU'].evaluate(templates['CU'].activation_energy, {}, eqgen, conditions)

    assert_allclose(d, 4.581566e-13, rtol=1e-3)
    assert_allclose(mq, -310840.27, rtol=1e-3)
    assert_allclose(d0, -16.4996237, rtol=1e-3)
    assert_allclose(q, 36468.03076, rtol=1e-3)

    # Test conditions with 1 dependent value
    conditions = {v.T: (2000, 3000, 100), v.P: 101325}
    t, d, variable = templates['CU'].evaluate(templates['CU'].mobility_function, {}, eqgen, conditions)
    assert len(t) == len(d)
    assert variable == v.T

def test_database_variable_utilities():
    phase_models = {
        "components": ["CU", "NI", "VA"],
        "phases": {
            "FCC_A1": {"sublattice_model": [["CU", "NI"], ["VA"]], "sublattice_site_ratios": [1, 1]},
        }
    }

    with PickleableTinyDB(storage=MemoryStorage) as datasets_db:
        for ds in CuNi_datasets:
            datasets_db.insert(ds)

        aicc = {'FCC_A1': {'TRACER_Q_CU': 2, 'TRACER_D0_CU': 2}}
        dbf = generate_mobility(phase_models, datasets_db, aicc_penalty_factor=aicc)

        last_variable = find_last_variable(dbf)
        # generate_mobility will generate 9 symbols, so next variable should be VV0009
        assert last_variable == 9

        # Only mixing parameters of Cu mobility (2 symbols in total)
        free_syms = get_used_database_symbols(dbf, ['CU', 'NI'], 'CU', ['FCC_A1'], include_subsystems=False)
        assert len(set(free_syms).symmetric_difference({'VV0007', 'VV0008'})) == 0

        # Endmember and mixing parameters of Cu mobility (6 symbols in total)
        free_syms_endmembers = get_used_database_symbols(dbf, ['CU', 'NI'], 'CU', ['FCC_A1'], include_subsystems=True)
        assert len(set(free_syms_endmembers).symmetric_difference({'VV0000', 'VV0001', 'VV0004', 'VV0005', 'VV0007', 'VV0008'})) == 0