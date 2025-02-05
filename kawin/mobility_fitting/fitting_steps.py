from espei.parameter_selection.fitting_steps import AbstractLinearPropertyStep, FittingStep
from pycalphad import Model, variables as v
import numpy as np
import symengine
from typing import Optional, Dict, Any, List
from numpy.typing import ArrayLike
from espei.utils import build_sitefractions
import itertools

Dataset = Dict[str, Any]

'''
Fitting steps for Mobility models from tracer diffusivity data

Tracer diffusivity can be defined as D* = D0*exp(-Q/RT)
    Where D0 is a pre-factor term (m/s^2) and Q is the activation energy (J/mol)

Mobility is related to tracer diffusivity by:
    D* = RTM = RT*(1/RT exp(MQ/RT)) = exp(MQ/RT)
        Where MQ is a Redlich-Kister polynomial

Option 1: Fit mobility to activation energy and pre-factor
    D* = D0 exp(-Q/RT) = exp(MQ/RT)
    ln(D0) + -Q/RT = MQ/RT
    RT ln(D0) - Q = MQ

    Then:
        R ln(D0) = dMQ/dT
        -Q = MQ - T dMQ/dT

    Some pre-processing of tracer diffusivity data is required to have input files for both
    activation energy and the pre-factor term, but this gives a better model fit

Option 2: Fit mobility directly to tracer diffusivity
    D* = exp(MQ/RT)
    RT ln(D*) = MQ

    Then we could fit MQ as a linear model where each RK term is in the form of A+B*T
    NOTE: This method is not recommended since it can lead to overfitting
'''
    
class StepD0(FittingStep):
    supported_reference_states: List[str] = [""]
    features: List[symengine.Expr] = [v.T]

    @staticmethod
    def transform_data(d: ArrayLike, model: Optional[Model] = None) -> ArrayLike:  # np.object_
        return np.array([symengine.log(di)*v.R for di in d])

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # Fitting to R ln(D0) = dMQ/dT
        return symengine.diff(expr, v.T)
    
    @classmethod
    def shift_reference_state(cls, desired_data: List[Dataset], fixed_model: Model, mole_atoms_per_mole_formula_unit: symengine.Expr) -> ArrayLike:  # np.object_
        """
        Shift _MIX or _FORM data to a common reference state in per mole-atom units.

        Parameters
        ----------
        desired_data : List[Dict[str, Any]]
            ESPEI single phase dataset
        fixed_model : pycalphad.Model
            Model with all lower order (in composition) terms already fit. Pure
            element reference state (GHSER functions) should be set to zero.
        mole_atoms_per_mole_formula_unit : float
            Number of moles of atoms in every mole atom unit.

        Returns
        -------
        np.ndarray
            Data for this feature in [qty]/mole-formula in a common reference state.

        Raises
        ------
        ValueError

        Notes
        -----
        pycalphad Model parameters are stored as per mole-formula quantites, but
        the calculated properties and our data are all in [qty]/mole-atoms. We
        multiply by mole-atoms/mole-formula to convert the units to
        [qty]/mole-formula.

        """
        total_response = []
        for dataset in desired_data:
            values = np.asarray(dataset['values'], dtype=np.object_)*mole_atoms_per_mole_formula_unit
            unique_excluded_contributions = set(dataset.get('excluded_model_contributions', []))
            for config_idx in range(len(dataset['solver']['sublattice_configurations'])):
                occupancy = dataset['solver'].get('sublattice_occupancies', None)
                if dataset['output'].endswith('_FORM'):
                    # we don't shift the reference state because we assume our
                    # models are already in the formation reference state (by us
                    # setting GHSERXX functions to zero explictly)
                    pass
                elif dataset['output'].endswith('_MIX'):
                    if occupancy is None:
                        raise ValueError('Cannot have a _MIX property without sublattice occupancies.')
                    else:
                        values[..., config_idx] += cls.transform_feature(fixed_model.models['ref'])*mole_atoms_per_mole_formula_unit
                else:
                    #raise ValueError(f'Unknown property to shift: {dataset["output"]}')
                    pass
                for excluded_contrib in unique_excluded_contributions:
                    values[..., config_idx] += cls.transform_feature(fixed_model.models[excluded_contrib])*mole_atoms_per_mole_formula_unit
            total_response.append(values.flatten())
        return total_response

    @classmethod
    def get_response_vector(cls, fixed_model: Model, fixed_portions: List[symengine.Basic], data: List[Dataset], sample_condition_dicts: list[dict[str, Any]]) -> ArrayLike:  # np.float64
        mole_atoms_per_mole_formula_unit = fixed_model._site_ratio_normalization
        # Define site fraction symbols that will be reused
        phase_name = fixed_model.phase_name

        # Construct flattened list of site fractions corresponding to the ravelled data (from shift_reference_state)
        site_fractions = []
        for ds in data:
            for _ in ds['conditions']['T']:
                sf = build_sitefractions(phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies', np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float64)))
                site_fractions.append(sf)
        site_fractions = list(itertools.chain(*site_fractions))

        #data_qtys = np.concatenate(cls.shift_reference_state(data, fixed_model, mole_atoms_per_mole_formula_unit), axis=-1)
        data_qtys = np.concatenate(cls.shift_reference_state(data, fixed_model, 1), axis=-1)
        data_qtys = cls.transform_data(data_qtys, fixed_model)
        # Remove existing partial model contributions from the data, convert to per mole-formula units
        data_qtys = data_qtys - cls.transform_feature(getattr(fixed_model, cls.parameter_name))*mole_atoms_per_mole_formula_unit
        # Subtract out high-order (in T) parameters we've already fit, already in per mole-formula units
        data_qtys = data_qtys - cls.transform_feature(sum(fixed_portions))
        # If any site fractions show up in our rhs that aren't in these
        # datasets' site fractions, set them to zero. This can happen if we're
        # fitting a multi-component model that has site fractions from
        # components that aren't in a particular dataset
        for sf, i, cond_dict in zip(site_fractions, data_qtys, sample_condition_dicts):
            missing_variables = symengine.S(i).atoms(v.SiteFraction) - set(sf.keys())
            sf.update({x: 0. for x in missing_variables})
            sf.update(cond_dict)
        # also replace with database symbols in case we did higher order fitting
        data_qtys = [fixed_model.symbol_replace(symengine.S(i).xreplace(sf), fixed_model._symbols).evalf() for i, sf in zip(data_qtys, site_fractions)]
        data_qtys = np.asarray(data_qtys, dtype=np.float64)
        return data_qtys
    
class StepQ(StepD0):
    features: List[symengine.Expr] = [symengine.S.One]

    @staticmethod
    def transform_data(d: ArrayLike, model: Optional[Model] = None) -> ArrayLike:  # np.object_
        return np.array([-di for di in d])

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # Fitting to -Q = MQ - T dMQ/dT
        return expr - v.T*symengine.diff(expr, v.T)
    
class StepTracerDiffusivity(AbstractLinearPropertyStep):
    features: List[symengine.Expr] = [symengine.S.One, v.T]
    
    @staticmethod
    def transform_data(d: ArrayLike, model: Optional[Model] = None) -> ArrayLike:  # np.object_
        # D* = exp(MQ/RT) -> RT D* = MQ
        return np.array([symengine.log(di)*v.R*v.T for di in d])