"""
Service layer that wraps the kawin precipitation simulation.
"""

import gc
import numpy as np
from app.config import settings
from app.models.simulation import (
    SimulationRequest,
    SimulationResponse,
    SimulationSummary,
    PhaseResult,
    PhaseTimeSeries,
    TimeSeriesData,
)
from kawin.run_simulation import PrecipitationSimulator


class SimulationService:
    """Service for running Al-Mg-Si-Cu precipitation simulations."""

    def __init__(self, almgsi_tdb: str = None, alcu_tdb: str = None):
        self.almgsi_tdb = almgsi_tdb or settings.TDB_FILE
        self.alcu_tdb = alcu_tdb or settings.ALCU_TDB_FILE

    def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """Run a precipitation simulation."""
        almgsi_composition = (
            [request.mg_content, request.si_content]
            if request.mg_content > 0 or request.si_content > 0
            else None
        )
        cu_composition = [request.cu_content] if request.cu_content > 0 else None

        simulator = PrecipitationSimulator(
            aging_temperature=request.aging_temperature,
            aging_time=request.aging_time,
            almgsi_tdb=self.almgsi_tdb if almgsi_composition is not None else None,
            alcu_tdb=self.alcu_tdb if cu_composition is not None else None,
            almgsi_composition=almgsi_composition,
            cu_composition=cu_composition,
        )

        try:
            results = simulator.solve(verbose=False)
            return self._build_response(results, request.aging_temperature)
        finally:
            del simulator
            del results
            gc.collect()

    def _build_response(self, results, aging_temperature: float) -> SimulationResponse:
        """Build SimulationResponse from simulation results."""
        phase_results = []

        if results.almgsi is not None:
            for i, phase in enumerate(results.almgsi.phases):
                phase_results.append(
                    PhaseResult(
                        name=phase,
                        diameter_nm=round(float(results.almgsi.diameter[-1, i]), 2),
                        volume_fraction_pct=round(float(results.almgsi.volume_fraction[-1, i]) * 100, 4),
                    )
                )

        if results.alcu is not None:
            for i, phase in enumerate(results.alcu.phases):
                phase_results.append(
                    PhaseResult(
                        name=phase,
                        diameter_nm=round(float(results.alcu.diameter[-1, i]), 2),
                        volume_fraction_pct=round(float(results.alcu.volume_fraction[-1, i]) * 100, 4),
                    )
                )

        summary = SimulationSummary(
            final_time_hours=round(float(results.time_hours[-1]), 2),
            temperature_c=aging_temperature,
            phases=phase_results,
            matrix_mg_wt_pct=round(float(results.almgsi.matrix_Mg_wt_pct[-1]), 4) if results.almgsi is not None else 0.0,
            matrix_si_wt_pct=round(float(results.almgsi.matrix_Si_wt_pct[-1]), 4) if results.almgsi is not None else 0.0,
            matrix_cu_wt_pct=round(float(results.alcu.matrix_Cu_wt_pct[-1]), 4) if results.alcu is not None else 0.0,
            orowan_strength_mpa=round(float(results.precipitate_strength[-1] / 1e6), 1),
            solid_solution_strength_mpa=round(float(results.solid_solution_strength[-1] / 1e6), 1),
            total_yield_strength_mpa=round(float(results.total_strength[-1] / 1e6), 1),
        )

        time_series = self._build_time_series(results)

        return SimulationResponse(
            status="completed",
            summary=summary,
            time_series=time_series,
        )

    def _build_time_series(self, results) -> TimeSeriesData:
        """Build TimeSeriesData from simulation results."""
        time_hours = [round(float(t), 4) for t in results.time_hours]

        phase_time_series = []

        if results.almgsi is not None:
            for i, phase in enumerate(results.almgsi.phases):
                phase_time_series.append(
                    PhaseTimeSeries(
                        name=phase,
                        diameter_nm=self._to_list(results.almgsi.diameter[:, i]),
                        volume_fraction_pct=self._to_list(results.almgsi.volume_fraction[:, i] * 100, 4),
                    )
                )

        if results.alcu is not None:
            for i, phase in enumerate(results.alcu.phases):
                phase_time_series.append(
                    PhaseTimeSeries(
                        name=phase,
                        diameter_nm=self._to_list(results.alcu.diameter[:, i]),
                        volume_fraction_pct=self._to_list(results.alcu.volume_fraction[:, i] * 100, 4),
                    )
                )

        return TimeSeriesData(
            time_hours=time_hours,
            phases=phase_time_series,
            matrix_mg_wt_pct=self._to_list(results.almgsi.matrix_Mg_wt_pct, 4) if results.almgsi is not None else [],
            matrix_si_wt_pct=self._to_list(results.almgsi.matrix_Si_wt_pct, 4) if results.almgsi is not None else [],
            matrix_cu_wt_pct=self._to_list(results.alcu.matrix_Cu_wt_pct, 4) if results.alcu is not None else [],
            orowan_strength_mpa=self._to_list(results.precipitate_strength / 1e6, 1),
            solid_solution_strength_mpa=self._to_list(results.solid_solution_strength / 1e6, 1),
            total_yield_strength_mpa=self._to_list(results.total_strength / 1e6, 1),
        )

    @staticmethod
    def _to_list(arr: np.ndarray, decimals: int = 2) -> list[float]:
        """Convert numpy array to list."""
        return [round(float(v), decimals) if np.isfinite(v) else 0.0 for v in arr]
