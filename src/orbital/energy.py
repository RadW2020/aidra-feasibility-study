"""
Perfil energetico del pipeline.

Estimacion de consumo energetico basada en tiempo de CPU y
TDP (Thermal Design Power) de referencia para procesadores
de vuelo conocidos.

La estimacion no es exacta (no medimos vatios reales), pero
permite comparar variantes de modelo y perfiles, y extrapolar
a hardware de vuelo especifico.

Usage:
    from src.orbital.energy import EnergyProfiler

    profiler = EnergyProfiler()
    estimate = profiler.estimate_inference_energy(
        cpu_time_seconds=0.15,
        cpu_cores_used=1.0,
    )
    print(estimate.energy_joules)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pydantic import BaseModel, Field

from src.db.models import ExecutionRecord
from src.orbital.orbit_params import (
    PROCESSOR_TDP_WATTS,
    SATELLITE_POWER_BUDGETS,
)

logger = logging.getLogger(__name__)

# Re-export for backward-compatibility so callers can import from here
__all__ = [
    "PROCESSOR_TDP_WATTS",
    "SATELLITE_POWER_BUDGETS",
    "EnergyProfiler",
    "EnergyEstimate",
    "PipelineEnergyEstimate",
    "OrbitalBudgetResult",
]


# ====================================================================
# Pydantic models
# ====================================================================


class EnergyEstimate(BaseModel):
    """Energy estimate for a single inference on a given processor."""

    processor: str
    tdp_watts: float
    cpu_time_seconds: float
    cpu_cores: float
    energy_joules: float
    energy_wh: float = Field(
        description="energy_joules / 3600",
    )
    equivalent_battery_percent: float = Field(
        description="Percentage of a typical satellite battery consumed",
    )


class PipelineEnergyEstimate(BaseModel):
    """Energy estimate for the full pipeline (preprocess + inference + postprocess)."""

    processor: str
    preprocessing_joules: float
    inference_joules: float
    postprocessing_joules: float
    total_joules: float
    total_wh: float
    breakdown_percent: dict[str, float] = Field(
        description='e.g. {"preprocessing": 15.0, "inference": 80.0, "postprocessing": 5.0}',
    )


class OrbitalBudgetResult(BaseModel):
    """Result of an orbital energy-budget calculation."""

    satellite_type: str
    battery_wh: float
    payload_power_w: float
    energy_per_image_wh: float
    max_images_per_orbit: int = Field(
        description="floor(available_wh / energy_per_image_wh)",
    )
    available_energy_wh: float = Field(
        description="payload_w * orbit_period_hours * sunlit_fraction",
    )
    utilization_percent: float = Field(
        description="(images_requested * energy_per_image_wh) / available_energy_wh * 100",
    )
    feasible: bool = Field(
        description="True if max_images_per_orbit >= 1",
    )
    notes: str


# ====================================================================
# EnergyProfiler
# ====================================================================


class EnergyProfiler:
    """Estimates energy consumption for the AIDRA pipeline on reference processors.

    Parameters
    ----------
    reference_processor:
        Default processor key used when no explicit processor is given.
    default_satellite:
        Default satellite type used for battery-percent calculations.
    """

    def __init__(
        self,
        reference_processor: str = "oci_arm_a1",
        default_satellite: str = "cubesat_6u",
    ) -> None:
        if reference_processor not in PROCESSOR_TDP_WATTS:
            raise KeyError(
                f"Unknown processor '{reference_processor}'. "
                f"Available: {', '.join(PROCESSOR_TDP_WATTS.keys())}"
            )
        self.reference_processor = reference_processor
        self.default_satellite = default_satellite
        logger.info(
            "EnergyProfiler initialised with reference_processor=%s",
            reference_processor,
        )

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def estimate_inference_energy(
        self,
        cpu_time_seconds: float,
        cpu_cores_used: float,
        processor: str | None = None,
        cpu_utilization: float = 1.0,
    ) -> EnergyEstimate:
        """Estimate energy consumed during a single inference.

        Parameters
        ----------
        cpu_time_seconds:
            Wall-clock time of the inference in seconds.
        cpu_cores_used:
            Number of CPU cores actively used (can be fractional).
        processor:
            Processor key in ``PROCESSOR_TDP_WATTS``.  Defaults to the
            reference processor given at construction time.
        cpu_utilization:
            Average CPU utilisation fraction (0.0-1.0).  A value of 0.5
            means the core was at 50 % load on average, so only 50 % of
            TDP is attributed.

        Returns
        -------
        EnergyEstimate
        """
        proc = processor or self.reference_processor
        tdp = self._get_tdp(proc)
        energy_j = cpu_time_seconds * cpu_cores_used * tdp * cpu_utilization
        energy_wh = energy_j / 3600.0
        battery_pct = self._battery_percent(energy_wh)

        return EnergyEstimate(
            processor=proc,
            tdp_watts=tdp,
            cpu_time_seconds=cpu_time_seconds,
            cpu_cores=cpu_cores_used,
            energy_joules=round(energy_j, 6),
            energy_wh=round(energy_wh, 8),
            equivalent_battery_percent=round(battery_pct, 4),
        )

    # ------------------------------------------------------------------
    # Full pipeline estimation
    # ------------------------------------------------------------------

    def estimate_pipeline_energy(
        self,
        execution_record: ExecutionRecord,
        processor: str | None = None,
    ) -> PipelineEnergyEstimate:
        """Estimate energy for the full pipeline from an ``ExecutionRecord``.

        Download time is treated as negligible (in orbit the sensor is
        on-board).  Preprocessing and postprocessing are CPU-bound.
        Inference dominates the energy budget.

        Parameters
        ----------
        execution_record:
            An ``ExecutionRecord`` containing timing breakdowns.
        processor:
            Processor key.  Defaults to reference processor.

        Returns
        -------
        PipelineEnergyEstimate
        """
        proc = processor or self.reference_processor
        tdp = self._get_tdp(proc)

        # Extract durations in seconds (stored as ms in the record)
        preprocess_s = (execution_record.preprocessing_ms or 0.0) / 1000.0
        inference_s = (execution_record.inference_ms or 0.0) / 1000.0
        postprocess_s = (execution_record.postprocessing_ms or 0.0) / 1000.0

        # CPU cores: infer from cpu_usage_pct if available, else default 1
        cpu_cores = 1.0
        cpu_util = 1.0
        if execution_record.cpu_usage_pct is not None:
            cpu_util = min(execution_record.cpu_usage_pct / 100.0, 1.0)

        pre_j = preprocess_s * cpu_cores * tdp * cpu_util
        inf_j = inference_s * cpu_cores * tdp * cpu_util
        post_j = postprocess_s * cpu_cores * tdp * cpu_util
        total_j = pre_j + inf_j + post_j

        # Breakdown percentages
        breakdown: dict[str, float] = {}
        if total_j > 0:
            breakdown = {
                "preprocessing": round(pre_j / total_j * 100.0, 1),
                "inference": round(inf_j / total_j * 100.0, 1),
                "postprocessing": round(post_j / total_j * 100.0, 1),
            }
        else:
            breakdown = {
                "preprocessing": 0.0,
                "inference": 0.0,
                "postprocessing": 0.0,
            }

        return PipelineEnergyEstimate(
            processor=proc,
            preprocessing_joules=round(pre_j, 6),
            inference_joules=round(inf_j, 6),
            postprocessing_joules=round(post_j, 6),
            total_joules=round(total_j, 6),
            total_wh=round(total_j / 3600.0, 8),
            breakdown_percent=breakdown,
        )

    # ------------------------------------------------------------------
    # TOPS/W
    # ------------------------------------------------------------------

    def calculate_tops_per_watt(
        self,
        model_flops: int,
        inference_seconds: float,
        processor: str | None = None,
    ) -> float:
        """Calculate Tera-Operations Per Second Per Watt (TOPS/W).

        TOPS/W = (FLOPs / inference_time) / TDP.  This is the standard
        metric for edge-AI energy efficiency.

        Parameters
        ----------
        model_flops:
            Total FLOPs for one forward pass of the model.
        inference_seconds:
            Inference wall-clock time in seconds.
        processor:
            Processor key.

        Returns
        -------
        float
            TOPS/W value.
        """
        proc = processor or self.reference_processor
        tdp = self._get_tdp(proc)
        if inference_seconds <= 0 or tdp <= 0:
            return 0.0
        ops_per_second = model_flops / inference_seconds
        tops = ops_per_second / 1e12
        tops_per_watt = tops / tdp
        logger.debug(
            "TOPS/W for %s: %.4f (FLOPs=%d, time=%.3fs, TDP=%.1fW)",
            proc,
            tops_per_watt,
            model_flops,
            inference_seconds,
            tdp,
        )
        return round(tops_per_watt, 6)

    # ------------------------------------------------------------------
    # Orbital budget
    # ------------------------------------------------------------------

    def calculate_orbital_budget(
        self,
        energy_per_image_joules: float,
        satellite_type: str = "cubesat_6u",
        images_per_orbit: int = 10,
    ) -> OrbitalBudgetResult:
        """Determine whether a satellite can process *images_per_orbit* images.

        Parameters
        ----------
        energy_per_image_joules:
            Energy needed to process one image (joules).
        satellite_type:
            Key in ``SATELLITE_POWER_BUDGETS``.
        images_per_orbit:
            Desired number of images to process per orbit.

        Returns
        -------
        OrbitalBudgetResult
        """
        budget = self._get_satellite_budget(satellite_type)
        payload_w: float = float(budget["payload_w"])
        orbit_period_min: float = float(budget["orbit_period_min"])
        sunlit_fraction: float = float(budget["sunlit_fraction"])
        battery_wh: float = float(budget["battery_wh"])

        orbit_period_hours = orbit_period_min / 60.0
        available_wh = payload_w * orbit_period_hours * sunlit_fraction
        energy_per_image_wh = energy_per_image_joules / 3600.0

        if energy_per_image_wh <= 0:
            max_images = 0
            utilization = 0.0
        else:
            max_images = math.floor(available_wh / energy_per_image_wh)
            utilization = (
                images_per_orbit * energy_per_image_wh / available_wh * 100.0
                if available_wh > 0
                else 0.0
            )

        feasible = max_images >= 1

        notes_parts: list[str] = []
        if feasible:
            notes_parts.append(
                f"Budget allows up to {max_images} images per orbit on {satellite_type}."
            )
        else:
            notes_parts.append(
                f"Insufficient energy on {satellite_type} to process even 1 image."
            )
        if utilization > 100.0:
            notes_parts.append(
                f"Requested {images_per_orbit} images would require "
                f"{utilization:.1f}% of the available energy."
            )

        return OrbitalBudgetResult(
            satellite_type=satellite_type,
            battery_wh=battery_wh,
            payload_power_w=payload_w,
            energy_per_image_wh=round(energy_per_image_wh, 6),
            max_images_per_orbit=max_images,
            available_energy_wh=round(available_wh, 6),
            utilization_percent=round(utilization, 2),
            feasible=feasible,
            notes=" ".join(notes_parts),
        )

    # ------------------------------------------------------------------
    # Extrapolation
    # ------------------------------------------------------------------

    def extrapolate_to_processor(
        self,
        measured_on: str,
        target_processor: str,
        measured_time_s: float,
        measured_cpu_cores: float,
    ) -> EnergyEstimate:
        """Extrapolate an energy measurement to a different processor.

        This is a coarse estimate — real performance depends on ISA,
        memory bandwidth, accelerator availability, etc.  However it
        gives an order-of-magnitude comparison useful for trade studies.

        The time is assumed to be the same (conservative).  Only the TDP
        changes, so energy scales linearly with the target's TDP.

        Parameters
        ----------
        measured_on:
            Processor where the measurement was taken.
        target_processor:
            Processor to extrapolate to.
        measured_time_s:
            Measured wall-clock time in seconds.
        measured_cpu_cores:
            Number of cores used during the measurement.

        Returns
        -------
        EnergyEstimate
            Estimated energy on *target_processor*.
        """
        target_tdp = self._get_tdp(target_processor)
        energy_j = measured_time_s * measured_cpu_cores * target_tdp
        energy_wh = energy_j / 3600.0

        # For battery percent, use default satellite
        battery_pct = self._battery_percent(energy_wh)

        logger.info(
            "Extrapolation %s -> %s: %.4f J (time=%.3fs, cores=%.1f)",
            measured_on,
            target_processor,
            energy_j,
            measured_time_s,
            measured_cpu_cores,
        )
        return EnergyEstimate(
            processor=target_processor,
            tdp_watts=target_tdp,
            cpu_time_seconds=measured_time_s,
            cpu_cores=measured_cpu_cores,
            energy_joules=round(energy_j, 6),
            energy_wh=round(energy_wh, 8),
            equivalent_battery_percent=round(battery_pct, 4),
        )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def compare_all_processors(
        self,
        cpu_time_seconds: float,
        cpu_cores_used: float,
        cpu_utilization: float = 1.0,
    ) -> list[EnergyEstimate]:
        """Generate a comparison table across all reference processors.

        Parameters
        ----------
        cpu_time_seconds:
            Wall-clock time of the inference.
        cpu_cores_used:
            Number of CPU cores used.
        cpu_utilization:
            Average CPU utilisation fraction (0.0-1.0).

        Returns
        -------
        list[EnergyEstimate]
            One entry per processor, sorted by energy (ascending).
        """
        estimates: list[EnergyEstimate] = []
        for proc_name in PROCESSOR_TDP_WATTS:
            est = self.estimate_inference_energy(
                cpu_time_seconds=cpu_time_seconds,
                cpu_cores_used=cpu_cores_used,
                processor=proc_name,
                cpu_utilization=cpu_utilization,
            )
            estimates.append(est)

        estimates.sort(key=lambda e: e.energy_joules)
        return estimates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tdp(self, processor: str) -> float:
        """Look up TDP for *processor*, raising on unknown names."""
        if processor not in PROCESSOR_TDP_WATTS:
            available = ", ".join(PROCESSOR_TDP_WATTS.keys())
            raise KeyError(
                f"Unknown processor '{processor}'. Available: {available}"
            )
        return PROCESSOR_TDP_WATTS[processor]

    @staticmethod
    def _get_satellite_budget(satellite_type: str) -> dict[str, Any]:
        """Look up power-budget dict for *satellite_type*."""
        if satellite_type not in SATELLITE_POWER_BUDGETS:
            available = ", ".join(SATELLITE_POWER_BUDGETS.keys())
            raise KeyError(
                f"Unknown satellite type '{satellite_type}'. "
                f"Available: {available}"
            )
        return SATELLITE_POWER_BUDGETS[satellite_type]

    def _battery_percent(self, energy_wh: float) -> float:
        """Return *energy_wh* as a percentage of the default satellite battery."""
        budget = SATELLITE_POWER_BUDGETS.get(self.default_satellite)
        if budget is None:
            return 0.0
        battery_wh = float(budget["battery_wh"])
        if battery_wh <= 0:
            return 0.0
        return energy_wh / battery_wh * 100.0
