"""
Tests for the orbital energy profiler module.

Covers:
- Energy estimation (positive values, scaling with time)
- TOPS/W calculation
- Orbital budget (cubesat, medium satellite)
- Comparison across all processors
"""

from __future__ import annotations

import pytest

from src.orbital.energy import EnergyProfiler
from src.orbital.orbit_params import PROCESSOR_TDP_WATTS

# ====================================================================
# Energy estimation
# ====================================================================


class TestEnergyEstimate:
    """Tests for estimate_inference_energy."""

    def test_energy_estimate_positive(self):
        """Energy must be positive for any non-zero CPU time."""
        profiler = EnergyProfiler(reference_processor="oci_arm_a1")
        estimate = profiler.estimate_inference_energy(
            cpu_time_seconds=0.15,
            cpu_cores_used=1.0,
        )

        assert estimate.energy_joules > 0
        assert estimate.energy_wh > 0
        assert estimate.equivalent_battery_percent > 0
        assert estimate.processor == "oci_arm_a1"

    def test_energy_scales_with_time(self):
        """Doubling the CPU time must double the energy consumption."""
        profiler = EnergyProfiler(reference_processor="oci_arm_a1")

        est_short = profiler.estimate_inference_energy(
            cpu_time_seconds=0.1,
            cpu_cores_used=1.0,
        )
        est_long = profiler.estimate_inference_energy(
            cpu_time_seconds=0.2,
            cpu_cores_used=1.0,
        )

        assert est_long.energy_joules > est_short.energy_joules
        # Should be approximately 2x (within floating point tolerance)
        ratio = est_long.energy_joules / est_short.energy_joules
        assert ratio == pytest.approx(2.0, rel=1e-4)

    def test_energy_scales_with_cores(self):
        """Using more cores must increase energy proportionally."""
        profiler = EnergyProfiler()
        est_1core = profiler.estimate_inference_energy(
            cpu_time_seconds=0.1, cpu_cores_used=1.0
        )
        est_2core = profiler.estimate_inference_energy(
            cpu_time_seconds=0.1, cpu_cores_used=2.0
        )
        assert est_2core.energy_joules == pytest.approx(
            est_1core.energy_joules * 2.0, rel=1e-4
        )

    def test_energy_utilization_factor(self):
        """50% CPU utilization should halve the energy."""
        profiler = EnergyProfiler()
        est_full = profiler.estimate_inference_energy(
            cpu_time_seconds=1.0, cpu_cores_used=1.0, cpu_utilization=1.0
        )
        est_half = profiler.estimate_inference_energy(
            cpu_time_seconds=1.0, cpu_cores_used=1.0, cpu_utilization=0.5
        )
        assert est_half.energy_joules == pytest.approx(
            est_full.energy_joules * 0.5, rel=1e-4
        )


# ====================================================================
# TOPS/W
# ====================================================================


class TestTOPSPerWatt:
    """Tests for calculate_tops_per_watt."""

    def test_tops_per_watt_positive(self):
        """TOPS/W must be positive for valid inputs."""
        profiler = EnergyProfiler(reference_processor="oci_arm_a1")

        # YOLOv8n has ~8.7 GFLOPs = 8.7e9 FLOPs
        tops_w = profiler.calculate_tops_per_watt(
            model_flops=int(8.7e9),
            inference_seconds=0.05,
            processor="oci_arm_a1",
        )

        assert tops_w > 0

    def test_tops_per_watt_zero_time(self):
        """Zero inference time must return 0 (avoid division by zero)."""
        profiler = EnergyProfiler()
        tops_w = profiler.calculate_tops_per_watt(
            model_flops=int(1e9),
            inference_seconds=0.0,
        )
        assert tops_w == 0.0


# ====================================================================
# Orbital budget
# ====================================================================


class TestOrbitalBudget:
    """Tests for calculate_orbital_budget."""

    def test_orbital_budget_cubesat(self):
        """A CubeSat 6U has limited capacity; budget must reflect that."""
        profiler = EnergyProfiler(reference_processor="oci_arm_a1")

        # Typical inference: 0.15s * 3W = 0.45 J per image
        budget = profiler.calculate_orbital_budget(
            energy_per_image_joules=0.45,
            satellite_type="cubesat_6u",
            images_per_orbit=10,
        )

        assert budget.satellite_type == "cubesat_6u"
        assert budget.battery_wh == 60.0
        assert budget.payload_power_w == 5.0
        assert budget.energy_per_image_wh > 0
        assert budget.max_images_per_orbit >= 1
        assert budget.feasible is True
        # CubeSat should have limited capacity
        assert budget.max_images_per_orbit < 100000

    def test_orbital_budget_medium_sat(self):
        """A medium satellite has significantly more capacity than a CubeSat."""
        profiler = EnergyProfiler()

        budget_cube = profiler.calculate_orbital_budget(
            energy_per_image_joules=0.45,
            satellite_type="cubesat_6u",
        )
        budget_med = profiler.calculate_orbital_budget(
            energy_per_image_joules=0.45,
            satellite_type="medium_sat",
        )

        assert budget_med.max_images_per_orbit > budget_cube.max_images_per_orbit
        assert budget_med.available_energy_wh > budget_cube.available_energy_wh
        assert budget_med.battery_wh > budget_cube.battery_wh

    def test_orbital_budget_infeasible(self):
        """If energy per image exceeds available energy, feasible=False."""
        profiler = EnergyProfiler()

        budget = profiler.calculate_orbital_budget(
            energy_per_image_joules=1e9,  # absurdly high
            satellite_type="cubesat_3u",
        )

        assert budget.feasible is False
        assert budget.max_images_per_orbit == 0


# ====================================================================
# Compare all processors
# ====================================================================


class TestCompareAllProcessors:
    """Tests for compare_all_processors."""

    def test_compare_all_processors(self):
        """Must return one entry per processor, sorted by energy."""
        profiler = EnergyProfiler(reference_processor="oci_arm_a1")

        estimates = profiler.compare_all_processors(
            cpu_time_seconds=0.1,
            cpu_cores_used=1.0,
        )

        assert len(estimates) == len(PROCESSOR_TDP_WATTS)

        # Verify sorted by energy ascending
        energies = [e.energy_joules for e in estimates]
        assert energies == sorted(energies)

        # Each estimate must have the correct processor name
        processor_names = {e.processor for e in estimates}
        assert processor_names == set(PROCESSOR_TDP_WATTS.keys())

    def test_compare_returns_8_entries(self):
        """There are exactly 8 reference processors defined."""
        profiler = EnergyProfiler()
        estimates = profiler.compare_all_processors(
            cpu_time_seconds=0.15, cpu_cores_used=1.0
        )
        assert len(estimates) == 8
