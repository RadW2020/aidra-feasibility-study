"""Tests for ResourceCollector enhancements (palanca L6).

Cover:
- ``_percentile`` helper (linear interpolation, edge cases).
- ``ResourceMetrics`` carries ``latency_p95_ms`` /
  ``energy_estimated_j`` / ``energy_method``.
- ``ResourceCollector.attach_energy_estimate`` derives Joules from
  the profile TDP.
"""

from __future__ import annotations

import pytest

from src.profiles.definitions import PROFILES
from src.profiles.metrics_collector import (
    ResourceCollector,
    ResourceMetrics,
    _percentile,
)


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 95.0) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50.0) == 42.0
        assert _percentile([42.0], 99.0) == 42.0

    def test_p50_matches_median(self):
        # Even-length linear-interpolated median.
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5

    def test_p95_close_to_max(self):
        values = list(range(1, 101))  # 1..100
        # Linear-interp p95 of 1..100 is 95.05.
        assert _percentile(values, 95.0) == pytest.approx(95.05, abs=0.05)

    def test_clamped_to_valid_range(self):
        values = [10.0, 20.0, 30.0]
        assert _percentile(values, -5.0) == 10.0
        assert _percentile(values, 200.0) == 30.0


class TestResourceMetricsFields:
    def test_default_values(self):
        m = ResourceMetrics()
        assert m.latency_p95_ms == 0.0
        assert m.energy_estimated_j is None
        assert m.energy_method == "unavailable"


class TestEnergyEstimate:
    """``attach_energy_estimate`` derives Joules from profile TDP and
    average CPU fraction. Profiles without ``tdp_watts`` stay
    ``unavailable`` so audits never confuse zero with missing."""

    def _metrics(self, avg_cpu: float, duration_ms: float) -> ResourceMetrics:
        return ResourceMetrics(
            duration_ms=duration_ms,
            avg_cpu_percent=avg_cpu,
        )

    def test_ground_profile_has_tdp(self):
        assert PROFILES["ground"].tdp_watts is not None

    def test_attach_for_sat_low_uses_tdp(self):
        # sat-low: cpu_limit=0.5, tdp_watts=2.5 W
        # avg_cpu_percent=50 → cpu_fraction = 50/(100*0.5) = 1.0
        # duration=1 s → energy = 1.0 * 2.5 * 1.0 = 2.5 J
        m = self._metrics(avg_cpu=50.0, duration_ms=1000.0)
        out = ResourceCollector.attach_energy_estimate(m, "sat-low")
        assert out.energy_method == "tdp_x_cpu_fraction"
        assert out.energy_estimated_j == pytest.approx(2.5, abs=1e-3)

    def test_attach_for_sat_high_uses_zynq_tdp(self):
        # sat-high: cpu_limit=2.0, tdp_watts=10 W
        # avg_cpu_percent=200 → cpu_fraction clamped to 1.0
        # duration=2 s → energy = 1.0 * 10 * 2 = 20 J
        m = self._metrics(avg_cpu=200.0, duration_ms=2000.0)
        out = ResourceCollector.attach_energy_estimate(m, "sat-high")
        assert out.energy_method == "tdp_x_cpu_fraction"
        assert out.energy_estimated_j == pytest.approx(20.0, abs=1e-3)

    def test_unknown_profile_marks_unavailable(self):
        m = self._metrics(avg_cpu=50.0, duration_ms=1000.0)
        out = ResourceCollector.attach_energy_estimate(m, "no-such-profile")
        assert out.energy_estimated_j is None
        assert out.energy_method == "unavailable"

    def test_idempotent_when_called_twice(self):
        m = self._metrics(avg_cpu=50.0, duration_ms=1000.0)
        out_a = ResourceCollector.attach_energy_estimate(m, "sat-mid")
        out_b = ResourceCollector.attach_energy_estimate(out_a, "sat-mid")
        assert out_a.energy_estimated_j == out_b.energy_estimated_j


class TestProfileTDPCoverage:
    """All defined profiles declare a TDP so energy is comparable."""

    def test_all_profiles_have_tdp(self):
        for name, profile in PROFILES.items():
            assert profile.tdp_watts is not None, (
                f"Profile '{name}' lacks tdp_watts — energy estimate "
                "would be unavailable for evaluation."
            )
            assert profile.tdp_watts > 0
