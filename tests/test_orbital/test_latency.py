"""
Tests for the orbital latency simulation module.

Covers:
- OBDP latency vs. without OBDP
- Actionability classification (high, low)
- Comparison across all scenarios
"""

from __future__ import annotations

from src.orbital.latency import OrbitalLatencySimulator

# ====================================================================
# With vs. without OBDP
# ====================================================================


class TestOBDPLatency:
    """Tests comparing latency with and without OBDP."""

    def test_obdp_faster(self):
        """Total latency with OBDP must be less than without OBDP.

        OBDP eliminates ground processing time and the large downlink,
        so total latency is dominated by the orbital wait for contact.
        """
        sim = OrbitalLatencySimulator()

        without = sim.simulate_without_obdp(
            orbit="sso_700",
            image_size_mb=800.0,
        )
        with_obdp = sim.simulate_with_obdp(
            orbit="sso_700",
            inference_ms=150.0,
            result_size_kb=10.0,
        )

        assert with_obdp.total_minutes < without.total_minutes
        assert with_obdp.ground_processing_s == 0.0
        assert with_obdp.onboard_processing_s > 0.0

    def test_without_obdp_includes_ground_processing(self):
        """Without OBDP, ground_processing_s must be positive (image
        is processed on the ground).
        """
        sim = OrbitalLatencySimulator()
        result = sim.simulate_without_obdp()
        assert result.ground_processing_s > 0

    def test_with_obdp_downlink_tiny(self):
        """With OBDP, the downlink time should be a fraction of a second
        for a ~10 KB result on any reasonable link.
        """
        sim = OrbitalLatencySimulator()
        result = sim.simulate_with_obdp(
            inference_ms=100.0,
            result_size_kb=10.0,
            downlink_profile="smallsat_xband",
        )

        assert result.downlink_s < 1.0


# ====================================================================
# Actionability
# ====================================================================


class TestActionability:
    """Tests for the actionability classification."""

    def test_actionability_high(self):
        """Low latency (< ~16 min at 20 knots) must yield high actionability.

        At 20 knots a vessel moves ~0.617 km/min.  To stay < 10 km
        the latency must be < ~16.2 min.
        """
        sim = OrbitalLatencySimulator()
        result = sim.calculate_actionability(
            latency_minutes=5.0,
            vessel_speed_knots=20.0,
        )

        assert result.actionability == "high"
        assert result.distance_moved_km < 10.0
        assert result.search_radius_km < 10.0

    def test_actionability_low(self):
        """High latency (> ~81 min at 20 knots) must yield low actionability.

        At 20 knots, 120 minutes -> ~74 km moved -> actionability "low".
        """
        sim = OrbitalLatencySimulator()
        result = sim.calculate_actionability(
            latency_minutes=120.0,
            vessel_speed_knots=20.0,
        )

        assert result.actionability == "low"
        assert result.distance_moved_km > 50.0

    def test_actionability_medium(self):
        """Intermediate latency should yield medium actionability."""
        sim = OrbitalLatencySimulator()
        # At 20 knots, 30 min -> ~18.5 km -> "medium"
        result = sim.calculate_actionability(
            latency_minutes=30.0,
            vessel_speed_knots=20.0,
        )

        assert result.actionability == "medium"
        assert 10.0 <= result.distance_moved_km <= 50.0


# ====================================================================
# Compare scenarios
# ====================================================================


class TestCompareScenarios:
    """Tests for compare_scenarios."""

    def test_compare_scenarios(self):
        """compare_scenarios must return multiple comparisons covering
        all orbit x downlink x ground chain combinations.
        """
        sim = OrbitalLatencySimulator()
        comparisons = sim.compare_scenarios(
            inference_ms=150.0,
            image_size_mb=800.0,
            result_size_kb=10.0,
        )

        assert len(comparisons) > 1

        # Each comparison should have a positive speedup
        for comp in comparisons:
            assert comp.speedup_factor >= 1.0
            assert comp.time_saved_minutes >= 0.0
            assert comp.actionability_with in ("high", "medium", "low")
            assert comp.actionability_without in ("high", "medium", "low")

    def test_compare_scenarios_sorted_by_speedup(self):
        """Results must be sorted by speedup_factor descending."""
        sim = OrbitalLatencySimulator()
        comparisons = sim.compare_scenarios(
            inference_ms=150.0,
            image_size_mb=800.0,
            result_size_kb=10.0,
        )

        speedups = [c.speedup_factor for c in comparisons]
        assert speedups == sorted(speedups, reverse=True)
