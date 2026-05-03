"""
Tests for the orbital resilience / bit-flip simulation module.

Covers:
- Single bit-flip injection (exactly one weight changes)
- Zero flips leaves weights unchanged
- MTBF estimation (positive value)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.orbital.resilience import BitFlipSimulator

# ====================================================================
# Helpers
# ====================================================================


def _make_sample_weights() -> dict[str, np.ndarray]:
    """Create a small set of model weights for testing."""
    rng = np.random.default_rng(42)
    return {
        "conv1.weight": rng.standard_normal((16, 3, 3, 3)).astype(np.float32),
        "conv1.bias": rng.standard_normal(16).astype(np.float32),
        "conv2.weight": rng.standard_normal((32, 16, 3, 3)).astype(np.float32),
        "conv2.bias": rng.standard_normal(32).astype(np.float32),
    }


# ====================================================================
# Bit-flip injection
# ====================================================================


class TestBitFlipInjection:
    """Tests for inject_bitflips."""

    def test_inject_single_bitflip(self):
        """Injecting 1 bit-flip must change exactly 1 weight value.

        We verify by counting the number of differing elements across
        all layers between the original and corrupted weights.
        """
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        corrupted, records = sim.inject_bitflips(
            num_flips=1,
            rng=np.random.default_rng(123),
        )

        # Exactly 1 record must be produced
        assert len(records) == 1

        # Count differing elements across all layers
        total_diff = 0
        for name in weights:
            diff = np.not_equal(weights[name], corrupted[name])
            total_diff += int(np.sum(diff))

        assert total_diff == 1, (
            f"Expected exactly 1 changed weight, found {total_diff}"
        )

        # The record must contain meaningful information
        rec = records[0]
        assert rec.layer_name in weights
        assert rec.original_value != rec.corrupted_value
        assert 0 <= rec.bit_position <= 31
        assert rec.bit_significance in ("sign", "exponent", "mantissa")

    def test_zero_flips_no_change(self):
        """Injecting 0 bit-flips must leave all weights identical."""
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        corrupted, records = sim.inject_bitflips(num_flips=0)

        assert len(records) == 0

        for name in weights:
            np.testing.assert_array_equal(
                weights[name],
                corrupted[name],
                err_msg=f"Layer {name} changed with 0 flips",
            )

    def test_multiple_flips(self):
        """Injecting N flips must produce N records.

        Note: in rare cases two flips can target the same element, so
        the number of changed elements may be <= N.
        """
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        n_flips = 10
        corrupted, records = sim.inject_bitflips(
            num_flips=n_flips,
            rng=np.random.default_rng(42),
        )

        assert len(records) == n_flips

    def test_targeted_layer(self):
        """When target_layers is specified, flips must only affect those layers."""
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        corrupted, records = sim.inject_bitflips(
            num_flips=5,
            target_layers=["conv1.weight"],
            rng=np.random.default_rng(42),
        )

        for rec in records:
            assert rec.layer_name == "conv1.weight"

        # conv2 layers must be untouched
        np.testing.assert_array_equal(
            weights["conv2.weight"], corrupted["conv2.weight"]
        )
        np.testing.assert_array_equal(
            weights["conv2.bias"], corrupted["conv2.bias"]
        )

    def test_msb_flip_high_impact(self):
        """Flipping the MSB (sign bit) should produce a large change
        in value magnitude (sign reversal).
        """
        weights = {"layer": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        sim = BitFlipSimulator(model_weights=weights)

        corrupted, records = sim.inject_bitflips(
            num_flips=1,
            bit_position="msb",
            rng=np.random.default_rng(42),
        )

        rec = records[0]
        assert rec.bit_position == 31
        assert rec.bit_significance == "sign"
        # Sign flip: the corrupted value should have opposite sign
        assert rec.original_value * rec.corrupted_value < 0


# ====================================================================
# MTBF estimation
# ====================================================================


class TestMTBF:
    """Tests for estimate_mtbf."""

    def test_mtbf_positive(self):
        """MTBF must be a positive number of days for any valid orbit."""
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        mtbf = sim.estimate_mtbf(
            orbit="leo_500",
            shielding_mm_al=1.0,
            critical_threshold=100,
        )

        assert mtbf.estimated_mtbf_days > 0
        assert mtbf.expected_flips_per_day > 0
        assert mtbf.expected_flips_per_orbit > 0
        assert mtbf.model_size_bits > 0
        assert mtbf.seu_rate_per_bit_per_day > 0

    def test_mtbf_shielding_effect(self):
        """More shielding must increase MTBF (lower SEU rate)."""
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        mtbf_thin = sim.estimate_mtbf(
            orbit="sso_700", shielding_mm_al=1.0, critical_threshold=100
        )
        mtbf_thick = sim.estimate_mtbf(
            orbit="sso_700", shielding_mm_al=5.0, critical_threshold=100
        )

        assert mtbf_thick.estimated_mtbf_days > mtbf_thin.estimated_mtbf_days

    def test_mtbf_recommendations(self):
        """MTBF result must include mitigation recommendations."""
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        mtbf = sim.estimate_mtbf(orbit="leo_500", critical_threshold=100)

        assert len(mtbf.mitigation_recommendations) >= 1

    def test_mtbf_invalid_orbit_raises(self):
        """An unknown orbit key must raise KeyError."""
        weights = _make_sample_weights()
        sim = BitFlipSimulator(model_weights=weights)

        with pytest.raises(KeyError, match="not found"):
            sim.estimate_mtbf(orbit="nonexistent_orbit")
