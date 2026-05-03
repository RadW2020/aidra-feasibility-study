"""Unit tests for :class:`src.profiles.throttle.CPUThrottle`.

The throttle is a soft-enforcement primitive used by sub-core
constraint profiles (sat-low, sat-extreme).  These tests exercise its
contract without depending on the rest of the pipeline.
"""

from __future__ import annotations

import time

import pytest

from src.profiles.throttle import CPUThrottle


# Burn ~``duration_s`` seconds of CPU time in a tight loop.  Used as a
# stand-in for an inference work unit.  Multiplications keep the JIT
# / interpreter from optimising the loop away.
def _busy(duration_s: float) -> float:
    end = time.monotonic() + duration_s
    x = 0.0
    while time.monotonic() < end:
        for _ in range(10_000):
            x = x * 1.000001 + 1.0
    return x


# ---------------------------------------------------------------------------
# Construction / disabled mode
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_target_one_disables_throttle(self) -> None:
        throttle = CPUThrottle(1.0)
        assert throttle.enabled is False

    def test_target_above_one_clamped_to_one(self) -> None:
        throttle = CPUThrottle(2.5)
        assert throttle.enabled is False
        assert throttle.target == 1.0

    def test_none_target_treated_as_full_speed(self) -> None:
        throttle = CPUThrottle(None)
        assert throttle.enabled is False

    def test_subcore_target_enables_throttle(self) -> None:
        throttle = CPUThrottle(0.25)
        assert throttle.enabled is True
        assert throttle.target == pytest.approx(0.25)

    def test_target_below_floor_clamped_up(self) -> None:
        throttle = CPUThrottle(0.001)
        assert throttle.enabled is True
        assert throttle.target >= 0.05  # floor enforced


# ---------------------------------------------------------------------------
# Tick behaviour
# ---------------------------------------------------------------------------


class TestTickNoOp:
    def test_disabled_throttle_does_not_sleep(self) -> None:
        throttle = CPUThrottle(1.0)
        wall_before = time.monotonic()
        for _ in range(100):
            throttle.tick()
        wall_after = time.monotonic()
        # 100 ticks on a no-op throttle must take << 100 ms.
        assert wall_after - wall_before < 0.1
        assert throttle.total_sleep_s == 0.0


@pytest.mark.invariant
class TestDutyCycleEnforcement:
    """Empirical: after several work + tick cycles, the wall-clock
    duty cycle should converge near the target fraction.

    Tolerances are intentionally loose because the test runs on
    shared CI hardware and ``time.process_time`` granularity varies
    between Linux kernels.
    """

    @pytest.mark.parametrize("target", [0.5, 0.25])
    def test_observed_fraction_close_to_target(self, target: float) -> None:
        throttle = CPUThrottle(target)
        # Do ~1 s of CPU work spread over 10 work units.
        for _ in range(10):
            _busy(0.10)
            throttle.tick()
        observed = throttle.observed_fraction()
        # Allow generous tolerance: throttle is soft, OS scheduling
        # noise is real.  Assert the throttle pulled the fraction
        # meaningfully down from ~1.0 and within a band around target.
        assert observed < 0.95, (
            f"throttle target={target} did not reduce duty cycle "
            f"(observed={observed:.2f})"
        )
        # Within +/- 50 % relative band of the target — wide enough
        # to survive flaky CI but tight enough to fail if the
        # throttle is broken.
        assert observed <= target * 1.5 + 0.10
        assert observed >= target * 0.5 - 0.05

    def test_total_sleep_grows_with_lower_target(self) -> None:
        light = CPUThrottle(0.5)
        heavy = CPUThrottle(0.1)
        for t in (light, heavy):
            for _ in range(10):
                _busy(0.05)
                t.tick()
        assert heavy.total_sleep_s > light.total_sleep_s


class TestReset:
    def test_reset_clears_total_sleep(self) -> None:
        throttle = CPUThrottle(0.25)
        _busy(0.05)
        throttle.tick()
        assert throttle.total_sleep_s > 0.0
        throttle.reset()
        assert throttle.total_sleep_s == 0.0
