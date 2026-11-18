"""Drift check: the z-score is reported even when it stays under the threshold."""

from __future__ import annotations

from types import SimpleNamespace

from src.orbital.decision_engine import DecisionEngine


def _runs(detections: list[int]) -> list[SimpleNamespace]:
    return [SimpleNamespace(num_detections=n, avg_confidence=0.6) for n in detections]


def test_no_drift_still_reports_how_far_recent_runs_moved() -> None:
    # Produccion 2026-11-17: 731 frente a 521 de media y el log decia z = 0.
    engine = DecisionEngine(models=[])
    recent = [731] * 10
    historical = [300, 400, 500, 600, 700, 800, 400, 500, 600, 410]

    result = engine.detect_drift(recent_executions=_runs(recent + historical), window_size=10)

    assert result.is_drifting is False
    assert result.metric == "all"
    assert result.z_score > 1.0
    assert result.recommendation == "continue"


def test_identical_windows_report_zero() -> None:
    engine = DecisionEngine(models=[])

    result = engine.detect_drift(recent_executions=_runs([500] * 20), window_size=10)

    assert result.is_drifting is False
    assert result.z_score == 0.0


def test_orbit_simulation_keeps_the_satellite_battery_capacity() -> None:
    # Produccion 2026-11-18: un small_sat de 300 Wh bajaba a 60 Wh en el
    # segundo paso porque la capacidad se quedaba en el valor por defecto.
    engine = DecisionEngine(models=[])

    result = engine.simulate_orbit_sequence(num_images=10, initial_battery_wh=300.0, solar_recharge_w=14.4)

    steps = result.battery_timeline
    assert steps[0] == 300.0
    assert max(abs(a - b) for a, b in zip(steps, steps[1:], strict=False)) < 60.0
    assert all(level <= 300.0 for level in steps)
