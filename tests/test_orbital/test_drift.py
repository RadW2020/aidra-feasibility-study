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
