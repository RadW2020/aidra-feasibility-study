"""Drift check: the z-score is reported even when it stays under the threshold."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from src.db.models import ModelInfo
from src.orbital.decision_engine import DecisionEngine, ModelCost


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


def _model(name: str, version: str, size_mb: float) -> ModelInfo:
    return ModelInfo(id=uuid4(), name=name, version=version, format="onnx", file_hash="0" * 64, size_mb=size_mb)


def test_orbit_simulation_charges_the_measured_scene_energy() -> None:
    # Produccion 2026-11-18: el simulador cobraba ~0.0025 Wh por imagen (el
    # coste de un tile) y la bateria no bajaba nunca. Con el coste medido por
    # escena la bateria baja en eclipse y se recupera al sol.
    fp32 = _model("vesseltracker-sar-yolov8", "v1.0", 49.62)
    int8 = _model("vesseltracker-sar-yolov8", "int8-static", 25.31)
    engine = DecisionEngine(
        models=[fp32, int8],
        model_costs={
            DecisionEngine.cost_key(fp32): ModelCost(energy_wh=3.11, latency_ms=3_193_000),
            DecisionEngine.cost_key(int8): ModelCost(energy_wh=1.19, latency_ms=401_000),
        },
    )

    result = engine.simulate_orbit_sequence(
        num_images=30,
        initial_battery_wh=300.0,
        solar_recharge_w=14.4,
        orbit_period_min=100,
    )

    # Same name, two versions: the report must say which one ran.
    assert result.models_used == {"vesseltracker-sar-yolov8:int8-static": 30}
    assert all(d.estimated_energy_wh == 1.19 for d in result.decisions)
    assert min(result.battery_timeline) < 299.0
    assert max(result.battery_timeline) == 300.0


def test_orbit_simulation_skips_once_the_reserve_is_reached() -> None:
    model = _model("vesseltracker-sar-yolov8", "v1.0", 49.62)
    engine = DecisionEngine(
        models=[model],
        model_costs={DecisionEngine.cost_key(model): ModelCost(energy_wh=3.11, latency_ms=3_193_000)},
    )

    # CubeSat 3U: 30 Wh, 1.08 W of recharge. 3.11 Wh per scene drains it.
    result = engine.simulate_orbit_sequence(num_images=30, initial_battery_wh=30.0, solar_recharge_w=1.08)

    assert result.skipped_images > 0
    assert result.processed_images > 0
    assert min(result.battery_timeline) >= 0.0


def test_unmeasured_model_keeps_its_plain_name_and_heuristic_cost() -> None:
    engine = DecisionEngine(models=[_model("yolov8n-sar", "v1", 6.2)])

    decision = engine.decide_model(available_cpu=1.0, available_ram_mb=1024, available_energy_wh=60.0)

    assert decision.selected_model == "yolov8n-sar"
    assert decision.estimated_energy_wh == 0.00004
