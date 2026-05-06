"""
Persistence tests for /api/orbital/resilience/* endpoints.

Migration 006_resilience.sql introduced three tables (bitflip_runs,
orbit_sim_runs, drift_alerts) that are written by the simulate-orbit,
drift, and bitflip endpoints respectively.  These tests verify that
each endpoint actually issues the expected INSERT against a mocked DB
with the right argument shape — guarding against silent regressions in
the persistence layer that would otherwise only surface as missing
rows on dashboard 08.

Tier 1 only: DB calls are mocked, so these tests do not require a
running Postgres.  Bitflip is not covered here because it loads a
real .pt YOLO weights file from disk; it has end-to-end coverage via
the production smoke tests instead.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from src.db.queries import (
    INSERT_BITFLIP_RUN,
    INSERT_DRIFT_ALERT,
    INSERT_ORBIT_SIM_RUN,
)

# ====================================================================
# Schema-shape sanity checks (catch breaking changes to the SQL strings)
# ====================================================================


def test_insert_bitflip_run_targets_correct_table():
    assert "INSERT INTO bitflip_runs" in INSERT_BITFLIP_RUN
    # 11 positional parameters per migration 006 schema
    for n in range(1, 12):
        assert f"${n}" in INSERT_BITFLIP_RUN, f"missing placeholder ${n}"
    assert "$12" not in INSERT_BITFLIP_RUN


def test_insert_orbit_sim_run_targets_correct_table_and_returns_id():
    assert "INSERT INTO orbit_sim_runs" in INSERT_ORBIT_SIM_RUN
    assert "RETURNING id" in INSERT_ORBIT_SIM_RUN
    # JSONB cast and DOUBLE PRECISION[] for battery_timeline must be present
    assert "$9::jsonb" in INSERT_ORBIT_SIM_RUN
    for n in range(1, 13):
        assert f"${n}" in INSERT_ORBIT_SIM_RUN, f"missing placeholder ${n}"


def test_insert_drift_alert_targets_correct_table():
    assert "INSERT INTO drift_alerts" in INSERT_DRIFT_ALERT
    for n in range(1, 8):
        assert f"${n}" in INSERT_DRIFT_ALERT, f"missing placeholder ${n}"
    assert "$8" not in INSERT_DRIFT_ALERT


# ====================================================================
# /api/orbital/resilience/simulate-orbit persists to orbit_sim_runs
# ====================================================================


async def test_simulate_orbit_persists_run(client, mock_db):
    """Successful orbit simulation issues INSERT_ORBIT_SIM_RUN with the
    right argument shape (satellite, totals, decision counters, JSONB
    models_used, battery_timeline array, energy_efficiency)."""
    # Return a single fake model for SELECT_ALL_MODELS so DecisionEngine
    # has something to schedule.
    mock_db.fetch = AsyncMock(
        return_value=[
            {
                "id": uuid4(),
                "name": "yolov8n",
                "version": "v1.0",
                "format": "pytorch",
                "file_hash": "0" * 64,
                "size_mb": 6.2,
                "base_model": None,
                "compression_technique": "none",
                "num_params": 3_000_000,
                "input_size": [640, 640],
                "classes": ["vessel"],
            }
        ]
    )
    mock_db.execute = AsyncMock(return_value="INSERT 0 1")

    resp = await client.post(
        "/api/orbital/resilience/simulate-orbit",
        json={"satellite": "small_sat", "num_images": 5},
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["total_images"] == 5
    assert body["processed_images"] >= 0  # decision engine may skip
    assert isinstance(body["battery_timeline"], list)
    assert "final_battery_wh" in body

    # Persistence call must have happened with INSERT_ORBIT_SIM_RUN.
    inserts = [
        c
        for c in mock_db.execute.call_args_list
        if c.args and c.args[0] is INSERT_ORBIT_SIM_RUN
    ]
    assert len(inserts) == 1, "expected exactly one orbit_sim_runs insert"

    args = inserts[0].args
    # Positional binding order matches queries.py:INSERT_ORBIT_SIM_RUN
    # (satellite, total, processed, skipped, fallback,
    #  process_count, fallback_cfar_count, skip_count,
    #  models_used JSON str, battery_timeline list, final_wh, energy_eff)
    assert args[1] == "small_sat"  # satellite
    assert args[2] == 5  # total_images
    assert args[3] >= 0  # processed_images
    assert args[5] + args[6] + args[7] == args[3] + args[4]  # decision sum sanity
    # models_used is serialised to a JSON string for the $9::jsonb cast
    assert isinstance(args[9], str)
    json.loads(args[9])  # must parse
    assert isinstance(args[10], list)  # battery_timeline


async def test_simulate_orbit_no_models_short_circuits(client, mock_db):
    """When models_registry is empty, the endpoint returns a friendly
    message and does NOT insert an orbit_sim_runs row (we have nothing
    meaningful to persist about a simulation that never ran)."""
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.execute = AsyncMock(return_value="INSERT 0 1")

    resp = await client.post(
        "/api/orbital/resilience/simulate-orbit",
        json={"satellite": "cubesat_3u", "num_images": 3},
    )
    assert resp.status_code == 200
    assert "No models registered" in resp.json().get("message", "")

    inserts = [
        c
        for c in mock_db.execute.call_args_list
        if c.args and c.args[0] is INSERT_ORBIT_SIM_RUN
    ]
    assert inserts == []


async def test_simulate_orbit_unknown_satellite_raises_400(client, mock_db):
    mock_db.fetch = AsyncMock(return_value=[])

    resp = await client.post(
        "/api/orbital/resilience/simulate-orbit",
        json={"satellite": "death_star", "num_images": 1},
    )
    assert resp.status_code == 400
    assert "Unknown satellite" in resp.json()["detail"]


# ====================================================================
# /api/orbital/resilience/drift persists to drift_alerts
# ====================================================================


def _fake_execution_row(idx: int, num_detections: int) -> dict:
    """Build a row shape matching what /api/orbital/resilience/drift
    expects from execution_log."""
    return {
        "id": uuid4(),
        "created_at": datetime.now(tz=UTC) - timedelta(hours=idx),
        "image_id": f"S1A_TEST_{idx:03d}",
        "image_hash": "a" * 64,
        "model_name": "yolov8n",
        "model_version": "v1.0",
        "model_hash": "b" * 64,
        "model_size_mb": 6.2,
        "num_detections": num_detections,
        "avg_confidence": 0.7,
        "output_hash": "c" * 64,
        "status": "success",
        "inference_ms": 50.0,
        "peak_ram_mb": 200.0,
        "cpu_usage_pct": 50.0,
    }


async def test_drift_persists_when_drifting(client, mock_db):
    """When detect_drift flags drift, drift_alerts gets a row whose
    is_drifting is True and metric/recommendation/window are populated."""
    # Recent window: high-detection executions; baseline: low-detection.
    rows = [_fake_execution_row(i, 4000) for i in range(10)]
    rows += [_fake_execution_row(i, 2000) for i in range(10, 25)]
    mock_db.fetch = AsyncMock(return_value=rows)
    mock_db.execute = AsyncMock(return_value="INSERT 0 1")

    resp = await client.get("/api/orbital/resilience/drift?window_size=10")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["is_drifting"] is True
    assert body["metric"] in {"num_detections", "avg_confidence", "spatial"}

    inserts = [
        c
        for c in mock_db.execute.call_args_list
        if c.args and c.args[0] is INSERT_DRIFT_ALERT
    ]
    assert len(inserts) == 1, "expected exactly one drift_alerts insert"

    args = inserts[0].args
    # (is_drifting, metric, z_score, recent_mean, historical_mean,
    #  recommendation, window_size)
    assert args[1] is True
    assert isinstance(args[2], str)
    assert args[7] == 10


async def test_drift_persists_even_when_not_drifting(client, mock_db):
    """A clean window must still write a row so the dashboard can show
    'no drift' baselines, not just gaps."""
    rows = [_fake_execution_row(i, 2400) for i in range(25)]
    mock_db.fetch = AsyncMock(return_value=rows)
    mock_db.execute = AsyncMock(return_value="INSERT 0 1")

    resp = await client.get("/api/orbital/resilience/drift?window_size=10")
    assert resp.status_code == 200

    body = resp.json()
    assert body["is_drifting"] is False

    inserts = [
        c
        for c in mock_db.execute.call_args_list
        if c.args and c.args[0] is INSERT_DRIFT_ALERT
    ]
    assert len(inserts) == 1, (
        "drift_alerts insert must happen on every drift check, "
        "drifting or not — guards the 'no drift' history line"
    )
    assert inserts[0].args[1] is False


async def test_drift_insufficient_data_does_not_persist(client, mock_db):
    """With fewer rows than window_size the endpoint short-circuits with
    a status='insufficient_data' message and skips persistence."""
    mock_db.fetch = AsyncMock(
        return_value=[_fake_execution_row(i, 1000) for i in range(3)]
    )
    mock_db.execute = AsyncMock(return_value="INSERT 0 1")

    resp = await client.get("/api/orbital/resilience/drift?window_size=10")
    assert resp.status_code == 200
    assert resp.json()["status"] == "insufficient_data"

    inserts = [
        c
        for c in mock_db.execute.call_args_list
        if c.args and c.args[0] is INSERT_DRIFT_ALERT
    ]
    assert inserts == []
