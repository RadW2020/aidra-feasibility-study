"""POST /api/pipeline/preview and the preflight-guarded trigger."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import src.api.pipeline as pipeline_mod
from tests.test_api.agent_helpers import model_row, route_db
from tests.test_api.conftest import FakeRecord

YOLO = "vesseltracker-sar-yolov8"


def _idle():
    pipeline_mod._pipeline_state.update(running=False, current_profile=None, progress=None, current_execution_id=None)


def _engine():
    return patch("src.api.pipeline._get_engine", return_value=MagicMock())


def _registry(*rows):
    return {"FROM models_registry": list(rows)}


async def test_preview_resolves_defaults_and_estimates(client, mock_db):
    _idle()
    route_db(
        mock_db,
        fetch=_registry(model_row(YOLO, "v1.0"), model_row(YOLO, "int8-static")),
        fetchrow={"PERCENTILE_CONT": FakeRecord(p50_ms=3_126_000.0, max_ms=3_200_000.0, runs=4)},
    )
    with patch("src.main.get_engine", return_value=MagicMock()):
        resp = await client.post("/api/pipeline/preview", json={"zone": "gibraltar", "model": YOLO, "profile": "ground"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["resolved_request"]["model_version"] == "v1.0"  # default model -> Settings.default_model_version
    assert data["resolved_request"]["thresholds_from"].startswith("Settings")
    assert data["estimate"]["duration_minutes_p50"] == 52.1
    assert data["profile"]["memory_limit_mb"] == 24576


async def test_preview_explains_a_tipcue_zone_and_never_starts(client, mock_db):
    _idle()
    route_db(mock_db, fetch=_registry(model_row(YOLO, "v1.0")))
    with patch("src.main.get_engine", return_value=MagicMock()):
        data = (await client.post("/api/pipeline/preview", json={"zone": "gibraltar_strait"})).json()
    assert data["ok"] is False
    issue = data["blocking_issues"][0]
    assert issue["code"] == "unknown_zone"
    assert "gibraltar" in issue["valid_values"]
    assert "search zone" in issue["hint"]
    assert pipeline_mod._pipeline_state["running"] is False


async def test_preview_reports_engine_unavailable_instead_of_raising(client, mock_db):
    _idle()
    with patch("src.main.get_engine", return_value=None):
        data = (await client.post("/api/pipeline/preview", json={})).json()
    assert any(i["code"] == "engine_unavailable" for i in data["blocking_issues"])


async def test_trigger_refuses_bad_zone_before_issuing_an_id(client, mock_db):
    _idle()
    route_db(mock_db, fetch=_registry(model_row(YOLO, "v1.0")))
    with _engine():
        resp = await client.post("/api/pipeline/trigger", json={"zone": "atlantis"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "unknown_zone"
    assert "execution_id" not in body
    assert pipeline_mod._pipeline_state["running"] is False


async def test_trigger_ambiguous_model_lists_versions(client, mock_db):
    _idle()
    route_db(mock_db, fetch=_registry(model_row("other-sar", "v1.0"), model_row("other-sar", "int8-static")))
    with _engine():
        resp = await client.post("/api/pipeline/trigger", json={"model": "other-sar"})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "ambiguous_model"
    assert err["valid_values"] == ["v1.0", "int8-static"]


async def test_trigger_blocks_coco_model_on_sar(client, mock_db):
    _idle()
    route_db(mock_db, fetch=_registry(model_row("yolov8n", "v1.0", classes=("person", "boat"))))
    with _engine():
        resp = await client.post("/api/pipeline/trigger", json={"model": "yolov8n", "model_version": "v1.0"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "model_not_sar_compatible"


async def test_trigger_rejected_variant_starts_with_a_warning(client, mock_db):
    _idle()
    route_db(mock_db, fetch=_registry(model_row(YOLO, "int8-dynamic", status="rejected", reason="x4.5 detections")))
    try:
        with _engine(), patch.object(pipeline_mod, "_run_pipeline_background"):
            resp = await client.post("/api/pipeline/trigger", json={"model": YOLO, "model_version": "int8-dynamic"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["warnings"][0]["code"] == "model_rejected"
        assert data["poll"]["url"] == f"/api/executions/{data['execution_id']}"
    finally:
        _idle()


async def test_preview_warns_when_the_same_configuration_last_ran_out_of_memory(client, mock_db):
    _idle()
    route_db(
        mock_db,
        fetch=_registry(model_row(YOLO, "int8-static")),
        fetchrow={"status IN ('success', 'error', 'failed', 'invalid')": FakeRecord(
            id=uuid4(), created_at=datetime.now(tz=UTC), status="error", peak_ram_mb=2650.0,
            error_message="MemoryError: memory budget exceeded: peak RSS 2650 MB > 2048 MB budget of profile 'sat-mid'",
        )},
    )
    with patch("src.main.get_engine", return_value=MagicMock()):
        data = (await client.post("/api/pipeline/preview",
                                  json={"model": YOLO, "model_version": "int8-static", "profile": "sat-mid"})).json()
    assert data["ok"] is True  # a warning, not a block: re-measuring the limit is legitimate
    warning = next(w for w in data["warnings"] if w["code"] == "previous_run_exceeded_memory_budget")
    assert "2650" in warning["message"] and "2048" in warning["message"]


async def test_trigger_sees_scheduled_run_in_flight(client, mock_db):
    """The in-memory flag is idle, but execution_log has a running scheduled scan."""
    _idle()
    live = FakeRecord(id=uuid4(), created_at=datetime.now(tz=UTC), status="running", trigger_type="scheduled",
                      constraint_profile="ground", model_name=YOLO, model_version="v1.0", search_zone="gibraltar")
    route_db(mock_db, fetch={"FROM models_registry": [model_row(YOLO, "v1.0")], "status IN ('pending', 'running')": [live]})
    with _engine():
        resp = await client.post("/api/pipeline/trigger", json={})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "pipeline_busy"
    assert body["error"]["retryable"] is True
    assert str(live["id"]) in body["detail"]
    assert "already running" in body["detail"].lower()


async def test_status_reports_db_in_flight_as_busy(client, mock_db):
    _idle()
    live = FakeRecord(id=uuid4(), created_at=datetime.now(tz=UTC), status="running", trigger_type="cue",
                      constraint_profile="ground", model_name=YOLO, model_version="v1.0", search_zone="gibraltar")
    route_db(mock_db, fetch={"status IN ('pending', 'running')": [live]})
    data = (await client.get("/api/pipeline/status")).json()
    assert data["running"] is False  # legacy field: API-triggered runs only
    assert data["busy"] is True
    assert data["in_flight"][0]["trigger_type"] == "cue"
