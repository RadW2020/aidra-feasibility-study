"""Cue validation/duplicates, detection filters, catalog and config for autonomous clients."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from tests.test_api.agent_helpers import model_row, route_db
from tests.test_api.conftest import FakeRecord

# ---------------------------------------------------------------------------
# POST /api/tasking/cue
# ---------------------------------------------------------------------------


async def test_cue_with_reversed_corners_is_refused_with_a_hint(client, mock_db):
    resp = await client.post("/api/tasking/cue", json={"bbox": [-5.3, 35.8, -5.6, 36.1]})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "invalid_bbox"
    assert "order of the corners" in err["hint"]
    out_of_range = await client.post("/api/tasking/cue", json={"bbox": [35.8, -5.6, 136.1, 95.0]})
    assert "longitude comes first" in out_of_range.json()["error"]["hint"]


async def test_duplicate_live_cue_is_refused_with_existing_id(client, mock_db):
    existing = uuid4()
    route_db(mock_db, fetchrow={"ST_Equals": FakeRecord(id=existing, created_at=datetime.now(tz=UTC),
                                                        status="pending", priority=2)})
    resp = await client.post("/api/tasking/cue", json={"bbox": [-5.6, 35.8, -5.3, 36.1], "zone": "gibraltar_strait"})
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert err["code"] == "duplicate_cue"
    assert err["existing_cue_id"] == str(existing)
    assert mock_db.fetchval.await_count == 0  # nothing inserted


async def test_cue_created_reports_zone_position_and_warning(client, mock_db):
    cue_id = uuid4()
    route_db(mock_db, fetchval={"INSERT INTO tasking_queue": cue_id},
             fetchrow={"AS position": FakeRecord(position=3)})
    resp = await client.post("/api/tasking/cue", json={"bbox": [12.1, 44.6, 12.4, 44.9], "priority": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cue_id"] == str(cue_id) and data["created"] is True
    assert data["queue_position"] == 3
    # No zone -> the processor searches the default zone, which doesn't cover the Adriatic.
    assert data["search_zone"] == "gibraltar"
    assert data["warnings"][0]["code"] == "bbox_outside_search_zone"


async def test_cue_unknown_zone_and_priority(client, mock_db):
    r1 = await client.post("/api/tasking/cue", json={"bbox": [-5.6, 35.8, -5.3, 36.1], "zone": "narnia"})
    assert r1.json()["error"]["code"] == "unknown_zone"
    r2 = await client.post("/api/tasking/cue", json={"bbox": [-5.6, 35.8, -5.3, 36.1], "priority": 99})
    assert r2.json()["error"]["code"] == "invalid_priority"


async def test_queue_status_filter_uses_real_values(client, mock_db):
    resp = await client.get("/api/tasking/queue", params={"status": "executing"})
    assert resp.status_code == 422
    assert "processing" in resp.json()["error"]["valid_values"]


# ---------------------------------------------------------------------------
# GET /api/detections
# ---------------------------------------------------------------------------


async def test_detections_filter_by_execution_and_tier_and_dedup(client, mock_db):
    eid = uuid4()
    resp = await client.get("/api/detections", params={
        "execution_id": str(eid), "tier": "high", "one_run_per_scene": "true", "sort": "recent",
    })
    assert resp.status_code == 200
    select_sql = mock_db.fetch.call_args[0][0]
    count_sql = mock_db.fetchval.call_args[0][0]
    for sql in (select_sql, count_sql):
        assert f"d.execution_id = '{eid}'::uuid" in sql
        assert "d.source = 'fused'" in sql
        assert "DISTINCT ON (x.image_id)" in sql
    assert "ORDER BY d.created_at DESC" in select_sql
    data = resp.json()
    assert data["filters"]["one_run_per_scene"] is True
    assert data["next_offset"] is None


async def test_detections_zone_becomes_bbox(client, mock_db):
    resp = await client.get("/api/detections", params={"zone": "algeciras_port"})
    assert resp.status_code == 200
    bbox_param = mock_db.fetch.call_args[0][6]
    assert '"coordinates"' in bbox_param and "-5.5" in bbox_param
    bad = await client.get("/api/detections", params={"zone": "nowhere"})
    assert bad.status_code == 422
    assert "gibraltar" in bad.json()["error"]["valid_values"]


async def test_detections_unknown_verdict(client, mock_db):
    resp = await client.get("/api/detections", params={"quality_verdict": "vessel"})
    assert resp.status_code == 422
    assert "valid_sea_target" in resp.json()["error"]["valid_values"]


# ---------------------------------------------------------------------------
# Discovery: models, catalog, vocabulary, config
# ---------------------------------------------------------------------------


def _registry_row(name, version, status="active", classes=("ship",), reason=None):
    r = model_row(name, version, status=status, classes=classes, reason=reason)
    r.update(id=uuid4(), file_hash=f"{name}{version}".ljust(64, "0"), size_mb=25.0, base_model=None,
             num_params=None, input_size=[640, 640])
    return r


async def test_models_expose_status_and_gates(client, mock_db):
    route_db(mock_db, fetch={"FROM models_registry": [
        _registry_row("vesseltracker-sar-yolov8", "int8-dynamic", status="rejected", reason="x4.5 detections"),
        _registry_row("yolov8n", "v1.0", classes=("person", "boat")),
    ]})
    models = {m["version"] if m["name"] != "yolov8n" else "coco": m for m in (await client.get("/api/models")).json()}
    assert models["int8-dynamic"]["status"] == "rejected"
    assert models["int8-dynamic"]["rejection_reason"] == "x4.5 detections"
    assert models["coco"]["sar_compatible"] is False


async def test_catalog_maps_tipcue_zones_to_search_zones(client, mock_db):
    data = (await client.get("/api/catalog")).json()
    tipcue = {z["id"]: z for z in data["zones"]["tipcue_zones"]}
    assert tipcue["gibraltar_strait"]["search_zone"] == "gibraltar"
    assert {p["name"] for p in data["profiles"]} == {"ground", "sat-high", "sat-mid", "sat-low", "sat-extreme"}
    assert data["defaults"]["model"]


async def test_config_masks_secrets_and_matches_bundle_hash(client, mock_db, monkeypatch):
    monkeypatch.setenv("AIDRA_API_TOKENS", "agent:run:super-secret")
    monkeypatch.setenv("COPERNICUS_PASSWORD", "hunter2")
    data = (await client.get("/api/config")).json()
    text = str(data)
    assert "super-secret" not in text and "hunter2" not in text
    from src.config import Settings
    from src.traceability.bundler import censored_settings_snapshot

    assert data["settings_hash"] == censored_settings_snapshot(Settings())[2]


async def test_vocabulary_endpoint(client, mock_db):
    data = (await client.get("/api/vocabulary")).json()
    assert {e["value"] for e in data["execution_statuses"]} >= {"pending", "success", "skipped", "failed"}
