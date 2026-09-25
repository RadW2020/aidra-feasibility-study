"""GET /api/executions and /api/executions/{id}: history and diagnosis for autonomous clients."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from tests.test_api.agent_helpers import route_db
from tests.test_api.conftest import FakeRecord


def _exec_row(**kw):
    now = datetime.now(tz=UTC)
    base = dict(
        id=uuid4(), created_at=now - timedelta(minutes=45), status="success", trigger_type="scheduled",
        triggered_by=None, search_zone="gibraltar", image_id="S1A_X", image_title="S1A_IW_GRDH_X",
        image_sensing_date=now - timedelta(days=1), model_name="vesseltracker-sar-yolov8",
        model_version="v1.0", compression_technique="none", constraint_profile="ground",
        memory_limit_mb=None, num_detections=40, num_valid_targets=12, total_duration_ms=380000.0,
        inference_p50_ms=252.0, inference_p95_ms=371.0, peak_ram_mb=2600.0, error_message=None, notes=None,
        image_hash="a" * 64, model_hash="b" * 64, output_hash="c" * 64, input_params_hash="d" * 64,
        commit_sha="e" * 40, image_size_mb=1700.0, model_format="pytorch", model_size_mb=49.6,
        confidence_threshold=0.25, iou_threshold=0.45, tile_size=640, tile_overlap=64,
    )
    base.update(kw)
    return FakeRecord(base)


async def test_list_includes_failures_with_outcome_and_pagination(client, mock_db):
    oom = _exec_row(
        status="error", constraint_profile="sat-mid", peak_ram_mb=2650.0,
        error_message="MemoryError: memory budget exceeded: peak RSS 2650 MB > 2048 MB budget of profile 'sat-mid'",
    )
    route_db(mock_db, fetch={"FROM execution_log e": [oom]}, fetchval={"COUNT(*)": 3})

    resp = await client.get("/api/executions", params={"status": "error,failed", "limit": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3 and data["next_offset"] == 1
    item = data["items"][0]
    assert item["outcome"]["category"] == "memory_budget_exceeded"
    assert item["memory_budget_mb"] == 2048
    # the status filter is passed to SQL as a list
    assert mock_db.fetch.call_args[0][1] == ["error", "failed"]


async def test_list_rejects_unknown_status_with_valid_values(client, mock_db):
    resp = await client.get("/api/executions", params={"status": "exploded"})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "unknown_status"
    assert "success" in err["valid_values"]


async def test_list_rejects_unknown_profile(client, mock_db):
    resp = await client.get("/api/executions", params={"profile": "sat_mid"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_profile"


async def test_get_execution_diagnosis_breakdown_and_lineage(client, mock_db):
    row = _exec_row()
    eid = row["id"]
    downstream = FakeRecord(
        id=uuid4(), created_at=datetime.now(tz=UTC), status="pending", target_zone="gibraltar_strait",
        priority=2, reason="cluster", execution_id=None, triggered_by=eid, result_status=None,
        confirmed_detections=None,
    )
    route_db(
        mock_db,
        fetchrow={"FROM execution_log": row, "FROM models_registry": FakeRecord(status="active", rejection_reason=None)},
        fetch={
            "GROUP BY source, quality_verdict": [
                FakeRecord(source="fused", quality_verdict="valid_sea_target", n=7),
                FakeRecord(source="cfar", quality_verdict="candidate", n=30),
                FakeRecord(source="cfar", quality_verdict="land_artifact", n=3),
            ],
            "ORDER BY confidence DESC": [FakeRecord(id=uuid4(), confidence=0.94, source="fused",
                                                   quality_verdict="valid_sea_target", on_land=False,
                                                   cluster_anomaly=False, longitude=-5.5, latitude=36.0)],
            "FROM tasking_queue": [downstream],
        },
    )
    resp = await client.get(f"/api/executions/{eid}", params={"top_detections": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"]["category"] == "succeeded"
    assert data["is_terminal"] is True and data["poll_after_seconds"] is None
    assert data["detections"]["by_source"] == {"fused": 7, "cfar": 33}
    assert data["detections"]["by_quality_verdict"]["land_artifact"] == 3
    assert len(data["detections"]["top"]) == 1
    assert data["provenance"]["complete"] is True
    assert data["lineage"]["downstream_cues"][0]["zone"] == "gibraltar_strait"
    assert data["model"]["registry_status"] == "active"


async def test_running_execution_says_when_to_poll(client, mock_db):
    row = _exec_row(status="running", created_at=datetime.now(tz=UTC) - timedelta(minutes=5))
    route_db(mock_db, fetchrow={"FROM execution_log": row})
    data = (await client.get(f"/api/executions/{row['id']}")).json()
    assert data["is_terminal"] is False
    assert data["poll_after_seconds"] == 120
    assert data["outcome"]["category"] == "in_progress"


async def test_unknown_execution_is_a_structured_404(client, mock_db):
    resp = await client.get(f"/api/executions/{uuid4()}")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "execution_not_found"
    assert "not found" in body["detail"]
    assert body["error"]["retryable"] is False


async def test_execution_that_died_before_its_row_is_explained(client, mock_db):
    from src.api.pipeline import record_early_failure

    eid = uuid4()
    record_early_failure(eid, "FileNotFoundError: Model not found: x:None")
    resp = await client.get(f"/api/executions/{eid}")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "execution_rejected_before_start"
    assert "Model not found" in err["message"]
