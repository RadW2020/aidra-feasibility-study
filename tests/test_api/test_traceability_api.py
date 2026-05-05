"""
Tests for /api/traceability endpoints.

Tier 1 (mocked DB/bundler): validates routing, response shape, and error
handling for GET /traceability/{id} and POST /traceability/bundle.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

# ---------------------------------------------------------------------------
# GET /traceability/{execution_id}
# ---------------------------------------------------------------------------


async def test_get_traceability_returns_full_chain(
    client, mock_db, fake_execution_row, fake_detection_row
):
    exec_id = fake_execution_row["id"]
    mock_db.fetchrow = AsyncMock(return_value=fake_execution_row)
    mock_db.fetch = AsyncMock(return_value=[fake_detection_row])

    resp = await client.get(f"/api/traceability/{exec_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "execution" in body
    assert "detections" in body
    assert "upstream_cue" in body
    assert "downstream_cues" in body
    assert body["execution"]["image_hash"] == "abc123def456"
    assert len(body["detections"]) == 1


async def test_get_traceability_not_found_returns_404(client, mock_db):
    mock_db.fetchrow = AsyncMock(return_value=None)
    resp = await client.get(f"/api/traceability/{uuid4()}")
    assert resp.status_code == 404


async def test_get_traceability_includes_traceability_fields(
    client, mock_db, fake_execution_row
):
    mock_db.fetchrow = AsyncMock(return_value=fake_execution_row)
    mock_db.fetch = AsyncMock(return_value=[])

    resp = await client.get(f"/api/traceability/{fake_execution_row['id']}")
    assert resp.status_code == 200
    execution = resp.json()["execution"]
    for field in ("image_hash", "model_hash", "output_hash", "input_params_hash",
                  "commit_sha", "pipeline_version", "status"):
        assert field in execution, f"missing traceability field: {field}"


# ---------------------------------------------------------------------------
# POST /traceability/bundle
# ---------------------------------------------------------------------------


async def test_post_bundle_returns_path_and_status(client, mock_db, tmp_path):
    fake_bundle_path = tmp_path / "d3-20260505T120000Z.tar.gz"
    fake_bundle_path.touch()

    with patch(
        "src.traceability.bundler.EvidenceBundler"
    ) as MockBundler:
        instance = MockBundler.return_value
        instance.build = AsyncMock(return_value=fake_bundle_path)

        resp = await client.post(
            "/api/traceability/bundle",
            json={"out_dir": str(tmp_path), "archive": True},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "bundle_path" in body


async def test_post_bundle_passes_filters_to_bundler(client, mock_db, tmp_path):
    fake_bundle_path = tmp_path / "d3-filtered.tar.gz"
    fake_bundle_path.touch()

    with patch(
        "src.traceability.bundler.EvidenceBundler"
    ) as MockBundler:
        instance = MockBundler.return_value
        instance.build = AsyncMock(return_value=fake_bundle_path)

        resp = await client.post(
            "/api/traceability/bundle",
            json={
                "out_dir": str(tmp_path),
                "zone": "gibraltar",
                "model": "vesseltracker-sar-yolov8",
                "profile": "ground",
                "date_from": "2026-04-01T00:00:00Z",
                "date_to": "2026-05-01T00:00:00Z",
            },
        )

    assert resp.status_code == 200
    build_call = instance.build.call_args
    assert build_call.kwargs["zone"] == "gibraltar"
    assert build_call.kwargs["model_name"] == "vesseltracker-sar-yolov8"
    assert build_call.kwargs["constraint_profile"] == "ground"
    assert build_call.kwargs["date_from"] is not None
    assert build_call.kwargs["date_to"] is not None


async def test_post_bundle_propagates_bundler_error_as_500(client, mock_db, tmp_path):
    with patch(
        "src.traceability.bundler.EvidenceBundler"
    ) as MockBundler:
        instance = MockBundler.return_value
        instance.build = AsyncMock(side_effect=RuntimeError("disk full"))

        resp = await client.post(
            "/api/traceability/bundle",
            json={"out_dir": str(tmp_path)},
        )

    assert resp.status_code == 500
    assert "disk full" in resp.json()["detail"]
