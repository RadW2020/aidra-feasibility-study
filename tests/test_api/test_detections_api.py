"""
Integration tests for /api/detections endpoints.

Tier 1: Mocked DB -- validates routing, pagination, 404 handling,
and profile filtering via mocked return values.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

# ------------------------------------------------------------------
# test_detections_list_empty
# ------------------------------------------------------------------


async def test_detections_list_empty(client, mock_db):
    """GET /api/detections returns an empty list when no data exists."""
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get("/api/detections")
    assert resp.status_code == 200

    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["limit"] == 50  # default
    assert data["offset"] == 0


# ------------------------------------------------------------------
# test_detections_list_pagination
# ------------------------------------------------------------------


async def test_detections_list_pagination(client, mock_db):
    """limit and offset query parameters are forwarded correctly."""
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get("/api/detections", params={"limit": 10, "offset": 20})
    assert resp.status_code == 200

    data = resp.json()
    assert data["limit"] == 10
    assert data["offset"] == 20

    # Verify fetch was called with the right limit/offset.
    # Call signature: query, profile, model, min_conf, dt_from, dt_to,
    # bbox, limit, offset, on_land, cluster_anomaly, quality_verdict
    # -> args[7]/args[8].
    call_args = mock_db.fetch.call_args
    positional = call_args[0]
    assert positional[7] == 10   # limit
    assert positional[8] == 20   # offset


# ------------------------------------------------------------------
# test_detections_list_with_results
# ------------------------------------------------------------------


async def test_detections_list_with_results(client, mock_db, fake_detection_row):
    """GET /api/detections returns properly serialized detection records."""
    mock_db.fetch = AsyncMock(return_value=[fake_detection_row])
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get("/api/detections")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1

    item = data["items"][0]
    assert item["confidence"] == 0.85
    assert item["source"] == "fused"
    assert item["quality_verdict"] == "valid_sea_target"
    assert item["longitude"] == -5.5
    assert item["latitude"] == 36.0


# ------------------------------------------------------------------
# test_detections_invalid_id_404
# ------------------------------------------------------------------


async def test_detections_invalid_id_404(client, mock_db):
    """GET /api/detections/{bad-uuid} returns 404 when not found."""
    mock_db.fetchrow = AsyncMock(return_value=None)

    bad_id = uuid4()
    resp = await client.get(f"/api/detections/{bad_id}")
    assert resp.status_code == 404

    data = resp.json()
    assert "not found" in data["detail"].lower()


# ------------------------------------------------------------------
# test_detections_invalid_uuid_format_422
# ------------------------------------------------------------------


async def test_detections_invalid_uuid_format_422(client, mock_db):
    """GET /api/detections/{not-a-uuid} returns 422 validation error."""
    resp = await client.get("/api/detections/not-a-uuid")
    assert resp.status_code == 422


# ------------------------------------------------------------------
# test_detections_filter_by_profile
# ------------------------------------------------------------------


async def test_detections_filter_by_profile(client, mock_db):
    """profile query param is forwarded as $1 to the SQL query."""
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get("/api/detections", params={"profile": "sat-low"})
    assert resp.status_code == 200

    # Verify the profile parameter was passed (first positional arg after query)
    call_args = mock_db.fetch.call_args[0]
    assert call_args[1] == "sat-low"


# ------------------------------------------------------------------
# test_detections_filter_by_min_confidence
# ------------------------------------------------------------------


async def test_detections_filter_by_min_confidence(client, mock_db):
    """min_confidence query param is forwarded to the SQL query."""
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get(
        "/api/detections", params={"min_confidence": 0.8}
    )
    assert resp.status_code == 200

    call_args = mock_db.fetch.call_args[0]
    # $3 position is min_confidence
    assert call_args[3] == 0.8


async def test_detections_filter_by_quality_verdict(client, mock_db):
    """quality_verdict query param is forwarded after anomaly flags."""
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get(
        "/api/detections", params={"quality_verdict": "valid_sea_target"}
    )
    assert resp.status_code == 200

    call_args = mock_db.fetch.call_args[0]
    count_args = mock_db.fetchval.call_args[0]
    assert call_args[11] == "valid_sea_target"
    assert count_args[9] == "valid_sea_target"


# ------------------------------------------------------------------
# test_detections_invalid_bbox_400
# ------------------------------------------------------------------


async def test_detections_invalid_bbox_400(client, mock_db):
    """Invalid bbox format returns 400."""
    resp = await client.get(
        "/api/detections", params={"bbox": "not,valid"}
    )
    assert resp.status_code == 400
    assert "bbox" in resp.json()["detail"].lower()


# ------------------------------------------------------------------
# test_detections_get_single_found
# ------------------------------------------------------------------


async def test_detections_get_single_found(client, mock_db, fake_detection_row):
    """GET /api/detections/{id} returns the record when it exists."""
    det_id = fake_detection_row["id"]
    mock_db.fetchrow = AsyncMock(return_value=fake_detection_row)

    resp = await client.get(f"/api/detections/{det_id}")
    assert resp.status_code == 200

    data = resp.json()
    assert data["confidence"] == 0.85
