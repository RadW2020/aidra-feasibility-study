"""
Integration tests for /api/tasking endpoints.

Tier 1: Mocked DB -- validates cue creation, queue listing, and validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

# ------------------------------------------------------------------
# test_create_cue_validation
# ------------------------------------------------------------------


async def test_create_cue_validation(client, mock_db):
    """POST /api/tasking/cue with valid bbox creates a cue entry."""
    cue_id = uuid4()
    mock_db.fetchval = AsyncMock(return_value=cue_id)

    resp = await client.post(
        "/api/tasking/cue",
        json={
            "bbox": [-5.8, 35.7, -5.2, 36.2],
            "priority": 3,
            "reason": "suspicious_activity",
            "zone": "gibraltar",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "cue_id" in data
    assert data["cue_id"] == str(cue_id)


# ------------------------------------------------------------------
# test_create_cue_default_values
# ------------------------------------------------------------------


async def test_create_cue_default_values(client, mock_db):
    """POST /api/tasking/cue with only bbox uses default priority and reason."""
    cue_id = uuid4()
    mock_db.fetchval = AsyncMock(return_value=cue_id)

    resp = await client.post(
        "/api/tasking/cue",
        json={"bbox": [1.0, 2.0, 3.0, 4.0]},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "cue_id" in data


# ------------------------------------------------------------------
# test_create_cue_invalid_bbox_empty
# ------------------------------------------------------------------


async def test_create_cue_invalid_bbox_empty(client, mock_db):
    """POST /api/tasking/cue with empty bbox returns 400."""
    resp = await client.post(
        "/api/tasking/cue",
        json={"bbox": []},
    )
    assert resp.status_code == 400
    assert "bbox" in resp.json()["detail"].lower()


# ------------------------------------------------------------------
# test_create_cue_invalid_bbox_wrong_length
# ------------------------------------------------------------------


async def test_create_cue_invalid_bbox_wrong_length(client, mock_db):
    """POST /api/tasking/cue with bbox of wrong length returns 400."""
    resp = await client.post(
        "/api/tasking/cue",
        json={"bbox": [1.0, 2.0, 3.0]},
    )
    assert resp.status_code == 400


# ------------------------------------------------------------------
# test_queue_empty
# ------------------------------------------------------------------


async def test_queue_empty(client, mock_db):
    """GET /api/tasking/queue returns empty list when no cues exist."""
    mock_db.fetch = AsyncMock(return_value=[])

    resp = await client.get("/api/tasking/queue")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 0


# ------------------------------------------------------------------
# test_queue_with_results
# ------------------------------------------------------------------


async def test_queue_with_results(client, mock_db, fake_tasking_row):
    """GET /api/tasking/queue returns properly serialized tasking entries."""
    mock_db.fetch = AsyncMock(return_value=[fake_tasking_row])

    resp = await client.get("/api/tasking/queue")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data) == 1

    entry = data[0]
    assert entry["status"] == "pending"
    assert entry["trigger_type"] == "cue"
    assert entry["priority"] == 1
    assert entry["target_zone"] == "gibraltar"


# ------------------------------------------------------------------
# test_queue_filter_by_status
# ------------------------------------------------------------------


async def test_queue_filter_by_status(client, mock_db):
    """GET /api/tasking/queue?status=pending forwards the filter."""
    mock_db.fetch = AsyncMock(return_value=[])

    resp = await client.get("/api/tasking/queue", params={"status": "pending"})
    assert resp.status_code == 200

    # Verify the status parameter was forwarded (first positional arg)
    call_args = mock_db.fetch.call_args[0]
    assert call_args[1] == "pending"


# ------------------------------------------------------------------
# test_queue_limit_param
# ------------------------------------------------------------------


async def test_queue_limit_param(client, mock_db):
    """GET /api/tasking/queue?limit=10 forwards the limit."""
    mock_db.fetch = AsyncMock(return_value=[])

    resp = await client.get("/api/tasking/queue", params={"limit": 10})
    assert resp.status_code == 200

    call_args = mock_db.fetch.call_args[0]
    # limit is the second positional arg ($2)
    assert call_args[2] == 10


# ------------------------------------------------------------------
# test_create_cue_db_error_500
# ------------------------------------------------------------------


async def test_create_cue_db_error_500(client, mock_db):
    """POST /api/tasking/cue returns 500 when DB insert fails."""
    mock_db.fetchval = AsyncMock(
        side_effect=RuntimeError("DB connection lost")
    )

    resp = await client.post(
        "/api/tasking/cue",
        json={"bbox": [-5.8, 35.7, -5.2, 36.2]},
    )
    assert resp.status_code == 500
