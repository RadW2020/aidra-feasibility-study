"""
Integration tests for GET /api/health.

Tier 1: Mocked DB -- validates routing, response shape, and field values.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

# ------------------------------------------------------------------
# test_health_returns_200
# ------------------------------------------------------------------


async def test_health_returns_200(client, mock_db):
    """GET /api/health returns HTTP 200 when DB is reachable."""
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get("/api/health")
    assert resp.status_code == 200


# ------------------------------------------------------------------
# test_health_response_shape
# ------------------------------------------------------------------


async def test_health_response_shape(client, mock_db):
    """Response body contains all required fields."""
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get("/api/health")
    assert resp.status_code == 200

    data = resp.json()
    expected_keys = {"status", "db", "models_loaded", "scheduler", "version"}
    assert expected_keys.issubset(data.keys()), (
        f"Missing keys: {expected_keys - data.keys()}"
    )

    # status must be a string
    assert isinstance(data["status"], str)
    # db must report connectivity
    assert isinstance(data["db"], str)
    # models_loaded must be a non-negative integer
    assert isinstance(data["models_loaded"], int)
    assert data["models_loaded"] >= 0
    # scheduler must be a string
    assert isinstance(data["scheduler"], str)
    # version must be a string
    assert isinstance(data["version"], str)


# ------------------------------------------------------------------
# test_health_version
# ------------------------------------------------------------------


async def test_health_version(client, mock_db):
    """Version field matches the hardcoded 1.0.0."""
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get("/api/health")
    data = resp.json()
    assert data["version"] == "1.0.0"


# ------------------------------------------------------------------
# test_health_db_connected
# ------------------------------------------------------------------


async def test_health_db_connected(client, mock_db):
    """When SELECT 1 succeeds, db field is 'connected' and status is 'ok'."""
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get("/api/health")
    data = resp.json()
    assert data["db"] == "connected"
    assert data["status"] == "ok"


# ------------------------------------------------------------------
# test_health_db_unreachable_503
# ------------------------------------------------------------------


async def test_health_db_unreachable_503(client, mock_db):
    """When the database is unreachable, health returns 503."""
    mock_db.fetchval = AsyncMock(
        side_effect=ConnectionRefusedError("Connection refused")
    )

    resp = await client.get("/api/health")
    assert resp.status_code == 503


# ------------------------------------------------------------------
# test_health_scheduler_stopped_by_default
# ------------------------------------------------------------------


async def test_health_scheduler_stopped_by_default(client, mock_db):
    """Without a running scheduler, scheduler field is 'stopped'."""
    mock_db.fetchval = AsyncMock(return_value=1)

    with patch("src.main.get_scheduler", return_value=None):
        resp = await client.get("/api/health")

    data = resp.json()
    assert data["scheduler"] == "stopped"
