"""
Integration tests for /api/pipeline endpoints.

Tier 1: Mocked DB and engine -- validates routing, request validation,
concurrency control (409), engine-unavailable (503), and status reporting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import src.api.pipeline as pipeline_mod

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _reset_pipeline_state():
    """Reset the in-memory pipeline state to idle."""
    pipeline_mod._pipeline_state.update(
        running=False,
        current_profile=None,
        progress=None,
        current_execution_id=None,
    )


def _mock_engine_available():
    """Return a patch context manager that makes _get_engine() succeed."""
    return patch(
        "src.api.pipeline._get_engine",
        return_value=MagicMock(),
    )


def _mock_engine_unavailable():
    """Return a patch context manager that makes _get_engine() raise 503."""
    from fastapi import HTTPException

    def raise_503():
        raise HTTPException(status_code=503, detail="Engine not available")

    return patch(
        "src.api.pipeline._get_engine",
        side_effect=raise_503,
    )


# ------------------------------------------------------------------
# test_trigger_validation_bad_profile
# ------------------------------------------------------------------


async def test_trigger_validation_bad_profile(client):
    """POST /api/pipeline/trigger with unknown profile returns 400."""
    _reset_pipeline_state()
    with _mock_engine_available():
        resp = await client.post(
            "/api/pipeline/trigger",
            json={"profile": "nonexistent-profile", "zone": "gibraltar"},
        )
    assert resp.status_code == 400
    assert "nonexistent-profile" in resp.json()["detail"]


# ------------------------------------------------------------------
# test_trigger_returns_started
# ------------------------------------------------------------------


async def test_trigger_returns_started(client):
    """POST /api/pipeline/trigger with valid params returns status='started'."""
    _reset_pipeline_state()
    with _mock_engine_available():
        resp = await client.post(
            "/api/pipeline/trigger",
            json={"profile": "ground", "zone": "gibraltar"},
        )
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "started"


# ------------------------------------------------------------------
# test_trigger_concurrent_409
# ------------------------------------------------------------------


async def test_trigger_concurrent_409(client):
    """Second trigger while first is running returns 409."""
    # Simulate a running pipeline
    pipeline_mod._pipeline_state.update(
        running=True,
        current_profile="ground",
        progress=0.5,
        current_execution_id=None,
    )

    try:
        with _mock_engine_available():
            resp = await client.post(
                "/api/pipeline/trigger",
                json={"profile": "sat-high", "zone": "gibraltar"},
            )
        assert resp.status_code == 409
        assert "already running" in resp.json()["detail"].lower()
    finally:
        _reset_pipeline_state()


# ------------------------------------------------------------------
# test_status_not_running
# ------------------------------------------------------------------


async def test_status_not_running(client):
    """GET /api/pipeline/status returns running=false when idle."""
    _reset_pipeline_state()

    resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200

    data = resp.json()
    assert data["running"] is False
    assert data["current_profile"] is None
    assert data["progress"] is None
    assert data["current_execution_id"] is None


# ------------------------------------------------------------------
# test_status_while_running
# ------------------------------------------------------------------


async def test_status_while_running(client):
    """GET /api/pipeline/status reports current state when a pipeline runs."""
    pipeline_mod._pipeline_state.update(
        running=True,
        current_profile="sat-mid",
        progress=0.6,
        current_execution_id=None,
    )

    try:
        resp = await client.get("/api/pipeline/status")
        assert resp.status_code == 200

        data = resp.json()
        assert data["running"] is True
        assert data["current_profile"] == "sat-mid"
        assert data["progress"] == 0.6
    finally:
        _reset_pipeline_state()


# ------------------------------------------------------------------
# test_trigger_engine_unavailable_503
# ------------------------------------------------------------------


async def test_trigger_engine_unavailable_503(client):
    """POST /api/pipeline/trigger returns 503 when engine is None."""
    _reset_pipeline_state()
    with _mock_engine_unavailable():
        resp = await client.post(
            "/api/pipeline/trigger",
            json={"profile": "ground", "zone": "gibraltar"},
        )
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"].lower()


# ------------------------------------------------------------------
# test_trigger_all_valid_profiles
# ------------------------------------------------------------------


async def test_trigger_all_valid_profiles(client):
    """Each of the 5 valid profiles is accepted by the trigger endpoint."""
    valid_profiles = ["ground", "sat-high", "sat-mid", "sat-low", "sat-extreme"]

    for profile in valid_profiles:
        _reset_pipeline_state()
        with _mock_engine_available():
            resp = await client.post(
                "/api/pipeline/trigger",
                json={"profile": profile, "zone": "gibraltar"},
            )
        assert resp.status_code == 200, (
            f"Profile '{profile}' was rejected with status {resp.status_code}"
        )


# ------------------------------------------------------------------
# test_trigger_request_defaults
# ------------------------------------------------------------------


async def test_trigger_request_defaults(client):
    """Trigger with minimal body uses default zone/model/profile."""
    _reset_pipeline_state()
    with _mock_engine_available():
        resp = await client.post(
            "/api/pipeline/trigger",
            json={},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
