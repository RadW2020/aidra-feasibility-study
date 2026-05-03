"""API bearer-token protection tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_write_endpoint_requires_bearer_token_when_configured(
    monkeypatch,
    mock_db,
) -> None:
    """POST /api/* is rejected when AIDRA_API_TOKEN is configured."""
    monkeypatch.setenv("AIDRA_API_TOKEN", "test-token")

    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/pipeline/trigger",
            json={"zone": "gibraltar", "profile": "ground"},
        )

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing or invalid API bearer token"


@pytest.mark.asyncio
async def test_write_endpoint_accepts_valid_bearer_token(
    monkeypatch,
    mock_db,
) -> None:
    """Auth passes through to the endpoint when the bearer token matches."""
    monkeypatch.setenv("AIDRA_API_TOKEN", "test-token")

    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/pipeline/trigger",
            headers={"Authorization": "Bearer test-token"},
            json={"zone": "gibraltar", "profile": "ground"},
        )

    # The test app has no lifespan-built engine, so reaching endpoint code returns 503.
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_read_endpoint_remains_public_when_token_configured(
    monkeypatch,
    mock_db,
) -> None:
    """GET /api/* remains public so dashboards and health checks can read."""
    monkeypatch.setenv("AIDRA_API_TOKEN", "test-token")

    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pipeline/status")

    assert resp.status_code == 200
