"""Scoped tokens, audit trail, idempotency, rate limit and path confinement (src.main.app)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

import src.api.pipeline as pipeline_mod
from src.api.auth import confined_dir, parse_tokens, required_scope, reset_rate_limits
from src.api.errors import ApiError
from src.config import Settings
from tests.test_api.agent_helpers import model_row, route_db
from tests.test_api.conftest import FakeRecord

TOKENS = "claude-agent:run:tok-run,dash-reader:read:tok-read,ci:admin:tok-admin"


@pytest.fixture
async def app_client(monkeypatch, mock_db):
    monkeypatch.setenv("AIDRA_API_TOKENS", TOKENS)
    monkeypatch.setenv("AIDRA_API_TOKEN", "")
    reset_rate_limits()
    pipeline_mod._pipeline_state.update(running=False, current_profile=None, progress=None, current_execution_id=None)
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    reset_rate_limits()
    pipeline_mod._pipeline_state.update(running=False, current_profile=None, progress=None, current_execution_id=None)


def _bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def _audit_rows(mock_db):
    return [c.args for c in mock_db.execute.await_args_list if "INSERT INTO api_audit_log" in c.args[0]]


# ---------------------------------------------------------------------------
# Scope table
# ---------------------------------------------------------------------------


def test_scope_table():
    assert required_scope("GET", "/api/executions") is None
    assert required_scope("POST", "/api/stac/search") is None
    assert required_scope("POST", "/api/pipeline/preview") is None
    assert required_scope("POST", "/api/pipeline/trigger") == "run"
    assert required_scope("POST", "/api/orbital/decision") == "run"
    assert required_scope("POST", "/api/models/fetch") == "admin"
    assert required_scope("POST", "/api/some/new/route") == "admin"  # secure default
    assert required_scope("GET", "/api/audit/actions") == "read"


def test_token_parsing_and_legacy_token():
    table = parse_tokens(Settings(aidra_api_tokens=TOKENS, aidra_api_token="legacy"))
    assert table["tok-run"].name == "claude-agent" and table["tok-run"].scope == "run"
    assert table["legacy"].name == "operator" and table["legacy"].scope == "admin"
    with pytest.raises(ValueError):
        parse_tokens(Settings(aidra_api_tokens="x:superuser:t"))


# ---------------------------------------------------------------------------
# Enforcement through the real middleware chain
# ---------------------------------------------------------------------------


async def test_run_token_may_trigger_but_not_fetch_weights(app_client, mock_db):
    ok = await app_client.post("/api/pipeline/trigger", json={}, headers=_bearer("tok-run"))
    assert ok.status_code == 503  # auth passed; no engine in the test app
    denied = await app_client.post(
        "/api/models/fetch",
        json={"filename": "x.pt", "url": "https://example.org/x.pt", "sha256": "0" * 64},
        headers=_bearer("tok-run"),
    )
    assert denied.status_code == 403
    err = denied.json()["error"]
    assert err["code"] == "insufficient_scope"
    assert err["required_scope"] == "admin" and err["granted_scope"] == "run"
    assert err["retryable"] is False


async def test_missing_token_is_401_with_required_scope(app_client, mock_db):
    resp = await app_client.post("/api/tasking/cue", json={"bbox": [-5.6, 35.8, -5.3, 36.1]})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing or invalid API bearer token"
    assert resp.json()["error"]["required_scope"] == "run"


async def test_reads_stay_public_but_audit_log_needs_a_token(app_client, mock_db):
    assert (await app_client.get("/api/vocabulary")).status_code == 200
    assert (await app_client.post("/api/stac/search", json={})).status_code == 200
    assert (await app_client.get("/api/audit/actions")).status_code == 401
    assert (await app_client.get("/api/audit/actions", headers=_bearer("tok-read"))).status_code == 200


async def test_every_response_has_a_request_id(app_client, mock_db):
    resp = await app_client.get("/api/vocabulary", headers={"X-Request-ID": "agent-req-12345"})
    assert resp.headers["X-Request-ID"] == "agent-req-12345"
    err = await app_client.get(f"/api/executions/{uuid4()}")
    assert err.json()["error"]["request_id"] == err.headers["X-Request-ID"]


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


async def test_refused_and_accepted_writes_are_audited(app_client, mock_db):
    cue_id = uuid4()
    route_db(mock_db, fetchval={"INSERT INTO tasking_queue": cue_id})
    await app_client.post("/api/models/fetch", json={"filename": "x.pt", "url": "https://e.org/x.pt",
                          "sha256": "0" * 64}, headers=_bearer("tok-run"))
    await app_client.post("/api/tasking/cue", json={"bbox": [-5.6, 35.8, -5.3, 36.1], "zone": "gibraltar_strait"},
                          headers={**_bearer("tok-run"), "X-AIDRA-Client": "aidra-mcp/1.0 via test"})
    rows = _audit_rows(mock_db)
    assert len(rows) == 2
    refused, created = rows
    # args: sql, request_id, actor, scope, authenticated, client, method, path, operation,
    #       status_code, outcome, error_code, duration_ms, resource_type, resource_id, key, replay
    assert refused[2] == "claude-agent" and refused[9] == 403 and refused[10] == "rejected"
    assert refused[11] == "insufficient_scope"
    assert created[5] == "aidra-mcp/1.0 via test"
    assert created[8] == "POST /api/tasking/cue"
    assert created[10] == "success"
    assert (created[13], created[14]) == ("cue", str(cue_id))


def test_operation_label_keeps_prefix_and_templates_path_parameters():
    from starlette.requests import Request

    from src.api.audit import _operation

    eid = str(uuid4())
    scope = {"type": "http", "method": "POST", "path": f"/api/executions/{eid}/retry", "headers": [],
             "query_string": b"", "path_params": {"execution_id": eid}}
    assert _operation(Request(scope)) == "POST /api/executions/{execution_id}/retry"
    scope = {**scope, "path": "/api/tasking/cue", "path_params": {}}
    assert _operation(Request(scope)) == "POST /api/tasking/cue"


async def test_audit_failure_never_breaks_the_request(app_client, mock_db):
    mock_db.execute = AsyncMock(side_effect=RuntimeError("db down"))
    resp = await app_client.post("/api/pipeline/preview", json={})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_retry_with_same_key_replays_the_original_run(app_client, mock_db):
    stored = {"execution_id": str(uuid4()), "status": "started"}
    route_db(mock_db, fetchval={"INSERT INTO api_idempotency_keys": None},
             fetchrow={"FROM api_idempotency_keys": lambda actor, key: FakeRecord(
                 method="POST", path="/api/pipeline/trigger", request_hash=_hash({}),
                 status_code=200, response_json=json.dumps(stored))})
    with patch("src.api.pipeline._get_engine", return_value=MagicMock()):
        resp = await app_client.post("/api/pipeline/trigger", json={},
                                     headers={**_bearer("tok-run"), "Idempotency-Key": "run-42"})
    assert resp.status_code == 200
    assert resp.json() == stored
    assert resp.headers["Idempotent-Replay"] == "true"
    assert pipeline_mod._pipeline_state["running"] is False  # nothing started


async def test_same_key_different_body_is_refused(app_client, mock_db):
    route_db(mock_db, fetchval={"INSERT INTO api_idempotency_keys": None},
             fetchrow={"FROM api_idempotency_keys": FakeRecord(
                 method="POST", path="/api/pipeline/trigger", request_hash="other",
                 status_code=200, response_json="{}")})
    with patch("src.api.pipeline._get_engine", return_value=MagicMock()):
        resp = await app_client.post("/api/pipeline/trigger", json={"profile": "sat-high"},
                                     headers={**_bearer("tok-run"), "Idempotency-Key": "run-42"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "idempotency_key_reused"


async def test_first_use_of_a_key_stores_the_response(app_client, mock_db):
    route_db(mock_db, fetch={"FROM models_registry": [model_row("vesseltracker-sar-yolov8", "v1.0")]},
             fetchval={"INSERT INTO api_idempotency_keys": "claude-agent"})
    with patch("src.api.pipeline._get_engine", return_value=MagicMock()), \
            patch.object(pipeline_mod, "_run_pipeline_background"):
        resp = await app_client.post("/api/pipeline/trigger", json={},
                                     headers={**_bearer("tok-run"), "Idempotency-Key": "run-43"})
    assert resp.status_code == 200
    stores = [c.args for c in mock_db.execute.await_args_list if "UPDATE api_idempotency_keys" in c.args[0]]
    assert stores and json.loads(stores[0][4])["execution_id"] == resp.json()["execution_id"]


def _hash(body):
    from src.api.idempotency import _request_hash

    return _request_hash("POST", "/api/pipeline/trigger", _trigger_body(body))


def _trigger_body(body):
    from src.db.models import PipelineTriggerRequest

    return PipelineTriggerRequest(**body).model_dump()


# ---------------------------------------------------------------------------
# Rate limit and path confinement
# ---------------------------------------------------------------------------


async def test_write_rate_limit(app_client, mock_db, monkeypatch):
    monkeypatch.setenv("API_WRITE_RATE_LIMIT_PER_MINUTE", "2")
    codes = []
    for _ in range(3):
        r = await app_client.post("/api/tasking/cue", json={"bbox": [1, 2]}, headers=_bearer("tok-run"))
        codes.append(r.status_code)
    assert codes == [400, 400, 429]
    assert r.json()["error"]["code"] == "rate_limited"
    assert int(r.headers["Retry-After"]) >= 1


def test_confined_dir(tmp_path):
    assert confined_dir(None, str(tmp_path)) == tmp_path.resolve()
    assert confined_dir("sub/run1", str(tmp_path)) == (tmp_path / "sub/run1").resolve()
    for bad in ("../escape", "/etc", str(tmp_path) + "/../x"):
        with pytest.raises(ApiError) as exc:
            confined_dir(bad, str(tmp_path))
        assert exc.value.code == "path_not_allowed"
