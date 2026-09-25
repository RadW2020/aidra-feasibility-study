"""The MCP adapter: tool surface, safety modes, error mapping, and independence from the backend."""

from __future__ import annotations

import json
import subprocess
import sys
import typing
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport

from src.mcp_server.client import AidraClient, AidraConfig
from src.mcp_server.server import ExecutionStatus, Profile, QualityVerdict, build_server

ROOT = Path(__file__).resolve().parents[2]
READ_TOOLS = {
    "get_system_status", "get_catalog", "list_executions", "get_execution", "search_detections",
    "get_detection", "compare_model_variants", "preview_detection_run", "list_recent_actions",
}
WRITE_TOOLS = {"start_detection_run", "request_observation"}


async def _tools(mode: str):
    return {t.name: t for t in await build_server(AidraConfig(mode=mode)).list_tools()}


async def test_read_only_mode_does_not_register_write_tools():
    assert set(await _tools("read-only")) == READ_TOOLS
    assert set(await _tools("operator")) == READ_TOOLS | WRITE_TOOLS


async def test_annotations_declare_mutation_semantics():
    tools = await _tools("operator")
    for name in READ_TOOLS:
        assert tools[name].annotations.read_only_hint is True, name
    for name in WRITE_TOOLS:
        a = tools[name].annotations
        assert a.read_only_hint is False and a.destructive_hint is False, name


async def test_every_tool_and_parameter_is_described():
    for tool in (await _tools("operator")).values():
        assert len(tool.description or "") > 60, tool.name
        for pname, spec in tool.input_schema.get("properties", {}).items():
            assert spec.get("description") or pname in {"limit", "offset", "min_confidence"}, (tool.name, pname)


def test_enums_match_the_backend_vocabulary():
    """The MCP layer can't import the backend, so its Literals are copies: keep them honest."""
    from src.profiles.definitions import PROFILES
    from src.vocabulary import EXECUTION_STATUSES, QUALITY_VERDICTS

    assert set(typing.get_args(Profile)) == set(PROFILES)
    assert set(typing.get_args(ExecutionStatus)) == set(EXECUTION_STATUSES)
    assert set(typing.get_args(QualityVerdict)) == set(QUALITY_VERDICTS)


def test_mcp_server_imports_nothing_from_the_backend():
    code = (
        "import sys, src.mcp_server.server, src.mcp_server.__main__; "
        "print([m for m in sys.modules if m.startswith('src.') and not m.startswith('src.mcp_server')])"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# ---------------------------------------------------------------------------
# Tools against the real FastAPI app (mocked DB tier, in-process transport)
# ---------------------------------------------------------------------------


@pytest.fixture
def server(test_app):
    api = AidraClient(AidraConfig(base_url="http://aidra.test", mode="operator"), transport=ASGITransport(app=test_app))
    return build_server(api=api)


def _payload(result):
    return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)


async def test_api_error_becomes_a_parseable_tool_error(server, mock_db):
    result = await server.call_tool("get_execution", {"execution_id": str(uuid4())})
    assert result.is_error is True
    error = json.loads(result.content[0].text)["error"]  # pure JSON, no SDK prefix
    assert error["code"] == "execution_not_found"
    assert error["http_status"] == 404
    assert "hint" in error


async def test_search_defaults_are_sea_only_and_deduplicated(server, mock_db):
    result = await server.call_tool("search_detections", {"zone": "gibraltar"})
    data = _payload(result)
    assert data["effective_filters"]["on_land"] is False
    assert data["effective_filters"]["one_run_per_scene"] is True
    sql = mock_db.fetch.call_args[0][0]
    assert "DISTINCT ON (x.image_id)" in sql


async def test_invalid_arguments_are_rejected_before_any_http_call(server, mock_db):
    # Over the wire this is an is_error result; in-process the SDK raises ToolError.
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="sat-mid"):  # the message lists the valid values
        await server.call_tool("list_executions", {"profile": "sat_mid"})
    assert mock_db.fetch.await_count == 0


async def test_missing_area_is_a_structured_error(server, mock_db):
    result = await server.call_tool("request_observation", {"reason": "urgent look"})
    assert result.is_error is True
    assert json.loads(result.content[0].text)["error"]["code"] == "missing_area"


async def test_unreachable_api_says_how_to_fix_it():
    api = AidraClient(AidraConfig(base_url="http://127.0.0.1:9", timeout_s=2))
    result = await build_server(api=api).call_tool("get_catalog", {})
    error = json.loads(result.content[0].text)["error"]
    assert error["code"] == "api_unreachable" and error["retryable"] is True
    assert "docker compose up" in error["hint"]


def test_client_sends_identity_and_idempotency_headers():
    api = AidraClient(AidraConfig(token="t0k", client_name="aidra-mcp/1.0.0"))
    api.peer = "claude-code/2.1"
    h = api._headers("key-1")
    assert h["Authorization"] == "Bearer t0k"
    assert h["X-AIDRA-Client"] == "aidra-mcp/1.0.0 via claude-code/2.1"
    assert h["Idempotency-Key"] == "key-1"
