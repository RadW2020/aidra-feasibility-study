"""Agent-interface evals in the test suite.

* The Claude agent's tool loop is checked with a fake Anthropic client (no
  network, no key): tool_use -> MCP call -> tool_result with is_error -> answer.
* The full scripted suite runs against real PostGIS only when
  ``EVAL_DATABASE_URL`` points at an eval/test database (CI's ``evals`` job).
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace

import pytest

from evals.framework import McpTools, Scenario, Transcript


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "get_execution":
            return SimpleNamespace(is_error=True, structured_content={"error": {"code": "execution_not_found"}}, content=[])
        return SimpleNamespace(is_error=False, structured_content={"ok": True}, content=[])


def _msg(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


async def test_claude_agent_loop_records_calls_and_errors(monkeypatch):
    replies = [
        _msg("tool_use", SimpleNamespace(type="tool_use", id="tu1", name="get_execution",
                                         input={"execution_id": "x"})),
        _msg("end_turn", SimpleNamespace(type="text", text="That execution was not found.")),
    ]
    sent = []

    class _Messages:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return replies.pop(0)

    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = lambda: SimpleNamespace(messages=_Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from evals.agents import ClaudeAgent

    transcript = Transcript()
    tools = McpTools(_FakeSession(), transcript, instructions="server primer")
    tools.tools = [SimpleNamespace(name="get_execution", description="d", input_schema={"type": "object"})]
    out = await ClaudeAgent(model="claude-opus-5")(Scenario("s", "UC2", "failure", "why?", []), tools)

    assert out.answer == "That execution was not found." and out.stop_reason == "end_turn"
    assert transcript.calls[0].error_code == "execution_not_found"
    tool_result = sent[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu1" and tool_result["is_error"] is True
    assert "server primer" in sent[0]["system"]
    assert out.usage == {"input_tokens": 20, "output_tokens": 10}


@pytest.mark.skipif(not os.environ.get("EVAL_DATABASE_URL"), reason="needs EVAL_DATABASE_URL (PostGIS eval database)")
def test_scripted_agent_passes_every_scenario():
    from evals.run import main

    assert asyncio.run(main(["--agent", "scripted"])) == 0
