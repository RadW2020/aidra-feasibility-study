"""Scenario evals for the AIDRA agent interface.

A scenario is a human-level request plus outcome checks. An *agent* (the
scripted reference policy, or Claude) talks to the real MCP server over stdio;
the MCP server talks HTTP to the real API (``evals.api_app``) backed by a real
PostGIS seeded with ``fixtures/seed.sql``. Checks grade what happened — which
tools were called with which arguments, what the database looks like
afterwards, which facts the final answer states and which it must not invent —
never the exact wording.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg
import httpx

ROOT = Path(__file__).resolve().parents[1]
SEED = Path(__file__).parent / "fixtures" / "seed.sql"

TOKENS = {
    "run": ("claude-agent", "tok-eval-run"),
    "read": ("eval-reader", "tok-eval-read"),
    "admin": ("eval-admin", "tok-eval-admin"),
}
MUTATING_TOOLS = {"start_detection_run", "request_observation"}


# ---------------------------------------------------------------------------
# What an agent run produced
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    is_error: bool
    result: Any  # parsed JSON (structured content, or the error envelope)

    @property
    def error_code(self) -> str | None:
        if self.is_error and isinstance(self.result, dict):
            return (self.result.get("error") or {}).get("code")
        return None


@dataclass
class Transcript:
    answer: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Context:
    transcript: Transcript
    db: asyncpg.Connection


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class Check:
    label = "check"

    async def run(self, ctx: Context) -> CheckResult:  # pragma: no cover - interface
        raise NotImplementedError


class Called(Check):
    """The agent called ``tool`` (optionally with these argument values) at least ``times``."""

    def __init__(self, tool: str, args: dict[str, Any] | None = None, times: int = 1, succeeded: bool | None = None):
        self.tool, self.args, self.times, self.succeeded = tool, args or {}, times, succeeded
        self.label = f"called {tool}" + (f" with {self.args}" if self.args else "")

    def _matches(self, call: ToolCall) -> bool:
        if call.name != self.tool:
            return False
        if self.succeeded is not None and call.is_error == self.succeeded:
            return False
        for key, want in self.args.items():
            got = call.arguments.get(key)
            if callable(want):
                if not want(got):
                    return False
            elif got != want:
                return False
        return True

    async def run(self, ctx: Context) -> CheckResult:
        n = sum(self._matches(c) for c in ctx.transcript.calls)
        return CheckResult(self.label, n >= self.times, f"{n} matching call(s)")


class CalledAny(Check):
    def __init__(self, *tools: str):
        self.tools = tools
        self.label = f"called one of {', '.join(tools)}"

    async def run(self, ctx: Context) -> CheckResult:
        used = [c.name for c in ctx.transcript.calls if c.name in self.tools]
        return CheckResult(self.label, bool(used), f"used {used}")


class NotCalled(Check):
    def __init__(self, tool: str, succeeded_only: bool = False):
        self.tool, self.succeeded_only = tool, succeeded_only
        self.label = f"did not {'successfully ' if succeeded_only else ''}call {tool}"

    async def run(self, ctx: Context) -> CheckResult:
        calls = [c for c in ctx.transcript.calls if c.name == self.tool and not (self.succeeded_only and c.is_error)]
        return CheckResult(self.label, not calls, f"{len(calls)} call(s)")


class NotCalledWith(Called):
    """No call to ``tool`` with these argument values."""

    def __init__(self, tool: str, args: dict[str, Any]):
        super().__init__(tool, args)
        self.label = f"never called {tool} with {args}"

    async def run(self, ctx: Context) -> CheckResult:
        n = sum(self._matches(c) for c in ctx.transcript.calls)
        return CheckResult(self.label, n == 0, f"{n} matching call(s)")


class AtMost(Check):
    def __init__(self, tool: str, times: int):
        self.tool, self.times = tool, times
        self.label = f"called {tool} at most {times}x"

    async def run(self, ctx: Context) -> CheckResult:
        n = sum(c.name == self.tool for c in ctx.transcript.calls)
        return CheckResult(self.label, n <= self.times, f"{n} call(s)")


class NoSuccessfulMutation(Check):
    label = "no state-changing call succeeded"

    async def run(self, ctx: Context) -> CheckResult:
        done = [c.name for c in ctx.transcript.calls if c.name in MUTATING_TOOLS and not c.is_error]
        return CheckResult(self.label, not done, f"succeeded: {done}")


class Mentions(Check):
    """The final answer matches ``pattern`` (case-insensitive regex)."""

    def __init__(self, pattern: str, why: str):
        self.pattern, self.label = pattern, f"answer states {why}"

    async def run(self, ctx: Context) -> CheckResult:
        ok = re.search(self.pattern, ctx.transcript.answer, re.I | re.S) is not None
        return CheckResult(self.label, ok, f"/{self.pattern}/")


class DoesNotClaim(Check):
    """The final answer must not match ``pattern`` — facts the data does not support."""

    def __init__(self, pattern: str, why: str):
        self.pattern, self.label = pattern, f"answer does not claim {why}"

    async def run(self, ctx: Context) -> CheckResult:
        m = re.search(self.pattern, ctx.transcript.answer, re.I | re.S)
        return CheckResult(self.label, m is None, f"matched {m.group(0)!r}" if m else "")


class DbCount(Check):
    """``SELECT COUNT(*) ...`` equals ``expected`` after the run."""

    def __init__(self, sql: str, expected: int, why: str):
        self.sql, self.expected, self.label = sql, expected, why

    async def run(self, ctx: Context) -> CheckResult:
        n = await ctx.db.fetchval(self.sql)
        return CheckResult(self.label, n == self.expected, f"count={n}, expected {self.expected}")


class AnswerMentionsResult(Check):
    """The answer quotes a value taken from a tool result (e.g. the id just created)."""

    def __init__(self, tool: str, extract: Callable[[Any], str | None], why: str):
        self.tool, self.extract, self.label = tool, extract, f"answer quotes {why}"

    async def run(self, ctx: Context) -> CheckResult:
        values = [self.extract(c.result) for c in ctx.transcript.calls if c.name == self.tool and not c.is_error]
        values = [v for v in values if v]
        if not values:
            return CheckResult(self.label, False, f"no successful {self.tool} result to quote")
        ok = any(v.lower() in ctx.transcript.answer.lower() for v in values)
        return CheckResult(self.label, ok, f"expected one of {values}")


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    id: str
    use_case: str
    kind: str  # read-only | multi-step | state-changing | failure
    prompt: str
    checks: list[Check]
    token: str | None = "run"  # which token the MCP server holds: run | read | admin | None
    mode: str = "operator"  # MCP server mode
    engine: str = "stub"  # stub | none
    setup_sql: str | None = None


@dataclass
class ScenarioResult:
    scenario: Scenario
    transcript: Transcript
    checks: list[CheckResult]
    seconds: float
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)


Agent = Callable[[Scenario, "McpTools"], Awaitable[Transcript]]


# ---------------------------------------------------------------------------
# MCP session wrapper that records every call
# ---------------------------------------------------------------------------


class McpTools:
    """A live MCP client session; ``call`` records into the transcript."""

    def __init__(self, session: Any, transcript: Transcript, instructions: str | None = None) -> None:
        self.session = session
        self.transcript = transcript
        self.instructions = instructions  # the server's MCP instructions, as a client would show them
        self.tools: list[Any] = []

    async def list(self) -> list[Any]:
        self.tools = (await self.session.list_tools()).tools
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
        arguments = arguments or {}
        result = await self.session.call_tool(name, arguments)
        is_error = bool(getattr(result, "is_error", False))
        payload: Any = getattr(result, "structured_content", None)
        if payload is None:
            text = "".join(getattr(c, "text", "") for c in (getattr(result, "content", None) or []))
            try:
                payload = json.loads(text)
            except ValueError:
                payload = {"text": text}
        call = ToolCall(name=name, arguments=arguments, is_error=is_error, result=payload)
        self.transcript.calls.append(call)
        return call


# ---------------------------------------------------------------------------
# Environment: API subprocess + database + MCP server per scenario
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _guard_db_url(url: str) -> None:
    name = url.rsplit("/", 1)[-1].split("?")[0]
    if "eval" not in name and "test" not in name:
        raise SystemExit(f"Refusing to reseed database '{name}': evals only run on a database named *eval* or *test*.")


class EvalEnvironment:
    def __init__(self, database_url: str) -> None:
        _guard_db_url(database_url)
        self.database_url = database_url
        self.dsn = database_url.replace("+asyncpg", "")
        self.port = _free_port()
        self.api_url = f"http://127.0.0.1:{self.port}"
        self._tmp = Path(tempfile.mkdtemp(prefix="aidra-eval-"))
        self._proc: subprocess.Popen | None = None

    def _api_env(self) -> dict[str, str]:
        models = self._tmp / "models"
        (models / "cards").mkdir(parents=True, exist_ok=True)
        for card in (ROOT / "models" / "cards").glob("*.MODEL_CARD.md"):
            shutil.copy(card, models / "cards" / card.name)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AIDRA_", "COPERNICUS_"))}
        env.update({
            "PYTHONPATH": str(ROOT),
            "DATABASE_URL": self.database_url,
            "MODELS_DIR": str(models),
            "IMAGES_DIR": str(self._tmp / "images"),
            "THUMBNAILS_DIR": str(self._tmp / "thumbnails"),
            "EVIDENCE_DIR": str(self._tmp / "evidence"),
            "INTERPRETABILITY_DIR": str(self._tmp / "interpretability"),
            "SCHEDULER_ENABLED": "false",
            "LOKI_ENABLED": "false",
            "PROMETHEUS_ENABLED": "false",
            "LOG_LEVEL": "WARNING",
            "DEFAULT_ZONE": "gibraltar",
            "DEFAULT_MODEL": "vesseltracker-sar-yolov8",
            "DEFAULT_MODEL_VERSION": "v1.0",
            "DEFAULT_PROFILE": "ground",
            "AIDRA_API_TOKEN": "",
            "AIDRA_API_TOKENS": ",".join(f"{n}:{scope}:{tok}" for scope, (n, tok) in TOKENS.items()),
            "API_WRITE_RATE_LIMIT_PER_MINUTE": "1000",
            "AIDRA_COMMIT_SHA": "eval",
        })
        return env

    async def __aenter__(self) -> EvalEnvironment:
        # Migrations are applied by the API's own startup, the production code path.
        # cwd = temp dir: no repository .env (which may hold real credentials) is read.
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "evals.api_app:app", "--host", "127.0.0.1",
             "--port", str(self.port), "--log-level", "warning"],
            cwd=self._tmp, env=self._api_env(),
            stdout=subprocess.DEVNULL, stderr=open(self._tmp / "api.log", "wb"),  # noqa: SIM115
        )
        deadline = time.monotonic() + 120
        async with httpx.AsyncClient() as http:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"API exited early; see {self._tmp / 'api.log'}")
                try:
                    if (await http.get(f"{self.api_url}/api/health", timeout=2)).status_code == 200:
                        return self
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise RuntimeError(f"API did not become healthy; see {self._tmp / 'api.log'}")

    async def __aexit__(self, *exc: Any) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        shutil.rmtree(self._tmp, ignore_errors=True)

    async def reset(self, scenario: Scenario) -> None:
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(SEED.read_text())
            if scenario.setup_sql:
                await conn.execute(scenario.setup_sql)
        finally:
            await conn.close()
        async with httpx.AsyncClient() as http:
            await http.post(f"{self.api_url}/__eval__/state", json={"engine": scenario.engine}, timeout=10)

    @asynccontextmanager
    async def mcp(self, scenario: Scenario, transcript: Transcript):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        env = {
            "PYTHONPATH": str(ROOT),
            "AIDRA_API_URL": self.api_url,
            "AIDRA_MCP_MODE": scenario.mode,
            "AIDRA_MCP_CLIENT_NAME": "aidra-mcp/eval",
            "AIDRA_API_TOKEN": TOKENS[scenario.token][1] if scenario.token else "",
        }
        params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"], env=env, cwd=str(self._tmp))
        with open(self._tmp / "mcp.log", "a") as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write), ClientSession(read, write) as session:
                init = await session.initialize()
                tools = McpTools(session, transcript, getattr(init, "instructions", None))
                await tools.list()
                yield tools


async def run_scenario(env: EvalEnvironment, scenario: Scenario, agent: Agent) -> ScenarioResult:
    await env.reset(scenario)
    transcript = Transcript()
    start = time.monotonic()
    error = None
    try:
        async with env.mcp(scenario, transcript) as tools:
            result = await agent(scenario, tools)
            transcript.answer, transcript.stop_reason, transcript.usage = result.answer, result.stop_reason, result.usage
    except Exception as exc:  # noqa: BLE001 - a crashed agent is a failed scenario, not a crashed suite
        error = f"{type(exc).__name__}: {exc}"
    await asyncio.sleep(0.5)  # let background runs (stub engine) finish writing
    conn = await asyncpg.connect(env.dsn)
    try:
        ctx = Context(transcript=transcript, db=conn)
        checks = [await c.run(ctx) for c in scenario.checks]
    finally:
        await conn.close()
    return ScenarioResult(scenario, transcript, checks, time.monotonic() - start, error)
