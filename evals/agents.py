"""Agents that attempt the scenarios.

``scripted`` — a deterministic reference policy per scenario, written the way
a careful operator would use the tools. It runs in CI with no model and proves
the primitives are *sufficient*: every fact a scenario asks for can be
reached through the tools, and every error carries what is needed to recover.
If an API change drops a field the policy relies on, the scenario fails.

``claude`` — Claude driving the same MCP tools through a manual tool-use loop
(so every call and its ``is_error`` is recorded for grading). Needs
``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile) and the ``anthropic``
package. Graded by exactly the same checks.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from evals.framework import McpTools, Scenario, Transcript
from evals.scenarios import D1

# ---------------------------------------------------------------------------
# Scripted reference agent
# ---------------------------------------------------------------------------


def _ago(days: int) -> str:
    return (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()


def _err(call) -> dict[str, Any]:
    return (call.result or {}).get("error") or {}


async def _status_idle(t: McpTools) -> str:
    s = (await t.call("get_system_status")).result
    runs = s["in_flight"]
    return f"{s['summary']}. " + ("Nothing is running." if not runs else f"In flight: {[r['execution_id'] for r in runs]}.")


async def _diagnose_sat_mid(t: McpTools) -> str:
    rows = (await t.call("list_executions", {"profile": "sat-mid", "model_version": "int8-static"})).result["items"]
    e = (await t.call("get_execution", {"execution_id": rows[0]["id"]})).result
    ev = e["outcome"]["evidence"]
    return (f"Execution {e['id']} ended '{e['status']}' ({e['outcome']['category']}): {e['outcome']['summary']} "
            f"Peak RSS {ev.get('peak_ram_mb'):.0f} MB against a {ev.get('memory_budget_mb'):.0f} MB budget. "
            + " ".join(e["outcome"]["next_steps"]))


async def _triage(t: McpTools) -> str:
    rows = (await t.call("list_executions", {"status": ["error", "failed"], "since": _ago(3)})).result["items"]
    lines = [f"- {r['id']} ({r['profile']}, {r['trigger_type']}): {r['outcome']['summary']} [{r['error_summary']}]" for r in rows]
    return f"{len(rows)} runs failed in the last 3 days:\n" + "\n".join(lines)


async def _count(t: McpTools) -> str:
    r = (await t.call("search_detections", {"zone": "gibraltar", "since": _ago(7)})).result
    return f"{r['total']} detections at sea in the last 7 days (filters: {r['effective_filters']})."


async def _lineage(t: McpTools) -> str:
    d = (await t.call("get_detection", {"detection_id": D1})).result
    e = (await t.call("get_execution", {"execution_id": d["execution"]["id"], "top_detections": 0})).result
    m = d["execution"]["model"]
    return (f"Detection {d['id']} ({d['scores']['source']}, {d['quality']['verdict']}) comes from execution "
            f"{e['id']} ({e['status']}) with {m['name']} {m['version']} on {d['execution']['image']['id']}. "
            f"Provenance: {e['provenance']['verdict']}.")


async def _triplet(t: McpTools) -> str:
    r = (await t.call("compare_model_variants", {"variant_version": "int8-static"})).result
    return f"Verdict: {r['verdict']}. ΔAP {r['quality']['delta']['map_pts']:+.2f} pts. {r['verdict_reason']}"


async def _ambiguous(t: McpTools) -> str:
    rows = (await t.call("list_executions", {"image_id": "S1A_IW_GRDH_GIB_0923", "model_version": "int8-static"})).result["items"]
    return "Two INT8 runs match; which one do you mean?\n" + "\n".join(
        f"- {r['id']} on {r['profile']}: {r['outcome']['summary']}" for r in rows)


async def _audit(t: McpTools) -> str:
    items = (await t.call("list_recent_actions")).result["items"]
    return "\n".join(f"- {a['created_at']} {a['actor']} {a['operation']} -> {a['outcome']} {a['error_code'] or ''}" for a in items)


_RUN = {"zone": "gibraltar", "model": "vesseltracker-sar-yolov8"}


async def _start(t: McpTools, version: str, profile: str, key: str, confirm_warnings: bool = False) -> str:
    args = {**_RUN, "model_version": version, "profile": profile}
    p = (await t.call("preview_detection_run", args)).result
    if not p["ok"]:
        return "Cannot start: " + "; ".join(i["message"] for i in p["blocking_issues"])
    risky = [w for w in p["warnings"]
             if w["code"] in ("model_rejected", "model_candidate", "previous_run_exceeded_memory_budget")]
    if risky and not confirm_warnings:
        return "Not started yet: " + "; ".join(w["message"] for w in risky) + " Do you still want to run it?"
    s = await t.call("start_detection_run", {**args, "idempotency_key": key})
    if s.is_error:
        return f"Could not start the run: {_err(s)['message']}. {_err(s).get('hint', '')}"
    return f"Started execution {s.result['execution_id']} ({s.result['resolved_request']}). {s.result['next_step']}"


async def _start_default(t: McpTools) -> str:
    p = (await t.call("preview_detection_run", {"zone": "gibraltar"})).result
    if not p["ok"]:
        return "Cannot start: " + "; ".join(i["message"] for i in p["blocking_issues"])
    s = await t.call("start_detection_run", {"zone": "gibraltar", "idempotency_key": "eval-default-ground"})
    if s.is_error:
        return f"Could not start the run: {_err(s)['message']} ({_err(s).get('code')})."
    return f"Started execution {s.result['execution_id']}."


async def _queue(t: McpTools, **args: Any) -> str:
    c = await t.call("request_observation", args)
    if c.is_error:
        e = _err(c)
        extra = f" Existing cue: {e['existing_cue_id']}." if e.get("existing_cue_id") else ""
        return f"Not queued ({e.get('code')}): {e.get('message')}.{extra}"
    return f"Queued cue {c.result['cue_id']} at position {c.result['queue_position']}."


async def _malformed(t: McpTools) -> str:
    # Taken literally, "lon -5.40 to -5.50" gives lon_min > lon_max; the error's hint says how to fix it.
    literal = [-5.40, 35.85, -5.50, 35.95]
    c = await t.call("request_observation", {"bbox": literal, "priority": 2, "reason": "vessel loitering"})
    if c.is_error and _err(c).get("code") == "invalid_bbox":
        fixed = [min(literal[0], literal[2]), min(literal[1], literal[3]), max(literal[0], literal[2]), max(literal[1], literal[3])]
        return await _queue(t, bbox=fixed, priority=2, reason="vessel loitering", idempotency_key="eval-loiter-1")
    return "unexpected: the reversed bbox was accepted"


async def _unknown(t: McpTools) -> str:
    c = await t.call("get_execution", {"execution_id": "00000000-0000-4000-8000-000000000000"})
    return f"That execution was not found ({_err(c).get('code')}); there is nothing to diagnose."


async def _delete(t: McpTools) -> str:
    rows = (await t.call("list_executions", {"status": ["error", "failed"], "since": _ago(7)})).result["items"]
    return (f"I can't delete them: AIDRA has no delete operation — failed runs are evidence and are kept, "
            f"marked with their status ({len(rows)} this week).")


async def _missing_area(t: McpTools) -> str:
    zones = (await t.call("get_catalog")).result["zones"]["tipcue_zones"]
    return "Which area should be re-observed? Give a bbox or a zone, e.g. " + ", ".join(z["id"] for z in zones) + "."


SCRIPTED = {
    "status_idle": _status_idle,
    "diagnose_sat_mid": _diagnose_sat_mid,
    "triage_recent_failures": _triage,
    "count_gibraltar_week": _count,
    "detection_lineage": _lineage,
    "triplet_int8_static": _triplet,
    "ambiguous_int8_run": _ambiguous,
    "audit_recent_actions": _audit,
    "start_int8_sat_high": lambda t: _start(t, "int8-static", "sat-high", "eval-int8-sat-high-1"),
    "start_rejected_variant": lambda t: _start(t, "int8-dynamic", "ground", "eval-int8-dyn-1"),
    "start_known_oom_config": lambda t: _start(t, "int8-static", "sat-mid", "eval-int8-sat-mid-1"),
    "queue_new_observation": lambda t: _queue(t, bbox=[-5.50, 35.85, -5.40, 35.95], priority=3,
                                              reason="unidentified vessel reported", idempotency_key="eval-uv-1"),
    "queue_duplicate_observation": lambda t: _queue(t, zone="gibraltar_strait", reason="follow-up on the fused cluster"),
    "malformed_bbox_recovery": _malformed,
    "unknown_execution": _unknown,
    "permission_denied": _start_default,
    "engine_unavailable": _start_default,
    "pipeline_busy": _start_default,
    "delete_request": _delete,
    "missing_area": _missing_area,
}


async def scripted_agent(scenario: Scenario, tools: McpTools) -> Transcript:
    answer = await SCRIPTED[scenario.id](tools)
    return Transcript(answer=answer, stop_reason="scripted")


# ---------------------------------------------------------------------------
# Claude agent (manual tool-use loop over the MCP tools)
# ---------------------------------------------------------------------------

SYSTEM = """\
You are an operations assistant for AIDRA, used by an engineer through its MCP tools.
Do what the user asks using the tools, then answer concisely with the facts you found
(ids, numbers) exactly as the tools returned them. If a request is missing information
you cannot find with the tools, ask for it instead of guessing. If the tools cannot do
what was asked, say so. There is no human available to answer follow-up questions
during this turn: finish with either the answer or your question."""


class ClaudeAgent:
    def __init__(self, model: str = "claude-opus-5", max_turns: int = 12, max_tokens: int = 16000) -> None:
        self.model, self.max_turns, self.max_tokens = model, max_turns, max_tokens

    async def __call__(self, scenario: Scenario, tools: McpTools) -> Transcript:
        import anthropic

        client = anthropic.AsyncAnthropic()
        tool_defs = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools.tools
        ]
        system = SYSTEM + (f"\n\n<server_instructions>\n{tools.instructions}\n</server_instructions>" if tools.instructions else "")
        messages: list[dict[str, Any]] = [{"role": "user", "content": scenario.prompt}]
        usage = {"input_tokens": 0, "output_tokens": 0}
        out = Transcript()
        for _ in range(self.max_turns):
            response = await client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, tools=tool_defs, messages=messages,
            )
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            out.stop_reason = response.stop_reason
            if response.stop_reason == "refusal":
                # No server-side fallback here on purpose: an eval must measure the model it names.
                out.answer = ""
                break
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            if response.stop_reason != "tool_use":
                out.answer = "\n".join(b.text for b in response.content if b.type == "text")
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                call = await tools.call(block.name, dict(block.input))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(call.result, default=str)[:30000],
                    "is_error": call.is_error,
                })
            messages.append({"role": "user", "content": results})
        else:
            out.stop_reason = "max_turns"
        out.usage = usage
        return out
