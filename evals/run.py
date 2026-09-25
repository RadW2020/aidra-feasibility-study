"""Run the agent-interface evals.

    python -m evals.run                         # scripted reference agent, all scenarios
    python -m evals.run --agent claude          # Claude over the same MCP tools
    python -m evals.run --agent claude --model claude-sonnet-5 --repeat 3 -s diagnose_sat_mid

Needs a PostGIS database whose name contains 'eval' or 'test' (it is re-seeded
before every scenario): ``make``-free one-liner in evals/README.md.
Exit code 1 if any scenario fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.agents import ClaudeAgent, scripted_agent
from evals.framework import EvalEnvironment, ScenarioResult, run_scenario
from evals.scenarios import BY_ID, SCENARIOS

DEFAULT_DB = "postgresql://aidra_eval:eval@127.0.0.1:5434/aidra_eval"


def _report(results: list[ScenarioResult], agent: str, model: str | None) -> dict:
    return {
        "agent": agent,
        "model": model,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "scenarios": [
            {
                "id": r.scenario.id,
                "use_case": r.scenario.use_case,
                "kind": r.scenario.kind,
                "passed": r.passed,
                "seconds": round(r.seconds, 1),
                "error": r.error,
                "stop_reason": r.transcript.stop_reason,
                "usage": r.transcript.usage,
                "checks": [{"check": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks],
                "tool_calls": [
                    {"tool": c.name, "arguments": c.arguments, "is_error": c.is_error, "error_code": c.error_code}
                    for c in r.transcript.calls
                ],
                "answer": r.transcript.answer,
            }
            for r in results
        ],
    }


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent", choices=["scripted", "claude"], default="scripted")
    parser.add_argument("--model", default="claude-opus-5", help="Claude model id for --agent claude")
    parser.add_argument("-s", "--scenario", action="append", help="Run only these scenario ids (repeatable)")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per scenario (LLM agents vary run to run)")
    parser.add_argument("--database-url", default=os.environ.get("EVAL_DATABASE_URL", DEFAULT_DB))
    parser.add_argument("--output", type=Path, help="Write the JSON report here")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print tool calls and answers")
    args = parser.parse_args(argv)

    scenarios = [BY_ID[s] for s in args.scenario] if args.scenario else SCENARIOS
    agent = scripted_agent if args.agent == "scripted" else ClaudeAgent(model=args.model)
    results: list[ScenarioResult] = []
    async with EvalEnvironment(args.database_url) as env:
        for scenario in scenarios:
            for _ in range(args.repeat):
                r = await run_scenario(env, scenario, agent)
                results.append(r)
                mark = "PASS" if r.passed else "FAIL"
                print(f"{mark}  {scenario.id:<30} {scenario.kind:<15} {len(r.transcript.calls):>2} calls  {r.seconds:5.1f}s")
                for c in r.checks:
                    if not c.passed or args.verbose:
                        print(f"        {'ok ' if c.passed else 'XX '} {c.name}  ({c.detail})")
                if r.error:
                    print(f"        XX  agent error: {r.error}")
                if args.verbose or not r.passed:
                    for call in r.transcript.calls:
                        print(f"        -> {call.name}({json.dumps(call.arguments)}){' ERROR ' + str(call.error_code) if call.is_error else ''}")
                    print("        answer:", (r.transcript.answer or "").replace("\n", "\n                "))

    report = _report(results, args.agent, args.model if args.agent == "claude" else None)
    print(f"\n{report['passed']}/{report['total']} scenarios passed ({args.agent}{' ' + args.model if args.agent == 'claude' else ''})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"report: {args.output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
