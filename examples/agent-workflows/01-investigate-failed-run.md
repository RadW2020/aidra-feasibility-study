# 01 — "Which runs failed this week, and was it the model or the hardware budget?"

**Kind:** read-only investigation · **Tools:** `get_system_status`, `list_executions`, `get_execution`
· **Evals:** `triage_recent_failures`, `diagnose_sat_mid`

Before this interface, answering this took Grafana admin credentials and SQL against
`execution_log`, plus knowing that `skipped` is not a failure, that the reaper writes
`reaped: stuck in …`, and that a memory-budget abort shows up as `status='error'`.

## What the agent can do

It lists failures directly. Each row already carries an `outcome.category`, so
triage doesn't need one call per run:

```jsonc
// list_executions {"status": ["error", "failed"], "since": "2026-09-22T00:00:00Z"}
{ "total": 3, "next_offset": null, "items": [
  { "id": "e1000000-…-000000000006", "status": "error", "trigger_type": "cue",
    "triggered_by": "e1000000-…-000000000001",
    "outcome": { "category": "ingestion_failed", "summary": "Failed while finding or downloading the scene." },
    "error_summary": "IngestionError: Copernicus search failed: 503 Service Unavailable" },
  { "id": "e1000000-…-000000000004", "status": "failed", "trigger_type": "scheduled",
    "outcome": { "category": "reaped_orphan", "summary": "Marked failed by the orphan reaper: it never reached a terminal status." } },
  { "id": "e1000000-…-000000000003", "status": "error", "profile": "sat-mid",
    "model": { "name": "vesseltracker-sar-yolov8", "version": "int8-static" },
    "peak_ram_mb": 2650.0, "memory_budget_mb": 2048,
    "outcome": { "category": "memory_budget_exceeded",
                 "summary": "Aborted: exceeded the 'sat-mid' memory budget (peak RSS 2650 MB > 2048 MB), emulating an on-board OOM kill." } } ] }
```

The skipped scheduled scan that ran in the same window isn't in this list: it
succeeded at doing nothing new, and the vocabulary says so.

For the one the user cares about, `get_execution` returns the evidence and a next step:

```jsonc
// get_execution {"execution_id": "e1000000-…-000000000003", "top_detections": 0}
{ "status": "error", "is_terminal": true, "poll_after_seconds": null,
  "outcome": {
    "category": "memory_budget_exceeded",
    "evidence": { "peak_ram_mb": 2650.0, "memory_budget_mb": 2048.0, "profile": "sat-mid",
                  "error_message": "MemoryError: memory budget exceeded: peak RSS 2650 MB > 2048 MB budget of profile 'sat-mid' (enforcement=abort)" },
    "next_steps": ["Needs a profile whose memory_limit_mb exceeds the peak, or a lower-footprint variant; list_executions(image_id=...) shows how other profiles fared on this scene."] },
  "model": { "version": "int8-static", "registry_status": "active" },
  "provenance": { "applicable": false, "verdict": "not_applicable: only a successful run is evidence" },
  … }
```

A natural follow-up with the same primitives is `list_executions(image_id=…)` on that
scene. It shows the same INT8 model **succeeding on sat-high** at 2.7 GB, which answers
the user's actual question: the 2 GB budget is the limit, not the model.

## A good answer contains

- the three failures with their causes in domain terms: an on-board OOM emulation, a
  container restart, a Copernicus outage on a Tip & Cue follow-up;
- the numbers as returned (2650 MB against 2048 MB) and the ids;
- no causes the data doesn't support. For an unknown id the tool returns
  `execution_not_found`, and the `unknown_execution` eval checks that no cause is
  invented.
