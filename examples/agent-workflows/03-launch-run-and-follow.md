# 03 — "Run static INT8 on the latest Gibraltar scene under sat-high and tell me what it found."

**Kind:** state-changing · **Scope:** `run` · **Tools:** `preview_detection_run`, `start_detection_run`, `get_execution`
· **Evals:** `start_int8_sat_high`, `pipeline_busy`, `permission_denied`, `engine_unavailable`

A run downloads ~1.7 GB and holds the host's only detection slot for 6–52 minutes.
The interface makes that cost visible before the agent commits, and safe to retry
afterwards.

## What the agent can do

**Preview (free, no side effects).** It shows what will actually run, including the
version and thresholds resolved from `Settings`, and how long that has taken before:

```jsonc
// preview_detection_run {"zone": "gibraltar", "model": "vesseltracker-sar-yolov8", "model_version": "int8-static", "profile": "sat-high"}
{ "ok": true, "blocking_issues": [], "warnings": [], "in_flight": [],
  "resolved_request": { "model_version": "int8-static", "profile": "sat-high",
                        "confidence_threshold": 0.25, "iou_threshold": 0.45, "thresholds_from": "Settings (I-DET-4)" },
  "model": { "registry_status": "active", "sar_compatible": true, "has_model_card": true },
  "profile": { "cpu_limit": 2.0, "memory_limit_mb": 4096 },
  "estimate": { "duration_minutes_p50": 6.3, "based_on_runs": 1 },
  "cost": "Searches Copernicus for the newest scene of the zone (last 7 days), downloads it (~1.7 GB …) …" }
```

**Start, with an idempotency key.** A timed-out call can be retried safely:

```jsonc
// start_detection_run {…same…, "idempotency_key": "int8-sat-high-0925"}
{ "execution_id": "795e112b-96a1-4651-8501-80cfe826b806", "status": "started",
  "poll": { "url": "/api/executions/795e112b-…", "after_seconds": 30, "expected_duration_minutes": 6.3 },
  "next_step": "Call get_execution('795e112b-…') after 30 s." }

// the same call again (a retry after a network timeout)
{ "execution_id": "795e112b-96a1-4651-8501-80cfe826b806", "status": "started", "idempotent_replay": true, … }
```

**Follow.** `get_execution` returns `is_terminal: false` and `poll_after_seconds: 120`
while the run is going. When it is terminal, the same call carries the outcome, the
detection breakdown (`by_source`, `by_quality_verdict`, `valid_sea_targets`) and
`provenance.verdict: "complete"`. That completes the "tell me what it found" part.

The audit log records the call: actor `claude-agent` (from the token), operation
`POST /api/pipeline/trigger`, outcome `success`, resource `execution 795e112b…`,
idempotency key, and client `aidra-mcp/1.0.0 via <name/version the MCP client reported at initialize>`.

## When it must not start

| Situation | What the agent sees | Expected behaviour |
|---|---|---|
| A scheduled scan is running | `blocking_issues: [{code: pipeline_busy, message: "A pipeline is already running (e9000000-…)"}]`, `retryable: true` | Report the blocking run; don't start another on a 4-CPU host |
| Token has scope `read` | `start_detection_run` → `{code: insufficient_scope, required_scope: "run", granted_scope: "read", retryable: false}` | One attempt, then tell the user; never retry |
| Engine not loaded | preview `blocking_issues: [{code: engine_unavailable}]` | Explain; nothing can start |
| Rejected or OOM-prone configuration | preview `warnings: [model_rejected \| previous_run_exceeded_memory_budget]` | Relay and ask before spending the run |
| Tip & Cue zone passed as a search zone | `{code: unknown_zone, hint: "'gibraltar_strait' is a Tip & Cue zone; the pipeline searches the 'gibraltar' search zone for it.", valid_values: [...]}` | Correct the argument and preview again |

Before this interface, the trigger answered `started` with an id for all of these,
even an unknown zone or a COCO model. The run then died before its row existed, and
the id returned 404 forever.
