# 04 — "A vessel was seen loitering at lat 35.85–35.95, lon −5.40 to −5.50. Get another look."

**Kind:** state-changing with error recovery · **Scope:** `run` · **Tool:** `request_observation`
· **Evals:** `malformed_bbox_recovery`, `queue_new_observation`, `queue_duplicate_observation`, `missing_area`

A manual Tip & Cue observation: a `tasking_queue` row that the cue processor turns
into a pipeline run within 15 minutes. Two things commonly go wrong: the box comes out
malformed, and a retry queues the same cue twice. Both are now visible to the agent.

## What the agent can do

Taken literally, "lon −5.40 to −5.50" gives `lon_min > lon_max`. The API refuses it
and says how to fix it:

```jsonc
// request_observation {"bbox": [-5.40, 35.85, -5.50, 35.95], "priority": 2, "reason": "vessel loitering"}
{ "error": { "code": "invalid_bbox", "message": "bbox needs lon_min < lon_max and lat_min < lat_max",
             "hint": "[lon_min, lat_min, lon_max, lat_max] in WGS-84 degrees; check the order of the corners.",
             "retryable": false, "http_status": 400 } }
```

It corrects the box and sends an idempotency key:

```jsonc
// request_observation {"bbox": [-5.50, 35.85, -5.40, 35.95], "priority": 2, "reason": "vessel loitering", "idempotency_key": "loiter-0925-1"}
{ "cue_id": "9bd52c85-4adc-4d5a-98e2-b8f5b76620bd", "created": true, "status": "pending",
  "search_zone": "gibraltar", "queue_position": 2, "warnings": [],
  "processed_by": "cue_processor job, every 15 minutes, highest priority first",
  "follow_up": "GET /api/tasking/queue?status=pending; the resulting run appears in GET /api/executions?trigger_type=cue" }
```

Retrying the same call with the same key returns the same `cue_id` with
`idempotent_replay: true`, and no second row is written.

If the user had named a whole zone that is already queued, the API refuses to
queue it twice:

```jsonc
// request_observation {"zone": "gibraltar_strait", "reason": "follow-up on the fused cluster"}
{ "error": { "code": "duplicate_cue", "http_status": 409,
             "message": "An identical cue is already pending: c0000000-0000-4000-8000-000000000002",
             "existing_cue_id": "c0000000-0000-4000-8000-000000000002",
             "hint": "Nothing was queued. Follow the existing cue in GET /api/tasking/queue, or pass allow_duplicate=true if a second observation is really wanted." } }
```

The right answer reports cue `c0000000…002` as already pending. It does not add
`allow_duplicate: true` on its own initiative, and the eval checks that.

If the cue's box lies outside the scenes its zone searches (the processor searches
by *search zone*, not by the cue's box), the response carries a
`bbox_outside_search_zone` warning. This is how the interface surfaced that the
`med_patrol` Tip & Cue zone maps to a search zone that doesn't cover it.

With no area at all ("queue a re-observation, it's urgent"), the tool returns
`missing_area` with the hint *"Ask the user which area to observe; do not invent
one."*
