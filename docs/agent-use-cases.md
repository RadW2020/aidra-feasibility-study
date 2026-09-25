# Agent use cases

The tasks an agent should be able to do against AIDRA. Each one is a job the
operator, evaluator or analyst already does by hand today (see
`agent-readiness-review.md`); none is invented for the exercise.

Permission scopes referenced below (defined in `docs/agent-first-plan.md`):
`public` (no token), `read` (authenticated read of sensitive views such as the
audit log), `run` (may launch runs and queue observations), `admin` (weights,
bundles, resets, validation imports).

---

## UC1 — "Is AIDRA healthy, and is anything running right now?"

| | |
|---|---|
| **User intent** | Know whether it is safe and useful to act: DB up, detector loaded, scheduler alive, no run in flight |
| **Required information** | None |
| **Operations** | `get_system_status` → health, engine availability, runs in flight *according to the database* (scheduled and cue runs included), next scheduler ticks, effective-config hash |
| **Expected output** | One-paragraph status with any in-flight execution id, its profile and age |
| **Failure cases** | DB unreachable → 503 `database_unavailable`, reported as a system outage and not retried in a loop; detector engine not loaded → reported as a degraded state, not "healthy" |
| **Permissions** | `public` |
| **Mutating?** | No |

## UC2 — "Why did the last run fail?" / "Why did the sat-mid run abort?"

| | |
|---|---|
| **User intent** | Root-cause a failed, skipped, reaped or empty execution |
| **Required information** | Optionally a profile, model, zone or time window. The agent finds the id itself |
| **Operations** | `list_executions(status=…, profile=…, since=…)` → pick the relevant row → `get_execution(id)` returns status meaning, error category, peak RAM against the profile budget, skip reason and timings |
| **Expected output** | The cause in domain terms ("peak RSS 2 650 MB exceeded the sat-mid budget of 2 048 MB; runtime floor, not the scene"), citing the execution id and the evidence fields |
| **Failure cases** | No matching execution → say so and don't guess; unknown id → 404 `execution_not_found`; several candidates → list them and ask instead of picking one silently; `status=running` → report it as in progress, not as a failure |
| **Permissions** | `public` |
| **Mutating?** | No |

## UC3 — "Which vessels were detected near Gibraltar last week?"

| | |
|---|---|
| **User intent** | A GEOINT question: detections in an area and a time window, counted correctly |
| **Required information** | Area (zone name or bbox) and time window. Tier and sea-only are optional; defaults are stated in the answer |
| **Operations** | `get_catalog` (zone → bbox, only when the name is ambiguous) → `search_detections(zone \| bbox, since, until, sea_only=true, one_run_per_scene=true)` → paginate with `next_offset` if needed |
| **Expected output** | Count and the top detections, with the filters used stated explicitly, plus the note that the `high` tier is an operating point (precision 0.31, Pd 0.12 on xView3) |
| **Failure cases** | Malformed bbox → 422 with the expected format; unknown zone → 422 `unknown_zone` with `valid_values`; zero results → report zero with the filters used, not "no vessels exist" |
| **Permissions** | `public` |
| **Mutating?** | No |

## UC4 — "Can I trust this detection as evidence? Show me its lineage."

| | |
|---|---|
| **User intent** | Explain where one detection came from and whether its provenance is complete (the core of tender criterion Q3) |
| **Required information** | Detection id, or enough to find it with `search_detections` |
| **Operations** | `get_detection(id)` → `get_execution(execution_id)`, whose `provenance` block checks image/model/output/params hashes and commit SHA → model status and card from `get_catalog` |
| **Expected output** | Lineage (scene → execution → model → hashes), the quality verdict and its meaning, and an explicit "provenance complete / incomplete because X" |
| **Failure cases** | Detection not found → 404; execution predates migration 017/019, so some fields are null → reported as missing, never inferred |
| **Permissions** | `public` |
| **Mutating?** | No |

## UC5 — "Is the static INT8 model good enough to replace FP32 on sat-high?"

| | |
|---|---|
| **User intent** | Evaluate a compression triplet `{baseline FP32, variant, profile}` against the declared degradation budget (I-MOD-1/2/3) |
| **Required information** | Baseline and variant model versions; profile optional |
| **Operations** | `compare_model_variants(model, baseline_version, variant_version, profile, dataset)` → quality deltas from `validation_runs` + latency/RAM from `execution_log` + verdict against ΔmAP ≤ 5 pts |
| **Expected output** | Verdict (`within_budget`, `exceeds_budget`, `insufficient_evidence`) with the numbers behind it and what is missing |
| **Failure cases** | No validation run for one side → `insufficient_evidence` naming the missing leg (never a verdict from latency alone); unknown version → 422 with `valid_values` |
| **Permissions** | `public` |
| **Mutating?** | No |

## UC6 — "Run the static INT8 model on the latest Gibraltar scene under sat-high."

| | |
|---|---|
| **User intent** | Produce new evidence: one pipeline run with an explicit configuration |
| **Required information** | Zone (or Copernicus `image_id`), model + version, profile. Thresholds default to `Settings` (I-DET-4) and the preview reports them |
| **Operations** | `preview_detection_run(...)` → resolved request, blocking issues, warnings, duration estimate from history → `start_detection_run(..., idempotency_key)` → `get_execution(id)` until it leaves `pending`/`running` (the response carries `poll_after_seconds`) |
| **Expected output** | Execution id, the resolved configuration actually used, the expected duration and — once finished — the outcome summary |
| **Failure cases** | Another run in flight → 409 `pipeline_busy` naming the blocking execution; ambiguous model name → 422 `ambiguous_model` with versions; rejected or candidate variant → warning surfaced to the user; COCO/non-SAR model → blocking `model_not_sar_compatible`; detector engine unavailable → 503 `engine_unavailable`; token without `run` scope → 403 `insufficient_scope`, reported, not retried; retry after timeout with the same idempotency key → the original execution, never a second run |
| **Permissions** | `run` |
| **Mutating?** | Yes — downloads a scene, consumes the box's CPU/RAM for 6–52 min, writes an `execution_log` row and detections |

## UC7 — "Queue a re-observation of this area."

| | |
|---|---|
| **User intent** | Manual Tip & Cue: ask the system to look again at an area |
| **Required information** | Area (bbox or a Tip & Cue zone), reason, priority (1–5) |
| **Operations** | `request_observation(bbox, reason, priority, idempotency_key)` → the response says whether a new cue was created or an equivalent pending cue already existed → `list_executions(trigger_type="cue")` later to see the result |
| **Expected output** | Cue id, whether it was new, its position in the queue, when the cue processor next runs (every 15 min) |
| **Failure cases** | Invalid bbox (lon/lat out of range, min > max, degenerate) → 422; an identical pending cue already queued → 409 `duplicate_cue` with the existing id (the agent reports it and does not force a duplicate); missing area → the agent asks instead of inventing one |
| **Permissions** | `run` |
| **Mutating?** | Yes — inserts a `tasking_queue` row that will trigger a pipeline run |

## UC8 — "What did the agents do today, and what failed?"

| | |
|---|---|
| **User intent** | Accountability: review mutating actions taken through the API by humans, CI or agents |
| **Required information** | Optional actor, time window, outcome |
| **Operations** | `list_recent_actions(since, actor, outcome)` → join with `get_execution` for the runs the actions created |
| **Expected output** | Timeline: who, which operation, which resource, result and error code |
| **Failure cases** | Caller not authenticated → 401 (the audit log is not public); empty window → "no recorded actions", not a guess |
| **Permissions** | `read` |
| **Mutating?** | No |

---

## Deliberately *not* agent use cases

| Request | Why it is not a tool |
|---|---|
| "Delete the failed runs / the rejected variant" | Evidence is never deleted (CLAUDE.md §8: mark, don't delete). No deletion API exists and none will be added; the agent must say so |
| "Replace the model weights", "reset the pipeline", "rebuild the D3 bundle", "import a validation report" | Admin operations that change what future evidence rests on. They stay in the REST API behind the `admin` scope for the operator and CI, and are not offered to agents as MCP tools |
| "Tune the CFAR thresholds" | A benchmarked decision (I-DET-4, I-MOD-3), made in versioned config and reviewed, not a runtime knob |
| "Chat with the detections" / RAG over reports | Nothing in the product needs retrieval over prose; the questions users ask are structured queries |
