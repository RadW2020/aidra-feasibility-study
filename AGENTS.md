# AGENTS.md

For AI agents that **operate** AIDRA (through MCP or the HTTP API) or **work on**
its code. Project-specific facts only. For coding contributions, the binding contract
is `CLAUDE.md` (invariants §5, definition of done §3).

## What AIDRA is

An evaluation study, not an operational service. It asks whether AI vessel detection
on Sentinel-1 SAR radar can run on satellite-class hardware, and what quality that
costs. A **run** (*execution*) downloads one Sentinel-1 scene for a **zone**, detects
vessels with CFAR + YOLOv8 under a **constraint profile** (a simulated CPU/RAM
budget), and persists every detection with SHA256 provenance. The product is the
evidence: failed runs and rejected variants are part of it.

## Vocabulary you will meet

| Term | Meaning |
|---|---|
| execution | One pipeline run, one `execution_log` row. Statuses: `pending`, `running` (not terminal), `success`, `skipped` (nothing new — **not a failure**), `invalid` (scene failed I-SAR-1), `error`, `failed` (reaped orphan). `GET /api/vocabulary` has the meanings |
| profile | `ground` (4 CPU / 24 GB) › `sat-high` (2 / 4 GB) › `sat-mid` (1 / 2 GB) › `sat-low` › `sat-extreme` (0.25 / 512 MB). A run whose peak RAM crosses the budget **aborts** (`status=error`, "memory budget exceeded"): that is the measurement working, not a bug |
| search zone vs Tip & Cue zone | Runs take a search zone (`gibraltar`, …). Cues use operational zones (`gibraltar_strait`, `algeciras_port`, `med_patrol`) that map to a search zone. `GET /api/catalog` has both and the mapping |
| model / version | `vesseltracker-sar-yolov8` has `v1.0` (FP32 baseline), `int8-static` (active), `int8-dynamic` (**rejected**, I-MOD-3). `cfar-default` is built in. `yolov8n` is a COCO model and the SAR gate refuses it |
| source / tier | `cfar`, `yolo`, `fused` (both agreed). `tier=high` = fused: precision 0.31, recall 0.12 on xView3. An operating point, not the full output |
| quality_verdict | `valid_sea_target` (YOLO-confirmed at sea) · `candidate` (CFAR-only) · `land_artifact` (kept, excluded from sea metrics, I-DET-2) · `cluster_artifact` · `outside_footprint` (historical rows) |
| triplet | {FP32 baseline, compressed variant, profile}. No compression claim without all three (I-MOD-1); degradation budget ΔmAP ≤ 5 pts (I-MOD-3) |
| cue | A Tip & Cue re-observation request in `tasking_queue`, processed every 15 min |

## Operating AIDRA through the agent interface

- **MCP server**: `src/mcp_server/`. It is a thin HTTP client of the API, with no
  business logic and no DB access, so it gets the same auth, audit and validation as
  every other client. `read-only` mode (the default) has 9 tools; `operator` mode adds
  `start_detection_run` and `request_observation`.
- **Discover first.** Call `get_catalog` before choosing zones, models or versions, and
  `get_system_status` before starting anything.
- **Preview, then start.** `preview_detection_run` is free and side-effect-free. Relay
  its `warnings` to the user: `model_rejected`, `model_candidate`,
  `previous_run_exceeded_memory_budget` and `recent_equivalent_run` all mean "ask
  before spending 6–52 minutes and a 1.7 GB download".
- **Poll, don't wait.** A run returns an `execution_id` immediately. Call
  `get_execution` no sooner than its `poll_after_seconds`.
- **Idempotency.** Send an `idempotency_key` (REST: `Idempotency-Key` header) on writes
  and reuse it on retry. A repeated key returns the original result with
  `idempotent_replay: true`.
- **Errors are data.** Every error has `error.code`, `message`, `hint`,
  `valid_values`, `retryable` and `request_id`. Branch on `code`:
  - `insufficient_scope` or `unauthenticated`: stop and tell the user. Never retry.
  - `pipeline_busy` or `rate_limited`: wait. The error names the blocking run.
  - `unknown_*`, `invalid_*` or `ambiguous_model`: fix the argument from `valid_values`.
  - `duplicate_cue`: the existing cue id is in the error. Don't force a duplicate.
- **Counting detections.** The same object is stored once per run that saw it: a scene
  processed under 5 profiles stores 5 copies. `search_detections` defaults to
  `sea_only` + `one_run_per_scene`; say which filters a number used.
- **Nothing is deleted.** There is no delete operation. Failed runs, rejected variants
  and land artefacts are evidence. If asked to delete, explain this.

### Permissions (`AIDRA_API_TOKENS=name:scope:token,…`)

| Scope | Can | Examples |
|---|---|---|
| none (public) | every `GET`, plus `POST /api/stac/search` and `POST /api/pipeline/preview` | status, history, detections, catalog |
| `read` | + audit log | `GET /api/audit/actions` |
| `run` | + produce evidence through the normal workflow | trigger a run, queue a cue, synthetic validation, orbital simulations |
| `admin` | + change what future evidence rests on | `POST /api/models/fetch` (overwrite weights), `/traceability/bundle`, `/validation/import`, `/pipeline/reset`, `/interpretability/run` |

Unknown mutating routes default to `admin`. The legacy `AIDRA_API_TOKEN` is
`operator:admin`. With no token configured (local dev) everything is open, as before.

### Dangerous or expensive operations

| Operation | Why it matters | Guard |
|---|---|---|
| `start_detection_run` | ~1.7 GB download, 6–52 min of the only detection slot; the OCI VNIC throttles above ~50 Mbps and takes down other services on the host | preview, one run at a time (409 `pipeline_busy`), `run` scope, idempotency |
| `POST /api/pipeline/trigger-all-profiles` | 5 sequential runs, hours | `run` scope; not an MCP tool |
| `POST /api/models/fetch` with `overwrite=true` | Replaces weights that evidence references by hash | `admin` only; not an MCP tool |
| Pushing to `main` | Coolify redeploys and **restarts the container, killing any run in progress** | Never push while `GET /api/pipeline/status` says `busy` |

## Working on the code

```bash
pip install -e '.[dev,mcp]'           # Python ≥ 3.11
ruff check . && pytest -q             # ~650 tests, mocked DB, ~40 s
pytest -k invariant -x                # CLAUDE.md §5 invariants
python -m evals.run                   # agent evals: real PostGIS + API + MCP (see evals/README.md)
```

| Path | What |
|---|---|
| `src/api/` | FastAPI routers. `errors.py` (envelope), `auth.py` (scopes, rate limit, `out_dir` confinement), `audit.py`, `idempotency.py`, `executions.py`, `catalog.py` |
| `src/pipeline/engine.py` | The 15-step run; `preflight.py` holds the checks shared by preview and trigger |
| `src/traceability/` | Hashes, recorder, D3 bundler; `diagnosis.py` turns a row into an outcome category |
| `src/vocabulary.py` | Every enum with its meaning: the single source of truth |
| `src/validation/triplet.py` | Compression-triplet grading |
| `src/mcp_server/` | MCP adapter (must not import other `src.*` modules; a test enforces it) |
| `evals/` | Scenario evals: seed, scenarios, scripted and Claude agents |
| `src/db/migrations/` | Applied in order at startup; never edit an applied one, add `0NN_*.sql` |

Conventions and traps:

- SQL is positional (`$1…`) and lives in `src/db/queries.py` or next to its router.
  Optional filters are spliced in with string replaces; any geometry parameter used
  twice must be cast both times (`$6::geometry`). One missed cast made every bbox
  query return 500 in production until the evals caught it.
- Thresholds come from `Settings`, never literals (I-DET-4). Changing a pipeline
  setting changes `input_params_hash`, and with it which runs are dedup-equivalent.
- `src/config.py` reads `.env` from the **current directory**. The repo's `.env` may
  hold production credentials; the evals start the API from a temp dir for this reason.
- Commit messages: imperative, no emojis, **no `Co-Authored-By`**.
- Production config lives in Coolify env vars, not `.env.example`. Read it with
  `GET /api/config`: its `settings_hash` matches the hash in D3 bundle manifests.

## Known edge cases

- `GET /api/pipeline/status.running` only covers runs started through this API process;
  `busy` / `in_flight` come from the DB and include scheduled and cue runs. Check `busy`.
- After a container restart, `pending`/`running` rows from before the restart are
  orphans: the reaper marks them `failed` after 240 min. They don't block new
  triggers.
- `med_patrol` (a Tip & Cue zone) maps to the `mediterranean_west` search zone, whose
  bbox doesn't cover it. Cues there re-observe the wrong scenes, and the cue endpoint
  now warns `bbox_outside_search_zone`.
- The live demonstrator (`https://aidra-api.uliber.com`) is public read-only. The
  agent endpoints (`/api/executions`, `/api/catalog`, `/api/audit/…`) exist only on
  deployments of this version or later (migration 021).
