# Agent-first plan

How AIDRA becomes a product that agents can operate, built on the audit
(`agent-readiness-review.md`) and the use cases (`agent-use-cases.md`).

## 1. Architecture

Agents are one more client of the same HTTP API that the dashboards, CI and the
operator already use. The MCP server is a **thin adapter**: it maps typed tool
arguments to REST calls, trims payloads to fit a context window and turns HTTP
errors into tool errors. Every rule — validation, preflight, verdicts,
permissions, idempotency, audit — lives in the backend, so a Codex session
using `curl` gets the same guarantees as Claude Code using MCP.

```
 Human (operator, evaluator)        GIS client / CI            AI agent (Claude Code, Codex, Cursor)
        │                                │                           │
   Grafana, curl                 STAC / OGC / REST            MCP (stdio or HTTP)  ── or plain REST
        │                                │                           │
        │                                │                  src/mcp_server  (typed tools, no domain logic)
        │                                │                           │
        └────────────────────────────────┴───────────────┬───────────┘
                                                         ▼
                          FastAPI /api  ── auth scopes · audit log · idempotency · error envelope
                                                         │
                  preflight · execution diagnosis · triplet verdict · vocabulary   (new, domain)
                                                         │
               PipelineEngine · ModelManager · ExecutionRecorder · EvidenceBundler  (existing)
                                                         │
                                         PostgreSQL + PostGIS  (execution_log, detections, …)
```

Why an HTTP client rather than an MCP server that imports the engine:

- The canonical interface is already the HTTP API (the no-SSH rule). An MCP
  server that imported `PipelineEngine` would be a second, unaudited write path
  into the database.
- The MCP process stays light (`mcp` + `httpx`, no torch/GDAL). It runs on a
  laptop against a local stack *or* the live deployment.
- One permission model, one audit trail, one set of tests.

## 2. Agent-facing operations

### 2.1 New or changed REST endpoints (backend)

| Endpoint | Kind | What it gives an agent | Reuses |
|---|---|---|---|
| `GET /api/executions` | new, read | List runs with filters (status, profile, model, version, trigger, zone, image, time window), paginated with `total` / `next_offset` | `execution_log` |
| `GET /api/executions/{id}` | new, read | One run as a *diagnosis*: status meaning, outcome category (`memory_budget_exceeded`, `skipped_duplicate`, `reaped_orphan`, …) with the evidence fields, detection breakdown (source, verdict, sea targets), provenance completeness check, lineage (cues), top-N detections, `poll_after_seconds` while in flight | `execution_log`, `detections`, `tasking_queue`, `PROFILES` |
| `GET /api/pipeline/status` | extended (additive) | Adds `in_flight` **from the database** (scheduled and cue runs included), `busy`, `engine_available`, scheduler jobs with `next_run_time` | APScheduler, `execution_log` |
| `POST /api/pipeline/preview` | new, read-only | Dry run: resolved request (model version, thresholds from `Settings`), `blocking_issues`, `warnings` (rejected/candidate variant, recent equivalent run), in-flight conflicts, duration estimate from history | `ModelManager` card/file lookups, `PipelineEngine._validate_model_for_sensor`, `SEARCH_ZONES`, `PROFILES` |
| `POST /api/pipeline/trigger` | changed | Runs the same preflight **before** issuing an `execution_id` (no more orphan ids); 409 if *any* run is in flight; honours `Idempotency-Key`; returns resolved request + warnings + poll hint | existing trigger |
| `POST /api/tasking/cue` | changed | Validates the bbox (ranges, ordering, area), refuses an identical pending cue with 409 `duplicate_cue` (+ existing id), honours `Idempotency-Key`, returns queue position | existing cue insert |
| `GET /api/detections` | extended (additive) | `execution_id`, `source`, `tier`, `zone`, `sort=confidence\|recent` and `one_run_per_scene` so counts stop duplicating across re-runs and profiles; fixes the broken STAC `thumbnail_gallery` link | `SELECT_DETECTIONS`, `tiers.py` |
| `GET /api/models` | extended (additive) | `status`, `rejection_reason`, `has_model_card`, `sar_compatible` | `models_registry`, card gate |
| `GET /api/benchmarks/triplet` | new, read | Compression triplet {baseline, variant, profile}: quality deltas from `validation_runs`, latency/RAM from `execution_log`, verdict against the declared ΔmAP budget (I-MOD-1/2/3), and the missing legs | `validation_runs`, `SELECT_BENCHMARKS_BY_MODEL` |
| `GET /api/catalog` | new, read | One call for zones (search zones **and** Tip & Cue zones with their mapping), profiles, models with status, defaults | `SEARCH_ZONES`, `tipcue.zones`, `PROFILES`, registry |
| `GET /api/vocabulary` | new, read | Every enum with its meaning: execution statuses (terminal or not), quality verdicts (counts as a sea target or not), sources/tiers, trigger types, cue and model statuses | new `src/vocabulary.py` (single source of truth) |
| `GET /api/config` | new, read | Effective `Settings`, secrets censored, with the **same `settings_hash` the D3 bundle manifest records**, plus `commit_sha` | `bundler._censor_secret` (factored out) |
| `GET /api/audit/actions` | new, read (`read` scope) | Who did what: actor, client, operation, resource, outcome, error code, duration, request id | new `api_audit_log` table |

Cross-cutting, applied to every route:

- **Error envelope.** Every error keeps its legacy `detail` field and gains
  `error: {code, message, hint?, valid_values?, retryable, request_id}`.
  `error.code` is the stable machine contract; status codes are preserved.
- **`X-Request-ID`** on every response, echoed in errors and in the audit log.
- **Scoped tokens** (`AIDRA_API_TOKENS=name:scope:token,…`); the legacy
  `AIDRA_API_TOKEN` keeps working as `operator:admin`.

### 2.2 MCP tools (`src/mcp_server`)

Eleven tools. Every one maps to one or two REST calls, and every one carries
MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) so a
client can auto-approve reads and prompt on writes.

| Tool | Mutates | REST | Notes |
|---|---|---|---|
| `get_system_status` | no | `GET /health`, `GET /pipeline/status`, `GET /config` | Health + in-flight runs + config hash in one answer |
| `get_catalog` | no | `GET /catalog`, `GET /vocabulary` | Valid values for every other tool |
| `list_executions` | no | `GET /executions` | Compact rows, `next_offset` |
| `get_execution` | no | `GET /executions/{id}` | The diagnosis view; `top_detections` is capped |
| `search_detections` | no | `GET /detections` | Defaults to sea-only, one run per scene, and says so |
| `get_detection` | no | `GET /detections/{id}` | Reshaped into detection / scores / quality / lineage sections; thumbnail URL instead of a server path |
| `compare_model_variants` | no | `GET /benchmarks/triplet` | Verdict + numbers + missing evidence |
| `preview_detection_run` | no | `POST /pipeline/preview` | Always safe; call before starting |
| `start_detection_run` | **yes** | `POST /pipeline/trigger` | Needs `run` scope; `idempotency_key` recommended |
| `request_observation` | **yes** | `POST /tasking/cue` | Needs `run` scope; duplicate-aware |
| `list_recent_actions` | no | `GET /audit/actions` | Needs `read` scope |

Server modes: `AIDRA_MCP_MODE=read-only` (default) doesn't register the two
mutating tools, so an agent pointed at the public deployment can't even try.
`operator` registers them; the backend scope check still applies.

Tool errors come back as `is_error` results whose text is JSON:
`{"error": {"code", "message", "hint", "valid_values", "retryable", "http_status", "request_id"}}`
— the same envelope as the REST API.

### 2.3 Design choices per operation

- **Preview and start are two tools, not a `dry_run` flag.** The permission
  differs (read vs `run`), and clients auto-approve by tool, not by argument.
- **No `wait_for_execution` tool.** Runs take 6–52 minutes; a blocking tool
  call would hang the agent. `get_execution` returns `poll_after_seconds`.
- **No generic `query`/`sql` tool.** Grafana's SQL proxy is exactly the thing
  that makes the product human-only today: it needs admin credentials and
  implicit knowledge of the schema.
- **Defaults are stated, not hidden.** `search_detections` defaults to
  `sea_only=true, one_run_per_scene=true` and echoes the effective filters in
  every response, so the agent can tell the user what "count" means.

## 3. Components reused

| Existing component | Used for |
|---|---|
| `PipelineEngine._validate_model_for_sensor`, `ModelManager._require_model_card`, `ModelManager._find_model_file` | Preflight checks: the same code that gates a real run |
| `src/pipeline/ingestion.SEARCH_ZONES`, `src/profiles/definitions.PROFILES`, `src/tipcue/zones` | Catalog and validation; `GET /zones` and `GET /profiles` now derive from them instead of hand copies |
| `src/api/tiers.py` | Source / tier filters on the list endpoint |
| `ExecutionRecorder` / `execution_log` | Execution list and diagnosis (no new run bookkeeping) |
| `validation_runs` + `SELECT_BENCHMARKS_BY_MODEL` | Triplet verdict |
| `bundler._censor_secret` + the bundle's settings hash | `GET /config` returns the hash that D3 bundles record |
| `StructuredLogger` → Loki, `prometheus_client` | Audit events go to Loki as JSON lines and to a counter `aidra_api_actions_total` |
| Existing bearer middleware | Extended into scopes, kept backward compatible |

## 4. Files

**New**

- `src/api/errors.py` — `ApiError`, exception handlers, request id.
- `src/api/auth.py` — token parsing, scope table, `Principal`.
- `src/api/audit.py` — audit middleware + `GET /audit/actions`.
- `src/api/idempotency.py` — `Idempotency-Key` store and replay.
- `src/vocabulary.py` — enums with meanings.
- `src/api/executions.py` — list + diagnosis routes.
- `src/api/catalog.py` — `/catalog`, `/vocabulary`, `/config`.
- `src/pipeline/preflight.py` — preview checks shared by preview and trigger.
- `src/traceability/diagnosis.py` — pure functions: outcome category, provenance check.
- `src/db/migrations/021_agent_interface.sql` — `api_audit_log`, `api_idempotency_keys`.
- `src/mcp_server/{__init__,__main__,client,server}.py` — the MCP adapter.
- `tests/test_api/test_agent_*.py`, `tests/test_mcp/` — unit and contract tests.
- `evals/` — scenario evals (see §6).
- `AGENTS.md`, `examples/agent-workflows/`, `docs/agent-portfolio-review.md`.

**Modified (additive)**

- `src/main.py` — middleware chain (request id → auth → audit), exception handlers.
- `src/api/router.py` — mount the new routers.
- `src/api/pipeline.py` — preflight, DB-aware concurrency, idempotency, extended status, `/preview`.
- `src/api/tasking.py` — bbox validation, duplicate check, idempotency.
- `src/api/detections.py` — extra filters and sort.
- `src/api/models_api.py` — status fields; profiles/zones from the domain modules.
- `src/api/benchmarks.py` — triplet route.
- `src/api/traceability.py`, `src/api/interpretability.py` — confine `out_dir`.
- `src/config.py` — `aidra_api_tokens`, `evidence_dir`, `triplet_max_delta_map_pts`, `api_write_rate_limit_per_minute`.
- `src/traceability/bundler.py` — factor out the censored-settings hash.
- `pyproject.toml` — `mcp` extra, `aidra-mcp` entry point.
- `docker-compose.yml` — optional `aidra-mcp` service (streamable HTTP, profile `agent`).
- `README.md` — "Agent-first architecture" section.

## 5. Security

| Risk | Control |
|---|---|
| One secret authorises everything | Scoped tokens: `read` < `run` < `admin`. Agents get `run`; weights, bundles, resets and imports need `admin`. Unknown mutating routes default to `admin` |
| No attribution | Every non-GET `/api` call writes an `api_audit_log` row (actor from the token, self-declared client from `X-AIDRA-Client`, route, resource, outcome, error code, duration, request id) and a Loki line |
| Retries duplicate work | `Idempotency-Key` on trigger and cue: same key + same body → replay of the original response (`Idempotent-Replay: true`); same key + different body → 422 `idempotency_key_reused`; concurrent duplicate → 409 `idempotency_in_progress` |
| Runs pile up on the box | Trigger refuses (409 `pipeline_busy`) while **any** run from this process is in flight — scheduled and cue runs included |
| Runaway agent loops | Per-actor limit on mutating calls (`api_write_rate_limit_per_minute`, default 30) → 429 `rate_limited` with `Retry-After` |
| Arbitrary write paths | `out_dir` of bundle / interpretability confined under `Settings.evidence_dir` / `interpretability_dir` → 422 `path_not_allowed` |
| Read-only POST blocked by the write guard | `POST /stac/search` and `POST /pipeline/preview` are explicitly public |
| Data leakage | `GET /config` uses the bundle's censoring (passwords, tokens, URL credentials). The audit log needs a token. Detection responses drop server filesystem paths in favour of URLs |
| Destructive operations | None exposed to agents. Deletion doesn't exist by design; weight overwrite, reset, bundle and import stay `admin`-only and are not MCP tools |
| Agent pointed at production by mistake | MCP defaults to `read-only` mode; write tools only register in `operator` mode, and the backend still checks scope |

What stays unchanged, deliberately: **reads are public**, as they are today —
the deployment is a public demonstrator (Grafana public dashboards, STAC/OGC for
QGIS). The audit log is the one exception.

## 6. Testing and eval strategy

**Unit / API tests** (mocked DB, same tier as the existing API tests):
diagnosis categories, provenance checks, preflight rules, error envelope,
scope table, idempotency replay/conflict, bbox validation, `out_dir`
confinement, vocabulary completeness (every status the code writes appears in
the vocabulary), MCP tool registration (names, annotations, schemas, read-only
mode hides writes), and that `src/mcp_server` imports nothing from the
backend.

**Evals** (`evals/`, real PostGIS, real FastAPI app, real MCP server over stdio):

- A seeded dataset that mirrors real production shapes: a successful ground
  run, a sat-mid memory-budget abort, a reaped orphan, a dedup skip, an
  ingestion failure, a Tip & Cue lineage, active/candidate/rejected/non-SAR
  models, and xView3 validation rows for both legs of the triplet.
- The heavy pipeline (Copernicus + inference) is replaced by a stub engine that
  writes a real pending → success row through the real `ExecutionRecorder`.
  Only the agent interface is under test.
- **Scenarios** state the task in human words plus outcome graders: tools that
  must / must not be called, DB state afterwards (exactly one cue, no
  duplicate, no run started), facts the answer must contain (ids, numbers) and
  claims it must not make.
- Two drivers: a **scripted reference agent** (deterministic, runs in CI,
  checks that the primitives are sufficient and the contracts hold) and a
  **Claude agent** (`--agent claude`, Anthropic API tool loop over the MCP
  tools, optional, needs `ANTHROPIC_API_KEY`), graded by the same graders.
- Failure scenarios: unknown execution, ambiguous request, token without scope,
  malformed bbox, engine unavailable, "delete the failed runs", missing area,
  run already in flight.

## 7. Priorities

| P | Item |
|---|---|
| **P0** | Error envelope + request id · `GET /executions`, `GET /executions/{id}` with diagnosis · DB-aware `/pipeline/status` · preflight + `/pipeline/preview` + trigger fix · models status fields · vocabulary + catalog · scoped tokens · audit log + `GET /audit/actions` · MCP server (11 tools, read-only/operator modes) · `AGENTS.md` · evals with scripted agent · workflow examples · README section |
| **P1** | Idempotency keys · cue validation + duplicate refusal · detections list filters + `one_run_per_scene` (fixes the STAC link) · triplet endpoint + tool · `GET /config` · `out_dir` confinement · public read-only POSTs · rate limit on writes · Claude agent driver for evals · `aidra-mcp` compose service |
| **P2** | OAuth for MCP-over-HTTP · per-request progress for runs (needs engine callbacks) · Loki query tool (logs for one execution) · tokens stored hashed in DB with rotation · MCP resources/prompts |

## 8. Deliberately not built

- **No LLM inside the product.** No chatbot, no summariser, no "AI insights"
  panel. The product stays fully useful with no model connected.
- **No vector database / RAG.** The questions users ask are structured queries
  over `execution_log` and `detections`; nothing needs semantic retrieval.
- **No LangChain / LangGraph / autonomous loop.** The agent is the client; the
  product doesn't orchestrate agents.
- **No SQL passthrough tool** (see §2.3).
- **No delete, retrain or threshold-tuning tools.** Evidence is marked, never
  deleted; thresholds are benchmarked decisions (I-DET-4, I-MOD-3).
- **No duplicated schemas in the MCP layer.** Tool inputs are typed; outputs
  are the backend's JSON, trimmed.
- **No second write path.** The MCP server never touches the database.
