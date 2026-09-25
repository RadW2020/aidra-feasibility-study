# Agent evals

Can an agent do real AIDRA tasks through the agent interface, and does that still hold
after a change? Each scenario is a request in plain words ("Why did the sat-mid run of
the static INT8 model fail?") plus **outcome checks**:

- which tools were called, with which arguments;
- what the database looks like afterwards (exactly one run created, no duplicate cue,
  nothing deleted, the audit row attributed to the agent);
- which facts the final answer states (ids, 2650 MB against 2048 MB, +0.61 AP pts);
- which claims it must not make (an invented cause, the double-counted total).

Wording is never graded.

```
agent (scripted | Claude)
   │  MCP over stdio — the real server, python -m src.mcp_server
   ▼
src/mcp_server  ──HTTP──▶  evals/api_app.py = src.main.app (real routes, auth, audit, preflight)
                                    │         + StubEngine: no Copernicus download, no inference
                                    ▼
                            PostGIS, re-seeded from fixtures/seed.sql before every scenario
```

Only the heavy pipeline (Copernicus + SAR inference) is stubbed. It still writes
pending → running → success rows through the real `ExecutionRecorder`. The API
runs as a subprocess from a temporary directory, so the repository `.env` (which may
hold production credentials) is never read.

## Run

```bash
pip install -e '.[dev,mcp]'

# A throwaway PostGIS (the name must contain "eval" or "test": it is wiped per scenario)
docker run -d --name aidra-eval-db -p 127.0.0.1:5434:5432 --tmpfs /var/lib/postgresql/data \
  -e POSTGRES_DB=aidra_eval -e POSTGRES_USER=aidra_eval -e POSTGRES_PASSWORD=eval \
  postgis/postgis:16-3.4          # on Apple Silicon: docker build -f Dockerfile.postgis .

python -m evals.run                                  # scripted reference agent, all scenarios
python -m evals.run -v -s diagnose_sat_mid           # one scenario, show calls and answer
python -m evals.run --output reports/evals/scripted.json
```

The default database URL is `postgresql://aidra_eval:eval@127.0.0.1:5434/aidra_eval`;
override it with `EVAL_DATABASE_URL` or `--database-url`. With `EVAL_DATABASE_URL`
set, `pytest tests/test_evals` runs the suite too. CI does this in the `evals` job and
uploads `eval-report.json`.

### With Claude as the agent

```bash
pip install -e '.[evals]'
export ANTHROPIC_API_KEY=...        # or an `ant auth login` profile
python -m evals.run --agent claude                           # default claude-opus-5
python -m evals.run --agent claude --model claude-sonnet-5 --repeat 3
```

Claude gets the MCP tool list, the server's instructions and a short system prompt,
and runs a manual tool-use loop (≤ 12 turns). Every call, its arguments and its
`is_error` are recorded, and the same checks grade the result. A `refusal` stop
reason is reported as a failure. No fallback model is configured, on purpose: the
eval measures the model it names. Use `--repeat` to see run-to-run variance before
trusting a single pass. **This spends API credits**: roughly 20 scenarios × 2–6
model calls per run.

## Two drivers, two questions

| Driver | Question it answers | When |
|---|---|---|
| `scripted` | *Are the primitives sufficient?* A deterministic policy per scenario, written the way a careful operator would use the tools. If an API change drops a field it relies on (`outcome.evidence.peak_ram_mb`, `error.existing_cue_id`, `valid_values`, …), the scenario fails | Every CI run |
| `claude` | *Does a model actually use them well?* Does it preview before starting, relay warnings, stop on `insufficient_scope`, ask when the area is missing, avoid inventing a cause? | Before releasing interface changes; when comparing tool descriptions or models |

## Scenarios

| id | kind | tests |
|---|---|---|
| `status_idle` | read-only | UC1: health + nothing in flight |
| `diagnose_sat_mid` | read-only | UC2: finds the run itself and cites peak RSS vs budget |
| `triage_recent_failures` | multi-step | UC2: memory abort, reaped orphan, Copernicus outage; skips are not failures |
| `count_gibraltar_week` | read-only | UC3: 5, not the 12 stored rows (duplicates across profiles, land) |
| `detection_lineage` | multi-step | UC4: detection → run → model → provenance verdict |
| `triplet_int8_static` | read-only | UC5: within budget, +0.61 AP pts, legs paired by `settings_hash` |
| `ambiguous_int8_run` | failure | Two INT8 runs match; both must be surfaced |
| `audit_recent_actions` | read-only | UC8: finds the rejected weight fetch and why |
| `start_int8_sat_high` | state-changing | UC6: preview → start → exactly one run, audited, id reported |
| `start_rejected_variant` | state-changing | Relays the I-MOD-3 rejection |
| `start_known_oom_config` | state-changing | Relays that this configuration aborted on its memory budget last time |
| `queue_new_observation` | state-changing | UC7: one cue with the requested priority, id reported |
| `queue_duplicate_observation` | failure | 409 `duplicate_cue`: reports the existing cue, does not force a duplicate |
| `malformed_bbox_recovery` | failure | Reversed corners → `invalid_bbox` hint → one correct cue |
| `unknown_execution` | failure | 404: says so, invents no cause |
| `permission_denied` | failure | `read` token: one attempt at most, reports the missing scope |
| `engine_unavailable` | failure | 503 / blocking preview: no run, says why |
| `pipeline_busy` | failure | Names the run already in flight, starts nothing |
| `delete_request` | failure | No delete tool: explains evidence is kept; DB unchanged |
| `missing_area` | failure | Asks which area instead of inventing one |

## Adding a scenario

1. Add rows to `fixtures/seed.sql` if the world needs new facts (fixed ids, times
   relative to `NOW()`).
2. Add a `Scenario` in `scenarios.py`: the prompt a user would write, the checks that
   define success, and `token` / `mode` / `engine` if it tests a failure path.
3. Add the reference policy to `SCRIPTED` in `agents.py`. If it can't be written
   with the existing tools, that gap is a finding about the interface, not the eval.

## What it found

The first run against a real database failed `count_gibraltar_week`: every bbox
query on `GET /api/detections`, `.geojson` and the OGC items endpoint returned
**500 in production**. A string-replace cast only the first of two geometry
parameters. The mocked-DB unit tests could never see it. It is now fixed, with a
regression test.
