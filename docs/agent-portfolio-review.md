# Agent portfolio review

A deliberately critical read of the agent-first work, from the point of view of a
hiring manager screening for *"built things agents actually use"*. It is written
against the repository as it stands after the change, not against the plan.

## The one-paragraph verdict

The design is right, and the evidence that it is right is real. Agents go through the
same API as every other client. Domain rules live in the backend, and the MCP layer
is thin and tested to stay that way. The interface is permission-scoped, idempotent,
audited and evaluated against a real database. The evals paid for themselves on their
first run: they caught a production bug that 574 mocked-DB tests had never seen. The
weakness is on the "actually use" half of the claim. **No real model has been run
through the evals yet** (no API key was available where this was built), and the
public deployment still serves the pre-agent API. Today the repository proves an agent
*could* operate AIDRA well. It doesn't yet show one doing it.

## Scores

| Dimension | Score | Evidence | What keeps it from the next level |
|---|---|---|---|
| Agent usability | **Good, unproven with models** | 11 tools with typed, described inputs and `Literal` enums; outcome categories instead of raw rows; `poll_after_seconds`; `valid_values` and `hint` in every error; server instructions that encode domain traps (dedup counts, `skipped` ≠ failure) | Zero measured runs of a real model. Tool descriptions were written, not tuned against observed agent mistakes |
| API / tool design | **Good** | Preview/start split by permission; idempotency with replay and conflict detection; `settings_hash` pairing for triplets; additive, backward-compatible changes (all 574 original tests still pass; seven were adjusted only to point the now-confined `out_dir` root at a temp directory) | Legacy status codes are inconsistent on purpose (400 vs 422). Tool outputs are `dict[str, Any]` with no output schemas. About a third of the old routes still lack response models |
| Product usefulness | **Strong for operators, thin for analysts** | Replaces Grafana-admin SQL for "what failed and why"; warns before a known memory abort costs a 1.7 GB download; surfaced three latent product bugs | GEOINT work (area/time queries) got only filters and dedup: no track/area aggregation, no "what changed between two passes" |
| Documentation | **Good** | `AGENTS.md` is project-specific (vocabulary, scopes, dangerous operations, traps); four workflows built on real responses; eval README; audit, use cases and plan | OpenAPI descriptions are still half Spanish; error codes aren't enumerated in OpenAPI `responses`; no rendered API reference |
| Evals | **Adequate** | 20 scenarios, 11 of them failure or safety cases; outcome graders (DB state, tool arguments, stated facts); real API, real PostGIS and real MCP over stdio; CI job; the harness itself is tested | The Claude driver has only been exercised with a fake client. Single-turn only (no "agent asks → user confirms → agent acts"). Answer checks are regexes, and a correct answer phrased unusually can fail. No pass@k / variance baseline |
| Safety | **Good for a demonstrator** | `read` < `run` < `admin` scopes; secure default for unknown routes; no destructive tool exists; `read-only` MCP mode by default; `out_dir` confinement; DB-aware single-run guard; write rate limit; secrets censored in `/config` | Tokens are plaintext env vars with no expiry, rotation or hashing. Rate limit and early-failure memory are in-process (lost on restart, not multi-replica). `X-AIDRA-Client` is self-declared |
| Observability | **Adequate** | Every mutating call, refused ones included, gets an audit row, a Loki JSON line and a Prometheus counter, with the request id echoed in errors | No Grafana panel for agent actions; reads aren't audited; no API for a run's logs, so an agent facing `failed_other` can't read the traceback |
| Developer experience | **Adequate** | `python -m src.mcp_server` needs only `mcp` + `httpx`; `claude mcp add …` one-liner; opt-in compose service over HTTP; `python -m evals.run` | The full stack is heavy (torch image, weights download, Copernicus credentials); on a fresh clone the engine is disabled and the DB is empty, so there's nothing for an agent to explore |
| Demo quality | **Weak until deployed** | The traces in `examples/agent-workflows/` are real responses | The public API (`aidra-api.uliber.com`) predates this work; there's no hosted MCP endpoint, so a reviewer can't try it in a minute |

## Weak areas → concrete next steps

1. **Measure a real agent.** Run `python -m evals.run --agent claude --repeat 3 --output reports/evals/claude-opus-5.json`
   and commit the report next to the scripted one. Track pass^3 per scenario. Then
   iterate the tool descriptions and server instructions on the failures, and keep
   each change only if the pass rate moves. That turns "designed for agents" into
   "measured with agents".
2. **Deploy, and host a read-only MCP endpoint.** Push to `main` when
   `GET /api/pipeline/status` is not `busy` (a deploy restarts the container). Then
   expose `aidra-mcp` behind Coolify in `read-only` mode:
   `claude mcp add --transport http aidra https://…/mcp` becomes the 30-second demo on
   real production data.
3. **Give fresh clones something to look at.** Add a `demo` seed: the eval fixture
   under a separate database name, loaded by one command, so `docker compose up`
   plus the MCP server gives an agent a populated world without Copernicus
   credentials or weights.
4. **Stop parsing error strings.** `diagnosis.py` classifies runs by matching
   `error_message` text. Have the engine persist an `error_category` column at write
   time (the categories already exist in `_classify_error` and the diagnosis module)
   and keep the string rules only for historical rows.
5. **Expose run logs.** `GET /api/executions/{id}/logs?limit=200` proxying the LogQL
   query that already works in Grafana (`{service="aidra"} | json | execution_id=…`),
   capped and redacted. It closes the last reason an agent would need the Grafana
   admin password.
6. **Real progress.** Record the stage (`searching`, `downloading n%`, `band k/N`) in
   `execution_log` as the engine goes. `poll_after_seconds` would then be informed
   rather than a constant, and an agent could report something better than "running".
7. **Harden tokens.** Store hashed tokens with expiry in the DB, add
   `POST /api/tokens` (admin) for issue/revoke, and move the rate limit and the
   early-failure memory to Postgres so they survive restarts.
8. **Machine-readable errors in OpenAPI.** Declare an `ErrorResponse` model in every
   route's `responses=`, publish the full error-code catalog in `/api/vocabulary`,
   and translate the remaining Spanish descriptions.
9. **Multi-turn evals.** Add a scripted *user* turn ("yes, run it anyway") so the
   confirmation flows (rejected variant, OOM-prone profile) are graded end to end, not
   only up to the question.

## What was deliberately not done, and still shouldn't be

No chatbot, no RAG, no agent framework, no SQL passthrough tool, no delete or
threshold-tuning tools. Each would add surface without adding product value, and the
last three would let an agent change or destroy the evidence the study exists to
produce.
