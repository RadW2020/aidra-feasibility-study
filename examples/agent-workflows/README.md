# Agent workflows

End-to-end examples of an agent using AIDRA's agent interface. Each one starts from a
request a person would actually make and shows the primitives the agent combined.
The tool responses are **real**: they were recorded from the evaluation world
(`evals/fixtures/seed.sql`, which mirrors production rows) and abridged with `…`.
Nothing here is a script the agent must follow. The eval scenario named in each file
checks outcomes, not steps.

| # | Request | Kind | Eval |
|---|---|---|---|
| [01](01-investigate-failed-run.md) | "Which runs failed this week, and was it the model or the hardware budget?" | read-only investigation | `triage_recent_failures`, `diagnose_sat_mid` |
| [02](02-evaluate-compression-variant.md) | "Is static INT8 good enough to replace FP32 on the satellite profiles? What evidence is missing?" | multi-step analysis | `triplet_int8_static`, `start_known_oom_config` |
| [03](03-launch-run-and-follow.md) | "Run static INT8 on the latest Gibraltar scene under sat-high and tell me what it found." | state-changing | `start_int8_sat_high`, `pipeline_busy`, `permission_denied` |
| [04](04-queue-reobservation.md) | "A vessel was seen loitering at lat 35.85–35.95, lon −5.40 to −5.50. Get another look." | state-changing + recovery | `queue_new_observation`, `queue_duplicate_observation`, `malformed_bbox_recovery` |

## Connecting an agent

```bash
docker compose up -d                        # API on :8000 (plus PostGIS, Grafana, Prometheus, Loki)
pip install "mcp>=2.2,<3" httpx             # the MCP server needs nothing else
```

**Claude Code** (stdio, from the repository root):

```bash
# read-only: 9 tools, safe against any deployment
claude mcp add aidra --env AIDRA_API_URL=http://localhost:8000 -- python -m src.mcp_server

# operator: + start_detection_run, request_observation (token needs scope 'run')
claude mcp add aidra-operator --env AIDRA_API_URL=http://localhost:8000 \
  --env AIDRA_MCP_MODE=operator --env AIDRA_API_TOKEN=$AGENT_TOKEN -- python -m src.mcp_server
```

**Any MCP client over HTTP** (Cursor, Codex, Claude Code), using the compose service:

```bash
docker compose --profile agent up -d aidra-mcp
claude mcp add --transport http aidra http://localhost:8765/mcp
```

Other clients take the same command in their MCP config, for example Cursor's
`.cursor/mcp.json`:

```json
{ "mcpServers": { "aidra": { "command": "python", "args": ["-m", "src.mcp_server"],
                             "env": { "AIDRA_API_URL": "http://localhost:8000" } } } }
```

**Without MCP.** Every tool is one or two REST calls, so a shell-only agent (Codex,
CI) gets the same behaviour:

```bash
curl -s localhost:8000/api/executions?status=error,failed | jq '.items[] | {id, outcome}'
curl -s -X POST localhost:8000/api/pipeline/preview -H 'content-type: application/json' \
     -d '{"zone":"gibraltar","model":"vesseltracker-sar-yolov8","model_version":"int8-static","profile":"sat-high"}'
```

Token setup for the operator mode (`.env` of the API):

```bash
AIDRA_API_TOKENS=claude-agent:run:<random>,dashboards:read:<random>,ci:admin:<random>
```
