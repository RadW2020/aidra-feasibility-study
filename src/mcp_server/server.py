"""AIDRA MCP server: the product's operations as typed tools for agents.

Read tools are always registered. The two write tools (``start_detection_run``,
``request_observation``) are registered only in ``operator`` mode, and the API
still checks the token's scope, so a read-only deployment can't be written
to through this server even by mistake.

Tool errors are ``is_error`` results whose text is JSON:
``{"error": {"code", "message", "hint", "valid_values", "retryable", "http_status", "request_id"}}``.
"""

import logging
import time
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from src.mcp_server import __version__
from src.mcp_server.client import AidraApiError, AidraClient, AidraConfig

logger = logging.getLogger("aidra.mcp")

Profile = Literal["ground", "sat-high", "sat-mid", "sat-low", "sat-extreme"]
ExecutionStatus = Literal["pending", "running", "success", "skipped", "invalid", "error", "failed"]
TriggerType = Literal["manual", "scheduled", "cue"]
QualityVerdict = Literal["valid_sea_target", "candidate", "land_artifact", "cluster_artifact", "outside_footprint"]
Bbox = Annotated[
    list[float],
    Field(min_length=4, max_length=4, description="[lon_min, lat_min, lon_max, lat_max] in WGS-84 degrees (longitude first)."),
]
IsoTime = Annotated[str, Field(description="ISO 8601 instant, e.g. 2026-09-18T00:00:00Z.")]

INSTRUCTIONS = """\
AIDRA runs Sentinel-1 radar (SAR) vessel detection (CFAR + YOLOv8) under simulated
satellite hardware budgets ("constraint profiles": ground > sat-high > sat-mid >
sat-low > sat-extreme) and records full provenance for every run. It is an
evaluation study: the evidence is the point.

How to work with it:
- Discover before acting: get_catalog lists zones, profiles, models (with status)
  and what every status/verdict means. Don't guess names.
- A run ("execution") takes 6-52 min. start_detection_run returns an execution_id;
  poll get_execution no faster than its poll_after_seconds. Never loop tightly.
- Always call preview_detection_run first and relay its warnings (rejected or
  candidate model variant, an equivalent run in the last 24 h) to the user.
- Writes: pass an idempotency_key and reuse it when retrying the same request;
  a repeated key returns the original run instead of starting another.
- Errors are JSON with error.code. insufficient_scope / unauthenticated: stop and
  tell the user, don't retry. pipeline_busy / rate_limited: wait. unknown_* /
  invalid_*: fix the argument using error.valid_values.
- Counting vessels: search_detections defaults to sea_only and one_run_per_scene
  (a scene processed under 5 profiles stores 5 copies). State the filters used.
  tier "high" (CFAR and YOLO agree) is a high-precision operating point
  (precision 0.31, recall 0.12 on xView3), not the full picture.
- Nothing is ever deleted: failed runs and rejected variants are evidence and are
  kept, marked. There is no delete tool; say so if asked.
- Report ids (execution, detection, cue) and numbers exactly as returned; if a
  field is null, say it is missing instead of estimating it.
"""


def _ok(data: Any, **extra: Any) -> dict[str, Any]:
    return {**data, **extra} if isinstance(data, dict) else {"result": data, **extra}


async def _call(tool: str, coro: Any) -> Any:
    """Await an API call; an API error becomes an ``is_error`` result whose text is pure JSON.

    Returned rather than raised: a raised ToolError reaches the model as
    ``"Error executing tool …: <text>"``, which a client can't parse as JSON.
    """
    start = time.perf_counter()
    try:
        result = await coro
    except AidraApiError as exc:
        logger.info("tool=%s outcome=error code=%s ms=%.0f", tool, exc.code, (time.perf_counter() - start) * 1000)
        return CallToolResult(
            content=[TextContent(type="text", text=exc.to_json())],
            structured_content={"error": exc.payload},
            is_error=True,
        )
    logger.info("tool=%s outcome=ok ms=%.0f", tool, (time.perf_counter() - start) * 1000)
    return result


def _peer(ctx: Context | None) -> str | None:
    try:
        info = ctx.session.client_params.client_info  # type: ignore[union-attr]
        return f"{info.name}/{info.version}" if info else None
    except Exception:  # noqa: BLE001 - informational only
        return None


def _compact_detection(d: dict[str, Any], api: AidraClient) -> dict[str, Any]:
    return {
        "id": d.get("id"),
        "execution_id": d.get("execution_id"),
        "detected_at": d.get("created_at"),
        "longitude": d.get("longitude"),
        "latitude": d.get("latitude"),
        "confidence": d.get("confidence"),
        "source": d.get("source"),
        "tier": d.get("tier"),
        "quality_verdict": d.get("quality_verdict"),
        "on_land": d.get("on_land"),
        "cluster_anomaly": d.get("cluster_anomaly"),
        "profile": d.get("constraint_profile"),
        "model": d.get("model_name"),
        "image_id": d.get("image_id"),
        "thumbnail_url": api.url(d["thumbnail_url"]) if d.get("thumbnail_url") else None,
    }


def build_server(config: AidraConfig | None = None, api: AidraClient | None = None) -> MCPServer:
    config = config or (api.config if api else AidraConfig.from_env())
    api = api or AidraClient(config)
    server = MCPServer(
        name="aidra",
        title="AIDRA — SAR vessel detection evaluation",
        instructions=INSTRUCTIONS,
        version=__version__,
    )
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    # ------------------------------------------------------------------ reads

    @server.tool(annotations=read)
    async def get_system_status() -> dict[str, Any]:
        """Health of AIDRA and what is running right now.

        Returns database/engine/scheduler health, runs in flight according to
        the database (scheduled and Tip & Cue runs included), the next scheduler
        ticks and the effective configuration hash. Call this before starting
        anything and when diagnosing "nothing happens".
        """
        async def gather() -> dict[str, Any]:
            try:
                health = await api.get("/api/health")
            except AidraApiError as exc:
                if exc.code == "api_unreachable":
                    raise
                health = {"status": "unhealthy", "error": exc.payload}
            status = await api.get("/api/pipeline/status")
            config_doc = await api.get("/api/config")
            s = config_doc.get("settings", {})
            busy = status.get("busy")
            summary = (
                f"{health.get('status', '?')}; db {health.get('db', '?')}; "
                f"engine {'loaded' if status.get('engine_available') else 'NOT loaded'}; "
                + (f"busy with {len(status.get('in_flight', []))} run(s)" if busy else "idle")
            )
            return {
                "api_url": config.base_url,
                "summary": summary,
                "health": health,
                "busy": busy,
                "in_flight": status.get("in_flight", []),
                "engine_available": status.get("engine_available"),
                "scheduler": status.get("scheduler"),
                "config": {
                    "commit_sha": config_doc.get("commit_sha"),
                    "settings_hash": config_doc.get("settings_hash"),
                    "default_zone": s.get("default_zone"),
                    "default_model": s.get("default_model"),
                    "default_model_version": s.get("default_model_version"),
                    "confidence_threshold": s.get("confidence_threshold"),
                    "fusion_mode": s.get("fusion_mode"),
                    "cfar_window": [s.get("cfar_guard_size"), s.get("cfar_training_size")],
                    "scheduler_interval_hours": s.get("scheduler_interval_hours"),
                },
                "mcp_mode": config.mode,
            }

        return await _call("get_system_status", gather())

    @server.tool(annotations=read)
    async def get_catalog() -> dict[str, Any]:
        """Valid values for every other tool: zones, profiles, models, and what each status means.

        Search zones are what a run takes; Tip & Cue zones are operational areas
        that map to a search zone. Models carry registry status (active,
        candidate, rejected + reason), SAR compatibility and AI Act card
        presence. The vocabulary explains execution statuses, quality verdicts,
        detection sources/tiers, trigger types and cue statuses.
        """
        async def gather() -> dict[str, Any]:
            catalog = await api.get("/api/catalog")
            vocabulary = await api.get("/api/vocabulary")
            models = [
                {k: m.get(k) for k in ("name", "version", "status", "rejection_reason", "compression_technique",
                                       "format", "size_mb", "sar_compatible", "has_model_card")}
                for m in catalog.get("models", [])
            ]
            return {**catalog, "models": models, "vocabulary": vocabulary}

        return await _call("get_catalog", gather())

    @server.tool(annotations=read)
    async def list_executions(
        status: Annotated[list[ExecutionStatus] | None, Field(description="Only these statuses, e.g. ['error', 'failed'].")] = None,
        profile: Annotated[Profile | None, Field(description="Constraint profile.")] = None,
        model: Annotated[str | None, Field(description="Model name, e.g. vesseltracker-sar-yolov8 or cfar-default.")] = None,
        model_version: Annotated[str | None, Field(description="Model version, e.g. v1.0 or int8-static.")] = None,
        trigger_type: Annotated[TriggerType | None, Field(description="manual, scheduled or cue.")] = None,
        zone: Annotated[str | None, Field(description="Search zone of the run, e.g. gibraltar.")] = None,
        image_id: Annotated[str | None, Field(description="Copernicus product id: all runs on one scene.")] = None,
        since: Annotated[IsoTime | None, Field(description="Runs created at or after this ISO 8601 instant.")] = None,
        until: Annotated[IsoTime | None, Field(description="Runs created at or before this ISO 8601 instant.")] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        offset: Annotated[int, Field(ge=0, description="Use next_offset from the previous page.")] = 0,
    ) -> dict[str, Any]:
        """Pipeline runs of every status (including failures, skips and reaped runs), newest first.

        Each row has a one-line outcome (e.g. memory_budget_exceeded,
        skipped_duplicate, succeeded) so failures can be triaged from the list.
        Follow up with get_execution for the evidence behind one row.
        """
        params = {
            "status": ",".join(status) if status else None, "profile": profile, "model": model,
            "model_version": model_version, "trigger_type": trigger_type, "zone": zone,
            "image_id": image_id, "since": since, "until": until, "limit": limit, "offset": offset,
        }
        return await _call("list_executions", api.get("/api/executions", params))

    @server.tool(annotations=read)
    async def get_execution(
        execution_id: Annotated[str, Field(description="Execution UUID from start_detection_run or list_executions.")],
        top_detections: Annotated[int, Field(ge=0, le=20, description="Highest-confidence detections to include.")] = 5,
    ) -> dict[str, Any]:
        """One run explained: status meaning, outcome with evidence, detections breakdown, provenance, lineage.

        outcome.category names the cause (memory_budget_exceeded, reaped_orphan,
        skipped_duplicate, ingestion_failed, ...) and outcome.evidence the fields
        that prove it. provenance.complete says whether the run qualifies as
        evidence (all hashes + commit present). While the run is not terminal,
        poll_after_seconds says when to look again.
        """
        return await _call(
            "get_execution",
            api.get(f"/api/executions/{execution_id}", {"top_detections": top_detections}),
        )

    @server.tool(annotations=read)
    async def search_detections(
        zone: Annotated[str | None, Field(description="Search zone or Tip & Cue zone name (see get_catalog). Use instead of bbox.")] = None,
        bbox: Annotated[Bbox | None, Field(description="Area filter; use instead of zone.")] = None,
        since: Annotated[IsoTime | None, Field(description="Detections persisted at or after this instant.")] = None,
        until: Annotated[IsoTime | None, Field(description="Detections persisted at or before this instant.")] = None,
        execution_id: Annotated[str | None, Field(description="Only detections of this run.")] = None,
        min_confidence: Annotated[float | None, Field(ge=0, le=1)] = None,
        tier: Annotated[Literal["high", "standard"] | None, Field(description="high = CFAR and YOLO agreed (source fused).")] = None,
        quality_verdict: Annotated[QualityVerdict | None, Field(description="e.g. valid_sea_target for YOLO-confirmed sea targets.")] = None,
        profile: Annotated[Profile | None, Field(description="Only runs under this constraint profile.")] = None,
        model: Annotated[str | None, Field(description="Only runs of this model name.")] = None,
        sea_only: Annotated[bool, Field(description="Exclude detections on land (kept in the DB as land_artifact, I-DET-2).")] = True,
        one_run_per_scene: Annotated[bool, Field(description="Keep one run per scene (ground preferred, newest) so a vessel is not counted once per profile or re-run.")] = True,
        sort: Annotated[Literal["confidence", "recent"], Field(description="Highest confidence first, or newest first.")] = "confidence",
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> dict[str, Any]:
        """Find vessel detections by area, time, run, confidence or tier; paginated, with the total count.

        Defaults (sea_only, one_run_per_scene) make `total` a defensible count of
        detections, not of database rows. The response echoes the effective
        filters: quote them when reporting a number.
        """
        params = {
            "zone": zone,
            "bbox": ",".join(str(v) for v in bbox) if bbox else None,
            "date_from": since, "date_to": until, "execution_id": execution_id,
            "min_confidence": min_confidence, "tier": tier, "quality_verdict": quality_verdict,
            "profile": profile, "model": model, "on_land": "false" if sea_only else None,
            "one_run_per_scene": "true" if one_run_per_scene else None, "sort": sort,
            "limit": limit, "offset": offset,
        }

        async def gather() -> dict[str, Any]:
            data = await api.get("/api/detections", params)
            return {
                "total": data.get("total"),
                "returned": len(data.get("items", [])),
                "next_offset": data.get("next_offset"),
                "effective_filters": {k: v for k, v in (data.get("filters") or {}).items() if v is not None},
                "items": [_compact_detection(d, api) for d in data.get("items", [])],
            }

        return await _call("search_detections", gather())

    @server.tool(annotations=read)
    async def get_detection(
        detection_id: Annotated[str, Field(description="Detection UUID from search_detections.")],
    ) -> dict[str, Any]:
        """One detection with its full lineage: scores, quality verdict, the run, the model and the hashes.

        Use it to answer "where does this detection come from and can it be
        trusted as evidence?"; get_execution(execution.id) adds the run-level
        provenance verdict.
        """
        async def gather() -> dict[str, Any]:
            r = await api.get(f"/api/detections/{detection_id}")
            return {
                "id": r.get("detection_id"),
                "detected_at": r.get("detection_created_at"),
                "location": {"longitude": r.get("longitude"), "latitude": r.get("latitude"),
                             "bbox_geojson": r.get("bbox_geojson")},
                "scores": {"confidence": r.get("confidence"), "source": r.get("source"), "tier": r.get("tier"),
                           "cfar_snr": r.get("cfar_snr"), "yolo_score": r.get("yolo_score")},
                "quality": {"verdict": r.get("quality_verdict"), "on_land": r.get("on_land"),
                            "cluster_anomaly": r.get("cluster_anomaly")},
                "pixel": {"bbox": r.get("bbox_pixel"), "tile_index": r.get("tile_index")},
                "execution": {
                    "id": r.get("execution_id"), "status": r.get("status"), "trigger_type": r.get("trigger_type"),
                    "zone": r.get("search_zone"), "profile": r.get("constraint_profile"),
                    "image": {"id": r.get("image_id"), "title": r.get("image_title"),
                              "sensing_date": r.get("image_sensing_date")},
                    "model": {"name": r.get("model_name"), "version": r.get("model_version"),
                              "compression_technique": r.get("compression_technique")},
                    "thresholds": {"confidence": r.get("confidence_threshold"), "iou": r.get("iou_threshold")},
                    "hashes": {"image": r.get("image_hash"), "model": r.get("model_hash"),
                               "output": r.get("output_hash"), "input_params": r.get("input_params_hash")},
                },
                "thumbnail_url": api.url(r["thumbnail_url"]) if r.get("thumbnail_url") else None,
            }

        return await _call("get_detection", gather())

    @server.tool(annotations=read)
    async def compare_model_variants(
        variant_version: Annotated[str, Field(description="Compressed variant version, e.g. int8-static.")],
        model: Annotated[str, Field(description="Base model name.")] = "vesseltracker-sar-yolov8",
        baseline_version: Annotated[str, Field(description="FP32 baseline version.")] = "v1.0",
        profile: Annotated[Profile | None, Field(description="Hardware profile for the latency/RAM leg; omit for all.")] = None,
        dataset: Annotated[str | None, Field(description="Validation dataset, e.g. xview3-sar/validation/adriatic; omit for the latest.")] = None,
        pipeline_path: Annotated[Literal["detector", "full"] | None, Field(description="detector = model alone; full = production pipeline.")] = None,
    ) -> dict[str, Any]:
        """Grade a compression triplet {FP32 baseline, variant, profile} against the declared ΔmAP budget.

        Returns verdict (within_budget / exceeds_budget / insufficient_evidence),
        quality deltas (AP, Pd, precision in points; FAR per km²) from legs
        paired by identical pipeline settings, per-profile latency / RAM /
        speed-up, and the evidence still missing. Never infer a verdict from
        latency alone.
        """
        params = {"model": model, "variant_version": variant_version, "baseline_version": baseline_version,
                  "profile": profile, "dataset": dataset, "pipeline_path": pipeline_path}
        return await _call("compare_model_variants", api.get("/api/benchmarks/triplet", params))

    @server.tool(annotations=read)
    async def preview_detection_run(
        zone: Annotated[str | None, Field(description="Search zone (get_catalog). Omit for the default zone.")] = None,
        image_id: Annotated[str | None, Field(description="Exact Copernicus product id instead of 'newest scene in zone'.")] = None,
        model: Annotated[str | None, Field(description="Model name; omit for the default model.")] = None,
        model_version: Annotated[str | None, Field(description="Required when the model has several versions.")] = None,
        profile: Annotated[Profile, Field(description="Hardware budget to simulate.")] = "ground",
        confidence_threshold: Annotated[float | None, Field(ge=0, le=1, description="Omit to use the configured value (I-DET-4).")] = None,
        aoi_bbox: Annotated[Bbox | None, Field(description="Optional sub-area of interest.")] = None,
    ) -> dict[str, Any]:
        """Dry run: what start_detection_run would do with these arguments. Starts and writes nothing.

        Returns ok, the resolved request (model version and thresholds actually
        used), blocking_issues (each with code, hint and valid_values), warnings
        to relay to the user, runs in flight, a duration estimate from history
        and the cost of a run.
        """
        body = _run_body(zone, image_id, model, model_version, profile, confidence_threshold, aoi_bbox)
        return await _call("preview_detection_run", api.post("/api/pipeline/preview", body))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
    async def list_recent_actions(
        since: Annotated[IsoTime | None, Field(description="Only actions at or after this instant.")] = None,
        actor: Annotated[str | None, Field(description="Token name, e.g. claude-agent or operator.")] = None,
        outcome: Annotated[Literal["success", "rejected", "error"] | None, Field(description="rejected = refused (4xx), error = server failure (5xx).")] = None,
        resource_id: Annotated[str | None, Field(description="Only actions on this execution/cue id.")] = None,
        limit: Annotated[int, Field(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        """Audit trail of mutating API calls: who (actor, client), what, on which resource, outcome, error code.

        Answers "what did the agents do?" and "why was this call rejected?".
        Needs a token with scope 'read' or higher.
        """
        params = {"since": since, "actor": actor, "outcome": outcome, "resource_id": resource_id, "limit": limit}
        return await _call("list_recent_actions", api.get("/api/audit/actions", params))

    # ----------------------------------------------------------------- writes

    if config.mode == "operator":

        @server.tool(annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True,
        ))
        async def start_detection_run(
            ctx: Context,
            zone: Annotated[str | None, Field(description="Search zone (get_catalog). Omit for the default zone.")] = None,
            image_id: Annotated[str | None, Field(description="Exact Copernicus product id instead of 'newest scene in zone'.")] = None,
            model: Annotated[str | None, Field(description="Model name; omit for the default model.")] = None,
            model_version: Annotated[str | None, Field(description="Required when the model has several versions.")] = None,
            profile: Annotated[Profile, Field(description="Hardware budget to simulate.")] = "ground",
            confidence_threshold: Annotated[float | None, Field(ge=0, le=1, description="Omit to use the configured value (I-DET-4).")] = None,
            aoi_bbox: Annotated[Bbox | None, Field(description="Optional sub-area of interest.")] = None,
            idempotency_key: Annotated[str | None, Field(
                max_length=200, pattern=r"^[A-Za-z0-9._:-]+$",
                description="Reuse the same key when retrying this exact request: the original run is returned instead of a second one.",
            )] = None,
        ) -> dict[str, Any]:
            """Start a detection run (mutating; needs a token with scope 'run').

            Downloads one Sentinel-1 scene (~1.7 GB) and occupies the host's
            detection capacity for 6-52 min; one run at a time. Call
            preview_detection_run first. Returns the execution_id, the resolved
            request, warnings and when to poll get_execution. Refused with
            pipeline_busy while any run is in flight.
            """
            api.peer = _peer(ctx) or api.peer
            body = _run_body(zone, image_id, model, model_version, profile, confidence_threshold, aoi_bbox)
            data = await _call("start_detection_run", api.post("/api/pipeline/trigger", body, idempotency_key))
            if isinstance(data, CallToolResult):
                return data
            return _ok(data, next_step=f"Call get_execution('{data.get('execution_id')}') after "
                                         f"{(data.get('poll') or {}).get('after_seconds', 30)} s.")

        @server.tool(annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False,
        ))
        async def request_observation(
            ctx: Context,
            reason: Annotated[str, Field(min_length=3, max_length=500, description="Why the area needs another look (recorded on the cue).")],
            bbox: Annotated[Bbox | None, Field(description="Area to re-observe. Omit only when zone is given.")] = None,
            zone: Annotated[str | None, Field(description="Tip & Cue zone or search zone; decides which scenes are searched. Its bbox is used when bbox is omitted.")] = None,
            priority: Annotated[int, Field(ge=0, le=10, description="Higher is served first; automatic cues use 1-3.")] = 1,
            allow_duplicate: Annotated[bool, Field(description="Queue even if an identical cue is already pending.")] = False,
            idempotency_key: Annotated[str | None, Field(
                max_length=200, pattern=r"^[A-Za-z0-9._:-]+$",
                description="Reuse when retrying: a repeated key never queues twice.",
            )] = None,
        ) -> dict[str, Any]:
            """Queue a Tip & Cue re-observation of an area (mutating; needs scope 'run').

            The cue processor runs every 15 minutes and starts a pipeline run on
            the newest scene of the cue's search zone. Refused with duplicate_cue
            (and the existing cue id) if an identical cue is already queued.
            """
            api.peer = _peer(ctx) or api.peer

            async def gather() -> dict[str, Any]:
                area = bbox
                if area is None:
                    if zone is None:
                        raise AidraApiError({
                            "code": "missing_area", "retryable": False,
                            "message": "request_observation needs bbox or zone",
                            "hint": "Ask the user which area to observe; do not invent one.",
                        })
                    catalog = await api.get("/api/catalog")
                    zones = catalog["zones"]["tipcue_zones"] + [
                        {"id": z["name"], "bbox": z["bbox"]} for z in catalog["zones"]["search_zones"]
                    ]
                    match = next((z for z in zones if z["id"] == zone), None)
                    if match is None:
                        raise AidraApiError({
                            "code": "unknown_zone", "retryable": False,
                            "message": f"Unknown zone '{zone}'", "valid_values": [z["id"] for z in zones],
                        })
                    area = match["bbox"]
                body = {"bbox": area, "zone": zone, "priority": priority, "reason": reason,
                        "allow_duplicate": allow_duplicate}
                return await api.post("/api/tasking/cue", body, idempotency_key)

            return await _call("request_observation", gather())

    return server


def _run_body(zone, image_id, model, model_version, profile, confidence_threshold, aoi_bbox) -> dict[str, Any]:
    body: dict[str, Any] = {"profile": profile}
    for key, value in (("zone", zone), ("image_id", image_id), ("model", model),
                       ("model_version", model_version), ("confidence_threshold", confidence_threshold),
                       ("aoi_bbox", aoi_bbox)):
        if value is not None:
            body[key] = value
    return body
