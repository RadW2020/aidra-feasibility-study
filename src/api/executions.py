"""Execution history and diagnosis: ``GET /api/executions[/{id}]``.

``/api/traceability/{id}`` answers "give me everything about this run" for a
human with a dashboard (every column, every detection). These two routes
answer the questions a client asks *before* it has an id and *after* a run
ends: which runs happened, which failed, and why — in domain terms, with the
evidence fields named and the next step suggested, and with payloads bounded
so an agent's context window survives a 4 000-detection scene.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query

from src.api.errors import ApiError
from src.config import Settings
from src.db.connection import db
from src.db.queries import (
    COUNT_EXECUTION_SUMMARIES,
    SELECT_CUE_LINEAGE,
    SELECT_DETECTION_BREAKDOWN,
    SELECT_EXECUTION_BY_ID,
    SELECT_EXECUTION_SUMMARIES,
    SELECT_MODEL_STATUS,
    SELECT_TOP_DETECTIONS,
)
from src.profiles.definitions import PROFILES
from src.traceability import diagnosis
from src.vocabulary import EXECUTION_STATUSES, TRIGGER_TYPES

logger = logging.getLogger("aidra.api.executions")

router = APIRouter(prefix="/executions", tags=["executions"])

_POLL_SECONDS = {"pending": 30, "running": 120}


def _parse_statuses(status: str | None) -> list[str] | None:
    if not status:
        return None
    values = [s.strip() for s in status.split(",") if s.strip()]
    unknown = [s for s in values if s not in EXECUTION_STATUSES]
    if unknown:
        raise ApiError(
            422, "unknown_status", f"Unknown execution status: {', '.join(unknown)}",
            valid_values=list(EXECUTION_STATUSES),
            hint="Comma-separate several statuses, e.g. status=error,failed.",
        )
    return values


def _check_enum(name: str, value: str | None, valid: list[str], code: str) -> None:
    if value is not None and value not in valid:
        raise ApiError(422, code, f"Unknown {name} '{value}'", valid_values=valid)


def _summary(row: dict[str, Any], reaper_minutes: int) -> dict[str, Any]:
    outcome = diagnosis.diagnose(row, reaper_threshold_minutes=reaper_minutes)
    error = row.get("error_message")
    return {
        "id": str(row["id"]),
        "created_at": row["created_at"],
        "status": row["status"],
        "outcome": {"category": outcome["category"], "summary": outcome["summary"]},
        "trigger_type": row.get("trigger_type"),
        "triggered_by": str(row["triggered_by"]) if row.get("triggered_by") else None,
        "zone": row.get("search_zone"),
        "image_id": row.get("image_id"),
        "image_sensing_date": row.get("image_sensing_date"),
        "model": {
            "name": row.get("model_name"),
            "version": row.get("model_version"),
            "compression_technique": row.get("compression_technique"),
        },
        "profile": row.get("constraint_profile"),
        "num_detections": row.get("num_detections"),
        "num_valid_targets": row.get("num_valid_targets"),
        "total_duration_ms": row.get("total_duration_ms"),
        "peak_ram_mb": row.get("peak_ram_mb"),
        "memory_budget_mb": diagnosis.memory_budget_mb(row),
        "error_summary": (error[:200] + "…") if error and len(error) > 200 else error,
    }


@router.get("")
async def list_executions(
    status: str | None = Query(
        None,
        description="Comma-separated execution statuses (pending, running, success, skipped, invalid, error, failed). See GET /api/vocabulary.",
    ),
    profile: str | None = Query(None, description="Constraint profile: ground, sat-high, sat-mid, sat-low, sat-extreme."),
    model: str | None = Query(None, description="Model name, e.g. vesseltracker-sar-yolov8."),
    model_version: str | None = Query(None, description="Model version, e.g. v1.0, int8-static."),
    trigger_type: str | None = Query(None, description="manual, scheduled or cue."),
    zone: str | None = Query(None, description="Search zone of the run, e.g. gibraltar."),
    image_id: str | None = Query(None, description="Copernicus product id: every run on one scene."),
    since: datetime | None = Query(None, description="ISO 8601; runs created at or after."),
    until: datetime | None = Query(None, description="ISO 8601; runs created at or before."),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Pipeline runs of every status, newest first, each with a one-line outcome.

    Unlike STAC items (successful runs only) this includes failures, skips
    and reaped orphans — the rows needed to answer "what failed and why".
    """
    statuses = _parse_statuses(status)
    _check_enum("profile", profile, list(PROFILES), "unknown_profile")
    _check_enum("trigger_type", trigger_type, list(TRIGGER_TYPES), "unknown_trigger_type")
    args = (statuses, profile, model, model_version, trigger_type, zone, image_id, since, until)
    rows = await db.fetch(SELECT_EXECUTION_SUMMARIES, *args, limit, offset)
    total = int(await db.fetchval(COUNT_EXECUTION_SUMMARIES, *args) or 0)
    reaper = Settings().orphan_reaper_threshold_minutes
    return {
        "items": [_summary(dict(r), reaper) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": offset + limit if offset + limit < total else None,
    }


@router.get("/{execution_id}")
async def get_execution(
    execution_id: UUID,
    top_detections: int = Query(5, ge=0, le=50, description="How many highest-confidence detections to include."),
) -> dict[str, Any]:
    """One run explained: outcome and its evidence, detection breakdown, provenance check, lineage.

    For the full row and every detection use ``GET /api/traceability/{id}``.
    """
    row = await db.fetchrow(SELECT_EXECUTION_BY_ID, execution_id)
    if row is None:
        from src.api.pipeline import early_failure

        rejected = early_failure(execution_id)
        if rejected is not None:
            raise ApiError(
                404, "execution_rejected_before_start",
                f"Execution {execution_id} was accepted by the API but failed before its record was created: {rejected['error']}",
                hint="Nothing ran and nothing was persisted. Fix the request (see POST /api/pipeline/preview) and trigger again.",
                failed_at=rejected["failed_at"],
            )
        raise ApiError(
            404, "execution_not_found", f"Execution {execution_id} not found",
            hint="List runs with GET /api/executions; ids come from the trigger response or that list.",
        )
    rec = dict(row)
    settings = Settings()
    outcome = diagnosis.diagnose(rec, reaper_threshold_minutes=settings.orphan_reaper_threshold_minutes)

    breakdown = await db.fetch(SELECT_DETECTION_BREAKDOWN, execution_id)
    by_source: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for b in breakdown:
        by_source[b["source"]] = by_source.get(b["source"], 0) + int(b["n"])
        by_verdict[b["quality_verdict"]] = by_verdict.get(b["quality_verdict"], 0) + int(b["n"])
    top = [dict(r) for r in await db.fetch(SELECT_TOP_DETECTIONS, execution_id, top_detections)] if top_detections else []
    for t in top:
        t["id"] = str(t["id"])

    cues = [dict(c) for c in await db.fetch(SELECT_CUE_LINEAGE, execution_id)]
    upstream = next((c for c in cues if c.get("execution_id") == execution_id), None)
    downstream = [c for c in cues if c.get("triggered_by") == execution_id]

    registry = await db.fetchrow(SELECT_MODEL_STATUS, rec.get("model_name"), rec.get("model_version"))
    meaning = diagnosis.status_meaning(rec.get("status"))
    eid = str(execution_id)
    return {
        "id": eid,
        "status": rec.get("status"),
        "status_meaning": meaning["meaning"],
        "is_terminal": meaning["terminal"],
        "poll_after_seconds": None if meaning["terminal"] else _POLL_SECONDS.get(rec.get("status"), 60),
        "outcome": outcome,
        "created_at": rec.get("created_at"),
        "trigger": {
            "type": rec.get("trigger_type"),
            "triggered_by_execution": str(rec["triggered_by"]) if rec.get("triggered_by") else None,
        },
        "zone": rec.get("search_zone"),
        "image": {
            "id": rec.get("image_id"),
            "title": rec.get("image_title"),
            "sensing_date": rec.get("image_sensing_date"),
            "size_mb": rec.get("image_size_mb"),
            "polarisation": rec.get("polarisation"),
            "orbit_direction": rec.get("orbit_direction"),
        },
        "model": {
            "name": rec.get("model_name"),
            "version": rec.get("model_version"),
            "format": rec.get("model_format"),
            "compression_technique": rec.get("compression_technique"),
            "size_mb": rec.get("model_size_mb"),
            "registry_status": registry["status"] if registry else None,
            "rejection_reason": registry["rejection_reason"] if registry else None,
        },
        "parameters": {
            "profile": rec.get("constraint_profile"),
            "confidence_threshold": rec.get("confidence_threshold"),
            "iou_threshold": rec.get("iou_threshold"),
            "tile_size": rec.get("tile_size"),
            "tile_overlap": rec.get("tile_overlap"),
        },
        "resources": {
            "peak_ram_mb": rec.get("peak_ram_mb"),
            "memory_budget_mb": diagnosis.memory_budget_mb(rec),
            "cpu_usage_pct": rec.get("cpu_usage_pct"),
            "num_tiles": rec.get("num_tiles"),
        },
        "timings_ms": {
            "total": rec.get("total_duration_ms"),
            "download": rec.get("download_ms"),
            "preprocessing": rec.get("preprocessing_ms"),
            "inference": rec.get("inference_ms"),
            "postprocessing": rec.get("postprocessing_ms"),
            "inference_per_tile_p50": rec.get("inference_p50_ms"),
            "inference_per_tile_p95": rec.get("inference_p95_ms"),
        },
        "detections": {
            "total": rec.get("num_detections"),
            "valid_sea_targets": rec.get("num_valid_targets"),
            "by_source": by_source,
            "by_quality_verdict": by_verdict,
            "top": top,
            "note": "valid_sea_targets excludes land and cluster artefacts (I-DET-2); tier 'high' = source fused.",
        },
        "provenance": diagnosis.provenance(rec),
        "lineage": {
            "upstream_cue": _cue(upstream),
            "downstream_cues": [_cue(c) for c in downstream],
        },
        "error_message": rec.get("error_message"),
        "notes": rec.get("notes"),
        "links": {
            "detections": f"/api/detections?execution_id={eid}",
            "detections_geojson": f"/api/detections.geojson?execution_id={eid}",
            "traceability": f"/api/traceability/{eid}",
            "stac_item": f"/api/stac/collections/detections/items/{eid}",
        },
    }


def _cue(c: dict[str, Any] | None) -> dict[str, Any] | None:
    if c is None:
        return None
    return {
        "id": str(c["id"]),
        "status": c.get("status"),
        "zone": c.get("target_zone"),
        "priority": c.get("priority"),
        "reason": c.get("reason"),
        "created_at": c.get("created_at"),
        "execution_id": str(c["execution_id"]) if c.get("execution_id") else None,
        "result_status": c.get("result_status"),
    }
