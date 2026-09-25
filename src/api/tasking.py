"""
Tip & Cue tasking endpoints.

Provides access to the tasking queue (pending, executing, completed
cues) and allows manual creation of cue entries to force observation
of a specific zone.
"""

from __future__ import annotations

import contextlib
import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.api import idempotency
from src.api.audit import note_resource
from src.api.errors import ApiError
from src.db.connection import db
from src.db.models import CueCreateRequest, TaskingEntry
from src.db.queries import INSERT_CUE
from src.vocabulary import CUE_STATUSES

logger = logging.getLogger("aidra.api.tasking")

router = APIRouter(tags=["tasking"])

# ---------------------------------------------------------------------------
# SQL for queue listing with optional status filter
# ---------------------------------------------------------------------------

_SELECT_TASKING_QUEUE = """
    SELECT *, ST_AsGeoJSON(target_bbox) AS target_bbox_geojson
    FROM tasking_queue
    WHERE ($1::text IS NULL OR status = $1)
    ORDER BY priority DESC, created_at
    LIMIT $2
"""


def _row_to_tasking_entry(row) -> TaskingEntry:  # type: ignore[no-untyped-def]
    """Convert an asyncpg Record to a TaskingEntry model."""
    bbox_geojson = None
    raw = row.get("target_bbox_geojson")
    if raw:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            bbox_geojson = json.loads(raw) if isinstance(raw, str) else raw

    return TaskingEntry(
        id=row["id"],
        created_at=row["created_at"],
        trigger_type=row["trigger_type"],
        triggered_by=row.get("triggered_by"),
        target_bbox_geojson=bbox_geojson,
        target_zone=row.get("target_zone"),
        priority=row.get("priority", 0),
        reason=row.get("reason"),
        status=row["status"],
        execution_id=row.get("execution_id"),
        result_status=row.get("result_status"),
        confirmed_detections=row.get("confirmed_detections"),
        attempts=row.get("attempts", 0),
    )


@router.get("/tasking/queue", response_model=list[TaskingEntry])
async def list_tasking_queue(
    status: str | None = Query(
        None,
        description="Filter by status: pending, processing, completed, failed",
    ),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
) -> list[TaskingEntry]:
    """List entries in the Tip & Cue tasking queue.

    Returns cue entries ordered by priority (descending) then creation
    time.  Optionally filter by *status* (``pending``, ``processing``,
    ``completed``, ``failed``).
    """
    if status is not None and status not in CUE_STATUSES:
        raise ApiError(422, "unknown_status", f"Unknown cue status '{status}'", valid_values=list(CUE_STATUSES))
    try:
        rows = await db.fetch(_SELECT_TASKING_QUEUE, status, limit)
        return [_row_to_tasking_entry(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to list tasking queue: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query tasking queue: {exc}",
        ) from exc


_SELECT_EQUIVALENT_LIVE_CUE = """
    SELECT id, created_at, status, priority
    FROM tasking_queue
    WHERE status IN ('pending', 'processing')
      AND ST_Equals(target_bbox, ST_GeomFromGeoJSON($1))
    ORDER BY created_at
    LIMIT 1
"""

_SELECT_QUEUE_POSITION = """
    SELECT COUNT(*) + 1 AS position
    FROM tasking_queue
    WHERE status = 'pending' AND id <> $1
      AND (priority > $2 OR (priority = $2 AND created_at < NOW()))
"""

_BBOX_FORMAT = "[lon_min, lat_min, lon_max, lat_max] in WGS-84 degrees"
MAX_PRIORITY = 10


def _validate_bbox(bbox: list[float]) -> None:
    """400 with an actionable message for anything PostGIS would store but the cue processor can't use."""
    if not bbox or len(bbox) != 4:
        raise ApiError(
            400, "invalid_bbox",
            "bbox must have exactly 4 values: [lon_min, lat_min, lon_max, lat_max]",
            hint=_BBOX_FORMAT,
        )
    lon_min, lat_min, lon_max, lat_max = bbox
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180 and -90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ApiError(400, "invalid_bbox", "bbox is outside lon [-180, 180] / lat [-90, 90]",
                       hint=f"{_BBOX_FORMAT}; longitude comes first.")
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ApiError(400, "invalid_bbox", "bbox needs lon_min < lon_max and lat_min < lat_max",
                       hint=f"{_BBOX_FORMAT}; check the order of the corners.")


def _zone_context(zone: str | None, bbox: list[float]) -> tuple[str, list[dict]]:
    """Resolve the search zone the cue processor will use, and warn if it can't see the bbox.

    The processor searches Copernicus by the *search zone's* bbox
    (``resolve_search_zone``), not by the cue's, so a cue outside that zone
    would re-process a scene that doesn't cover it.
    """
    from src.pipeline.ingestion import SEARCH_ZONES
    from src.tipcue.zones import DEFAULT_ZONES, resolve_search_zone

    tipcue_ids = [z.id for z in DEFAULT_ZONES]
    if zone is not None and zone not in SEARCH_ZONES and zone not in tipcue_ids:
        raise ApiError(
            400, "unknown_zone", f"Unknown zone '{zone}'",
            hint="Use a Tip & Cue zone id or a pipeline search zone (GET /api/catalog), or omit it.",
            valid_values=tipcue_ids + list(SEARCH_ZONES),
        )
    search_zone = resolve_search_zone(zone)
    warnings: list[dict] = []
    zb = SEARCH_ZONES.get(search_zone, {}).get("bbox")
    if zb and not (bbox[0] < zb[2] and bbox[2] > zb[0] and bbox[1] < zb[3] and bbox[3] > zb[1]):
        warnings.append({
            "code": "bbox_outside_search_zone",
            "message": (
                f"The cue processor searches scenes of '{search_zone}' {zb}; this bbox lies outside it, "
                "so the re-observation will not cover the requested area."
            ),
            "hint": "Pass the zone that contains the bbox (GET /api/catalog lists zones with bboxes).",
        })
    return search_zone, warnings


@router.post("/tasking/cue")
async def create_manual_cue(request: CueCreateRequest, http_request: Request) -> dict:
    """Queue a manual Tip & Cue observation of an area (scope ``run``).

    Bypasses the automatic TipEvaluator to force a re-observation. The cue
    processor picks pending cues every 15 minutes, highest priority first,
    and runs the pipeline on the newest scene of the cue's search zone.

    Refuses an identical cue that is still pending or processing (409
    ``duplicate_cue`` with its id) unless ``allow_duplicate`` is true, and
    honours ``Idempotency-Key`` so a retried call never queues twice.

    Args:
        request: includes ``bbox`` (``[lon_min, lat_min, lon_max, lat_max]``),
                 ``priority`` (0-10), ``reason``, optional ``zone`` and
                 ``allow_duplicate``.

    Returns:
        ``cue_id`` plus ``created``, ``status``, ``queue_position``, the
        ``search_zone`` the processor will use and any ``warnings``.

    Raises:
        HTTPException 400: if the bounding box, priority or zone is invalid.
        HTTPException 409: if an identical cue is already queued.
    """
    _validate_bbox(request.bbox)
    if not 0 <= request.priority <= MAX_PRIORITY:
        raise ApiError(400, "invalid_priority", f"priority must be between 0 and {MAX_PRIORITY}",
                       hint="Automatic cues use 1-3; reserve higher values for urgent manual requests.")
    search_zone, warnings = _zone_context(request.zone, request.bbox)

    claim = await idempotency.begin(http_request, request.model_dump())
    if claim.replay is not None:
        return claim.replay

    lon_min, lat_min, lon_max, lat_max = request.bbox

    bbox_geojson = json.dumps(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
            ],
        }
    )

    if not request.allow_duplicate:
        existing = await db.fetchrow(_SELECT_EQUIVALENT_LIVE_CUE, bbox_geojson)
        if existing is not None:
            await idempotency.release(claim)
            raise ApiError(
                409, "duplicate_cue",
                f"An identical cue is already {existing['status']}: {existing['id']}",
                hint="Nothing was queued. Follow the existing cue in GET /api/tasking/queue, "
                     "or pass allow_duplicate=true if a second observation is really wanted.",
                existing_cue_id=str(existing["id"]),
                existing_status=existing["status"],
            )

    try:
        cue_id: UUID = await db.fetchval(
            INSERT_CUE,
            None,           # $1 triggered_by (manual — no parent execution)
            None,           # $2 triggering_detections
            bbox_geojson,   # $3 target_bbox (GeoJSON)
            request.zone,   # $4 target_zone
            request.priority,  # $5 priority
            request.reason,    # $6 reason
        )
    except Exception as exc:
        logger.error("Failed to create cue: %s", exc, exc_info=True)
        await idempotency.release(claim)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create cue: {exc}",
        ) from exc

    note_resource(http_request, "cue", cue_id)
    position = None
    try:
        row = await db.fetchrow(_SELECT_QUEUE_POSITION, cue_id, request.priority)
        position = int(row["position"]) if row is not None else None
    except Exception:  # noqa: BLE001 - advisory
        logger.debug("Queue position lookup failed", exc_info=True)

    response = {
        "cue_id": str(cue_id),
        "created": True,
        "status": "pending",
        "priority": request.priority,
        "search_zone": search_zone,
        "queue_position": position,
        "processed_by": "cue_processor job, every 15 minutes, highest priority first",
        "warnings": warnings,
        "follow_up": "GET /api/tasking/queue?status=pending; the resulting run appears in GET /api/executions?trigger_type=cue",
    }
    await idempotency.complete(claim, 200, response)
    return response
