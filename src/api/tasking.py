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

from fastapi import APIRouter, HTTPException, Query

from src.db.connection import db
from src.db.models import CueCreateRequest, TaskingEntry
from src.db.queries import INSERT_CUE

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
        description="Filter by status: pending, executing, completed, failed",
    ),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
) -> list[TaskingEntry]:
    """List entries in the Tip & Cue tasking queue.

    Returns cue entries ordered by priority (descending) then creation
    time.  Optionally filter by *status* (``pending``, ``executing``,
    ``completed``, ``failed``).
    """
    try:
        rows = await db.fetch(_SELECT_TASKING_QUEUE, status, limit)
        return [_row_to_tasking_entry(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to list tasking queue: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query tasking queue: {exc}",
        ) from exc


@router.post("/tasking/cue")
async def create_manual_cue(request: CueCreateRequest) -> dict:
    """Create a manual cue entry in the tasking queue.

    This bypasses the automatic TipEvaluator and allows forcing an
    observation of a specific area.  The bounding box is stored as a
    PostGIS polygon.

    Args:
        request: includes ``bbox`` (``[lon_min, lat_min, lon_max, lat_max]``),
                 ``priority``, ``reason``, and optional ``zone``.

    Returns:
        A dictionary with the created ``cue_id``.

    Raises:
        HTTPException 400: if the bounding box is malformed.
    """
    if not request.bbox or len(request.bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must have exactly 4 values: [lon_min, lat_min, lon_max, lat_max]",
        )

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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create cue: {exc}",
        ) from exc

    return {"cue_id": str(cue_id)}
