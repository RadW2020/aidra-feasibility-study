"""
Detection endpoints.

Provides paginated listing of vessel detections with rich filtering
(by profile, model, confidence, bounding box, date range) and a
detail endpoint that returns the full traceability chain for a
single detection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from src.db.connection import db
from src.db.models import DetectionRecord, PaginatedResponse
from src.db.queries import SELECT_DETECTION_BY_ID, SELECT_DETECTIONS
from src.pipeline.postprocessing import detections_to_geojson

logger = logging.getLogger("aidra.api.detections")

router = APIRouter(tags=["detections"])

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

COUNT_DETECTIONS = """
    SELECT COUNT(*)
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE ($1::text IS NULL OR e.constraint_profile = $1)
      AND ($2::text IS NULL OR e.model_name = $2)
      AND ($3::real IS NULL OR d.confidence >= $3)
      AND ($4::timestamptz IS NULL OR d.created_at >= $4)
      AND ($5::timestamptz IS NULL OR d.created_at <= $5)
      AND ($6::geometry IS NULL OR ST_Intersects(d.center_geo, $6))
      AND ($7::boolean IS NULL OR d.on_land = $7)
      AND ($8::boolean IS NULL OR d.cluster_anomaly = $8)
      AND ($9::text IS NULL OR d.quality_verdict = $9)
"""


def _parse_bbox(bbox_str: str | None):
    """Parse a ``lon_min,lat_min,lon_max,lat_max`` string into a PostGIS
    envelope GeoJSON, or return ``None`` if *bbox_str* is empty/invalid.
    """
    if not bbox_str:
        return None
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have exactly 4 values")
        lon_min, lat_min, lon_max, lat_max = parts
        envelope = {
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
        return json.dumps(envelope)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bbox format (expected lon_min,lat_min,lon_max,lat_max): {exc}",
        ) from exc


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse an ISO-8601 date string into a *datetime*, or return ``None``."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format (expected ISO 8601): {exc}",
        ) from exc


def _row_to_detection(row) -> DetectionRecord:  # type: ignore[no-untyped-def]
    """Convert an asyncpg Record to a DetectionRecord."""
    return DetectionRecord(
        id=row["id"],
        execution_id=row["execution_id"],
        created_at=row["created_at"],
        longitude=row["longitude"],
        latitude=row["latitude"],
        bbox_pixel=list(row["bbox_pixel"]) if row["bbox_pixel"] else [],
        confidence=row["confidence"],
        source=row["source"],
        cfar_snr=row.get("cfar_snr"),
        yolo_score=row.get("yolo_score"),
        class_name=row.get("class_name", "vessel"),
        tile_index=row["tile_index"],
        constraint_profile=row.get("constraint_profile"),
        model_name=row.get("model_name"),
        image_id=row.get("image_id"),
        on_land=bool(row.get("on_land", False)),
        cluster_anomaly=bool(row.get("cluster_anomaly", False)),
        quality_verdict=row.get("quality_verdict", "candidate"),
        thumbnail_path=row.get("thumbnail_path"),
        has_thumbnail=row.get("thumbnail_path") is not None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/detections", response_model=PaginatedResponse)
async def list_detections(
    limit: int = Query(50, ge=1, le=500, description="Max items per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    profile: str | None = Query(
        None, description="Filter by constraint profile (e.g. ground, sat-high)"
    ),
    model: str | None = Query(
        None, description="Filter by model name (e.g. yolov8n-sar)"
    ),
    min_confidence: float | None = Query(
        None, ge=0, le=1, description="Minimum confidence threshold"
    ),
    bbox: str | None = Query(
        None,
        description=(
            "Bounding box filter: lon_min,lat_min,lon_max,lat_max. "
            "Only detections whose center falls within this box are returned."
        ),
    ),
    date_from: str | None = Query(
        None, description="Start date (ISO 8601)"
    ),
    date_to: str | None = Query(
        None, description="End date (ISO 8601)"
    ),
    on_land: bool | None = Query(
        None,
        description="Filter by on_land flag (I-DET-2). True only land, False only sea, omit for all.",
    ),
    cluster_anomaly: bool | None = Query(
        None,
        description="Filter by cluster_anomaly flag (I-DET-3). True keeps only anomalies, False excludes them.",
    ),
    quality_verdict: str | None = Query(
        None,
        description=(
            "Filter by quality verdict: valid_sea_target, candidate, "
            "land_artifact, cluster_artifact, outside_footprint."
        ),
    ),
) -> PaginatedResponse:
    """List vessel detections with optional filters.

    Supports filtering by constraint profile, model name, minimum
    confidence, geospatial bounding box and date range.  Results are
    ordered by descending confidence and paginated via *limit* / *offset*.
    """
    dt_from = _parse_date(date_from)
    dt_to = _parse_date(date_to)
    bbox_geojson = _parse_bbox(bbox)

    try:
        # Build the geometry parameter for PostGIS
        # When bbox is provided, we pass the GeoJSON string and let PostGIS
        # cast it via ST_GeomFromGeoJSON inside the query.  The query already
        # expects $6::geometry which asyncpg will handle when we pass None or
        # the text.  We use a wrapper query to handle the ST_GeomFromGeoJSON
        # call inline.
        if bbox_geojson is not None:
            # Use a modified query that converts the GeoJSON text inline
            select_q = SELECT_DETECTIONS.replace(
                "$6::geometry", "ST_GeomFromGeoJSON($6::text)"
            )
            count_q = COUNT_DETECTIONS.replace(
                "$6::geometry", "ST_GeomFromGeoJSON($6::text)"
            )
        else:
            select_q = SELECT_DETECTIONS
            count_q = COUNT_DETECTIONS

        rows = await db.fetch(
            select_q,
            profile,
            model,
            min_confidence,
            dt_from,
            dt_to,
            bbox_geojson,
            limit,
            offset,
            on_land,
            cluster_anomaly,
            quality_verdict,
        )

        total = await db.fetchval(
            count_q,
            profile,
            model,
            min_confidence,
            dt_from,
            dt_to,
            bbox_geojson,
            on_land,
            cluster_anomaly,
            quality_verdict,
        )

        items = [_row_to_detection(r) for r in rows]

        return PaginatedResponse(
            items=items,
            total=total or 0,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("Failed to list detections: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query detections: {exc}",
        ) from exc


@router.get("/detections.geojson")
async def list_detections_geojson(
    limit: int = Query(5000, ge=1, le=50000, description="Hard cap on features"),
    profile: str | None = Query(None),
    model: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0, le=1),
    bbox: str | None = Query(
        None,
        description="lon_min,lat_min,lon_max,lat_max (WGS-84)",
    ),
    date_from: str | None = Query(None, description="ISO 8601"),
    date_to: str | None = Query(None, description="ISO 8601"),
    on_land: bool | None = Query(None, description="Filter on_land flag (I-DET-2)"),
    cluster_anomaly: bool | None = Query(
        None, description="Filter cluster_anomaly flag (I-DET-3)"
    ),
    quality_verdict: str | None = Query(
        None,
        description=(
            "Filter by quality verdict: valid_sea_target, candidate, "
            "land_artifact, cluster_artifact, outside_footprint."
        ),
    ),
    execution_id: UUID | None = Query(
        None,
        description="Restrict to detections from a single pipeline execution.",
    ),
) -> Response:
    """Return detections as RFC 7946 GeoJSON FeatureCollection.

    Same filters as ``GET /detections`` but emits a single
    ``application/geo+json`` document — drop-in for QGIS, Leaflet,
    Mapbox, ogr2ogr, etc.
    """
    dt_from = _parse_date(date_from)
    dt_to = _parse_date(date_to)
    bbox_geojson = _parse_bbox(bbox)
    select_q = (
        SELECT_DETECTIONS.replace("$6::geometry", "ST_GeomFromGeoJSON($6::text)")
        if bbox_geojson is not None
        else SELECT_DETECTIONS
    )
    # Append execution_id filter via inline SQL to avoid breaking the
    # positional contract of SELECT_DETECTIONS used elsewhere.
    if execution_id is not None:
        select_q = select_q.replace(
            "ORDER BY d.confidence DESC",
            f"AND d.execution_id = '{execution_id}'::uuid\nORDER BY d.confidence DESC",
        )
    rows = await db.fetch(
        select_q, profile, model, min_confidence, dt_from, dt_to,
        bbox_geojson, limit, 0, on_land, cluster_anomaly, quality_verdict,
    )
    features_input: list[dict] = []
    for r in rows:
        features_input.append({
            "id": str(r["id"]),
            "execution_id": str(r["execution_id"]),
            "geometry": json.loads(r["center_geojson"]) if r["center_geojson"] else None,
            "bbox_geo": (
                [
                    *json.loads(r["bbox_geojson"])["coordinates"][0][0],
                    *json.loads(r["bbox_geojson"])["coordinates"][0][2],
                ]
                if r.get("bbox_geojson") else []
            ),
            "confidence": float(r["confidence"]),
            "source": r["source"],
            "cfar_snr": r.get("cfar_snr"),
            "yolo_score": r.get("yolo_score"),
            "class_name": r.get("class_name", "vessel"),
            "on_land": bool(r.get("on_land", False)),
            "cluster_anomaly": bool(r.get("cluster_anomaly", False)),
            "quality_verdict": r.get("quality_verdict", "candidate"),
            "model_name": r.get("model_name"),
            "constraint_profile": r.get("constraint_profile"),
            "image_id": str(r["image_id"]) if r.get("image_id") else None,
            "detected_at": r["created_at"].isoformat() if r.get("created_at") else None,
        })
    fc = detections_to_geojson(features_input)
    return Response(
        content=json.dumps(fc, default=str),
        media_type="application/geo+json",
        headers={"Content-Disposition": "inline; filename=\"detections.geojson\""},
    )


@router.get("/detections/{detection_id}")
async def get_detection(detection_id: UUID) -> dict:
    """Return a single detection with its full traceability chain.

    Joins the ``detections`` table with ``execution_log`` to include
    all provenance data: image hash, model hash, output hash, timing
    metrics, constraint profile, etc.

    Raises:
        HTTPException 404: if the detection ID does not exist.
    """
    try:
        row = await db.fetchrow(SELECT_DETECTION_BY_ID, detection_id)
    except Exception as exc:
        logger.error(
            "Failed to fetch detection %s: %s", detection_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {exc}",
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Detection {detection_id} not found",
        )

    # Convert the joined row to a rich response dict
    record = dict(row)

    # Convert geometry columns to serializable values
    for key in list(record.keys()):
        # Remove raw geometry columns that are not JSON-serializable
        val = record[key]
        if hasattr(val, "__geo_interface__") or (
            isinstance(val, (bytes, memoryview))
        ):
            record.pop(key, None)

    return record


# ---------------------------------------------------------------------------
# Thumbnail endpoint (Wow effect #1)
# ---------------------------------------------------------------------------

_FETCH_THUMBNAIL_PATH = """
    SELECT thumbnail_path FROM detections WHERE id = $1
"""


@router.get(
    "/detections/{detection_id}/thumbnail.png",
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Detection or thumbnail not found"},
    },
)
async def get_detection_thumbnail(detection_id: UUID) -> FileResponse:
    """Serves the SAR crop PNG for a single detection.

    Wow effect #1: visual proof of detection. Each PNG is a tight
    crop of the calibrated SAR tile around the detection bbox,
    log-stretched and percentile-clipped for visibility.
    """
    row = await db.fetchrow(_FETCH_THUMBNAIL_PATH, detection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    path_str = row["thumbnail_path"]
    if not path_str:
        raise HTTPException(
            status_code=404,
            detail="Thumbnail not generated for this detection",
        )
    path = Path(path_str)
    if not path.exists():
        logger.warning("Thumbnail referenced but missing on disk: %s", path)
        raise HTTPException(
            status_code=410,
            detail="Thumbnail file is gone (regenerate via re-run)",
        )
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
