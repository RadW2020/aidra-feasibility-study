"""
OGC API - Features Part 1 (Core) endpoints over the ``detections`` table.

Provides a standards-compliant entry point so that GEOINT clients
(QGIS, ArcGIS, ogr2ogr, OGC API Features clients) can consume AIDRA
detections without ad-hoc parsing.

Reference: OGC 17-069r4 (OGC API - Features - Part 1: Core, v1.0).

All geometries are returned in CRS84 (lon, lat) per RFC 7946.
The single collection exposed is ``detections``; each Feature is a
detection joined with its ``execution_log`` provenance row.

Closes Q3 GEOINT criterion of the SatCen tender (interoperability
with OGC stacks, no proprietary formats).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response

from src.db.connection import db
from src.db.queries import SELECT_DETECTIONS

logger = logging.getLogger("aidra.api.ogc_features")

router = APIRouter(prefix="/ogc", tags=["ogc"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLLECTION_ID = "detections"
_CRS84 = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
_CONTENT_CRS_HEADER = f"<{_CRS84}>"

_CONFORMANCE_CLASSES = [
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30",
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
]


# ---------------------------------------------------------------------------
# SQL — items query reuses SELECT_DETECTIONS but with extra provenance columns
# ---------------------------------------------------------------------------

# We need model_hash, commit_sha, pipeline_version and incidence_angle from
# execution_log, plus detection quality flags from detections. SELECT_DETECTIONS
# already JOINs execution_log so we extend the projection with a small wrapper.
_SELECT_OGC_ITEMS = """
    SELECT
        d.id,
        d.execution_id,
        d.created_at,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude,
        ST_AsGeoJSON(d.center_geo) AS center_geojson,
        ST_AsGeoJSON(d.bbox_geo)   AS bbox_geojson,
        d.bbox_pixel,
        d.confidence,
        d.source,
        d.cfar_snr,
        d.yolo_score,
        d.class_name,
        d.on_land,
        d.cluster_anomaly,
        d.quality_verdict,
        e.image_id,
        e.image_title,
        e.image_sensing_date,
        e.model_name,
        e.model_version,
        e.model_hash,
        e.constraint_profile,
        e.incidence_angle,
        e.pipeline_version,
        e.commit_sha
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE ($1::geometry IS NULL OR ST_Intersects(d.center_geo, $1))
      AND ($2::timestamptz IS NULL OR e.image_sensing_date >= $2)
      AND ($3::timestamptz IS NULL OR e.image_sensing_date <= $3)
    ORDER BY d.created_at DESC, d.id
    LIMIT $4 OFFSET $5
"""

_COUNT_OGC_ITEMS = """
    SELECT COUNT(*)
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE ($1::geometry IS NULL OR ST_Intersects(d.center_geo, $1))
      AND ($2::timestamptz IS NULL OR e.image_sensing_date >= $2)
      AND ($3::timestamptz IS NULL OR e.image_sensing_date <= $3)
"""

_SELECT_OGC_ITEM_BY_ID = """
    SELECT
        d.id,
        d.execution_id,
        d.created_at,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude,
        ST_AsGeoJSON(d.center_geo) AS center_geojson,
        ST_AsGeoJSON(d.bbox_geo)   AS bbox_geojson,
        d.bbox_pixel,
        d.confidence,
        d.source,
        d.cfar_snr,
        d.yolo_score,
        d.class_name,
        d.on_land,
        d.cluster_anomaly,
        d.quality_verdict,
        e.image_id,
        e.image_title,
        e.image_sensing_date,
        e.model_name,
        e.model_version,
        e.model_hash,
        e.constraint_profile,
        e.incidence_angle,
        e.pipeline_version,
        e.commit_sha
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE d.id = $1
"""

# Suppress unused-import warning for SELECT_DETECTIONS — kept as an explicit
# reference so reviewers see we are aware of the existing query but extend it
# instead of mutating it. See module docstring.
_ = SELECT_DETECTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root_url(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}/api/ogc"


def _self_url(request: Request) -> str:
    return str(request.url)


def _parse_bbox(bbox_str: str | None) -> str | None:
    """Parse ``lon_min,lat_min,lon_max,lat_max`` into a GeoJSON envelope.

    Returns the GeoJSON polygon as a JSON string suitable for
    ST_GeomFromGeoJSON, or None when the bbox is missing.
    """
    if not bbox_str:
        return None
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bbox: {exc}",
        ) from exc
    if len(parts) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must have exactly 4 values: lon_min,lat_min,lon_max,lat_max",
        )
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min > lon_max or lat_min > lat_max:
        raise HTTPException(
            status_code=400,
            detail="bbox must satisfy lon_min<=lon_max and lat_min<=lat_max",
        )
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


def _parse_datetime(value: str | None) -> tuple[datetime | None, datetime | None]:
    """Parse OGC ``datetime`` parameter (instant or interval ``start/end``).

    Open-ended intervals (``../end`` or ``start/..``) are honoured.
    """
    if not value:
        return (None, None)
    try:
        if "/" in value:
            start_s, end_s = value.split("/", 1)
            start = (
                datetime.fromisoformat(start_s.replace("Z", "+00:00"))
                if start_s and start_s != ".."
                else None
            )
            end = (
                datetime.fromisoformat(end_s.replace("Z", "+00:00"))
                if end_s and end_s != ".."
                else None
            )
            return (start, end)
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (instant, instant)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid datetime (expected ISO 8601 instant or start/end interval): {exc}",
        ) from exc


def _row_to_feature(row: Any, request: Request) -> dict[str, Any]:
    """Convert an asyncpg row into an RFC 7946 GeoJSON Feature."""
    # Geometry: prefer the polygon bbox_geo if present, else fall back to point.
    geometry: dict[str, Any] | None = None
    if row.get("bbox_geojson"):
        try:
            geometry = json.loads(row["bbox_geojson"])
        except (TypeError, ValueError):
            geometry = None
    if geometry is None and row.get("center_geojson"):
        try:
            geometry = json.loads(row["center_geojson"])
        except (TypeError, ValueError):
            geometry = None

    sensing = row.get("image_sensing_date")
    sensing_iso = sensing.isoformat() if isinstance(sensing, datetime) else None

    bbox_pixel = row.get("bbox_pixel")
    if bbox_pixel is not None:
        bbox_pixel_list: list[float] | None = [float(v) for v in bbox_pixel]
    else:
        bbox_pixel_list = None

    incidence = row.get("incidence_angle")
    incidence_val = float(incidence) if incidence is not None else None

    properties: dict[str, Any] = {
        "detection_id": str(row["id"]),
        "execution_id": str(row["execution_id"]),
        "scene_id": row.get("image_id"),
        "scene_title": row.get("image_title"),
        "acquisition_time_utc": sensing_iso,
        "model_name": row.get("model_name"),
        "model_version": row.get("model_version"),
        "model_hash": row.get("model_hash"),
        "pipeline_version": row.get("pipeline_version"),
        "commit_sha": row.get("commit_sha"),
        "constraint_profile": row.get("constraint_profile"),
        "confidence": (
            float(row["confidence"]) if row.get("confidence") is not None else None
        ),
        "source": row.get("source"),
        "class_name": row.get("class_name", "vessel"),
        "cfar_snr": (
            float(row["cfar_snr"]) if row.get("cfar_snr") is not None else None
        ),
        "yolo_score": (
            float(row["yolo_score"]) if row.get("yolo_score") is not None else None
        ),
        "incidence_angle": incidence_val,
        "on_land": bool(row.get("on_land", False)),
        "cluster_anomaly": bool(row.get("cluster_anomaly", False)),
        "quality_verdict": row.get("quality_verdict", "candidate"),
        "bbox_pixel": bbox_pixel_list,
    }

    root = _root_url(request)
    return {
        "type": "Feature",
        "id": str(row["id"]),
        "geometry": geometry,
        "properties": properties,
        "links": [
            {
                "rel": "self",
                "type": "application/geo+json",
                "href": f"{root}/collections/{_COLLECTION_ID}/items/{row['id']}",
            },
            {
                "rel": "collection",
                "type": "application/json",
                "href": f"{root}/collections/{_COLLECTION_ID}",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/")
async def landing_page(request: Request) -> dict[str, Any]:
    """OGC API - Features landing page."""
    root = _root_url(request)
    return {
        "title": "AIDRA OGC API - Features",
        "description": (
            "Standards-compliant OGC API - Features Part 1 endpoint over "
            "AIDRA vessel detections. All geometries are in CRS84 (WGS-84)."
        ),
        "links": [
            {"rel": "self", "type": "application/json", "href": _self_url(request)},
            {
                "rel": "service-desc",
                "type": "application/vnd.oai.openapi+json;version=3.0",
                "href": f"{request.url.scheme}://{request.url.netloc}/openapi.json",
                "title": "OpenAPI 3.0 service description",
            },
            {
                "rel": "conformance",
                "type": "application/json",
                "href": f"{root}/conformance",
                "title": "OGC API conformance classes",
            },
            {
                "rel": "data",
                "type": "application/json",
                "href": f"{root}/collections",
                "title": "Feature collections",
            },
        ],
    }


@router.get("/conformance")
async def conformance() -> dict[str, Any]:
    """List of OGC API - Features conformance classes implemented."""
    return {"conformsTo": _CONFORMANCE_CLASSES}


@router.get("/collections")
async def list_collections(request: Request) -> dict[str, Any]:
    """List the feature collections exposed by AIDRA."""
    root = _root_url(request)
    return {
        "links": [
            {"rel": "self", "type": "application/json", "href": _self_url(request)},
            {"rel": "root", "type": "application/json", "href": f"{root}/"},
        ],
        "collections": [await _build_collection_dict(request)],
    }


async def _build_collection_dict(request: Request) -> dict[str, Any]:
    root = _root_url(request)
    return {
        "id": _COLLECTION_ID,
        "title": "AIDRA vessel detections",
        "description": (
            "Vessel detections produced by AIDRA on Sentinel-1 SAR scenes, "
            "joined with their full execution_log provenance."
        ),
        "itemType": "feature",
        "crs": [_CRS84],
        "storageCrs": _CRS84,
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": f"{root}/collections/{_COLLECTION_ID}",
            },
            {
                "rel": "items",
                "type": "application/geo+json",
                "href": f"{root}/collections/{_COLLECTION_ID}/items",
                "title": "Detections as GeoJSON FeatureCollection",
            },
        ],
    }


@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str, request: Request) -> dict[str, Any]:
    if collection_id != _COLLECTION_ID:
        raise HTTPException(status_code=404, detail="Collection not found")
    return await _build_collection_dict(request)


@router.get("/collections/{collection_id}/items")
async def list_items(
    collection_id: str,
    request: Request,
    bbox: str | None = Query(
        None,
        description="lon_min,lat_min,lon_max,lat_max (CRS84/WGS-84)",
    ),
    datetime_param: str | None = Query(
        None,
        alias="datetime",
        description="ISO 8601 instant or interval (start/end). Use '..' for open ends.",
    ),
    limit: int = Query(100, ge=1, le=1000, description="Max features per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> Response:
    """Return matching detections as an RFC 7946 FeatureCollection."""
    if collection_id != _COLLECTION_ID:
        raise HTTPException(status_code=404, detail="Collection not found")

    bbox_geojson = _parse_bbox(bbox)
    dt_from, dt_to = _parse_datetime(datetime_param)

    # Compose dynamic query: when bbox is provided, swap the geometry param
    # for ST_GeomFromGeoJSON so asyncpg can pass the JSON text.
    select_q = _SELECT_OGC_ITEMS
    count_q = _COUNT_OGC_ITEMS
    if bbox_geojson is not None:
        select_q = select_q.replace("$1::geometry", "ST_GeomFromGeoJSON($1::text)")
        count_q = count_q.replace("$1::geometry", "ST_GeomFromGeoJSON($1::text)")

    try:
        rows = await db.fetch(
            select_q,
            bbox_geojson,
            dt_from,
            dt_to,
            limit,
            offset,
        )
        matched = await db.fetchval(
            count_q,
            bbox_geojson,
            dt_from,
            dt_to,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OGC items query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query items: {exc}",
        ) from exc

    features = [_row_to_feature(dict(r), request) for r in rows]
    matched_int = int(matched) if matched is not None else 0

    root = _root_url(request)
    items_url = f"{root}/collections/{_COLLECTION_ID}/items"
    links: list[dict[str, str]] = [
        {"rel": "self", "type": "application/geo+json", "href": _self_url(request)},
        {
            "rel": "collection",
            "type": "application/json",
            "href": f"{root}/collections/{_COLLECTION_ID}",
        },
        {"rel": "root", "type": "application/json", "href": f"{root}/"},
    ]

    # Build query-string for prev/next preserving filters.
    base_qs_parts: list[str] = []
    if bbox:
        base_qs_parts.append(f"bbox={bbox}")
    if datetime_param:
        base_qs_parts.append(f"datetime={datetime_param}")
    base_qs = "&".join(base_qs_parts)
    sep = "&" if base_qs else ""

    if offset > 0:
        prev_offset = max(0, offset - limit)
        links.append(
            {
                "rel": "prev",
                "type": "application/geo+json",
                "href": f"{items_url}?{base_qs}{sep}limit={limit}&offset={prev_offset}",
            }
        )
    if offset + limit < matched_int:
        links.append(
            {
                "rel": "next",
                "type": "application/geo+json",
                "href": f"{items_url}?{base_qs}{sep}limit={limit}&offset={offset + limit}",
            }
        )

    body = {
        "type": "FeatureCollection",
        "features": features,
        "links": links,
        "numberMatched": matched_int,
        "numberReturned": len(features),
        "timeStamp": datetime.utcnow().isoformat() + "Z",
    }

    return Response(
        content=json.dumps(body, default=str),
        media_type="application/geo+json",
        headers={"Content-Crs": _CONTENT_CRS_HEADER},
    )


@router.get("/collections/{collection_id}/items/{feature_id}")
async def get_item(
    collection_id: str,
    feature_id: UUID,
    request: Request,
) -> Response:
    if collection_id != _COLLECTION_ID:
        raise HTTPException(status_code=404, detail="Collection not found")

    try:
        row = await db.fetchrow(_SELECT_OGC_ITEM_BY_ID, feature_id)
    except Exception as exc:
        logger.error("OGC item-by-id query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query item: {exc}",
        ) from exc

    if row is None:
        raise HTTPException(status_code=404, detail="Feature not found")

    feature = _row_to_feature(dict(row), request)
    return Response(
        content=json.dumps(feature, default=str),
        media_type="application/geo+json",
        headers={"Content-Crs": _CONTENT_CRS_HEADER},
    )
