"""
STAC (SpatioTemporal Asset Catalog) endpoints — version minima.

Implementa el subset de STAC 1.0.0 necesario para que la integracion
GEOINT (SatCen, QGIS plugin, pystac) descubra runs y detecciones de
AIDRA via API:

  - GET /api/stac/catalog.json
  - GET /api/stac/collections
  - GET /api/stac/collections/detections
  - GET /api/stac/collections/detections/items
  - GET /api/stac/collections/detections/items/{execution_id}

Cada Item STAC corresponde a una entrada de ``execution_log`` (una
escena procesada). Sus properties incluyen extension ``sar``
(polarisation, orbit, incidence_angle, product_type).

Cierra criterio Q3 GEOINT — interoperabilidad OGC/STAC.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.db.connection import db

logger = logging.getLogger("aidra.api.stac")

router = APIRouter(prefix="/stac", tags=["stac"])


_STAC_VERSION = "1.0.0"
_COLLECTION_ID = "detections"


def _self_url(request: Request) -> str:
    return str(request.url)


def _root_url(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}/api/stac"


# ---------------------------------------------------------------------------
# Catalog root
# ---------------------------------------------------------------------------


@router.get("/catalog.json")
async def stac_catalog(request: Request) -> dict[str, Any]:
    root = _root_url(request)
    return {
        "type": "Catalog",
        "stac_version": _STAC_VERSION,
        "id": "aidra",
        "title": "AIDRA — Vessel detections (SAR)",
        "description": (
            "Catalog of Sentinel-1 SAR scenes processed by AIDRA, "
            "with vessel detections geocoded in WGS-84."
        ),
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/item-search",
            "https://api.stacspec.org/v1.0.0/collections",
            "https://api.stacspec.org/v1.0.0/ogcapi-features",
        ],
        "links": [
            {"rel": "self", "href": _self_url(request)},
            {"rel": "root", "href": f"{root}/catalog.json"},
            {
                "rel": "child",
                "type": "application/json",
                "href": f"{root}/collections/{_COLLECTION_ID}",
                "title": "Vessel detections",
            },
            {
                "rel": "data",
                "href": f"{root}/collections",
            },
            {
                "rel": "search",
                "type": "application/json",
                "method": "POST",
                "href": f"{root}/search",
                "title": "STAC Item Search (POST)",
            },
            {
                "rel": "search",
                "type": "application/json",
                "method": "GET",
                "href": f"{root}/search",
                "title": "STAC Item Search (GET)",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


_SELECT_COLLECTION_EXTENT = """
    SELECT
        ST_XMin(ST_Extent(image_bbox)) AS lon_min,
        ST_YMin(ST_Extent(image_bbox)) AS lat_min,
        ST_XMax(ST_Extent(image_bbox)) AS lon_max,
        ST_YMax(ST_Extent(image_bbox)) AS lat_max,
        MIN(image_sensing_date)        AS sensing_min,
        MAX(image_sensing_date)        AS sensing_max,
        COUNT(*)                       AS n_items
    FROM execution_log
    WHERE status = 'success' AND image_bbox IS NOT NULL
"""


async def _build_collection_dict(root: str) -> dict[str, Any]:
    # Compute extent dynamically from execution_log.
    bbox_extent: list[list[float]] = [[-180.0, -90.0, 180.0, 90.0]]
    temporal_extent: list[list[str | None]] = [[None, None]]
    try:
        row = await db.fetchrow(_SELECT_COLLECTION_EXTENT)
        if row is not None and row.get("lon_min") is not None:
            bbox_extent = [[
                float(row["lon_min"]),
                float(row["lat_min"]),
                float(row["lon_max"]),
                float(row["lat_max"]),
            ]]
            sensing_min = row.get("sensing_min")
            sensing_max = row.get("sensing_max")
            temporal_extent = [[
                sensing_min.isoformat() if sensing_min else None,
                sensing_max.isoformat() if sensing_max else None,
            ]]
    except Exception as exc:
        logger.warning("Could not compute dynamic extent: %s", exc)

    return {
        "type": "Collection",
        "stac_version": _STAC_VERSION,
        "id": _COLLECTION_ID,
        "title": "AIDRA vessel detections",
        "description": (
            "One Item per Sentinel-1 scene processed by AIDRA. "
            "Properties expose SAR metadata (polarisation, orbit, "
            "incidence angle, product type) and detection counts."
        ),
        "license": "CC-BY-SA-3.0-IGO",
        "extent": {
            "spatial": {"bbox": bbox_extent},
            "temporal": {"interval": temporal_extent},
        },
        "stac_extensions": [
            "https://stac-extensions.github.io/sar/v1.0.0/schema.json",
            "https://stac-extensions.github.io/sat/v1.0.0/schema.json",
            "https://stac-extensions.github.io/view/v1.0.0/schema.json",
        ],
        "links": [
            {"rel": "self", "href": f"{root}/collections/{_COLLECTION_ID}"},
            {"rel": "root", "href": f"{root}/catalog.json"},
            {"rel": "parent", "href": f"{root}/catalog.json"},
            {
                "rel": "items",
                "href": f"{root}/collections/{_COLLECTION_ID}/items",
            },
        ],
    }


@router.get("/collections")
async def list_collections(request: Request) -> dict[str, Any]:
    root = _root_url(request)
    return {
        "collections": [await _build_collection_dict(root)],
        "links": [
            {"rel": "self", "href": _self_url(request)},
            {"rel": "root", "href": f"{root}/catalog.json"},
        ],
    }


@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str, request: Request) -> dict[str, Any]:
    if collection_id != _COLLECTION_ID:
        raise HTTPException(status_code=404, detail="Collection not found")
    return await _build_collection_dict(_root_url(request))


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


_SELECT_ITEMS = """
    SELECT
        e.id,
        e.created_at,
        e.image_id,
        e.image_title,
        e.image_hash,
        ST_AsGeoJSON(e.image_bbox) AS image_bbox_geojson,
        e.image_sensing_date,
        e.search_zone,
        e.model_name,
        e.model_version,
        e.constraint_profile,
        e.num_detections,
        e.avg_confidence,
        e.output_hash,
        e.input_params_hash,
        e.commit_sha,
        e.incidence_angle,
        e.polarisation,
        e.orbit_direction,
        e.relative_orbit,
        e.product_type,
        e.pixel_spacing,
        e.status
    FROM execution_log e
    WHERE e.status = 'success'
    ORDER BY e.created_at DESC
    LIMIT $1 OFFSET $2
"""

_SELECT_ITEM_BY_ID = _SELECT_ITEMS.replace(
    "WHERE e.status = 'success'\n    ORDER BY e.created_at DESC\n    LIMIT $1 OFFSET $2",
    "WHERE e.id = $1",
)


def _row_to_item(row: Any, root: str) -> dict[str, Any]:
    bbox_geom = (
        json.loads(row["image_bbox_geojson"])
        if row.get("image_bbox_geojson")
        else None
    )
    if bbox_geom and bbox_geom.get("type") == "Polygon":
        coords = bbox_geom["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
    else:
        bbox = [-180.0, -90.0, 180.0, 90.0]

    sensing = row.get("image_sensing_date") or row.get("created_at")
    sensing_iso = (
        sensing.isoformat() if isinstance(sensing, datetime) else None
    )

    properties: dict[str, Any] = {
        "datetime": sensing_iso,
        "title": row.get("image_title") or row.get("image_id"),
        "platform": "sentinel-1",
        "constellation": "sentinel-1",
        # SAR extension
        "sar:instrument_mode": "IW",
        "sar:product_type": row.get("product_type"),
        "sar:polarizations": (
            row["polarisation"].split("+")
            if row.get("polarisation")
            else None
        ),
        "sar:relative_orbit": row.get("relative_orbit"),
        "sat:orbit_state": (
            row["orbit_direction"].lower()
            if row.get("orbit_direction")
            else None
        ),
        "view:incidence_angle": row.get("incidence_angle"),
        # AIDRA-specific traceability
        "aidra:model_name": row.get("model_name"),
        "aidra:model_version": row.get("model_version"),
        "aidra:constraint_profile": row.get("constraint_profile"),
        "aidra:image_hash": row.get("image_hash"),
        "aidra:output_hash": row.get("output_hash"),
        "aidra:input_params_hash": row.get("input_params_hash"),
        "aidra:commit_sha": row.get("commit_sha"),
        "aidra:num_detections": row.get("num_detections"),
        "aidra:avg_confidence": (
            float(row["avg_confidence"])
            if row.get("avg_confidence") is not None
            else None
        ),
        "aidra:search_zone": row.get("search_zone"),
    }

    return {
        "type": "Feature",
        "stac_version": _STAC_VERSION,
        "stac_extensions": [
            "https://stac-extensions.github.io/sar/v1.0.0/schema.json",
            "https://stac-extensions.github.io/sat/v1.0.0/schema.json",
            "https://stac-extensions.github.io/view/v1.0.0/schema.json",
        ],
        "id": str(row["id"]),
        "collection": _COLLECTION_ID,
        "geometry": bbox_geom,
        "bbox": bbox,
        "properties": properties,
        "assets": {
            "detections": {
                "href": (
                    f"{root.replace('/stac', '')}/detections.geojson"
                    f"?execution_id={row['id']}"
                ),
                "type": "application/geo+json",
                "title": "Detections (GeoJSON FeatureCollection)",
                "roles": ["data"],
            },
            "thumbnail_gallery": {
                "href": (
                    f"{root.replace('/stac', '')}/detections"
                    f"?execution_id={row['id']}&limit=50"
                ),
                "type": "application/json",
                "title": "Top-confidence detection list with thumbnail URLs",
                "roles": ["overview"],
            },
        },
        "links": [
            {
                "rel": "self",
                "href": f"{root}/collections/{_COLLECTION_ID}/items/{row['id']}",
            },
            {"rel": "root", "href": f"{root}/catalog.json"},
            {
                "rel": "parent",
                "href": f"{root}/collections/{_COLLECTION_ID}",
            },
            {
                "rel": "collection",
                "href": f"{root}/collections/{_COLLECTION_ID}",
            },
        ],
    }


@router.get("/collections/{collection_id}/items")
async def list_items(
    collection_id: str,
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    if collection_id != _COLLECTION_ID:
        raise HTTPException(status_code=404, detail="Collection not found")
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be in [1, 500]")

    root = _root_url(request)
    rows = await db.fetch(_SELECT_ITEMS, limit, offset)
    features = [_row_to_item(dict(r), root) for r in rows]

    matched = await db.fetchval(
        "SELECT COUNT(*) FROM execution_log WHERE status = 'success'"
    )

    items_url = f"{root}/collections/{_COLLECTION_ID}/items"
    links = [
        {"rel": "self", "href": _self_url(request)},
        {"rel": "root", "href": f"{root}/catalog.json"},
        {
            "rel": "parent",
            "href": f"{root}/collections/{_COLLECTION_ID}",
        },
    ]
    if offset > 0:
        prev_offset = max(0, offset - limit)
        links.append({
            "rel": "prev",
            "href": f"{items_url}?limit={limit}&offset={prev_offset}",
        })
    if matched is not None and offset + limit < int(matched):
        links.append({
            "rel": "next",
            "href": f"{items_url}?limit={limit}&offset={offset + limit}",
        })

    matched_int = int(matched) if matched is not None else 0
    return {
        "type": "FeatureCollection",
        "stac_version": _STAC_VERSION,
        "features": features,
        "links": links,
        # OGC API Features 10.4 / STAC API Item Search numbers.
        "numberMatched": matched_int,
        "numberReturned": len(features),
        # Legacy STAC `context` extension (kept for back-compat with
        # older pystac-client / EODAG versions).
        "context": {
            "returned": len(features),
            "limit": limit,
            "matched": matched_int,
        },
        "timeStamp": datetime.now(tz=UTC).isoformat(),
    }


@router.get("/collections/{collection_id}/items/{execution_id}")
async def get_item(
    collection_id: str, execution_id: str, request: Request
) -> dict[str, Any]:
    if collection_id != _COLLECTION_ID:
        raise HTTPException(status_code=404, detail="Collection not found")
    row = await db.fetchrow(_SELECT_ITEM_BY_ID, execution_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return _row_to_item(dict(row), _root_url(request))


# ---------------------------------------------------------------------------
# Item Search (STAC API - Item Search)
# ---------------------------------------------------------------------------


_SELECT_ITEMS_SEARCH = """
    SELECT
        e.id,
        e.created_at,
        e.image_id,
        e.image_title,
        e.image_hash,
        ST_AsGeoJSON(e.image_bbox) AS image_bbox_geojson,
        e.image_sensing_date,
        e.search_zone,
        e.model_name,
        e.model_version,
        e.constraint_profile,
        e.num_detections,
        e.avg_confidence,
        e.output_hash,
        e.input_params_hash,
        e.commit_sha,
        e.incidence_angle,
        e.polarisation,
        e.orbit_direction,
        e.relative_orbit,
        e.product_type,
        e.pixel_spacing,
        e.status
    FROM execution_log e
    WHERE e.status = 'success'
      AND (
            $1::double precision IS NULL
            OR ST_Intersects(
                e.image_bbox,
                ST_MakeEnvelope($1, $2, $3, $4, 4326)
            )
          )
      AND ($5::timestamptz IS NULL OR e.image_sensing_date >= $5)
      AND ($6::timestamptz IS NULL OR e.image_sensing_date <= $6)
    ORDER BY e.image_sensing_date DESC NULLS LAST, e.created_at DESC
    LIMIT $7 OFFSET $8
"""

_COUNT_ITEMS_SEARCH = """
    SELECT COUNT(*)
    FROM execution_log e
    WHERE e.status = 'success'
      AND (
            $1::double precision IS NULL
            OR ST_Intersects(
                e.image_bbox,
                ST_MakeEnvelope($1, $2, $3, $4, 4326)
            )
          )
      AND ($5::timestamptz IS NULL OR e.image_sensing_date >= $5)
      AND ($6::timestamptz IS NULL OR e.image_sensing_date <= $6)
"""


class StacSearchBody(BaseModel):
    """Request body for ``POST /api/stac/search``.

    Mirrors a minimal subset of the STAC API Item Search spec
    (https://api.stacspec.org/v1.0.0/item-search/).
    """

    bbox: list[float] | None = Field(
        default=None,
        description="[lon_min, lat_min, lon_max, lat_max] in WGS-84.",
    )
    datetime: str | None = Field(
        default=None,
        description="ISO 8601 instant or interval (start/end). '..' = open end.",
    )
    collections: list[str] | None = Field(
        default=None,
        description="Restrict to collection ids. Only 'detections' supported.",
    )
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    query: dict[str, Any] | None = Field(
        default=None,
        description="Property-level filter (CQL-lite). Currently advisory only.",
    )


def _parse_stac_datetime(value: str | None) -> tuple[datetime | None, datetime | None]:
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
            detail=f"Invalid datetime: {exc}",
        ) from exc


async def _do_stac_search(
    request: Request,
    bbox: list[float] | None,
    datetime_str: str | None,
    collections: list[str] | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    if collections and any(c != _COLLECTION_ID for c in collections):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown collection in 'collections'; only '{_COLLECTION_ID}' is supported.",
        )

    if bbox is not None:
        if len(bbox) != 4:
            raise HTTPException(
                status_code=400,
                detail="bbox must have exactly 4 floats: [lon_min, lat_min, lon_max, lat_max]",
            )
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
        if lon_min > lon_max or lat_min > lat_max:
            raise HTTPException(
                status_code=400,
                detail="bbox must satisfy lon_min<=lon_max and lat_min<=lat_max",
            )
    else:
        lon_min = lat_min = lon_max = lat_max = None  # type: ignore[assignment]

    dt_from, dt_to = _parse_stac_datetime(datetime_str)

    try:
        rows = await db.fetch(
            _SELECT_ITEMS_SEARCH,
            lon_min, lat_min, lon_max, lat_max,
            dt_from, dt_to,
            limit, offset,
        )
        matched = await db.fetchval(
            _COUNT_ITEMS_SEARCH,
            lon_min, lat_min, lon_max, lat_max,
            dt_from, dt_to,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("STAC search query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"STAC search failed: {exc}",
        ) from exc

    root = _root_url(request)
    features = [_row_to_item(dict(r), root) for r in rows]
    matched_int = int(matched) if matched is not None else 0

    search_url = f"{root}/search"
    links: list[dict[str, Any]] = [
        {"rel": "self", "href": _self_url(request)},
        {"rel": "root", "href": f"{root}/catalog.json"},
    ]
    if offset > 0:
        prev_offset = max(0, offset - limit)
        links.append({
            "rel": "prev",
            "href": f"{search_url}?limit={limit}&offset={prev_offset}",
        })
    if offset + limit < matched_int:
        links.append({
            "rel": "next",
            "href": f"{search_url}?limit={limit}&offset={offset + limit}",
        })

    return {
        "type": "FeatureCollection",
        "stac_version": _STAC_VERSION,
        "features": features,
        "links": links,
        "numberMatched": matched_int,
        "numberReturned": len(features),
        "context": {
            "returned": len(features),
            "limit": limit,
            "matched": matched_int,
        },
        "timeStamp": datetime.now(tz=UTC).isoformat(),
    }


@router.post("/search")
async def stac_search_post(
    request: Request,
    body: StacSearchBody | None = None,
) -> dict[str, Any]:
    """STAC API - Item Search (POST). Body filters: bbox, datetime, collections, limit, query."""
    if body is None:
        body = StacSearchBody()
    return await _do_stac_search(
        request,
        bbox=body.bbox,
        datetime_str=body.datetime,
        collections=body.collections,
        limit=body.limit,
        offset=body.offset,
    )


@router.get("/search")
async def stac_search_get(
    request: Request,
    bbox: str | None = Query(
        None,
        description="Comma-separated lon_min,lat_min,lon_max,lat_max",
    ),
    datetime: str | None = Query(  # noqa: A002 - STAC spec name
        None,
        description="ISO 8601 instant or start/end interval",
    ),
    collections: str | None = Query(
        None,
        description="Comma-separated collection ids (only 'detections' supported)",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """STAC API - Item Search (GET) for clients like EODAG / pystac-client."""
    bbox_list: list[float] | None = None
    if bbox:
        try:
            bbox_list = [float(x.strip()) for x in bbox.split(",")]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid bbox: {exc}",
            ) from exc
    coll_list = (
        [c.strip() for c in collections.split(",") if c.strip()]
        if collections
        else None
    )
    return await _do_stac_search(
        request,
        bbox=bbox_list,
        datetime_str=datetime,
        collections=coll_list,
        limit=limit,
        offset=offset,
    )
