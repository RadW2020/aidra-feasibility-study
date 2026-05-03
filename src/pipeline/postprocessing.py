"""
Postprocesamiento de detecciones.

Responsabilidades:
1. Non-Maximum Suppression (NMS) final sobre detecciones fusionadas
2. Conversion de coordenadas pixel -> geo (lat/lon WGS-84)
3. Filtrado por confianza minima
4. Generacion de GeoJSON de salida
5. Calculo de estadisticas agregadas
6. Fusion de detecciones de multiples tiles eliminando duplicados en
   zonas de solapamiento
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================
# IoU
# ====================================================================


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union between two bounding boxes.

    Parameters
    ----------
    box_a, box_b:
        Bounding boxes in ``[x_min, y_min, x_max, y_max]`` format.

    Returns
    -------
    IoU value in ``[0, 1]``.
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return float(inter_area / union) if union > 0.0 else 0.0


# ====================================================================
# NMS
# ====================================================================


def apply_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Non-Maximum Suppression to remove duplicate detections.

    Algorithm:
    1. Sort detections by confidence (descending).
    2. Starting from the highest confidence, compute IoU against all
       remaining detections of lower confidence.
    3. Suppress (remove) any detection whose IoU with the current one
       exceeds *iou_threshold*.

    Parameters
    ----------
    detections:
        List of detection dicts.  Each must have ``"bbox"``
        ``[x_min, y_min, x_max, y_max]`` and ``"confidence"`` keys.
    iou_threshold:
        Maximum IoU allowed between two surviving detections.

    Returns
    -------
    Filtered list of detections (order preserved by descending confidence).
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    keep: list[dict[str, Any]] = []
    suppressed: set[int] = set()

    for i, det_i in enumerate(sorted_dets):
        if i in suppressed:
            continue
        keep.append(det_i)
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            iou = compute_iou(det_i["bbox"], sorted_dets[j]["bbox"])
            if iou >= iou_threshold:
                suppressed.add(j)

    logger.info(
        "NMS: %d -> %d detections (IoU threshold=%.2f, suppressed %d)",
        len(detections),
        len(keep),
        iou_threshold,
        len(suppressed),
    )
    return keep


# ====================================================================
# Coordinate conversion
# ====================================================================


def pixel_to_geo(
    bbox_pixel: list[float],
    tile_row_offset: int,
    tile_col_offset: int,
    geo_transform: tuple[float, ...],
    crs: str = "EPSG:4326",
) -> dict[str, Any]:
    """Convert a pixel-space bounding box to geographic coordinates.

    Uses the 6-element GDAL-style affine geo-transform to map pixel
    coordinates to the image's coordinate reference system.

    Parameters
    ----------
    bbox_pixel:
        ``[x_min, y_min, x_max, y_max]`` in tile-local pixel coords.
    tile_row_offset:
        Row offset of the tile within the full image.
    tile_col_offset:
        Column offset of the tile within the full image.
    geo_transform:
        ``(origin_x, pixel_size_x, 0, origin_y, 0, -pixel_size_y)``.
    crs:
        Coordinate reference system of the source image.

    Returns
    -------
    Dictionary with ``bbox_geo``, ``center_geo``, ``geometry_point``
    (GeoJSON Point), and ``geometry_polygon`` (GeoJSON Polygon).
    """
    origin_x, px_x, rot_lon, origin_y, rot_lat, px_y = geo_transform[:6]

    x_min, y_min, x_max, y_max = bbox_pixel

    # Convert tile-local to image-global pixel coordinates, then project
    # through the full 6-element affine (rotation included).
    global_x_min = tile_col_offset + x_min
    global_x_max = tile_col_offset + x_max
    global_y_min = tile_row_offset + y_min
    global_y_max = tile_row_offset + y_max

    corner_lons: list[float] = []
    corner_lats: list[float] = []
    for c, r in (
        (global_x_min, global_y_min),
        (global_x_max, global_y_min),
        (global_x_max, global_y_max),
        (global_x_min, global_y_max),
    ):
        corner_lons.append(origin_x + c * px_x + r * rot_lon)
        corner_lats.append(origin_y + c * rot_lat + r * px_y)

    lon_min = float(min(corner_lons))
    lon_max = float(max(corner_lons))
    lat_min = float(min(corner_lats))
    lat_max = float(max(corner_lats))

    cx = (global_x_min + global_x_max) / 2.0
    cy = (global_y_min + global_y_max) / 2.0
    center_lon = origin_x + cx * px_x + cy * rot_lon
    center_lat = origin_y + cx * rot_lat + cy * px_y

    # GeoJSON polygon (closed ring, counter-clockwise)
    polygon_coords = [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],  # close ring
    ]

    result: dict[str, Any] = {
        "bbox_geo": [lon_min, lat_min, lon_max, lat_max],
        "center_geo": [center_lon, center_lat],
        "geometry_point": {
            "type": "Point",
            "coordinates": [center_lon, center_lat],
        },
        "geometry_polygon": {
            "type": "Polygon",
            "coordinates": [polygon_coords],
        },
    }

    logger.debug(
        "pixel_to_geo: pixel [%.0f,%.0f,%.0f,%.0f] -> geo [%.6f,%.6f,%.6f,%.6f] (crs=%s)",
        x_min,
        y_min,
        x_max,
        y_max,
        lon_min,
        lat_min,
        lon_max,
        lat_max,
        crs,
    )
    return result


# ====================================================================
# Merge tile detections
# ====================================================================


def merge_tile_detections(
    tile_detections: list[list[dict[str, Any]]],
    tile_infos: list[dict[str, Any]],
    overlap: int = 64,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge detections from all tiles, removing duplicates in overlap zones.

    Steps:
    1. Convert each detection's ``bbox`` from tile-local to image-global
       pixel coordinates using the tile's ``row_offset`` / ``col_offset``.
    2. Apply global NMS to remove duplicates where tiles overlap.
    3. Return unique detections with updated ``bbox`` in global pixel
       coordinates.

    Parameters
    ----------
    tile_detections:
        List of detection lists, one per tile.
    tile_infos:
        List of tile metadata dicts matching *tile_detections* by index.
        Each must contain ``row_offset`` and ``col_offset`` (ints).
    overlap:
        Pixel overlap between adjacent tiles (used for logging only;
        deduplication relies on IoU).
    iou_threshold:
        IoU threshold for the global NMS pass.

    Returns
    -------
    Deduplicated list of detections with ``bbox_global`` added.
    """
    if len(tile_detections) != len(tile_infos):
        raise ValueError(
            f"tile_detections ({len(tile_detections)}) and tile_infos "
            f"({len(tile_infos)}) must have the same length"
        )

    all_dets: list[dict[str, Any]] = []

    for tile_dets, tile_info in zip(tile_detections, tile_infos, strict=False):
        row_off = tile_info.get("row_offset", 0)
        col_off = tile_info.get("col_offset", 0)

        for det in tile_dets:
            det = dict(det)  # shallow copy to avoid mutating originals
            bbox = det["bbox"]
            det["bbox_global"] = [
                bbox[0] + col_off,
                bbox[1] + row_off,
                bbox[2] + col_off,
                bbox[3] + row_off,
            ]
            # NMS will use bbox_global for overlap comparison
            det["_original_bbox"] = det["bbox"]
            det["bbox"] = det["bbox_global"]
            all_dets.append(det)

    total_before = len(all_dets)

    # Global NMS to remove duplicates in overlap regions
    merged = apply_nms(all_dets, iou_threshold=iou_threshold)

    logger.info(
        "merge_tile_detections: %d tiles, overlap=%d px, %d -> %d detections "
        "(removed %d overlap duplicates)",
        len(tile_detections),
        overlap,
        total_before,
        len(merged),
        total_before - len(merged),
    )
    return merged


# ====================================================================
# Cluster anomaly flagging (I-DET-3)
# ====================================================================


def flag_cluster_anomaly(
    detections: list[Any],
    radius_deg: float = 0.01,
    min_neighbours: int = 8,
) -> int:
    """Marca como ``cluster_anomaly`` las detecciones con vecindario denso.

    Heuristica O(n^2) sobre lon/lat: una deteccion se marca si dentro
    de un radio de ``radius_deg`` (~1.1 km a 0.01 deg en latitud media)
    hay al menos ``min_neighbours`` detecciones. Densidades muy
    elevadas suelen ser artefactos de borde de swath, speckle agudo
    o land returns no filtrados (I-DET-3).

    Parameters
    ----------
    detections:
        Lista de objetos ``Detection`` (con atributos ``center_geo``,
        ``cluster_anomaly``).
    radius_deg:
        Radio de vecindad en grados.
    min_neighbours:
        Numero minimo de vecinos (incluida la propia deteccion) para
        considerar la zona anomala.

    Returns
    -------
    int
        Numero de detecciones marcadas.
    """
    if len(detections) < min_neighbours:
        return 0

    coords: list[tuple[float, float]] = []
    valid_idx: list[int] = []
    for i, det in enumerate(detections):
        center = getattr(det, "center_geo", None)
        if center and len(center) == 2:
            coords.append((float(center[0]), float(center[1])))
            valid_idx.append(i)

    if len(coords) < min_neighbours:
        return 0

    arr = np.asarray(coords)
    flagged = 0
    r2 = radius_deg * radius_deg
    for k, idx in enumerate(valid_idx):
        delta = arr - arr[k]
        d2 = (delta[:, 0] * delta[:, 0]) + (delta[:, 1] * delta[:, 1])
        neighbours = int((d2 <= r2).sum())
        if neighbours >= min_neighbours:
            try:
                detections[idx].cluster_anomaly = True
                flagged += 1
            except (AttributeError, TypeError):
                continue

    if flagged:
        logger.info(
            "Cluster anomaly: flagged=%d / total=%d (radius=%.4f deg, min_n=%d)",
            flagged,
            len(detections),
            radius_deg,
            min_neighbours,
        )
    return flagged


# ====================================================================
# GeoJSON export
# ====================================================================


def detections_to_geojson(
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a GeoJSON FeatureCollection from a list of detections.

    Each detection becomes a GeoJSON Feature.  The ``geometry`` field
    uses a Point at the detection centre if ``center_geo`` or
    ``geometry`` is available; otherwise it falls back to
    ``bbox_geo`` as a Polygon.

    Parameters
    ----------
    detections:
        List of detection dicts.  Each should have at least
        ``"confidence"``, ``"source"``, and one of ``"center_geo"``,
        ``"geometry"``, or ``"bbox_geo"``.

    Returns
    -------
    GeoJSON-compliant ``FeatureCollection`` dict.
    """
    features: list[dict[str, Any]] = []

    for det in detections:
        # Build geometry
        geometry: dict[str, Any] | None = None
        if "geometry" in det and det["geometry"]:
            geometry = det["geometry"]
        elif "center_geo" in det and len(det.get("center_geo", [])) == 2:
            geometry = {
                "type": "Point",
                "coordinates": det["center_geo"],
            }
        elif "bbox_geo" in det and len(det.get("bbox_geo", [])) == 4:
            lon_min, lat_min, lon_max, lat_max = det["bbox_geo"]
            geometry = {
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

        # Build properties (exclude large/internal fields)
        props: dict[str, Any] = {}
        skip_keys = {"geometry", "bbox_global", "_original_bbox", "data"}
        for key, val in det.items():
            if key in skip_keys:
                continue
            # Ensure JSON-serialisable values
            if isinstance(val, (str, int, float, bool, type(None))):
                props[key] = val
            elif isinstance(val, (list, tuple)):
                props[key] = list(val)
            elif isinstance(val, dict):
                props[key] = val
            else:
                props[key] = str(val)

        feature: dict[str, Any] = {
            "type": "Feature",
            "geometry": geometry,
            "properties": props,
        }
        features.append(feature)

    geojson: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }

    logger.info(
        "detections_to_geojson: generated FeatureCollection with %d features",
        len(features),
    )
    return geojson


# ====================================================================
# Aggregate statistics
# ====================================================================


def compute_detection_stats(
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics over a set of detections.

    Parameters
    ----------
    detections:
        List of detection dicts.  Each should have ``"confidence"``,
        ``"source"``, and optionally ``"bbox_geo"`` or ``"center_geo"``.

    Returns
    -------
    Dictionary with ``total``, ``avg_confidence``, ``max_confidence``,
    ``min_confidence``, ``by_source``, and ``spatial_extent``.
    """
    if not detections:
        return {
            "total": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "by_source": {"cfar": 0, "yolo": 0, "fused": 0},
            "spatial_extent": [],
        }

    confidences = [d.get("confidence", 0.0) for d in detections]
    conf_array = np.array(confidences, dtype=np.float64)

    # Count by source
    by_source: dict[str, int] = {"cfar": 0, "yolo": 0, "fused": 0}
    for d in detections:
        src = d.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    # Spatial extent from bbox_geo or center_geo
    lons: list[float] = []
    lats: list[float] = []
    for d in detections:
        if "bbox_geo" in d and len(d["bbox_geo"]) == 4:
            lon_min, lat_min, lon_max, lat_max = d["bbox_geo"]
            lons.extend([lon_min, lon_max])
            lats.extend([lat_min, lat_max])
        elif "center_geo" in d and len(d.get("center_geo", [])) == 2:
            lons.append(d["center_geo"][0])
            lats.append(d["center_geo"][1])

    spatial_extent: list[float] = []
    if lons and lats:
        spatial_extent = [
            float(min(lons)),
            float(min(lats)),
            float(max(lons)),
            float(max(lats)),
        ]

    stats: dict[str, Any] = {
        "total": len(detections),
        "avg_confidence": float(np.mean(conf_array)),
        "max_confidence": float(np.max(conf_array)),
        "min_confidence": float(np.min(conf_array)),
        "by_source": by_source,
        "spatial_extent": spatial_extent,
    }

    logger.info(
        "Detection stats: total=%d, avg_conf=%.3f, by_source=%s",
        stats["total"],
        stats["avg_confidence"],
        stats["by_source"],
    )
    return stats
