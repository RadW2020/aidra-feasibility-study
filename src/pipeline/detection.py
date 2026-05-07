"""
Orquestador de deteccion.  Combina CFAR y YOLO.

Flujo:
1. Recibe tiles preprocesados
2. Ejecuta CFAR en cada tile (rapido, alta recall)
3. Ejecuta YOLO en cada tile (preciso, costoso)
4. Fusiona resultados
5. Registra metricas de rendimiento

Logica de fusion:
- CFAR + YOLO en la misma zona (IoU > 0.3): alta confianza  -> "fused"
- Solo YOLO: confianza media (depende del score YOLO)         -> "yolo"
- Solo CFAR: confianza baja (posible falso positivo)           -> "cfar"
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from src.models.base import BaseDetector
from src.models.cfar import CFARDetector

logger = logging.getLogger(__name__)


# Lazy-imported global-land-mask cache.  None means "not yet attempted",
# False means "import attempted and failed", an object means "ready".
_GLOBE_LAND: Any = None


def _get_globe() -> Any:
    """Return the global_land_mask globe object, or None if unavailable.

    Caches the import so repeat calls in the per-tile loop are free.
    """
    global _GLOBE_LAND
    if _GLOBE_LAND is None:
        try:
            from global_land_mask import globe  # type: ignore[import-not-found]
            _GLOBE_LAND = globe
        except ImportError:
            _GLOBE_LAND = False
    return _GLOBE_LAND if _GLOBE_LAND is not False else None


def _build_sea_mask(
    geo_bounds: dict[str, float | None],
    tile_shape: tuple[int, int],
    coarse: int = 32,
) -> NDArray[np.bool_] | None:
    """Build a sea-only mask for a SAR tile from its lat/lon bounds.

    The mask is True where the pixel falls on sea/ocean and False on land.
    CFAR can use this to skip land pixels entirely — CFAR's Rayleigh
    sea-clutter assumption breaks on land and produces ~90% false
    positives on mixed-coverage scenes (e.g. Strait of Gibraltar).

    The lookup uses a coarse grid (default 32×32, ~200 m per cell on a
    640-pixel tile) and nearest-neighbour upsamples to the full tile
    resolution.  global-land-mask's underlying NOAA dataset is ~1.85 km
    so finer sampling gives no real precision gain.

    Returns None when global-land-mask is not available or when the tile
    has no geocoding info — callers should treat None as "no masking,
    let CFAR run on the whole tile".
    """
    globe = _get_globe()
    if globe is None:
        return None

    lon_min = geo_bounds.get("lon_min")
    lon_max = geo_bounds.get("lon_max")
    lat_min = geo_bounds.get("lat_min")
    lat_max = geo_bounds.get("lat_max")
    if None in (lon_min, lon_max, lat_min, lat_max):
        return None

    rows, cols = tile_shape
    # Coarse lat/lon grid.  Note: global_land_mask expects (lat, lon).
    coarse_lats = np.linspace(lat_min, lat_max, coarse)
    coarse_lons = np.linspace(lon_min, lon_max, coarse)
    lat_grid, lon_grid = np.meshgrid(coarse_lats, coarse_lons, indexing="ij")
    try:
        coarse_sea = np.asarray(globe.is_ocean(lat_grid, lon_grid), dtype=bool)
    except Exception:
        return None

    # Nearest-neighbour upsample to tile resolution.  Image arrays use
    # row=0 at the TOP, but lat increases northwards — flip the lat axis
    # so row 0 corresponds to lat_max.
    coarse_sea = np.flipud(coarse_sea)
    row_idx = np.clip(
        (np.arange(rows) * coarse / rows).astype(np.int64), 0, coarse - 1
    )
    col_idx = np.clip(
        (np.arange(cols) * coarse / cols).astype(np.int64), 0, coarse - 1
    )
    return coarse_sea[np.ix_(row_idx, col_idx)]


# ====================================================================
# Pydantic models
# ====================================================================


class Detection(BaseModel):
    """Single vessel detection after fusion and geolocation."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    bbox_pixel: list[float] = Field(
        ..., description="[x_min, y_min, x_max, y_max] in pixel coords"
    )
    bbox_geo: list[float] = Field(
        default_factory=list,
        description="[lon_min, lat_min, lon_max, lat_max] WGS-84",
    )
    center_geo: list[float] = Field(
        default_factory=list, description="[lon, lat] centre point"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(..., description='"cfar", "yolo", or "fused"')
    cfar_snr: float | None = None
    yolo_score: float | None = None
    tile_index: int = 0
    class_name: str = "vessel"
    geometry: dict[str, Any] = Field(
        default_factory=dict, description="GeoJSON Point"
    )
    on_land: bool = Field(
        default=False,
        description="True si la deteccion cae sobre tierra segun footprint mask (I-DET-2). Excluida de metricas de mar.",
    )
    cluster_anomaly: bool = Field(
        default=False,
        description="True si forma parte de un cluster con densidad > umbral por km2 (I-DET-3). Probable artefacto.",
    )
    thumbnail_path: str | None = Field(
        default=None,
        description="Ruta absoluta al PNG con el crop SAR de la deteccion. Wow effect #1.",
    )


class DetectionMetrics(BaseModel):
    """Performance metrics collected during the detection phase."""

    total_inference_ms: float = 0.0
    cfar_ms: float = 0.0
    yolo_ms: float = 0.0
    fusion_ms: float = 0.0
    peak_ram_mb: float = 0.0
    cpu_percent: float = 0.0
    num_tiles: int = 0
    num_detections_cfar: int = 0
    num_detections_yolo: int = 0
    num_detections_fused: int = 0


class DetectionResult(BaseModel):
    """Complete output of a detection run."""

    detections: list[Detection] = Field(default_factory=list)
    metrics: DetectionMetrics = Field(default_factory=DetectionMetrics)
    cfar_raw: list[dict[str, Any]] = Field(default_factory=list)
    yolo_raw: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None  # constraint-profile observations (e.g. budget breach)


# ====================================================================
# Detection engine
# ====================================================================


class DetectionEngine:
    """Orchestrates dynamic multimodal detection pipelines.

    Can run pure YOLO (optical), pure CFAR, or fused (SAR) logic
    depending on the loaded model.
    """

    def __init__(
        self,
        fusion_iou_threshold: float = 0.3,
        edge_buffer_px: int = 0,
    ) -> None:
        self.fusion_iou_threshold = fusion_iou_threshold
        # I-SAR-2: pixel buffer around scene edges. Detections whose
        # pixel center falls inside the buffer are dropped before
        # geolocation. ``0`` disables the filter (legacy behaviour).
        self.edge_buffer_px = max(0, int(edge_buffer_px))
        logger.info(
            "DetectionEngine initialised (fusion IoU=%.2f, edge_buffer_px=%d)",
            fusion_iou_threshold,
            self.edge_buffer_px,
        )

    def run(
        self,
        tiles: list[dict[str, Any]],
        detector: BaseDetector,
        cfar: CFARDetector | None = None,
        constraint_profile: str = "ground",
        scene_shape: tuple[int, int] | None = None,
        cpu_throttle: Any = None,
    ) -> DetectionResult:
        """Execute detection on a set of tiles using the provided detector.

        ``cpu_throttle`` is an optional :class:`src.profiles.throttle.CPUThrottle`
        instance.  When provided, ``tick()`` is called after each per-tile
        detector invocation so that sub-core constraint profiles can
        emulate fractional-OCPU hardware via wall-clock duty cycling.
        """
        process = psutil.Process()
        process.cpu_percent(interval=None)

        t_start = time.perf_counter()
        ram_peak: float = 0.0

        all_cfar_raw: list[dict[str, Any]] = []
        all_yolo_raw: list[dict[str, Any]] = []
        all_detections: list[Detection] = []

        # --- CFAR pass (optional, typical for SAR) --------------------
        t_cfar_start = time.perf_counter()
        if cpu_throttle is not None:
            cpu_throttle.reset()
        cfar_land_masked_tiles = 0
        if cfar is not None:
            for tile in tiles:
                tile_data: NDArray = tile.get("data", tile.get("array"))
                tile_idx: int = tile.get("tile_index", 0)
                # Build a sea-only mask from the tile's geocoded bounds so
                # CFAR's Rayleigh sea-clutter assumption holds — without
                # this, ~90% of CFAR detections fall on land features
                # (buildings, terrain) on mixed-coverage scenes.
                sea_mask = _build_sea_mask(
                    tile.get("geo_bounds", {}),
                    tile_data.shape,
                )
                if sea_mask is not None:
                    cfar_land_masked_tiles += 1
                # Tighter clustering + SNR gate suppresses sea/edge clutter.
                # min_mean_snr=2.0 → ≥3 dB above local background (vessels are
                # typically 10–30 dB brighter than calm sea).
                cfar_dets = cfar.detect_with_clustering(
                    tile_data,
                    min_cluster_size=5,
                    eps=1.5,
                    min_mean_snr=2.0,
                    valid_mask=sea_mask,
                )
                for d in cfar_dets:
                    d["tile_index"] = tile_idx
                all_cfar_raw.extend(cfar_dets)
                ram_peak = max(ram_peak, process.memory_info().rss / (1024 * 1024))
                if cpu_throttle is not None:
                    cpu_throttle.tick()
            if cfar_land_masked_tiles > 0:
                logger.info(
                    "CFAR land-mask applied to %d/%d tiles "
                    "(remaining tiles had no geocoding)",
                    cfar_land_masked_tiles,
                    len(tiles),
                )
        t_cfar_end = time.perf_counter()

        # --- Primary Detector pass (YOLO/Custom) ----------------------
        t_det_start = time.perf_counter()
        if cpu_throttle is not None:
            cpu_throttle.reset()
        for tile in tiles:
            tile_data = tile.get("data", tile.get("array"))
            tile_idx = tile.get("tile_index", 0)

            # Adapt input shape and dtype for YOLO.
            # SAR tiles are float32 linear sigma0 (positive, ~0..10);
            # YOLO expects uint8 [0,255] RGB.  Apply a SAR-standard log
            # stretch (sigma0_dB ∈ [-25, 0]) and scale to 0..255.
            input_data = tile_data
            if input_data.ndim == 2 and input_data.dtype != np.uint8:
                input_data = _sar_linear_to_uint8_rgb(input_data)
            elif input_data.ndim == 2:
                input_data = np.stack([input_data] * 3, axis=-1)
            elif input_data.ndim == 3 and input_data.shape[2] == 1:
                input_data = np.concatenate([input_data] * 3, axis=-1)

            # Polymorphic predict call
            dets = detector.predict(input_data)
            for d in dets:
                d["tile_index"] = tile_idx
            all_yolo_raw.extend(dets)

            ram_peak = max(ram_peak, process.memory_info().rss / (1024 * 1024))
            if cpu_throttle is not None:
                cpu_throttle.tick()
        t_det_end = time.perf_counter()

        # --- Fusion --------------------------------------------------
        t_fusion_start = time.perf_counter()

        # Group raw detections by tile for per-tile fusion
        cfar_by_tile: dict[int, list[dict]] = {}
        for d in all_cfar_raw:
            cfar_by_tile.setdefault(d["tile_index"], []).append(d)

        yolo_by_tile: dict[int, list[dict]] = {}
        for d in all_yolo_raw:
            yolo_by_tile.setdefault(d["tile_index"], []).append(d)

        tile_indices = set(cfar_by_tile.keys()) | set(yolo_by_tile.keys())
        for tile_idx in sorted(tile_indices):
            tile_cfar = cfar_by_tile.get(tile_idx, [])
            tile_yolo = yolo_by_tile.get(tile_idx, [])

            # Find the tile info for geolocation
            tile_info = next(
                (t for t in tiles if t.get("tile_index") == tile_idx),
                None,
            )

            fused = self._fuse_detections(tile_cfar, tile_yolo, tile_idx)

            # Geolocate if geo_transform is available
            if tile_info and "geo_transform" in tile_info:
                for det in fused:
                    self._geolocate(det, tile_info)

            all_detections.extend(fused)

        # --- Edge swath filter (I-SAR-2) -----------------------------
        # Drop detections whose pixel center lies within
        # ``edge_buffer_px`` of any scene edge. Buffer is in scene
        # pixel coordinates: per-tile bbox + tile offsets.
        if self.edge_buffer_px > 0:
            before_edge = len(all_detections)
            all_detections, edge_dropped = _apply_edge_swath_filter(
                all_detections,
                tiles,
                edge_buffer_px=self.edge_buffer_px,
                scene_shape=scene_shape,
            )
            if edge_dropped:
                logger.info(
                    "Edge swath filter (I-SAR-2): %d → %d detections "
                    "(dropped %d within %d px of swath edge)",
                    before_edge,
                    len(all_detections),
                    edge_dropped,
                    self.edge_buffer_px,
                )

        # --- Cross-tile deduplication --------------------------------
        # Vessels falling in the 64 px overlap between adjacent tiles are
        # detected twice.  Suppress geographic duplicates (~50 m on the
        # ground for typical S1 GRD pixel size).
        before_dedup = len(all_detections)
        all_detections = _dedup_geo_detections(
            all_detections, max_distance_deg=5e-4
        )
        if before_dedup > len(all_detections):
            logger.info(
                "Cross-tile NMS: %d → %d detections (removed %d duplicates)",
                before_dedup,
                len(all_detections),
                before_dedup - len(all_detections),
            )

        t_fusion_end = time.perf_counter()

        cpu_pct = process.cpu_percent(interval=None)
        t_end = time.perf_counter()

        metrics = DetectionMetrics(
            total_inference_ms=(t_end - t_start) * 1000,
            cfar_ms=(t_cfar_end - t_cfar_start) * 1000,
            yolo_ms=(t_det_end - t_det_start) * 1000,
            fusion_ms=(t_fusion_end - t_fusion_start) * 1000,
            peak_ram_mb=ram_peak,
            cpu_percent=cpu_pct if cpu_pct is not None else 0.0,
            num_tiles=len(tiles),
            num_detections_cfar=len(all_cfar_raw),
            num_detections_yolo=len(all_yolo_raw),
            num_detections_fused=len(all_detections),
        )

        logger.info(
            "Detection complete: %d tiles -> %d cfar, %d yolo, %d fused "
            "(%.0f ms total, peak %.0f MB RAM)",
            metrics.num_tiles,
            metrics.num_detections_cfar,
            metrics.num_detections_yolo,
            metrics.num_detections_fused,
            metrics.total_inference_ms,
            metrics.peak_ram_mb,
        )

        return DetectionResult(
            detections=all_detections,
            metrics=metrics,
            cfar_raw=all_cfar_raw,
            yolo_raw=all_yolo_raw,
        )

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------

    def _fuse_detections(
        self,
        cfar_dets: list[dict[str, Any]],
        yolo_dets: list[dict[str, Any]],
        tile_index: int,
    ) -> list[Detection]:
        """Merge CFAR and YOLO detections for a single tile.

        Matching rules:
        - CFAR bbox overlapping a YOLO bbox (IoU >= threshold) are fused
          into a single high-confidence detection.
        - Unmatched YOLO detections are kept with their original score.
        - Unmatched CFAR detections are kept with reduced confidence.
        """
        fused: list[Detection] = []
        matched_cfar: set[int] = set()
        matched_yolo: set[int] = set()

        for ci, c_det in enumerate(cfar_dets):
            c_bbox = c_det["bbox"]
            best_iou = 0.0
            best_yi: int | None = None

            for yi, y_det in enumerate(yolo_dets):
                if yi in matched_yolo:
                    continue
                iou = _compute_iou(c_bbox, y_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_yi = yi

            if best_iou >= self.fusion_iou_threshold and best_yi is not None:
                y_det = yolo_dets[best_yi]
                # Fused: boost confidence
                fused_conf = min(
                    1.0,
                    0.5 * y_det["confidence"] + 0.5 * _snr_to_confidence(c_det["mean_snr"]),
                )
                fused.append(
                    Detection(
                        bbox_pixel=y_det["bbox"],  # prefer YOLO bbox (tighter)
                        confidence=fused_conf,
                        source="fused",
                        cfar_snr=c_det.get("mean_snr"),
                        yolo_score=y_det["confidence"],
                        tile_index=tile_index,
                        class_name=y_det.get("class_name", "vessel"),
                    )
                )
                matched_cfar.add(ci)
                matched_yolo.add(best_yi)

        # Unmatched YOLO detections
        for yi, y_det in enumerate(yolo_dets):
            if yi in matched_yolo:
                continue
            fused.append(
                Detection(
                    bbox_pixel=y_det["bbox"],
                    confidence=y_det["confidence"],
                    source="yolo",
                    yolo_score=y_det["confidence"],
                    tile_index=tile_index,
                    class_name=y_det.get("class_name", "vessel"),
                )
            )

        # Unmatched CFAR detections (lower confidence)
        for ci, c_det in enumerate(cfar_dets):
            if ci in matched_cfar:
                continue
            cfar_conf = _snr_to_confidence(c_det["mean_snr"]) * 0.6
            fused.append(
                Detection(
                    bbox_pixel=c_det["bbox"],
                    confidence=cfar_conf,
                    source="cfar",
                    cfar_snr=c_det.get("mean_snr"),
                    tile_index=tile_index,
                )
            )

        logger.debug(
            "Fusion tile %d: %d cfar + %d yolo -> %d fused "
            "(%d matched, %d yolo-only, %d cfar-only)",
            tile_index,
            len(cfar_dets),
            len(yolo_dets),
            len(fused),
            len(matched_cfar),
            len(yolo_dets) - len(matched_yolo),
            len(cfar_dets) - len(matched_cfar),
        )
        return fused

    # ------------------------------------------------------------------
    # Geolocation
    # ------------------------------------------------------------------

    def _geolocate(
        self, detection: Detection, tile_info: dict[str, Any]
    ) -> None:
        """Convert pixel coordinates to geographic (WGS-84) in-place.

        Uses the affine ``geo_transform`` from the tile metadata to map
        pixel (x, y) to (longitude, latitude).

        Parameters
        ----------
        detection:
            Detection to update (mutated in place).
        tile_info:
            Tile metadata containing ``row_offset``, ``col_offset``, and
            ``geo_transform``.
        """
        gt = tile_info["geo_transform"]
        # 6-element affine: (origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h)
        origin_x, px_x, rot_lon, origin_y, rot_lat, px_y = gt

        row_off = tile_info.get("row_offset", 0)
        col_off = tile_info.get("col_offset", 0)

        x_min, y_min, x_max, y_max = detection.bbox_pixel

        # Project the four bbox corners through the full affine so the
        # geographic bbox stays correct under orbit rotation.
        corners_pixel = [
            (col_off + x_min, row_off + y_min),
            (col_off + x_max, row_off + y_min),
            (col_off + x_max, row_off + y_max),
            (col_off + x_min, row_off + y_max),
        ]
        lons: list[float] = []
        lats: list[float] = []
        for c, r in corners_pixel:
            lons.append(origin_x + c * px_x + r * rot_lon)
            lats.append(origin_y + c * rot_lat + r * px_y)

        lon_min = float(min(lons))
        lon_max = float(max(lons))
        lat_min = float(min(lats))
        lat_max = float(max(lats))

        # Centre uses the bbox-center pixel, not the corner average — for
        # rotated affines the two differ by a few metres.
        cx = col_off + (x_min + x_max) / 2.0
        cy = row_off + (y_min + y_max) / 2.0
        center_lon = origin_x + cx * px_x + cy * rot_lon
        center_lat = origin_y + cx * rot_lat + cy * px_y

        detection.bbox_geo = [lon_min, lat_min, lon_max, lat_max]
        detection.center_geo = [center_lon, center_lat]
        detection.geometry = {
            "type": "Point",
            "coordinates": [center_lon, center_lat],
        }


# ====================================================================
# Module-level helpers
# ====================================================================


def _dedup_geo_detections(
    detections: list[Detection],
    max_distance_deg: float = 5e-4,
) -> list[Detection]:
    """Keep the highest-confidence detection within each spatial cluster.

    Two detections are considered duplicates when their ``center_geo``
    points fall inside a square of side ``max_distance_deg`` (≈55 m at
    the equator).  Detections without geolocation are passed through
    unchanged so they remain visible for diagnostics.
    """
    if not detections:
        return detections

    geo: list[tuple[int, float, float]] = []
    no_geo: list[Detection] = []
    for i, det in enumerate(detections):
        if det.center_geo and len(det.center_geo) == 2:
            geo.append((i, float(det.center_geo[0]), float(det.center_geo[1])))
        else:
            no_geo.append(det)

    # Sort by confidence descending so the kept survivor is the strongest
    geo.sort(key=lambda t: detections[t[0]].confidence, reverse=True)

    kept_idx: list[int] = []
    kept_pts: list[tuple[float, float]] = []
    for i, lon, lat in geo:
        duplicate = False
        for klon, klat in kept_pts:
            if (
                abs(lon - klon) <= max_distance_deg
                and abs(lat - klat) <= max_distance_deg
            ):
                duplicate = True
                break
        if not duplicate:
            kept_idx.append(i)
            kept_pts.append((lon, lat))

    survivors = [detections[i] for i in sorted(kept_idx)]
    return survivors + no_geo


def _sar_linear_to_uint8_rgb(
    tile: NDArray[np.floating],
    db_min: float = -25.0,
    db_max: float = 0.0,
) -> NDArray[np.uint8]:
    """Convert linear sigma0 → uint8 RGB for optical YOLO input.

    SAR linear-power tiles span many orders of magnitude (typically
    1e-3..10).  A log stretch to dB followed by clipping to a fixed
    visualization range produces a well-conditioned uint8 image YOLO can
    actually process.
    """
    safe = np.clip(tile.astype(np.float32), 1e-10, None)
    db = 10.0 * np.log10(safe)
    db = np.clip(db, db_min, db_max)
    norm = (db - db_min) / (db_max - db_min)  # 0..1
    gray = (norm * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-Union for two ``[x_min, y_min, x_max, y_max]`` boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _snr_to_confidence(snr: float) -> float:
    """Map CFAR signal-to-noise ratio to a [0, 1] confidence score.

    Uses a simple sigmoid-like mapping:  conf = 1 - exp(-snr / 10).
    """
    if snr <= 0:
        return 0.0
    return float(min(1.0, 1.0 - np.exp(-snr / 10.0)))


def _apply_edge_swath_filter(
    detections: list[Detection],
    tiles: list[dict[str, Any]],
    edge_buffer_px: int,
    scene_shape: tuple[int, int] | None = None,
) -> tuple[list[Detection], int]:
    """Drop detections within ``edge_buffer_px`` of the scene swath edge.

    Implements invariant **I-SAR-2** declared in CLAUDE.md §5.1.
    Sentinel-1 GRD scenes concentrate speckle, range/azimuth ambiguity
    ghosts and antenna-pattern artefacts on the first/last pixel rows
    and columns. Vessel-detection pipelines suppress these by clipping
    the bbox of inferable area inwards by a fixed pixel buffer.

    Scene extent is inferred (in priority order):

    1. Explicit ``scene_shape`` argument as ``(rows, cols)``.
    2. ``scene_shape`` key on any tile (preprocessing usually attaches it).
    3. ``scene_height`` / ``scene_width`` on any tile.
    4. Maximum of ``row_offset + tile_rows`` and ``col_offset + tile_cols``
       across the supplied tiles. Always available as last resort.

    Returns the kept detections plus the count of dropped ones.
    """
    if edge_buffer_px <= 0 or not detections:
        return detections, 0

    # ---- Resolve scene shape ----------------------------------------
    rows: int | None = None
    cols: int | None = None
    if scene_shape is not None:
        rows, cols = int(scene_shape[0]), int(scene_shape[1])
    else:
        for t in tiles:
            ss = t.get("scene_shape")
            if ss is not None and len(ss) >= 2:
                rows, cols = int(ss[0]), int(ss[1])
                break
            sh = t.get("scene_height")
            sw = t.get("scene_width")
            if sh is not None and sw is not None:
                rows, cols = int(sh), int(sw)
                break

    # Build per-tile bbox map (row_offset, col_offset, tile_rows, tile_cols)
    tile_extents: dict[int, tuple[int, int, int, int]] = {}
    inferred_rows = 0
    inferred_cols = 0
    for t in tiles:
        idx = int(t.get("tile_index", 0))
        row_off = int(t.get("row_offset", 0))
        col_off = int(t.get("col_offset", 0))
        arr = t.get("data", t.get("array"))
        if arr is not None and hasattr(arr, "shape") and len(arr.shape) >= 2:
            tile_rows = int(arr.shape[0])
            tile_cols = int(arr.shape[1])
        else:
            tile_rows = tile_cols = 0
        tile_extents[idx] = (row_off, col_off, tile_rows, tile_cols)
        inferred_rows = max(inferred_rows, row_off + tile_rows)
        inferred_cols = max(inferred_cols, col_off + tile_cols)

    if rows is None or cols is None:
        rows = inferred_rows
        cols = inferred_cols

    if rows <= 0 or cols <= 0:
        return detections, 0

    # ---- Filter -----------------------------------------------------
    kept: list[Detection] = []
    dropped = 0
    for det in detections:
        ext = tile_extents.get(det.tile_index)
        if ext is None:
            kept.append(det)
            continue
        row_off, col_off, _, _ = ext
        x_min, y_min, x_max, y_max = det.bbox_pixel
        cx = col_off + (x_min + x_max) / 2.0
        cy = row_off + (y_min + y_max) / 2.0
        if (
            cx < edge_buffer_px
            or cx > cols - edge_buffer_px
            or cy < edge_buffer_px
            or cy > rows - edge_buffer_px
        ):
            dropped += 1
            continue
        kept.append(det)
    return kept, dropped
