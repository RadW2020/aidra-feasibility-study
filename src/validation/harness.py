"""Inference adapters for the D2 validation harness.

Two ways of producing predictions for a labelled scene raster:

``run_detector_only``
    Feeds the raster straight into one detector (CFAR or YOLO). No Lee
    filter, no sea mask, no edge filter, no fusion. This is what the
    2026-04 reports in ``reports/validation_xview3_med_*.json`` used and
    it measures the *detector in isolation* (R13 in RISK_REGISTER.md).

``run_full_pipeline``
    Mirrors the production ``PipelineEngine`` path as closely as a
    pre-calibrated raster allows: dB -> linear sigma0, Lee 7x7 on linear
    power, tiles with the production ``tile_size``/``tile_overlap``,
    rotation-aware lon/lat affine per tile, and a single
    :class:`~src.pipeline.detection.DetectionEngine` pass (CFAR with the
    pre-inference sea mask, YOLO, fusion, edge-swath filter, cross-tile
    dedup). From that one pass four prediction sets are derived under
    identical settings so they are directly comparable:

    - ``cfar``       raw CFAR clusters (before fusion), SNR -> confidence
    - ``yolo``       raw YOLO boxes (before fusion)
    - ``aidra``      the production output (union of unfused + fused)
    - ``fused_only`` the ``source == "fused"`` subset (CFAR ∩ YOLO)

    Every prediction carries ``on_land`` computed with the same
    global-land-mask used in production, so sea-only metrics (I-DET-2)
    can be reported next to the raw ones.

The xView3-SAR rasters are already calibrated and terrain-corrected by
the dataset providers (SNAP, EPSG:326xx, dB), so orbit correction,
radiometric calibration and terrain correction cannot be exercised here;
``steps`` in the returned provenance says exactly what ran.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("aidra.validation.harness")

PREDICTION_SETS: tuple[str, ...] = ("cfar", "yolo", "aidra", "fused_only")

# ---------------------------------------------------------------------------
# Small helpers shared by both paths
# ---------------------------------------------------------------------------


def tile_indices(
    height: int, width: int, tile: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Yield ``(row_off, col_off, h, w)`` covering the array with overlap."""
    if tile <= 0:
        return [(0, 0, height, width)]
    step = max(1, tile - overlap)
    rows: list[int] = list(range(0, max(1, height - tile + 1), step))
    if rows[-1] + tile < height:
        rows.append(max(0, height - tile))
    cols: list[int] = list(range(0, max(1, width - tile + 1), step))
    if cols[-1] + tile < width:
        cols.append(max(0, width - tile))
    out: list[tuple[int, int, int, int]] = []
    for r in rows:
        for c in cols:
            h = min(tile, height - r)
            w = min(tile, width - c)
            if h <= 0 or w <= 0:
                continue
            out.append((r, c, h, w))
    return out


def sar_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Same dB stretch used in production (``detection._sar_linear_to_uint8_rgb``)."""
    from src.pipeline.detection import _sar_linear_to_uint8_rgb

    return _sar_linear_to_uint8_rgb(arr)


def db_to_linear(arr: np.ndarray, nodata_below: float = -1000.0) -> np.ndarray:
    """dB -> linear sigma0. Values at/below ``nodata_below`` become 0."""
    nodata_mask = arr <= nodata_below
    arr_db = np.clip(arr, -50.0, 30.0)
    return np.where(nodata_mask, 0.0, np.power(10.0, arr_db / 10.0)).astype(np.float32)


def seed_everything(seed: int) -> dict[str, Any]:
    """Pin every RNG the harness can reach. Returns what was applied."""
    random.seed(seed)
    np.random.seed(seed)
    applied: dict[str, Any] = {"seed": seed, "numpy": True, "random": True}
    try:
        import torch

        torch.manual_seed(seed)
        applied["torch"] = True
        try:
            torch.use_deterministic_algorithms(True)
            applied["torch_deterministic_algorithms"] = True
        except Exception as exc:  # pragma: no cover - backend dependent
            applied["torch_deterministic_algorithms"] = f"unavailable: {exc}"
    except ImportError:  # pragma: no cover
        applied["torch"] = False
    return applied


def settings_fingerprint(settings: Any) -> str:
    """SHA256 of the non-secret Settings fields (I-TRACE-4 for reports)."""
    import hashlib
    import json

    secret_markers = ("password", "token", "secret", "url", "key", "user")
    payload = {
        k: v
        for k, v in settings.model_dump().items()
        if not any(m in k.lower() for m in secret_markers)
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def build_provenance(
    *,
    settings: Any,
    model_name: str,
    model_path: Path | None,
    pipeline_path: str,
    steps: list[str],
    seed_info: dict[str, Any],
    harness: str,
    cfar_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block persisted with every report."""
    from src.traceability.hasher import compute_sha256, get_commit_sha

    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "ultralytics", "numpy", "rasterio", "onnxruntime"):
        with contextlib.suppress(Exception):  # optional deps
            versions[mod] = __import__(mod).__version__

    prov: dict[str, Any] = {
        "commit_sha": get_commit_sha(),
        "settings_hash": settings_fingerprint(settings),
        "model_name": model_name,
        "model_hash": compute_sha256(model_path) if model_path else None,
        "model_path": str(model_path) if model_path else None,
        "pipeline_path": pipeline_path,
        "steps": steps,
        "seed": seed_info,
        "harness": harness,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "versions": versions,
        "image_hashes": {},
    }
    if cfar_params is not None:
        prov["cfar_params"] = cfar_params
    return prov


def cfar_params_of(cfar: Any) -> dict[str, Any]:
    """Describe the CFAR detector actually used (guard/training/pfa/method)."""
    return {
        "guard_size": getattr(cfar, "guard_size", None),
        "training_size": getattr(cfar, "training_size", None),
        "pfa": getattr(cfar, "pfa", None),
        "method": getattr(cfar, "method", None),
        "distribution": getattr(cfar, "distribution", None),
    }


# ---------------------------------------------------------------------------
# Model loading (DB-free, card-gated)
# ---------------------------------------------------------------------------


def _offline_manager(settings: Any) -> Any:
    from src.models.manager import ModelManager

    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = Path(settings.models_dir)
    manager._cache = {}  # type: ignore[attr-defined]
    manager._load_order = []  # type: ignore[attr-defined]
    manager.max_cached_models = 1  # type: ignore[attr-defined]
    return manager


def load_yolo(settings: Any, model_name: str, confidence_threshold: float) -> tuple[Any, Path]:
    """Resolve ``model_name`` under ``Settings.models_dir`` behind the I-AIA-1 gate."""
    from src.models.yolo import YOLODetector

    manager = _offline_manager(settings)
    model_path = manager._find_model_file(model_name, version=None)
    if model_path is None:
        raise FileNotFoundError(
            f"No model file matching '{model_name}' under {manager.models_dir}."
        )
    manager._require_model_card(model_name, model_path)
    detector = YOLODetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=settings.iou_threshold,
    )
    return detector, model_path


def load_cfar(settings: Any, model_name: str = "cfar-default") -> Any:
    """Build the CFAR detector exactly as ``ModelManager._load_cfar_detector`` does.

    Both go through ``CFARDetector.from_settings`` (I-DET-4); the window
    geometry actually used is recorded via :func:`cfar_params_of`.
    """
    from src.models.cfar import CFARDetector

    manager = _offline_manager(settings)
    manager._require_model_card(model_name, manager.models_dir / f"{model_name}.pt")
    return CFARDetector.from_settings(settings)


# ---------------------------------------------------------------------------
# Path 1 — detector only (legacy behaviour, kept for comparability)
# ---------------------------------------------------------------------------


def run_detector_only(
    image_path: Path,
    model_name: str,
    confidence_threshold: float | None = None,
    raster_in_db: bool | None = None,
    tile_size: int = 0,
    tile_overlap: int = 64,
    settings: Any = None,
) -> list[dict[str, Any]]:
    """Run one detector on a single raster without the AIDRA pipeline.

    CFAR detectors (name starts with ``cfar``) are fed linear sigma0 and
    clustered with the production ``Settings.cfar_*`` cluster parameters;
    anything else is loaded as YOLO and fed the production uint8 stretch.
    Confidence threshold defaults to ``Settings.confidence_threshold``
    (I-DET-4).
    """
    import rasterio
    import rasterio.windows

    from src.config import Settings

    settings = settings or Settings()
    if confidence_threshold is None:
        confidence_threshold = settings.confidence_threshold

    is_cfar = model_name.lower().startswith("cfar")
    detector = load_cfar(settings, model_name) if is_cfar else load_yolo(
        settings, model_name, confidence_threshold
    )[0]

    if raster_in_db is None:
        raster_in_db = "_dB" in image_path.name or "_db" in image_path.name

    with rasterio.open(image_path) as src:
        height = int(src.height)
        width = int(src.width)
        tile_arr_full = None if tile_size > 0 else src.read(1).astype(np.float32)

    def _to_linear(arr: np.ndarray) -> np.ndarray:
        return db_to_linear(arr) if raster_in_db else arr

    def _cfar_on_tile(tile_arr: np.ndarray) -> list[dict[str, Any]]:
        # Saturation guard: on port/urban tiles CFAR fires on > 5 % of the
        # pixels and DBSCAN hangs for minutes. Without a sea mask (this
        # path has none) the tile is treated as saturated clutter. The
        # full-pipeline path does NOT have this guard — it relies on the
        # production sea mask instead.
        valid = (tile_arr > 0).sum()
        if valid == 0:
            return []
        pixel_hits = detector.detect(tile_arr)
        if len(pixel_hits) > 0.05 * valid:
            return []
        raw = detector.detect_with_clustering(
            tile_arr,
            min_cluster_size=settings.cfar_min_cluster_size,
            eps=settings.cfar_cluster_eps,
            min_mean_snr=settings.cfar_min_mean_snr,
        )
        from src.pipeline.detection import _snr_to_confidence

        local: list[dict[str, Any]] = []
        for d in raw:
            confidence = _snr_to_confidence(float(d.get("mean_snr", d.get("snr", 0.0))))
            if confidence < confidence_threshold:
                continue
            bbox = d.get("bbox")
            if bbox is None:
                continue
            local.append({
                "bbox": list(map(float, bbox)),
                "confidence": confidence,
                "class_name": d.get("class_name", "vessel"),
            })
        return local

    def _yolo_on_tile(tile_arr: np.ndarray) -> list[dict[str, Any]]:
        raw_dets = detector.predict(sar_to_uint8(tile_arr))
        return [
            {
                "bbox": d["bbox"],
                "confidence": float(d.get("confidence", 0.0)),
                "class_name": d.get("class_name", "vessel"),
            }
            for d in raw_dets
            if float(d.get("confidence", 0.0)) >= confidence_threshold
        ]

    out: list[dict[str, Any]] = []
    if tile_size > 0:
        for r_off, c_off, h, w in tile_indices(height, width, tile_size, tile_overlap):
            with rasterio.open(image_path) as src:
                tile_arr = src.read(
                    1, window=rasterio.windows.Window(c_off, r_off, w, h)
                ).astype(np.float32)
            tile_lin = _to_linear(tile_arr)
            tile_dets = _cfar_on_tile(tile_lin) if is_cfar else _yolo_on_tile(tile_lin)
            for det in tile_dets:
                bx = det["bbox"]
                det["bbox"] = [
                    float(bx[0]) + c_off,
                    float(bx[1]) + r_off,
                    float(bx[2]) + c_off,
                    float(bx[3]) + r_off,
                ]
                out.append(det)
        return out

    arr_lin = _to_linear(tile_arr_full)
    return _cfar_on_tile(arr_lin) if is_cfar else _yolo_on_tile(arr_lin)


# ---------------------------------------------------------------------------
# Path 2 — full AIDRA pipeline on a pre-calibrated raster
# ---------------------------------------------------------------------------


@dataclass
class ScenePredictions:
    """Everything the full-pipeline path produces for one scene."""

    scene_shape: tuple[int, int]
    lonlat_affine: tuple[float, float, float, float, float, float]
    sets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    def is_on_land(self, points_px: np.ndarray) -> np.ndarray:
        """``on_land`` for an ``(N, 2)`` array of ``(x=col, y=row)`` pixel centres."""
        return points_on_land(self.lonlat_affine, points_px)


def lonlat_affine_from_raster(src: Any, grid: int = 12) -> tuple[float, ...]:
    """Least-squares 6-parameter lon/lat affine for a projected raster.

    Production geolocates through a GDAL-order affine
    ``(origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h)`` fitted on
    the Sentinel-1 GCPs. xView3 rasters are UTM GeoTIFFs, so the same
    kind of affine is fitted here on a grid of projected -> WGS-84 points.
    Residuals are a few metres across a 300 km scene — irrelevant at the
    ~1.85 km resolution of the land mask and unused for pixel matching.
    """
    from pyproj import Transformer

    rows = np.linspace(0, src.height, grid)
    cols = np.linspace(0, src.width, grid)
    cg, rg = np.meshgrid(cols, rows)
    xs, ys = src.transform * (cg.ravel(), rg.ravel())
    if src.crs is None or src.crs.to_epsg() == 4326:
        lons, lats = np.asarray(xs), np.asarray(ys)
    else:
        tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lons, lats = tr.transform(np.asarray(xs), np.asarray(ys))
    A = np.column_stack([np.ones_like(cg.ravel()), cg.ravel(), rg.ravel()])
    (o_x, p_w, r_lon), *_ = np.linalg.lstsq(A, np.asarray(lons), rcond=None)
    (o_y, r_lat, p_h), *_ = np.linalg.lstsq(A, np.asarray(lats), rcond=None)
    return (float(o_x), float(p_w), float(r_lon), float(o_y), float(r_lat), float(p_h))


def points_on_land(
    affine: tuple[float, ...], points_px: np.ndarray
) -> np.ndarray:
    """global-land-mask lookup for ``(N, 2)`` ``(col, row)`` pixel centres."""
    from src.pipeline.detection import _get_globe

    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    globe = _get_globe()
    if globe is None:
        return np.zeros(pts.shape[0], dtype=bool)
    o_x, p_w, r_lon, o_y, r_lat, p_h = affine[:6]
    cols, rows = pts[:, 0], pts[:, 1]
    lons = o_x + cols * p_w + rows * r_lon
    lats = o_y + cols * r_lat + rows * p_h
    return ~np.asarray(globe.is_ocean(lats, lons), dtype=bool)


def _tile_geo_bounds(
    affine: tuple[float, ...], col_off: int, row_off: int, size: int
) -> dict[str, float]:
    o_x, p_w, r_lon, o_y, r_lat, p_h = affine[:6]
    lons: list[float] = []
    lats: list[float] = []
    for c, r in (
        (col_off, row_off),
        (col_off + size, row_off),
        (col_off + size, row_off + size),
        (col_off, row_off + size),
    ):
        lons.append(o_x + c * p_w + r * r_lon)
        lats.append(o_y + c * r_lat + r * p_h)
    return {
        "lon_min": min(lons),
        "lon_max": max(lons),
        "lat_min": min(lats),
        "lat_max": max(lats),
    }


def valid_data_distance_map(
    src: Any, *, raster_in_db: bool, downsample: int = 8
) -> tuple[np.ndarray, int]:
    """Distance (px, full-res units) from every pixel to the nearest nodata.

    xView3 rasters are UTM-projected, so the swath edge is a rotated
    boundary running through the raster interior, not the raster border.
    CFAR's training window straddling that step (sea -> nodata zeros) fires
    along the whole boundary; production removes those with footprint
    clipping (I-SAR-3) plus the edge buffer (I-SAR-2) in
    ``_save_detections``. Computed on a ``downsample``-times smaller grid;
    the result is in full-resolution pixels.
    """
    from rasterio.enums import Resampling
    from scipy.ndimage import distance_transform_edt

    h = max(1, int(src.height) // downsample)
    w = max(1, int(src.width) // downsample)
    small = src.read(1, out_shape=(h, w), resampling=Resampling.nearest).astype(np.float32)
    if raster_in_db:
        nodata = small <= -1000.0
    elif src.nodata is not None:
        nodata = small == src.nodata
    else:
        nodata = small == 0.0
    if not nodata.any():
        return np.full(small.shape, np.inf, dtype=np.float32), downsample
    dist = distance_transform_edt(~nodata) * float(downsample)
    return dist.astype(np.float32), downsample


def _swath_edge_flags(
    dist_map: np.ndarray, downsample: int, centres_px: np.ndarray, buffer_px: int
) -> np.ndarray:
    """True where a ``(col, row)`` centre is inside nodata or within ``buffer_px`` of it."""
    pts = np.asarray(centres_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    rr = np.clip((pts[:, 1] // downsample).astype(int), 0, dist_map.shape[0] - 1)
    cc = np.clip((pts[:, 0] // downsample).astype(int), 0, dist_map.shape[1] - 1)
    return dist_map[rr, cc] < float(buffer_px)


def run_full_pipeline(
    image_path: Path,
    *,
    engine: Any,
    yolo: Any,
    cfar: Any,
    confidence_threshold: float,
    tile_size: int,
    tile_overlap: int,
    band_tile_rows: int = 4,
    lee_window: int = 7,
    raster_in_db: bool | None = None,
    log_every_bands: int = 5,
) -> ScenePredictions:
    """Run the production ``DetectionEngine`` over a pre-calibrated raster.

    Tiles are streamed in horizontal bands of ``band_tile_rows`` tile rows
    to bound peak RAM (a full 30k x 22k scene as float32 tiles is ~2.5 GB).
    ``engine.run`` applies the edge filter against ``scene_shape`` and a
    per-band cross-tile dedup; the production dedup is applied once more
    over the concatenated bands so band seams do not double count.
    """
    import rasterio
    import rasterio.windows

    from src.pipeline.detection import (
        _dedup_geo_detections,
        _snr_to_confidence,
        sar_linear_to_uint8_gray,
    )
    from src.pipeline.preprocessing import apply_lee_filter

    t0 = time.perf_counter()
    if raster_in_db is None:
        raster_in_db = "_dB" in image_path.name or "_db" in image_path.name

    with rasterio.open(image_path) as src:
        height, width = int(src.height), int(src.width)
        affine = lonlat_affine_from_raster(src)
        dist_map, dist_ds = valid_data_distance_map(src, raster_in_db=raster_in_db)

    scene_shape = (height, width)
    edge_buffer_px = int(getattr(engine, "edge_buffer_px", 0) or 0)
    windows = tile_indices(height, width, tile_size, tile_overlap)
    # Group windows by their row offset -> bands of tile rows.
    row_offsets = sorted({w[0] for w in windows})
    bands = [
        row_offsets[i : i + band_tile_rows]
        for i in range(0, len(row_offsets), band_tile_rows)
    ]
    margin = lee_window // 2

    tile_offsets: dict[int, tuple[int, int]] = {}
    all_dets: list[Any] = []
    cfar_raw: list[dict[str, Any]] = []
    yolo_raw: list[dict[str, Any]] = []
    masked_tiles = 0
    tile_index = 0
    t_prep = 0.0
    t_engine = 0.0

    with rasterio.open(image_path) as src:
        for b_idx, band_rows in enumerate(bands):
            tp0 = time.perf_counter()
            tiles: list[dict[str, Any]] = []
            for r_off, c_off, h, w in windows:
                if r_off not in band_rows:
                    continue
                # Read with a Lee margin so the filter has context at tile
                # borders, mirroring preprocess_full.
                r0 = max(0, r_off - margin)
                c0 = max(0, c_off - margin)
                r1 = min(height, r_off + h + margin)
                c1 = min(width, c_off + w + margin)
                chunk = src.read(
                    1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                ).astype(np.float32)
                linear = db_to_linear(chunk) if raster_in_db else chunk
                filtered = apply_lee_filter(linear, window_size=lee_window)
                center = filtered[r_off - r0 : r_off - r0 + h, c_off - c0 : c_off - c0 + w]
                tile = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile[: center.shape[0], : center.shape[1]] = center
                # R15 parity with preprocess_full: unfiltered uint8 for YOLO.
                raw_center = linear[r_off - r0 : r_off - r0 + h, c_off - c0 : c_off - c0 + w]
                yolo_gray = np.zeros((tile_size, tile_size), dtype=np.uint8)
                yolo_gray[: raw_center.shape[0], : raw_center.shape[1]] = (
                    sar_linear_to_uint8_gray(raw_center)
                )
                tiles.append(
                    {
                        "data": tile,
                        "yolo_input": yolo_gray,
                        "tile_index": tile_index,
                        "row_offset": r_off,
                        "col_offset": c_off,
                        "geo_transform": affine,
                        "geo_bounds": _tile_geo_bounds(affine, c_off, r_off, tile_size),
                        "scene_shape": scene_shape,
                    }
                )
                tile_offsets[tile_index] = (r_off, c_off)
                tile_index += 1
            t_prep += time.perf_counter() - tp0

            te0 = time.perf_counter()
            result = engine.run(
                tiles, detector=yolo, cfar=cfar, scene_shape=scene_shape
            )
            t_engine += time.perf_counter() - te0

            all_dets.extend(result.detections)
            cfar_raw.extend(result.cfar_raw)
            yolo_raw.extend(result.yolo_raw)
            masked_tiles += sum(1 for t in tiles if t.get("geo_transform") is not None)
            if b_idx % log_every_bands == 0 or b_idx == len(bands) - 1:
                logger.info(
                    "  band %d/%d: %d tiles, cfar_raw=%d yolo_raw=%d aidra=%d",
                    b_idx + 1,
                    len(bands),
                    len(tiles),
                    len(result.cfar_raw),
                    len(result.yolo_raw),
                    len(result.detections),
                )
            del tiles

    # Production dedup once more across band seams (idempotent otherwise).
    before = len(all_dets)
    all_dets = _dedup_geo_detections(all_dets, max_distance_deg=5e-4)

    def _to_scene(bbox: list[float], idx: int) -> list[float]:
        r_off, c_off = tile_offsets.get(int(idx), (0, 0))
        return [
            float(bbox[0]) + c_off,
            float(bbox[1]) + r_off,
            float(bbox[2]) + c_off,
            float(bbox[3]) + r_off,
        ]

    sets: dict[str, list[dict[str, Any]]] = {k: [] for k in PREDICTION_SETS}
    for d in cfar_raw:
        conf = _snr_to_confidence(float(d.get("mean_snr", d.get("snr", 0.0))))
        if conf < confidence_threshold or d.get("bbox") is None:
            continue
        sets["cfar"].append({
            "bbox": _to_scene(d["bbox"], d.get("tile_index", 0)),
            "confidence": conf,
            "source": "cfar",
        })
    for d in yolo_raw:
        conf = float(d.get("confidence", 0.0))
        if conf < confidence_threshold or d.get("bbox") is None:
            continue
        sets["yolo"].append({
            "bbox": _to_scene(d["bbox"], d.get("tile_index", 0)),
            "confidence": conf,
            "source": "yolo",
        })
    for det in all_dets:
        pred = {
            "bbox": _to_scene(det.bbox_pixel, det.tile_index),
            "confidence": float(det.confidence),
            "source": det.source,
        }
        sets["aidra"].append(pred)
        if det.source == "fused":
            sets["fused_only"].append(dict(pred))

    # Footprint clipping against the valid-data boundary (I-SAR-3) with the
    # production edge buffer (I-SAR-2). Production applies this in
    # _save_detections; the harness mirrors it here so CFAR streaks along
    # the rotated nodata edge of projected rasters are not scored as FP.
    swath_dropped: dict[str, int] = {}
    if edge_buffer_px > 0:
        for name, preds in sets.items():
            if not preds:
                swath_dropped[name] = 0
                continue
            centres = np.array(
                [[(p["bbox"][0] + p["bbox"][2]) / 2.0, (p["bbox"][1] + p["bbox"][3]) / 2.0] for p in preds]
            )
            flags = _swath_edge_flags(dist_map, dist_ds, centres, edge_buffer_px)
            sets[name] = [p for p, f in zip(preds, flags, strict=True) if not f]
            swath_dropped[name] = int(flags.sum())

    # on_land for every prediction (I-DET-2), same land mask as production.
    for preds in sets.values():
        if not preds:
            continue
        centres = np.array(
            [[(p["bbox"][0] + p["bbox"][2]) / 2.0, (p["bbox"][1] + p["bbox"][3]) / 2.0] for p in preds]
        )
        flags = points_on_land(affine, centres)
        for p, flag in zip(preds, flags, strict=True):
            p["on_land"] = bool(flag)

    return ScenePredictions(
        scene_shape=scene_shape,
        lonlat_affine=affine,  # type: ignore[arg-type]
        sets=sets,
        stats={
            "num_tiles": tile_index,
            "num_bands": len(bands),
            "tiles_with_geocoding": masked_tiles,
            "cfar_raw": len(cfar_raw),
            "yolo_raw": len(yolo_raw),
            "aidra_before_seam_dedup": before,
            "aidra": len(all_dets),
            "fused_only": len(sets["fused_only"]),
            "swath_edge_buffer_px": edge_buffer_px,
            "swath_edge_dropped": swath_dropped,
            "fusion_mode": getattr(engine, "fusion_mode", None),
            "fusion_center_tolerance_px": getattr(engine, "fusion_center_tolerance_px", None),
            "yolo_input": getattr(engine, "yolo_input", None),
        },
        timings_s={
            "preprocess": round(t_prep, 3),
            "engine": round(t_engine, 3),
            "total": round(time.perf_counter() - t0, 3),
        },
    )


FULL_PIPELINE_STEPS: list[str] = [
    "db_to_linear_sigma0",
    "lee_filter_7x7_linear (CFAR input)",
    "yolo_input_uint8_from_unfiltered_or_filtered_per_Settings.yolo_input",
    "tiles_settings_tile_size_overlap",
    "lonlat_affine_rotation_aware",
    "cfar_sea_mask_pre_inference",
    "cfar_dbscan_clustering",
    "yolo_uint8_db_stretch",
    "fusion_per_Settings.fusion_mode (center|iou)",
    "edge_swath_filter_settings",
    "footprint_clip_valid_data_boundary_edge_buffer",
    "cross_tile_geo_dedup",
    "on_land_flag_global_land_mask",
]

NOT_EXERCISED_STEPS: list[str] = [
    "orbit_correction (raster pre-processed by xView3)",
    "radiometric_calibration_sigma0 (raster pre-processed by xView3)",
    "terrain_correction (excluded from MVP, R8; xView3 rasters are TC'd)",
]
