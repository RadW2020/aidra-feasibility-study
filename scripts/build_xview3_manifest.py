"""
Build an AIDRA validation manifest from a downloaded xView3-SAR
subset (palanca L19).

Inputs
------
  --xview-dir       Directory with the xView3 metadata files
                    (validation.csv at minimum).
  --scenes-dir      Directory containing the **decompressed** scene
                    folders (one per scene_id, e.g. ``264ed833...v/``).
                    Each folder is expected to expose at least
                    ``VH_dB.tif`` (or ``VV_dB.tif`` as fallback).
  --out             Path of the manifest JSON to write.
  --confidence      xView3 confidence levels to include
                    (LOW / MEDIUM / HIGH). Default: ``HIGH,MEDIUM``.
  --vessels-only    If set, drop ``is_vessel=False`` ground truths
                    (platforms / fixed infrastructure).

Output
------
A JSON manifest compatible with ``scripts/run_validation.py``::

    [
      {
        "image_id":  "264ed833a13b7f2av",
        "image_path": "/abs/path/.../VH_dB.tif",
        "scene_area_km2": 22431.4,
        "ground_truth": [
          {"bbox": [left, top, right, bottom],
           "vessel_length_m": 80.0,
           "is_vessel": true,
           "confidence": "HIGH"},
          ...
        ],
        "aoi_label": "adriatic",
        "polarisation_band": "VH_dB",
        "centroid_lat": 44.46,
        "centroid_lon": 12.66
      },
      ...
    ]

Design notes
------------
- xView3 stores backscatter as **decibels** (``*_dB.tif``). The
  harness, when it sees ``"polarisation_band": *_dB``, will exponentiate
  back to linear sigma0 before feeding CFAR (which expects linear)
  and skip the dB stretch when feeding YOLO (already in dB-friendly
  scale).
- Scene area is computed from the raster shape × 10 m pixel
  spacing² (Sentinel-1 IW GRD-H pixel spacing).
- Scenes without their decompressed folder are skipped with a
  warning so the manifest stays consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import median

import rasterio

logger = logging.getLogger("aidra.xview3_manifest")

S1_PIXEL_SPACING_M = 10.0
DEFAULT_CONFIDENCE = "HIGH,MEDIUM"


def _detect_aoi(lat: float, lon: float) -> str:
    if 35 <= lat <= 37 and -7 <= lon <= -3:
        return "gibraltar"
    if 28 <= lat <= 33 and 31 <= lon <= 33:
        return "suez-canal"
    if 12 <= lat <= 30 and 32 <= lon <= 44:
        return "red-sea"
    if 49 <= lat <= 51 and -6 <= lon <= 2:
        return "english-channel"
    if 39 <= lat <= 46 and 12 <= lon <= 21:
        return "adriatic"
    if 35 <= lat <= 41 and 22 <= lon <= 30:
        return "aegean"
    if 30 <= lat <= 44 and -6 <= lon <= 12:
        return "west-med"
    if 30 <= lat <= 38 and 22 <= lon <= 36:
        return "east-med"
    return "other"


def _resolve_raster(scene_dir: Path) -> Path | None:
    """Pick the SAR raster for inference, preferring VH then VV."""
    for name in ("VH_dB.tif", "VV_dB.tif", "VH.tif", "VV.tif"):
        p = scene_dir / name
        if p.exists():
            return p
    # Some xView3 deliveries lower-case the band name.
    candidates = sorted(scene_dir.glob("V*_dB.tif")) + sorted(scene_dir.glob("V*.tif"))
    return candidates[0] if candidates else None


def _scene_area_km2(raster: Path) -> float:
    """Return the **valid** SAR footprint area, not the bounding raster.

    xView3-SAR rasters are clipped + projected, so the bounding box
    contains a large no-data margin (sentinel = -32768 in float16
    dB scale). Using the full raster shape over-estimates coverage
    by 30-50 % and flatters FAR/km² accordingly. We sample the
    raster and count valid pixels.
    """
    with rasterio.open(raster) as src:
        # Use the raster overview if available, else stride-read.
        arr = src.read(1, out_shape=(min(2048, src.height), min(2048, src.width)))
        scale = (src.height / arr.shape[0]) * (src.width / arr.shape[1])
        valid = (arr > -1000.0) & (arr != 0)
    valid_px = int(valid.sum()) * scale
    return round(valid_px * (S1_PIXEL_SPACING_M / 1000.0) ** 2, 1)


def _row_to_bbox(row: dict) -> list[float] | None:
    try:
        left = float(row["left"])
        top = float(row["top"])
        right = float(row["right"])
        bottom = float(row["bottom"])
    except (KeyError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xview-dir", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence", type=str, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--vessels-only", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    confidence_filter = {c.strip().upper() for c in args.confidence.split(",") if c.strip()}
    if not confidence_filter:
        confidence_filter = {"HIGH", "MEDIUM"}

    val_csv = args.xview_dir / "validation.csv"
    if not val_csv.exists():
        logger.error("Missing %s", val_csv)
        return 2

    by_scene: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    n_kept_conf = 0
    n_kept_vessel = 0
    with val_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            confidence = (row.get("confidence") or "").upper()
            if confidence not in confidence_filter:
                continue
            n_kept_conf += 1
            if args.vessels_only and row.get("is_vessel", "True") != "True":
                continue
            n_kept_vessel += 1
            by_scene[row["scene_id"]].append(row)

    logger.info(
        "Detections: total=%d, after-confidence=%d, after-vessel=%d",
        n_total,
        n_kept_conf,
        n_kept_vessel,
    )

    manifest: list[dict] = []
    skipped_missing: list[str] = []
    for scene_id, rows in sorted(by_scene.items()):
        scene_dir = args.scenes_dir / scene_id
        if not scene_dir.is_dir():
            skipped_missing.append(scene_id)
            continue
        raster = _resolve_raster(scene_dir)
        if raster is None:
            skipped_missing.append(scene_id)
            continue

        ground_truth: list[dict] = []
        for r in rows:
            bbox = _row_to_bbox(r)
            if bbox is None:
                continue
            vessel_length = r.get("vessel_length_m") or ""
            ground_truth.append({
                "bbox": bbox,
                "is_vessel": r.get("is_vessel", "True") == "True",
                "is_fishing": r.get("is_fishing", "False") == "True",
                "vessel_length_m": float(vessel_length) if vessel_length else None,
                "confidence": r.get("confidence", "HIGH"),
                "source": r.get("source", "manual"),
            })

        if not ground_truth:
            continue

        lats = [float(r["detect_lat"]) for r in rows]
        lons = [float(r["detect_lon"]) for r in rows]
        c_lat = float(median(lats))
        c_lon = float(median(lons))

        manifest.append({
            "image_id": scene_id,
            "image_path": str(raster.resolve()),
            "scene_area_km2": round(_scene_area_km2(raster), 1),
            "ground_truth": ground_truth,
            "aoi_label": _detect_aoi(c_lat, c_lon),
            "polarisation_band": raster.stem,
            "centroid_lat": round(c_lat, 4),
            "centroid_lon": round(c_lon, 4),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))

    n_gt = sum(len(s["ground_truth"]) for s in manifest)
    total_area = sum(s["scene_area_km2"] for s in manifest)
    logger.info(
        "Manifest written: %d scenes, %d ground truths, %.0f km² total area",
        len(manifest),
        n_gt,
        total_area,
    )
    if skipped_missing:
        logger.warning(
            "Skipped %d scenes without decompressed folder/raster: %s%s",
            len(skipped_missing),
            ", ".join(skipped_missing[:5]),
            " ..." if len(skipped_missing) > 5 else "",
        )
    logger.info("Output: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
