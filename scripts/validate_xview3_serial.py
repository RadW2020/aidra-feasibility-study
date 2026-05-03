"""
Serial driver for xView3-SAR validation under disk-tight conditions.

For each of the 11 Mediterranean scenes::

  1. Extract **only** ``<scene_id>/VH_dB.tif`` from its tar.gz
     (skip bathymetry/owi*/VV_dB ⇒ ~70 % less disk).
  2. Build a single-scene manifest pointing at the extracted file.
  3. Run ``scripts.run_validation`` with tiling.
  4. Delete the extracted file.
  5. Aggregate per-scene results into a combined report.

This keeps peak disk usage under ~1.5 GB (one VH_dB.tif at a time)
even when validating the full Med subset (~30 GB decompressed).

Run::

    python -m scripts.validate_xview3_serial \
        --xview-dir x-view-us-data \
        --tar-dir data/xview3/scenes \
        --tmp-dir data/xview3/scratch \
        --model cfar-default \
        --output reports/validation_xview3_med.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from statistics import median

logger = logging.getLogger("aidra.xview3_serial")

S1_PIXEL_SPACING_M = 10.0
DEFAULT_CONFIDENCE = "HIGH,MEDIUM"

# Reuse the AOI labelling from build_xview3_manifest.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.build_xview3_manifest import _detect_aoi  # noqa: E402
from scripts.run_validation import (  # noqa: E402
    ValidationReport,
    _format_markdown,
    _match_predictions,
    _pr_curve_from_scored,
    _run_inference,
)


def _list_med_scene_ids(med_manifest: Path) -> list[str]:
    """Pull the Mediterranean scene_ids out of L17's manifest_med.json."""
    data = json.loads(med_manifest.read_text())
    return [r["scene_id"] for r in data if r.get("in_mediterranean")]


def _load_gt_for_scene(
    val_csv: Path,
    scene_id: str,
    confidence_filter: set[str],
    vessels_only: bool,
) -> tuple[list[dict], float, float]:
    """Return ``(ground_truth_list, centroid_lat, centroid_lon)``."""
    gt: list[dict] = []
    lats: list[float] = []
    lons: list[float] = []
    with val_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["scene_id"] != scene_id:
                continue
            if (row.get("confidence") or "").upper() not in confidence_filter:
                continue
            if vessels_only and row.get("is_vessel", "True") != "True":
                continue
            try:
                left = float(row["left"])
                top = float(row["top"])
                right = float(row["right"])
                bottom = float(row["bottom"])
            except (KeyError, ValueError):
                continue
            if right <= left or bottom <= top:
                continue
            gt.append({
                "bbox": [left, top, right, bottom],
                "is_vessel": row.get("is_vessel", "True") == "True",
                "confidence": row.get("confidence", "HIGH"),
                "vessel_length_m": (
                    float(row["vessel_length_m"])
                    if row.get("vessel_length_m") else None
                ),
            })
            try:
                lats.append(float(row["detect_lat"]))
                lons.append(float(row["detect_lon"]))
            except (KeyError, ValueError):
                pass
    centroid_lat = float(median(lats)) if lats else 0.0
    centroid_lon = float(median(lons)) if lons else 0.0
    return gt, centroid_lat, centroid_lon


def _extract_band(tar_path: Path, scene_id: str, out_dir: Path, band: str) -> Path:
    """Extract a single ``<scene_id>/<band>`` from the tarball.

    Saves only the requested band to keep disk usage minimal.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_member = f"{scene_id}/{band}"
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.getmember(target_member)
        tf.extract(member, out_dir)
    return out_dir / target_member


def _scene_area_km2(raster: Path) -> float:
    """Mirror build_xview3_manifest._scene_area_km2 (valid pixels only)."""
    import rasterio
    with rasterio.open(raster) as src:
        arr = src.read(1, out_shape=(min(2048, src.height), min(2048, src.width)))
        scale = (src.height / arr.shape[0]) * (src.width / arr.shape[1])
        valid = (arr > -1000.0) & (arr != 0)
    valid_px = int(valid.sum()) * scale
    return round(valid_px * (S1_PIXEL_SPACING_M / 1000.0) ** 2, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xview-dir", type=Path, required=True)
    parser.add_argument("--tar-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--med-manifest", type=Path, default=Path("data/xview3/manifest_med.json"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=str, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--vessels-only", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--match-mode", choices=["iou", "center"], default="center")
    parser.add_argument("--center-tolerance-px", type=float, default=20.0)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--band", type=str, default="VH_dB.tif")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    confidence_filter = {c.strip().upper() for c in args.confidence.split(",") if c.strip()}
    val_csv = args.xview_dir / "validation.csv"
    scene_ids = _list_med_scene_ids(args.med_manifest)
    logger.info("Mediterranean scenes to evaluate: %d", len(scene_ids))

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    per_scene_scored: list[list[tuple[float, bool]]] = []
    per_class_count: defaultdict[str, int] = defaultdict(int)
    tp_total = fp_total = fn_total = 0
    gt_total = pred_total = 0
    area_total = 0.0
    per_scene_records: list[dict] = []

    for i, sid in enumerate(scene_ids, 1):
        tar_path = args.tar_dir / f"{sid}.tar.gz"
        if not tar_path.exists():
            logger.warning("Missing tar for %s — skipping", sid)
            continue

        logger.info("[%d/%d] %s — extracting %s", i, len(scene_ids), sid, args.band)
        try:
            raster_path = _extract_band(tar_path, sid, args.tmp_dir, args.band)
        except KeyError:
            logger.warning("Band %s not present in %s — skipping", args.band, sid)
            continue

        gt, c_lat, c_lon = _load_gt_for_scene(
            val_csv, sid, confidence_filter, args.vessels_only
        )
        if not gt:
            logger.info("  no GT after filters — skipping")
            shutil.rmtree(raster_path.parent, ignore_errors=True)
            continue

        scene_area = _scene_area_km2(raster_path)
        try:
            preds = _run_inference(
                raster_path,
                args.model,
                args.confidence_threshold,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
        except Exception as exc:
            logger.error("Inference failed on %s: %s", sid, exc)
            shutil.rmtree(raster_path.parent, ignore_errors=True)
            return 3

        tp, fp, fn, scored = _match_predictions(
            preds,
            gt,
            args.iou_threshold,
            match_mode=args.match_mode,
            center_tolerance_px=args.center_tolerance_px,
        )
        per_scene_scored.append(scored)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        gt_total += len(gt)
        pred_total += len(preds)
        area_total += scene_area
        for g in gt:
            per_class_count[g.get("class_name", "vessel")] += 1
        per_scene_records.append({
            "scene_id": sid,
            "aoi_label": _detect_aoi(c_lat, c_lon),
            "centroid_lat": round(c_lat, 4),
            "centroid_lon": round(c_lon, 4),
            "scene_area_km2": scene_area,
            "n_gt": len(gt),
            "n_preds": len(preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        logger.info(
            "  %s gt=%d preds=%d tp=%d fp=%d fn=%d  pd=%.3f  far/km²=%.3f",
            sid,
            len(gt),
            len(preds),
            tp,
            fp,
            fn,
            tp / max(1, len(gt)),
            fp / max(1.0, scene_area),
        )

        # Free disk before the next scene.
        shutil.rmtree(raster_path.parent, ignore_errors=True)

    # Optional cleanup of empty tmp dir.
    if args.tmp_dir.exists() and not any(args.tmp_dir.iterdir()):
        args.tmp_dir.rmdir()

    pr_curve = _pr_curve_from_scored(per_scene_scored, gt_total)
    report = ValidationReport(
        model_name=args.model,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        num_scenes=len(per_scene_records),
        num_ground_truth=gt_total,
        num_predictions=pred_total,
        true_positives=tp_total,
        false_positives=fp_total,
        false_negatives=fn_total,
        total_area_km2=area_total,
        pr_curve=pr_curve,
        match_mode=args.match_mode,
        center_tolerance_px=args.center_tolerance_px,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["per_scene"] = per_scene_records
    payload["dataset"] = "xview3-sar/validation/mediterranean"
    args.output.write_text(json.dumps(payload, indent=2))
    md = _format_markdown(report)
    args.output.with_suffix(".md").write_text(md)

    logger.info("Report: %s", args.output)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
