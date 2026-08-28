"""
Serial driver for xView3-SAR validation under disk-tight conditions.

For each Adriatic/Mediterranean scene in ``manifest_med.json``::

  1. Extract **only** ``<scene_id>/VH_dB.tif`` from its tar.gz
     (skip bathymetry/owi*/VV_dB => ~70 % less disk).
  2. Load the xView3 ground truth for that scene (HIGH+MEDIUM vessels).
  3. Run inference — either the detector alone (``--pipeline-path
     detector``, the 2026-04 behaviour) or the production
     ``DetectionEngine`` (``--pipeline-path full``: Lee, sea mask, fusion,
     edge filter). The full path scores four prediction sets from ONE pass
     under identical settings: ``cfar``, ``yolo``, ``aidra`` (production
     output) and ``fused_only`` (CFAR ∩ YOLO).
  4. Score raw and *sea-only* (predictions and GT with ``on_land`` dropped,
     I-DET-2) against the GT.
  5. Delete the extracted raster and move on.

Every report carries ``provenance`` (commit_sha, model_hash,
settings_hash, seed, per-scene tar SHA256) and ``params`` (pipeline path,
tile size, thresholds, steps) so it can be imported through
``POST /api/validation/import`` and audited later.

Run::

    python -m scripts.validate_xview3_serial \
        --xview-dir x-view-us-data \
        --tar-dir data/xview3/scenes \
        --tmp-dir data/xview3/scratch \
        --pipeline-path full --model all \
        --models-dir models \
        --output reports/validation_xview3_adriatic_full.json

With ``--model all`` the output stem receives one ``_<set>.json`` /
``.md`` pair per prediction set.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

logger = logging.getLogger("aidra.xview3_serial")

S1_PIXEL_SPACING_M = 10.0
DEFAULT_CONFIDENCE = "HIGH,MEDIUM"

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.build_xview3_manifest import _detect_aoi  # noqa: E402
from scripts.run_validation import (  # noqa: E402
    FULL_PATH_SETS,
    _format_markdown,
)
from src.validation.harness import (  # noqa: E402
    FULL_PIPELINE_STEPS,
    NOT_EXERCISED_STEPS,
    PREDICTION_SETS,
    build_provenance,
    cfar_params_of,
    run_detector_only,
)
from src.validation.metrics import (  # noqa: E402
    ValidationReport,
    match_predictions,
    pr_curve_from_scored,
)

SET_LABELS: dict[str, str] = {
    "cfar": "cfar-default",
    "yolo": "vesseltracker-sar-yolov8",
    "aidra": "aidra-cfar+yolo-union",
    "fused_only": "aidra-fused-only",
}


def _list_med_scene_ids(med_manifest: Path) -> list[str]:
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
                    float(row["vessel_length_m"]) if row.get("vessel_length_m") else None
                ),
                "distance_from_shore_km": (
                    float(row["distance_from_shore_km"])
                    if row.get("distance_from_shore_km")
                    else None
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
    out_dir.mkdir(parents=True, exist_ok=True)
    target_member = f"{scene_id}/{band}"
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.getmember(target_member)
        tf.extract(member, out_dir)
    return out_dir / target_member


def _scene_area_km2(raster: Path) -> float:
    import rasterio

    with rasterio.open(raster) as src:
        arr = src.read(1, out_shape=(min(2048, src.height), min(2048, src.width)))
        scale = (src.height / arr.shape[0]) * (src.width / arr.shape[1])
        valid = (arr > -1000.0) & (arr != 0)
    valid_px = int(valid.sum()) * scale
    return round(valid_px * (S1_PIXEL_SPACING_M / 1000.0) ** 2, 1)


class _Accumulator:
    """Per-prediction-set running totals (raw and sea-only)."""

    def __init__(self) -> None:
        self.scored: list[list[tuple[float, bool]]] = []
        self.scored_sea: list[list[tuple[float, bool]]] = []
        self.tp = self.fp = self.fn = 0
        self.tp_sea = self.fp_sea = self.fn_sea = 0
        self.gt = self.gt_sea = 0
        self.preds = self.preds_sea = 0
        self.per_scene: list[dict[str, Any]] = []


def _score(
    preds: list[dict[str, Any]],
    gt: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[int, int, int, list[tuple[float, bool]]]:
    return match_predictions(
        preds,
        gt,
        args.iou_threshold,
        match_mode=args.match_mode,
        center_tolerance_px=args.center_tolerance_px,
    )


def _report(
    acc: _Accumulator, *, sea_only: bool, label: str, args: argparse.Namespace,
    confidence_threshold: float, params: dict[str, Any], provenance: dict[str, Any],
) -> ValidationReport:
    if sea_only:
        tp, fp, fn, gt_n, pred_n, scored = (
            acc.tp_sea, acc.fp_sea, acc.fn_sea, acc.gt_sea, acc.preds_sea, acc.scored_sea
        )
    else:
        tp, fp, fn, gt_n, pred_n, scored = acc.tp, acc.fp, acc.fn, acc.gt, acc.preds, acc.scored
    return ValidationReport(
        model_name=label,
        iou_threshold=args.iou_threshold,
        confidence_threshold=confidence_threshold,
        num_scenes=len(acc.per_scene),
        num_ground_truth=gt_n,
        num_predictions=pred_n,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_area_km2=sum(r["scene_area_km2"] for r in acc.per_scene),
        pr_curve=pr_curve_from_scored(scored, gt_n),
        match_mode=args.match_mode,
        center_tolerance_px=args.center_tolerance_px,
        params={**params, "sea_only": sea_only},
        provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xview-dir", type=Path, required=True)
    parser.add_argument("--tar-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument(
        "--med-manifest", type=Path, default=Path("data/xview3/manifest_med.json")
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "detector path: model name. full path: one of "
            + ", ".join(FULL_PATH_SETS)
            + " or 'all'"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-path", choices=["detector", "full"], default="detector")
    parser.add_argument("--confidence", type=str, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--vessels-only", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Defaults to Settings.confidence_threshold (I-DET-4).",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--match-mode", choices=["iou", "center"], default="center")
    parser.add_argument("--center-tolerance-px", type=float, default=20.0)
    parser.add_argument("--tile-size", type=int, default=None, help="Defaults to Settings.tile_size.")
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--band-tile-rows", type=int, default=4)
    parser.add_argument("--band", type=str, default="VH_dB.tif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None, help="Debug: only first N scenes.")
    parser.add_argument("--keep-raster", action="store_true", help="Debug: do not delete extracted rasters.")
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help=(
            "Directory for per-scene JSON dumps of every prediction set and the GT "
            "(bbox, confidence, source, on_land). Enables offline analysis (fusion "
            "co-occurrence, FP/FN sampling for D4) without re-running inference."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.models_dir is not None:
        os.environ["MODELS_DIR"] = str(args.models_dir)

    from src.config import Settings
    from src.traceability.hasher import compute_sha256
    from src.validation.harness import seed_everything

    settings = Settings()
    seed_info = seed_everything(args.seed)
    confidence_threshold = (
        settings.confidence_threshold
        if args.confidence_threshold is None
        else args.confidence_threshold
    )
    tile_size = settings.tile_size if args.tile_size is None else args.tile_size
    tile_overlap = settings.tile_overlap if args.tile_overlap is None else args.tile_overlap

    full = args.pipeline_path == "full"
    if full:
        if args.model == "all":
            set_names = list(PREDICTION_SETS)
        elif args.model in FULL_PATH_SETS:
            set_names = [FULL_PATH_SETS[args.model]]
        else:
            logger.error("--pipeline-path full needs --model in %s or 'all'", list(FULL_PATH_SETS))
            return 2
    else:
        set_names = ["detector"]

    # Build detectors once.
    engine = yolo = cfar = None
    model_path: Path | None = None
    if full:
        from src.pipeline.detection import DetectionEngine
        from src.validation.harness import load_cfar, load_yolo

        yolo, model_path = load_yolo(settings, "vesseltracker-sar-yolov8", confidence_threshold)
        cfar = load_cfar(settings)
        engine = DetectionEngine(edge_buffer_px=settings.edge_buffer_px)
        steps = FULL_PIPELINE_STEPS
    else:
        steps = ["detector_only_on_provided_raster"]
        if not args.model.lower().startswith("cfar"):
            from src.validation.harness import _offline_manager

            model_path = _offline_manager(settings)._find_model_file(args.model, version=None)

    provenance = build_provenance(
        settings=settings,
        model_name=args.model,
        model_path=model_path,
        pipeline_path=args.pipeline_path,
        steps=steps,
        seed_info=seed_info,
        harness="scripts/validate_xview3_serial.py",
        cfar_params=cfar_params_of(cfar) if cfar is not None else None,
    )
    provenance["steps_not_exercised"] = (
        NOT_EXERCISED_STEPS
        if full
        else ["preprocess_full (Lee, sea mask, edge filter, fusion) — detector-only path"]
    )
    provenance["ground_truth"] = {
        "source": str(args.xview_dir / "validation.csv"),
        "confidence_filter": sorted({c.strip().upper() for c in args.confidence.split(",")}),
        "vessels_only": bool(args.vessels_only),
        "band": args.band,
    }
    params: dict[str, Any] = {
        "pipeline_path": args.pipeline_path,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "band_tile_rows": args.band_tile_rows if full else None,
        "confidence_threshold": confidence_threshold,
        "iou_threshold_nms": settings.iou_threshold,
        "edge_buffer_px": settings.edge_buffer_px if full else None,
        "fusion_iou_threshold": settings.fusion_iou_threshold if full else None,
        "fusion_yolo_weight": settings.fusion_yolo_weight if full else None,
        "cfar_cluster": {
            "min_cluster_size": settings.cfar_min_cluster_size,
            "eps": settings.cfar_cluster_eps,
            "min_mean_snr": settings.cfar_min_mean_snr,
        },
        "steps": steps,
        "match_mode": args.match_mode,
        "center_tolerance_px": args.center_tolerance_px,
        "iou_threshold_match": args.iou_threshold,
    }

    confidence_filter = {c.strip().upper() for c in args.confidence.split(",") if c.strip()}
    val_csv = args.xview_dir / "validation.csv"
    scene_ids = _list_med_scene_ids(args.med_manifest)
    if args.limit_scenes:
        scene_ids = scene_ids[: args.limit_scenes]
    logger.info("Scenes to evaluate: %d (%s)", len(scene_ids), args.pipeline_path)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    accs: dict[str, _Accumulator] = {name: _Accumulator() for name in set_names}
    run_started = time.perf_counter()

    for i, sid in enumerate(scene_ids, 1):
        tar_path = args.tar_dir / f"{sid}.tar.gz"
        if not tar_path.exists():
            logger.warning("Missing tar for %s — skipping", sid)
            continue
        logger.info("[%d/%d] %s — hashing tar + extracting %s", i, len(scene_ids), sid, args.band)
        tar_hash = compute_sha256(tar_path)
        provenance["image_hashes"][sid] = {
            "tar_sha256": tar_hash,
            "member": f"{sid}/{args.band}",
        }
        try:
            raster_path = _extract_band(tar_path, sid, args.tmp_dir, args.band)
        except KeyError:
            logger.warning("Band %s not present in %s — skipping", args.band, sid)
            continue

        gt, c_lat, c_lon = _load_gt_for_scene(val_csv, sid, confidence_filter, args.vessels_only)
        if not gt:
            logger.info("  no GT after filters — skipping")
            if not args.keep_raster:
                shutil.rmtree(raster_path.parent, ignore_errors=True)
            continue
        scene_area = _scene_area_km2(raster_path)

        t_scene = time.perf_counter()
        try:
            if full:
                from src.validation.harness import run_full_pipeline

                scene = run_full_pipeline(
                    raster_path,
                    engine=engine,
                    yolo=yolo,
                    cfar=cfar,
                    confidence_threshold=confidence_threshold,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    band_tile_rows=args.band_tile_rows,
                )
                pred_sets = {name: scene.sets[name] for name in set_names}
                gt_centres = np.array(
                    [[(g["bbox"][0] + g["bbox"][2]) / 2.0, (g["bbox"][1] + g["bbox"][3]) / 2.0] for g in gt]
                )
                gt_on_land = scene.is_on_land(gt_centres)
                scene_stats = {**scene.stats, "timings_s": scene.timings_s}
            else:
                preds = run_detector_only(
                    raster_path,
                    args.model,
                    confidence_threshold,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    settings=settings,
                )
                pred_sets = {"detector": preds}
                gt_on_land = np.zeros(len(gt), dtype=bool)
                scene_stats = {}
        except Exception:
            logger.exception("Inference failed on %s", sid)
            if not args.keep_raster:
                shutil.rmtree(raster_path.parent, ignore_errors=True)
            return 3
        scene_seconds = round(time.perf_counter() - t_scene, 1)

        for g, flag in zip(gt, gt_on_land, strict=True):
            g["on_land"] = bool(flag)
        gt_sea = [g for g in gt if not g["on_land"]]

        if args.dump_predictions is not None:
            args.dump_predictions.mkdir(parents=True, exist_ok=True)
            dump: dict[str, Any] = {
                "scene_id": sid,
                "tar_sha256": tar_hash,
                "pipeline_path": args.pipeline_path,
                "confidence_threshold": confidence_threshold,
                "ground_truth": gt,
                "sets": pred_sets,
            }
            if full:
                dump["scene_shape"] = list(scene.scene_shape)
                dump["lonlat_affine"] = list(scene.lonlat_affine)
                dump["stats"] = scene.stats
            (args.dump_predictions / f"{sid}.json").write_text(json.dumps(dump))

        for name, preds in pred_sets.items():
            acc = accs[name]
            tp, fp, fn, scored = _score(preds, gt, args)
            preds_sea = [p for p in preds if not p.get("on_land", False)]
            tp_s, fp_s, fn_s, scored_s = _score(preds_sea, gt_sea, args)
            acc.scored.append(scored)
            acc.scored_sea.append(scored_s)
            acc.tp += tp
            acc.fp += fp
            acc.fn += fn
            acc.tp_sea += tp_s
            acc.fp_sea += fp_s
            acc.fn_sea += fn_s
            acc.gt += len(gt)
            acc.gt_sea += len(gt_sea)
            acc.preds += len(preds)
            acc.preds_sea += len(preds_sea)
            acc.per_scene.append({
                "scene_id": sid,
                "aoi_label": _detect_aoi(c_lat, c_lon),
                "centroid_lat": round(c_lat, 4),
                "centroid_lon": round(c_lon, 4),
                "scene_area_km2": scene_area,
                "n_gt": len(gt),
                "n_gt_on_land": int(len(gt) - len(gt_sea)),
                "n_preds": len(preds),
                "n_preds_on_land": int(len(preds) - len(preds_sea)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sea_only": {"tp": tp_s, "fp": fp_s, "fn": fn_s},
                "seconds": scene_seconds,
                **({"pipeline_stats": scene_stats} if scene_stats else {}),
            })
            logger.info(
                "  %s [%s] gt=%d preds=%d tp=%d fp=%d fn=%d  pd=%.3f far/km²=%.4f | sea: pd=%.3f far=%.4f",
                sid,
                name,
                len(gt),
                len(preds),
                tp,
                fp,
                fn,
                tp / max(1, len(gt)),
                fp / max(1.0, scene_area),
                tp_s / max(1, len(gt_sea)),
                fp_s / max(1.0, scene_area),
            )
        logger.info("  scene wall time: %.0fs", scene_seconds)

        if not args.keep_raster:
            shutil.rmtree(raster_path.parent, ignore_errors=True)

    if args.tmp_dir.exists() and not any(args.tmp_dir.iterdir()):
        args.tmp_dir.rmdir()

    provenance["wall_time_s"] = round(time.perf_counter() - run_started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = "xview3-sar/validation/adriatic"

    for name, acc in accs.items():
        label = args.model if name == "detector" else SET_LABELS[name]
        report = _report(
            acc, sea_only=False, label=label, args=args,
            confidence_threshold=confidence_threshold, params=params, provenance=provenance,
        )
        report_sea = _report(
            acc, sea_only=True, label=label, args=args,
            confidence_threshold=confidence_threshold, params=params, provenance=provenance,
        )
        payload = report.as_dict()
        sea_payload = report_sea.as_dict()
        sea_payload.pop("pr_curve", None)
        sea_payload.pop("provenance", None)
        payload["sea_only"] = sea_payload
        payload["per_scene"] = acc.per_scene
        payload["dataset"] = dataset
        payload["dataset_split"] = "validation"
        payload["prediction_set"] = name

        out = args.output if len(accs) == 1 else args.output.with_name(
            f"{args.output.stem}_{name}{args.output.suffix}"
        )
        out.write_text(json.dumps(payload, indent=2))
        md = _format_markdown(report)
        md += (
            f"\n**Sea-only (I-DET-2, on_land excluded from preds and GT):** "
            f"AP {report_sea.map_at_iou:.4f} · Pd {report_sea.pd_recall:.4f} · "
            f"FAR/km² {report_sea.far_per_km2:.4f} · Precision {report_sea.precision:.4f} · "
            f"GT {report_sea.num_ground_truth} · preds {report_sea.num_predictions}\n"
        )
        out.with_suffix(".md").write_text(md)
        logger.info("Report [%s]: %s", name, out)
        sys.stdout.write(f"\n### {label}\n{md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
