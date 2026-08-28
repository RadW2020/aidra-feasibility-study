"""
D2 — Formal validation harness for AIDRA detectors (manifest-driven CLI).

Computes the metrics required by every MODEL_CARD's "Métricas de
validación" section and by the SatCen Q3 rubric:

    - AP              (VOC all-points; at IoU or at centre tolerance)
    - Pd              (probability of detection = recall)
    - FAR / km²       (false alarms per square kilometre)
    - precision, F1
    - PR curve        (sample of operating points)

Inputs
------
A *labels manifest* JSON with an array of items:

    [
      {
        "image_path": "/path/to/scene_001.tif",
        "scene_area_km2": 256.0,
        "ground_truth": [
          {"bbox": [x_min, y_min, x_max, y_max]},
          ...
        ]
      },
      ...
    ]

    bbox is in **pixel** coordinates of the image. Optional fields
    accepted: ``image_id``, ``aoi_name``, ``incidence_angle``.

Run
---
    python -m scripts.run_validation \
        --manifest data/validation/gibraltar_test.json \
        --model vesseltracker-sar-yolov8 \
        --output reports/validation_gibraltar.json

``--pipeline-path detector`` (default) feeds the raster straight into one
detector — the detector in isolation. ``--pipeline-path full`` runs the
production ``DetectionEngine`` (Lee, sea mask, fusion, edge filter) and
``--model`` selects which prediction set is scored (``cfar-default``,
``vesseltracker-sar-yolov8``, ``aidra`` or ``fused``). The heavy lifting
lives in :mod:`src.validation.harness`; this file is the CLI.

Thresholds default to ``Settings`` (I-DET-4). Every report carries a
``provenance`` block (commit_sha, model_hash, settings_hash, seed,
image hashes) and a ``params`` block (tile size, pipeline path, steps).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.validation.harness import (
    FULL_PIPELINE_STEPS,
    NOT_EXERCISED_STEPS,
    build_provenance,
    cfar_params_of,
    run_detector_only,
    sar_to_uint8,
    tile_indices,
)
from src.validation.metrics import (
    ValidationReport,
    bbox_iou,
    match_predictions,
    pr_curve_from_scored,
)

logger = logging.getLogger("aidra.validation")

# Backwards-compatible aliases: tests and validate_xview3_serial import
# these private names from here.
_bbox_iou = bbox_iou
_match_predictions = match_predictions
_pr_curve_from_scored = pr_curve_from_scored
_run_inference = run_detector_only
_tile_indices = tile_indices
_sar_to_uint8 = sar_to_uint8

FULL_PATH_SETS: dict[str, str] = {
    "cfar-default": "cfar",
    "vesseltracker-sar-yolov8": "yolo",
    "aidra": "aidra",
    "fused": "fused_only",
}


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Build one with "
            "scripts/build_xview3_manifest.py or hand-curate JSON. See "
            "scripts/run_validation.py docstring for the schema."
        )
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"Manifest {manifest_path} is empty or not a list.")
    for entry in data:
        if "image_path" not in entry:
            raise ValueError("Manifest entry missing 'image_path'.")
        if "scene_area_km2" not in entry:
            raise ValueError("Manifest entry missing 'scene_area_km2'.")
        if "ground_truth" not in entry:
            raise ValueError("Manifest entry missing 'ground_truth'.")
    return data


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_markdown(report: ValidationReport) -> str:
    if report.match_mode == "center":
        match_label = (
            f"distancia al centro ≤ {report.center_tolerance_px:.0f} px "
            "(xView3-SAR convention)"
        )
    else:
        match_label = f"IoU ≥ {report.iou_threshold:.2f}"
    prov = report.provenance or {}
    params = report.params or {}
    lines = [
        "## Métricas de validación (D2 — `scripts/run_validation.py`)\n",
        f"- **Match mode**: {match_label}",
        f"- **AP**: {report.map_at_iou:.4f}",
        f"- **Pd (recall)**: {report.pd_recall:.4f}",
        f"- **FAR / km²**: {report.far_per_km2:.4f}",
        f"- **Precision**: {report.precision:.4f}",
        f"- **F1**: {report.f1:.4f}",
        f"- Escenas evaluadas: {report.num_scenes}",
        f"- Ground-truth total: {report.num_ground_truth}",
        f"- Predicciones (post-confidence ≥ "
        f"{report.confidence_threshold:.2f}): {report.num_predictions}",
        f"- Área cubierta: {report.total_area_km2:.1f} km²",
    ]
    if params:
        lines.append(
            f"- Pipeline path: `{params.get('pipeline_path', 'detector')}`, "
            f"tile {params.get('tile_size')}px / overlap {params.get('tile_overlap')}px"
        )
    if prov:
        lines.append(
            f"- Provenance: commit `{str(prov.get('commit_sha', ''))[:12]}`, "
            f"model_hash `{str(prov.get('model_hash') or 'n/a')[:16]}`, "
            f"settings_hash `{str(prov.get('settings_hash', ''))[:16]}`, "
            f"seed {prov.get('seed', {}).get('seed') if isinstance(prov.get('seed'), dict) else prov.get('seed')}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal D2 validation against a labelled manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Detector-only path: model name in models/. Full path: one of "
            + ", ".join(FULL_PATH_SETS)
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pipeline-path",
        choices=["detector", "full"],
        default="detector",
        help="'detector' = detector in isolation; 'full' = production DetectionEngine.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Defaults to Settings.confidence_threshold (I-DET-4).",
    )
    parser.add_argument("--match-mode", choices=["iou", "center"], default="iou")
    parser.add_argument("--center-tolerance-px", type=float, default=20.0)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Defaults to Settings.tile_size. 0 = whole raster (detector path only).",
    )
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override Settings.models_dir (sets MODELS_DIR before Settings loads).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        manifest = _load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    if args.dry_run:
        total_gt = sum(len(e["ground_truth"]) for e in manifest)
        total_area = sum(float(e["scene_area_km2"]) for e in manifest)
        logger.info(
            "Dry run OK — %d scenes, %d ground truths, %.1f km² total area.",
            len(manifest),
            total_gt,
            total_area,
        )
        return 0

    if args.models_dir is not None:
        os.environ["MODELS_DIR"] = str(args.models_dir)

    from src.config import Settings
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

    if args.pipeline_path == "full" and args.model not in FULL_PATH_SETS:
        logger.error("--pipeline-path full requires --model in %s", list(FULL_PATH_SETS))
        return 2

    # Detectors / engine for the full path are built once.
    engine = yolo = cfar = None
    model_path: Path | None = None
    steps: list[str]
    if args.pipeline_path == "full":
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
        harness="scripts/run_validation.py",
        cfar_params=cfar_params_of(cfar) if cfar is not None else None,
    )
    provenance["steps_not_exercised"] = NOT_EXERCISED_STEPS if args.pipeline_path == "full" else [
        "preprocess_full (Lee, sea mask, edge filter, fusion) — detector-only path"
    ]

    per_scene_scored: list[list[tuple[float, bool]]] = []
    tp_total = fp_total = fn_total = 0
    gt_total = pred_total = 0
    area_total = 0.0

    for entry in manifest:
        image_path = Path(entry["image_path"])
        gt = entry["ground_truth"]
        gt_total += len(gt)
        area_total += float(entry["scene_area_km2"])
        try:
            if args.pipeline_path == "full":
                from src.validation.harness import run_full_pipeline

                scene = run_full_pipeline(
                    image_path,
                    engine=engine,
                    yolo=yolo,
                    cfar=cfar,
                    confidence_threshold=confidence_threshold,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
                preds = scene.sets[FULL_PATH_SETS[args.model]]
            else:
                preds = run_detector_only(
                    image_path,
                    args.model,
                    confidence_threshold,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    settings=settings,
                )
        except Exception as exc:
            logger.error("Inference failed on %s: %s", image_path, exc)
            return 3
        from src.traceability.hasher import compute_sha256

        provenance["image_hashes"][entry.get("image_id") or image_path.name] = compute_sha256(
            image_path
        )
        pred_total += len(preds)
        tp, fp, fn, scored = match_predictions(
            preds,
            gt,
            args.iou_threshold,
            match_mode=args.match_mode,
            center_tolerance_px=args.center_tolerance_px,
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
        per_scene_scored.append(scored)
        logger.info(
            "scene=%s gt=%d preds=%d tp=%d fp=%d fn=%d",
            entry.get("image_id") or image_path.name,
            len(gt),
            len(preds),
            tp,
            fp,
            fn,
        )

    report = ValidationReport(
        model_name=args.model,
        iou_threshold=args.iou_threshold,
        confidence_threshold=confidence_threshold,
        num_scenes=len(manifest),
        num_ground_truth=gt_total,
        num_predictions=pred_total,
        true_positives=tp_total,
        false_positives=fp_total,
        false_negatives=fn_total,
        total_area_km2=area_total,
        pr_curve=pr_curve_from_scored(per_scene_scored, gt_total),
        match_mode=args.match_mode,
        center_tolerance_px=args.center_tolerance_px,
        params={
            "pipeline_path": args.pipeline_path,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "confidence_threshold": confidence_threshold,
            "iou_threshold_nms": settings.iou_threshold,
            "edge_buffer_px": settings.edge_buffer_px if args.pipeline_path == "full" else None,
            "fusion_iou_threshold": settings.fusion_iou_threshold if args.pipeline_path == "full" else None,
            "steps": steps,
        },
        provenance=provenance,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.as_dict(), indent=2))
    md_block = _format_markdown(report)
    args.output.with_suffix(".md").write_text(md_block)
    logger.info("Report written: %s", args.output)
    sys.stdout.write(md_block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
