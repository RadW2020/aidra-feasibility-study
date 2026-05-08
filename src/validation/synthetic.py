"""Synthetic-GT validation runner.

Generates deterministic synthetic SAR tiles with known ground-truth
vessel positions, runs CFAR + a YOLO detector against them, and
returns a :class:`ValidationReport`. Used by the
``POST /api/validation/synthetic`` endpoint to populate
``validation_runs`` rows when a real labelled dataset (xView3-SAR,
HRSID, OpenSARShip) is not yet wired up.

The numbers are honest in the sense that the matcher and the metric
formulas are identical to the manifest-driven harness in
``scripts/run_validation.py`` — only the data source is synthetic.
The dataset label persisted is ``synthetic-seed-<seed>`` so an
evaluator can tell synthetic runs apart from real ones at a glance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.cfar import CFARDetector
from src.models.yolo import YOLODetector
from src.pipeline.detection import DetectionEngine
from src.pipeline.preprocessing import generate_synthetic_sar_tile
from src.validation.metrics import (
    ValidationReport,
    match_predictions,
    pr_curve_from_scored,
)

logger = logging.getLogger(__name__)


def _synthetic_tile(size: int, num_vessels: int, seed: int) -> tuple[
    np.ndarray, list[dict[str, Any]]
]:
    """Generate one synthetic tile + GT bboxes in pixel coordinates.

    The preprocessing helper returns vessel centres + half-widths;
    we project to ``[x_min, y_min, x_max, y_max]`` so the matcher in
    :func:`match_predictions` can score them with bbox IoU.
    """
    image, gt = generate_synthetic_sar_tile(
        size=size, num_vessels=num_vessels, seed=seed
    )
    gt_bboxes: list[dict[str, Any]] = []
    for v in gt:
        bbox = v.get("bbox")
        if bbox is None:
            cx, cy = v["center"]
            w = v.get("width", 8)
            h = v.get("height", 8)
            bbox = [cx - w, cy - h, cx + w, cy + h]
        gt_bboxes.append({
            "bbox": [float(x) for x in bbox],
            "class_name": v.get("class_name", "vessel"),
        })
    return image, gt_bboxes


def _build_tile_payload(
    image: np.ndarray, tile_index: int
) -> dict[str, Any]:
    """Wrap a numpy tile in the dict shape DetectionEngine expects."""
    return {
        "data": image,
        "tile_index": tile_index,
        # Synthetic tile has no geocoding; DetectionEngine.run handles
        # the missing geo_bounds gracefully (sea_mask is just None).
        "geo_bounds": {},
        "tile_row_offset": 0,
        "tile_col_offset": 0,
    }


async def run_synthetic_validation(
    yolo_detector: YOLODetector,
    detection_engine: DetectionEngine,
    *,
    num_scenes: int = 5,
    num_vessels: int = 8,
    tile_size: int = 640,
    seed: int = 42,
    iou_threshold: float = 0.3,
    confidence_threshold: float = 0.0,
    pixel_spacing_m: float = 10.0,
) -> ValidationReport:
    """Run end-to-end synthetic validation and return the report.

    Each scene is a single ``tile_size``-sized tile, so the synthetic
    "scene area" is ``(tile_size * pixel_spacing_m / 1000) ** 2`` km²
    per scene. Default 640 × 10 m = 6.4 km → 40.96 km²/scene.

    The CFAR detector is built fresh with default settings; the YOLO
    detector is supplied by the caller so a model already loaded in
    memory (e.g. by ModelManager) is reused.
    """
    cfar = CFARDetector()
    model_info = yolo_detector.get_model_info()
    model_name = model_info.get("name", "unknown")
    scene_area_km2 = (tile_size * pixel_spacing_m / 1000.0) ** 2

    per_scene_scored: list[list[tuple[float, bool]]] = []
    tp_total = fp_total = fn_total = 0
    gt_total = pred_total = 0
    area_total = 0.0

    for i in range(num_scenes):
        image, gt = _synthetic_tile(tile_size, num_vessels, seed + i)
        tile = _build_tile_payload(image, tile_index=i)
        result = detection_engine.run(
            tiles=[tile],
            detector=yolo_detector,
            cfar=cfar,
            constraint_profile="ground",
            scene_shape=(image.shape[0], image.shape[1]),
        )
        # Predictions come back as Detection objects — project to the
        # dict shape match_predictions expects (bbox + confidence).
        preds = [
            {
                "bbox": [float(x) for x in d.bbox_pixel],
                "confidence": float(d.confidence),
                "class_name": d.class_name,
            }
            for d in result.detections
            if float(d.confidence) >= confidence_threshold
        ]
        tp, fp, fn, scored = match_predictions(
            preds,
            gt,
            iou_threshold=iou_threshold,
            match_mode="iou",
        )
        gt_total += len(gt)
        pred_total += len(preds)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        area_total += scene_area_km2
        per_scene_scored.append(scored)
        logger.info(
            "synthetic scene %d/%d: gt=%d preds=%d tp=%d fp=%d fn=%d",
            i + 1, num_scenes, len(gt), len(preds), tp, fp, fn,
        )

    pr_curve = pr_curve_from_scored(per_scene_scored, gt_total)
    return ValidationReport(
        model_name=model_name,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        num_scenes=num_scenes,
        num_ground_truth=gt_total,
        num_predictions=pred_total,
        true_positives=tp_total,
        false_positives=fp_total,
        false_negatives=fn_total,
        total_area_km2=area_total,
        pr_curve=pr_curve,
        match_mode="iou",
    )
