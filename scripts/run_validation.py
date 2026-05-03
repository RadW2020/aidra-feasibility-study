"""
D2 — Formal validation harness for AIDRA detectors.

Computes the metrics required by every MODEL_CARD's "Métricas de
validación" section and by the SatCen Q3 rubric:

    - mAP@0.5         (PASCAL-VOC style, single class "vessel")
    - Pd              (probability of detection = recall @ IoU 0.5)
    - FAR / km²       (false alarms per square kilometre)
    - precision       (sanity check)
    - per-confidence  (PR curve, sample of operating points)

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

If the manifest is missing the script emits an explicit "no dataset"
report rather than fabricating numbers — keeps L3 honest.

Output
------
JSON file with the metrics + a Markdown block ready to paste into
``models/cards/<name>.MODEL_CARD.md``.

Notes
-----
- Single-class evaluation (vessel). Multi-class would require label
  taxonomy harmonisation across xView3-SAR / HRSID / OpenSARShip.
- IoU threshold is 0.5 (default mAP@0.5). Higher thresholds tighten
  the metric — left as CLI flag.
- This harness intentionally does **not** download any dataset: it
  consumes a manifest produced by ``scripts/prepare_validation.py``
  (post-MVP) or hand-curated by the analyst.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("aidra.validation")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregated metrics over a manifest."""

    model_name: str
    iou_threshold: float
    confidence_threshold: float
    num_scenes: int
    num_ground_truth: int
    num_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    total_area_km2: float
    pr_curve: list[dict[str, float]] = field(default_factory=list)
    match_mode: str = "iou"
    center_tolerance_px: float = 20.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def pd_recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def far_per_km2(self) -> float:
        return (
            self.false_positives / self.total_area_km2
            if self.total_area_km2 > 0
            else 0.0
        )

    @property
    def map_at_iou(self) -> float:
        """Single-class mAP from the PR curve (interpolated AP)."""
        if not self.pr_curve:
            return 0.0
        # Sort by recall ascending, then compute interpolated AP.
        pts = sorted(self.pr_curve, key=lambda p: p["recall"])
        ap = 0.0
        prev_recall = 0.0
        for pt in pts:
            ap += (pt["recall"] - prev_recall) * pt["precision"]
            prev_recall = pt["recall"]
        return ap

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "match_mode": self.match_mode,
            "iou_threshold": self.iou_threshold,
            "center_tolerance_px": self.center_tolerance_px,
            "confidence_threshold": self.confidence_threshold,
            "num_scenes": self.num_scenes,
            "num_ground_truth": self.num_ground_truth,
            "num_predictions": self.num_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_area_km2": self.total_area_km2,
            "precision": round(self.precision, 4),
            "pd_recall": round(self.pd_recall, 4),
            "far_per_km2": round(self.far_per_km2, 4),
            "map_at_iou": round(self.map_at_iou, 4),
            "pr_curve": self.pr_curve,
            "computed_at_utc": datetime.now(tz=UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _bbox_iou(a: list[float], b: list[float]) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _match_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
    match_mode: str = "iou",
    center_tolerance_px: float = 20.0,
) -> tuple[int, int, int, list[tuple[float, bool]]]:
    """Greedy 1-to-1 matching by descending confidence.

    ``match_mode``:
      - ``"iou"``: classic bbox IoU >= ``iou_threshold`` (PASCAL-VOC).
      - ``"center"``: Euclidean distance between bbox centres
        <= ``center_tolerance_px``. Used by xView3-SAR (200 m at 10 m
        GRD pixel spacing → 20 px) for ship detection scoring, where
        the detector returns a centroid rather than an exact bbox.

    Returns ``(tp, fp, fn, scored)``. ``scored`` is a list of
    ``(confidence, is_true_positive)`` for PR-curve aggregation.
    """
    sorted_preds = sorted(
        predictions, key=lambda p: float(p.get("confidence", 0.0)), reverse=True
    )
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    scored: list[tuple[float, bool]] = []

    for pred in sorted_preds:
        best_score: float | None = None
        best_gt: int | None = None
        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if match_mode == "center":
                d = _bbox_center_distance(pred["bbox"], gt["bbox"])
                if d <= center_tolerance_px and (
                    best_score is None or d < best_score
                ):
                    best_score = d
                    best_gt = gi
            else:
                iou = _bbox_iou(pred["bbox"], gt["bbox"])
                if iou >= iou_threshold and (
                    best_score is None or iou > best_score
                ):
                    best_score = iou
                    best_gt = gi

        if best_gt is not None:
            matched_gt.add(best_gt)
            tp += 1
            scored.append((float(pred.get("confidence", 0.0)), True))
        else:
            fp += 1
            scored.append((float(pred.get("confidence", 0.0)), False))

    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn, scored


def _pr_curve_from_scored(
    scored_per_scene: list[list[tuple[float, bool]]],
    total_gt: int,
) -> list[dict[str, float]]:
    """Aggregate per-scene scored detections into a PR curve."""
    if total_gt == 0:
        return []
    flat = [s for scene in scored_per_scene for s in scene]
    flat.sort(key=lambda x: x[0], reverse=True)
    cum_tp = 0
    cum_fp = 0
    curve: list[dict[str, float]] = []
    for confidence, is_tp in flat:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        precision = cum_tp / max(cum_tp + cum_fp, 1)
        recall = cum_tp / total_gt
        curve.append({
            "confidence": round(confidence, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    return curve


# ---------------------------------------------------------------------------
# Inference adapter
# ---------------------------------------------------------------------------


def _tile_indices(
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


def _run_inference(
    image_path: Path,
    model_name: str,
    confidence_threshold: float,
    raster_in_db: bool | None = None,
    tile_size: int = 0,
    tile_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Run inference on a single preprocessed sigma0 raster.

    The harness is DB-free: it resolves the model file directly under
    ``Settings.models_dir`` and enforces the I-AIA-1 gate via
    ``ModelManager._require_model_card`` so a missing card aborts the
    run before any inference runs.

    CFAR detectors (name starts with ``cfar``) are loaded with default
    parameters and fed linear sigma0. Anything else is loaded as YOLO
    against the resolved ``.pt`` and fed uint8 RGB derived from the
    same SAR-standard log stretch the production pipeline uses.
    """
    # Lazy imports keep ``--help`` cheap.
    import numpy as np
    import rasterio
    import rasterio.windows  # noqa: F401  (used inside the tiling branch)

    from src.config import Settings
    from src.models.manager import ModelManager

    settings = Settings()
    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = Path(settings.models_dir)
    manager._cache = {}  # type: ignore[attr-defined]
    manager._load_order = []  # type: ignore[attr-defined]
    manager.max_cached_models = 1  # type: ignore[attr-defined]

    is_cfar = model_name.lower().startswith("cfar")

    if is_cfar:
        # I-AIA-1: even built-in detectors must have a card.
        manager._require_model_card(
            model_name, manager.models_dir / f"{model_name}.pt"
        )
        from src.models.cfar import CFARDetector

        detector = CFARDetector()
    else:
        model_path = manager._find_model_file(model_name, version=None)
        if model_path is None:
            raise FileNotFoundError(
                f"No model file matching '{model_name}' under "
                f"{manager.models_dir}. Did you archive it?"
            )
        manager._require_model_card(model_name, model_path)
        from src.models.yolo import YOLODetector

        detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=0.45,
        )

    if raster_in_db is None:
        raster_in_db = "_dB" in image_path.name or "_db" in image_path.name

    with rasterio.open(image_path) as src:
        height = int(src.height)
        width = int(src.width)
        # When tiling, read each window separately to bound peak RAM
        # to (tile_size² × dtype). When not, read the whole array.
        tile_arr_full = (
            None if tile_size > 0 else src.read(1).astype(np.float32)
        )

    def _to_linear(arr: np.ndarray) -> np.ndarray:
        if not raster_in_db:
            return arr
        nodata_mask = arr <= -1000.0
        arr_db = np.clip(arr, -50.0, 30.0)
        return np.where(
            nodata_mask, 0.0, np.power(10.0, arr_db / 10.0)
        ).astype(np.float32)

    def _cfar_on_tile(tile_arr: np.ndarray) -> list[dict[str, Any]]:
        # CFAR pixel-detection guard. Production xView3 scenes contain
        # ports/urban/coastal patches where CFAR fires on >5 % of the
        # tile, producing 100k+ candidate pixels that overwhelm DBSCAN
        # (multi-minute hang per tile, no useful output). When the
        # raw pixel-hit fraction is implausibly high we treat the tile
        # as saturated clutter and emit zero detections — same
        # behaviour as the production engine deferring to land-mask /
        # cluster_anomaly downstream.
        valid = (tile_arr > 0).sum()
        if valid == 0:
            return []
        # Cheap pre-pass: a single CFAR call to count pixel hits.
        from src.models.cfar import CFARDetector

        if isinstance(detector, CFARDetector):
            pixel_hits = detector.detect(tile_arr)
            if len(pixel_hits) > 0.05 * valid:
                return []
        raw = detector.detect_with_clustering(
            tile_arr,
            min_cluster_size=5,
            eps=1.5,
            min_mean_snr=2.0,
        )
        local: list[dict[str, Any]] = []
        for d in raw:
            snr = float(d.get("mean_snr", d.get("snr", 0.0)))
            confidence = float(min(1.0, max(0.0, 1.0 - np.exp(-snr / 10.0))))
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
        tile_uint8 = _sar_to_uint8(tile_arr)
        raw_dets = detector.predict(tile_uint8)
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
    windows = _tile_indices(height, width, tile_size, tile_overlap)

    if tile_size > 0:
        # Stream each tile from disk to keep peak RAM bounded.
        for r_off, c_off, h, w in windows:
            with rasterio.open(image_path) as src:
                tile_arr = src.read(
                    1,
                    window=rasterio.windows.Window(c_off, r_off, w, h),
                ).astype(np.float32)
            tile_lin = _to_linear(tile_arr)
            tile_dets = _cfar_on_tile(tile_lin) if is_cfar else _yolo_on_tile(tile_lin)
            for det in tile_dets:
                bx = det["bbox"]
                # Map tile-local pixel bbox to scene coordinates.
                det["bbox"] = [
                    float(bx[0]) + c_off,
                    float(bx[1]) + r_off,
                    float(bx[2]) + c_off,
                    float(bx[3]) + r_off,
                ]
                out.append(det)
        return out

    # No tiling — operate on the full array.
    arr_lin = _to_linear(tile_arr_full)
    return _cfar_on_tile(arr_lin) if is_cfar else _yolo_on_tile(arr_lin)


def _sar_to_uint8(arr: Any) -> Any:
    """Same dB stretch used in production — see detection.py."""
    import numpy as np

    safe = np.clip(arr.astype(np.float32), 1e-10, None)
    db = 10.0 * np.log10(safe)
    db = np.clip(db, -25.0, 0.0)
    norm = (db + 25.0) / 25.0
    gray = (norm * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Build one with "
            "scripts/prepare_validation.py (post-MVP) or hand-curate "
            "JSON. See scripts/run_validation.py docstring for the "
            "schema."
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
# Main
# ---------------------------------------------------------------------------


def _format_markdown(report: ValidationReport) -> str:
    if report.match_mode == "center":
        match_label = (
            f"distancia al centro ≤ {report.center_tolerance_px:.0f} px "
            "(xView3-SAR convention)"
        )
    else:
        match_label = f"IoU ≥ {report.iou_threshold:.2f}"
    return (
        "## Métricas de validación (D2 — `scripts/run_validation.py`)\n\n"
        f"- **Match mode**: {match_label}\n"
        f"- **mAP**: {report.map_at_iou:.4f}\n"
        f"- **Pd (recall)**: {report.pd_recall:.4f}\n"
        f"- **FAR / km²**: {report.far_per_km2:.4f}\n"
        f"- **Precision**: {report.precision:.4f}\n"
        f"- Escenas evaluadas: {report.num_scenes}\n"
        f"- Ground-truth total: {report.num_ground_truth}\n"
        f"- Predicciones (post-confidence ≥ "
        f"{report.confidence_threshold:.2f}): {report.num_predictions}\n"
        f"- Área cubierta: {report.total_area_km2:.1f} km²\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run formal D2 validation against a labelled manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the JSON labels manifest (see module docstring).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name in models/ (e.g. vesseltracker-sar-yolov8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.25
    )
    parser.add_argument(
        "--match-mode",
        choices=["iou", "center"],
        default="iou",
        help=(
            "Detection-vs-GT matching mode. 'iou' is PASCAL-VOC bbox "
            "IoU. 'center' uses Euclidean distance between bbox "
            "centres (xView3-SAR style for centroid-only detectors)."
        ),
    )
    parser.add_argument(
        "--center-tolerance-px",
        type=float,
        default=20.0,
        help=(
            "Pixel tolerance for --match-mode=center. Default 20 px = "
            "200 m at 10 m GRD pixel spacing (xView3-SAR convention)."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help=(
            "If > 0, run inference over sliding tiles of this pixel "
            "size. Required for full Sentinel-1 GRD scenes (≈ 22k×30k) "
            "where loading the full raster as float32 would consume "
            "5+ GB. Recommended 1024-2048 for CFAR, 640 for YOLO."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Overlap (px) between consecutive tiles to avoid edge misses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest schema and exit without inference.",
    )
    args = parser.parse_args(argv)

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

    per_scene_scored: list[list[tuple[float, bool]]] = []
    per_class_count: defaultdict[str, int] = defaultdict(int)
    tp_total = fp_total = fn_total = 0
    gt_total = pred_total = 0
    area_total = 0.0

    for entry in manifest:
        image_path = Path(entry["image_path"])
        gt = entry["ground_truth"]
        gt_total += len(gt)
        area_total += float(entry["scene_area_km2"])
        for g in gt:
            per_class_count[g.get("class_name", "vessel")] += 1
        try:
            preds = _run_inference(
                image_path,
                args.model,
                args.confidence_threshold,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
        except Exception as exc:
            logger.error("Inference failed on %s: %s", image_path, exc)
            return 3
        pred_total += len(preds)
        tp, fp, fn, scored = _match_predictions(
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

    pr_curve = _pr_curve_from_scored(per_scene_scored, gt_total)
    report = ValidationReport(
        model_name=args.model,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        num_scenes=len(manifest),
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
    args.output.write_text(json.dumps(report.as_dict(), indent=2))

    md_block = _format_markdown(report)
    md_path = args.output.with_suffix(".md")
    md_path.write_text(md_block)

    logger.info("Report written: %s", args.output)
    logger.info("Markdown ready for MODEL_CARD: %s", md_path)
    sys.stdout.write(md_block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
