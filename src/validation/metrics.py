"""Matcher and aggregator helpers for AIDRA validation.

Extracted from ``scripts/run_validation.py`` so the logic is reusable
from both the CLI harness and the in-process API endpoint. Behaviour
is identical to the original module — see the script docstring for the
PASCAL-VOC / xView3-SAR conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ValidationReport:
    """Aggregated metrics over a manifest or synthetic batch."""

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


def bbox_iou(a: list[float], b: list[float]) -> float:
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


def bbox_center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


# ---------------------------------------------------------------------------
# Matching + PR-curve aggregation
# ---------------------------------------------------------------------------


def match_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
    match_mode: str = "iou",
    center_tolerance_px: float = 20.0,
) -> tuple[int, int, int, list[tuple[float, bool]]]:
    """Greedy 1-to-1 matching by descending confidence.

    Two matchers are supported:

    * ``"iou"`` — classic bbox IoU >= ``iou_threshold`` (PASCAL-VOC).
    * ``"center"`` — Euclidean distance between bbox centres
      <= ``center_tolerance_px``. xView3-SAR convention (200 m at
      10 m GRD pixel spacing -> 20 px) for centroid-only detectors.

    Returns ``(tp, fp, fn, scored)`` where ``scored`` is the per-
    prediction ``(confidence, is_true_positive)`` list used by
    ``pr_curve_from_scored``.
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
                d = bbox_center_distance(pred["bbox"], gt["bbox"])
                if d <= center_tolerance_px and (
                    best_score is None or d < best_score
                ):
                    best_score = d
                    best_gt = gi
            else:
                iou = bbox_iou(pred["bbox"], gt["bbox"])
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


def pr_curve_from_scored(
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
