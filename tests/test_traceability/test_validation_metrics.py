"""Tests for src.validation.metrics matchers and aggregator.

The matcher logic was previously inlined in ``scripts/run_validation.py``
and only exercised end-to-end. Moving it to ``src/validation/`` exposes
the helpers so unit tests can pin the contract that both the CLI and
the API endpoint depend on.
"""

from __future__ import annotations

from src.validation.metrics import (
    ValidationReport,
    bbox_iou,
    match_predictions,
    pr_curve_from_scored,
)


class TestBboxIou:
    def test_full_overlap(self):
        assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_no_overlap(self):
        assert bbox_iou([0, 0, 5, 5], [10, 10, 15, 15]) == 0.0

    def test_half_overlap(self):
        # [0,0,10,10] area=100; [5,0,15,10] area=100; overlap=[5,0,10,10] area=50
        # IoU = 50 / (100+100-50) = 50/150 ≈ 0.333
        assert abs(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


class TestMatchPredictions:
    def test_perfect_match(self):
        gt = [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 60, 60]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"bbox": [50, 50, 60, 60], "confidence": 0.8},
        ]
        tp, fp, fn, scored = match_predictions(preds, gt, iou_threshold=0.5)
        assert tp == 2
        assert fp == 0
        assert fn == 0
        assert all(is_tp for _, is_tp in scored)

    def test_one_false_positive(self):
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9},  # match
            {"bbox": [100, 100, 110, 110], "confidence": 0.5},  # FP
        ]
        tp, fp, fn, _ = match_predictions(preds, gt, iou_threshold=0.5)
        assert (tp, fp, fn) == (1, 1, 0)

    def test_one_false_negative(self):
        gt = [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 60, 60]}]
        preds = [{"bbox": [0, 0, 10, 10], "confidence": 0.9}]
        tp, fp, fn, _ = match_predictions(preds, gt, iou_threshold=0.5)
        assert (tp, fp, fn) == (1, 0, 1)

    def test_higher_confidence_wins_assignment(self):
        """When two predictions cover the same GT, the higher-confidence
        one must claim it (greedy by descending confidence)."""
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.6},
            {"bbox": [1, 1, 11, 11], "confidence": 0.9},
        ]
        tp, fp, fn, scored = match_predictions(preds, gt, iou_threshold=0.5)
        assert (tp, fp, fn) == (1, 1, 0)
        # The 0.9 prediction should be the TP, the 0.6 the FP.
        scored.sort(key=lambda x: x[0], reverse=True)
        assert scored[0] == (0.9, True)
        assert scored[1] == (0.6, False)

    def test_center_mode(self):
        gt = [{"bbox": [40, 40, 60, 60]}]  # centre (50, 50)
        preds = [
            {"bbox": [55, 55, 65, 65], "confidence": 0.9},  # centre (60,60), d≈14.1
            {"bbox": [200, 200, 210, 210], "confidence": 0.5},  # far
        ]
        tp, fp, fn, _ = match_predictions(
            preds, gt, iou_threshold=0.0, match_mode="center",
            center_tolerance_px=20.0,
        )
        assert (tp, fp, fn) == (1, 1, 0)


class TestPRCurveAndReport:
    def test_pr_curve_monotonic_recall(self):
        scored = [[(0.9, True), (0.8, False), (0.7, True)]]
        curve = pr_curve_from_scored(scored, total_gt=2)
        assert len(curve) == 3
        # Recall must be non-decreasing as we walk descending confidence.
        recalls = [pt["recall"] for pt in curve]
        assert recalls == sorted(recalls)
        # Precision at 0.9 conf: 1 TP / 1 pred = 1.0
        assert curve[0]["precision"] == 1.0

    def test_report_metrics_when_perfect(self):
        report = ValidationReport(
            model_name="m",
            iou_threshold=0.5,
            confidence_threshold=0.0,
            num_scenes=1,
            num_ground_truth=10,
            num_predictions=10,
            true_positives=10,
            false_positives=0,
            false_negatives=0,
            total_area_km2=40.0,
            pr_curve=[{"confidence": 1.0, "precision": 1.0, "recall": 1.0}],
        )
        assert report.precision == 1.0
        assert report.pd_recall == 1.0
        assert report.far_per_km2 == 0.0
        assert report.map_at_iou == 1.0

    def test_report_far_per_km2(self):
        report = ValidationReport(
            model_name="m",
            iou_threshold=0.5,
            confidence_threshold=0.0,
            num_scenes=2,
            num_ground_truth=4,
            num_predictions=6,
            true_positives=4,
            false_positives=2,
            false_negatives=0,
            total_area_km2=40.0,
        )
        assert report.far_per_km2 == 2 / 40.0
