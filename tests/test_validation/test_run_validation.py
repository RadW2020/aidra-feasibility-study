"""Tests for the D2 validation harness (palanca L3).

Cover the deterministic pieces — matching, PR aggregation, manifest
loader, dry-run path — without actually loading any model weight.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable as ``run_validation``.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_validation as rv  # noqa: E402


class TestBboxIoU:
    def test_disjoint_boxes(self):
        assert rv._bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_identical_boxes(self):
        assert rv._bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_partial_overlap(self):
        # 50% overlap by area → IoU = 1/3.
        iou = rv._bbox_iou([0, 0, 10, 10], [5, 0, 15, 10])
        assert iou == pytest.approx(1 / 3, abs=1e-6)


class TestMatchPredictions:
    def test_perfect_match_all_tp(self):
        gt = [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 60, 60]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"bbox": [50, 50, 60, 60], "confidence": 0.7},
        ]
        tp, fp, fn, _scored = rv._match_predictions(preds, gt, 0.5)
        assert (tp, fp, fn) == (2, 0, 0)

    def test_unmatched_pred_is_fp(self):
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [{"bbox": [100, 100, 110, 110], "confidence": 0.8}]
        tp, fp, fn, _ = rv._match_predictions(preds, gt, 0.5)
        assert (tp, fp, fn) == (0, 1, 1)

    def test_low_iou_below_threshold_is_fp(self):
        gt = [{"bbox": [0, 0, 10, 10]}]
        # Slightly overlapping → IoU ≈ 0.21 < 0.5
        preds = [{"bbox": [8, 0, 18, 10], "confidence": 0.9}]
        tp, fp, fn, _ = rv._match_predictions(preds, gt, 0.5)
        assert (tp, fp, fn) == (0, 1, 1)

    def test_higher_confidence_wins(self):
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.4},
            {"bbox": [0, 0, 10, 10], "confidence": 0.9},
        ]
        tp, fp, fn, scored = rv._match_predictions(preds, gt, 0.5)
        # Only one TP (greedy 1-to-1) — the higher-confidence one.
        assert tp == 1
        assert fp == 1
        # The TP is tagged with confidence 0.9.
        tp_scored = [s for s in scored if s[1]]
        assert tp_scored[0][0] == 0.9


class TestCenterMatching:
    """xView3-SAR-style centroid matching for centroid-only detectors."""

    def test_center_match_accepts_inside_pred(self):
        # Pred bbox is tiny but centred on the GT.
        gt = [{"bbox": [50, 50, 70, 70]}]  # 20×20 box centred at (60,60)
        preds = [{"bbox": [59, 59, 61, 61], "confidence": 0.7}]  # centre (60,60)
        tp, fp, fn, _ = rv._match_predictions(
            preds, gt, iou_threshold=0.5, match_mode="center",
            center_tolerance_px=20.0,
        )
        assert (tp, fp, fn) == (1, 0, 0)

    def test_center_match_rejects_far_pred(self):
        gt = [{"bbox": [50, 50, 70, 70]}]
        # Centre 50 px away
        preds = [{"bbox": [109, 60, 111, 62], "confidence": 0.7}]
        tp, fp, fn, _ = rv._match_predictions(
            preds, gt, iou_threshold=0.5, match_mode="center",
            center_tolerance_px=20.0,
        )
        assert (tp, fp, fn) == (0, 1, 1)

    def test_center_match_picks_nearest_when_multiple(self):
        # Two preds share the same confidence; expect the nearest to GT to win.
        gt = [{"bbox": [100, 100, 120, 120]}]
        preds = [
            {"bbox": [108, 110, 112, 110], "confidence": 0.8},  # centre 4 px off
            {"bbox": [115, 117, 119, 117], "confidence": 0.8},  # centre 9 px off
        ]
        tp, fp, fn, _ = rv._match_predictions(
            preds, gt, iou_threshold=0.5, match_mode="center",
            center_tolerance_px=20.0,
        )
        # 1 GT only → exactly 1 TP regardless of which one matches.
        assert tp == 1
        assert fp == 1
        assert fn == 0


class TestPRCurve:
    def test_monotonic_recall_after_sort(self):
        scored_per_scene = [
            [(0.9, True), (0.8, False), (0.7, True)],
            [(0.6, False), (0.5, True)],
        ]
        curve = rv._pr_curve_from_scored(scored_per_scene, total_gt=3)
        recalls = [pt["recall"] for pt in curve]
        assert recalls == sorted(recalls)
        # Last point covers all TPs.
        assert curve[-1]["recall"] == pytest.approx(1.0)

    def test_empty_when_no_gt(self):
        assert rv._pr_curve_from_scored([], total_gt=0) == []


class TestManifestLoader:
    def test_missing_manifest_explicit_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            rv._load_manifest(tmp_path / "nope.json")

    def test_empty_list_rejected(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="empty"):
            rv._load_manifest(p)

    def test_missing_field_rejected(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"image_path": "x.tif"}]))
        with pytest.raises(ValueError, match="scene_area_km2"):
            rv._load_manifest(p)

    def test_well_formed_manifest_accepted(self, tmp_path: Path):
        p = tmp_path / "ok.json"
        p.write_text(
            json.dumps(
                [
                    {
                        "image_path": "scene_001.tif",
                        "scene_area_km2": 256.0,
                        "ground_truth": [{"bbox": [0, 0, 10, 10]}],
                    }
                ]
            )
        )
        data = rv._load_manifest(p)
        assert len(data) == 1
        assert data[0]["scene_area_km2"] == 256.0


class TestValidationReport:
    def test_metrics_when_no_data(self):
        r = rv.ValidationReport(
            model_name="x",
            iou_threshold=0.5,
            confidence_threshold=0.25,
            num_scenes=0,
            num_ground_truth=0,
            num_predictions=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            total_area_km2=0.0,
        )
        assert r.precision == 0.0
        assert r.pd_recall == 0.0
        assert r.far_per_km2 == 0.0
        assert r.map_at_iou == 0.0

    def test_metrics_for_perfect_run(self):
        r = rv.ValidationReport(
            model_name="x",
            iou_threshold=0.5,
            confidence_threshold=0.25,
            num_scenes=2,
            num_ground_truth=10,
            num_predictions=10,
            true_positives=10,
            false_positives=0,
            false_negatives=0,
            total_area_km2=100.0,
            pr_curve=[
                {"confidence": 0.9, "precision": 1.0, "recall": 0.5},
                {"confidence": 0.8, "precision": 1.0, "recall": 1.0},
            ],
        )
        assert r.precision == 1.0
        assert r.pd_recall == 1.0
        assert r.far_per_km2 == 0.0
        assert r.map_at_iou == pytest.approx(1.0, abs=1e-6)


class TestDryRunCLI:
    """CLI dry-run validates a manifest without spinning up any model."""

    def test_dry_run_returns_zero_on_valid_manifest(
        self, tmp_path: Path, capsys
    ):
        manifest = tmp_path / "m.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "image_path": "scene_001.tif",
                        "scene_area_km2": 256.0,
                        "ground_truth": [{"bbox": [0, 0, 10, 10]}],
                    }
                ]
            )
        )
        rc = rv.main(
            [
                "--manifest",
                str(manifest),
                "--model",
                "vesseltracker-sar-yolov8",
                "--output",
                str(tmp_path / "out.json"),
                "--dry-run",
            ]
        )
        assert rc == 0

    def test_dry_run_returns_nonzero_on_missing_manifest(self, tmp_path: Path):
        rc = rv.main(
            [
                "--manifest",
                str(tmp_path / "nonexistent.json"),
                "--model",
                "vesseltracker-sar-yolov8",
                "--output",
                str(tmp_path / "out.json"),
                "--dry-run",
            ]
        )
        assert rc == 2
