"""
Reproducibility tests for traceability — covers gate:reproducibility
declared in CLAUDE.md section 6.

Invariant under audit: same input -> same output_hash. Without this
guarantee, the D3 evidence bundle cannot be re-verified by a third
party (SatCen) and the I-TRACE-1/I-TRACE-4 invariants collapse.

These tests run entirely on synthetic data so they are CI-friendly:
no DB, no model weights, no Sentinel-1 download.
"""

from __future__ import annotations

import pytest

from src.models.cfar import CFARDetector
from src.pipeline.preprocessing import generate_synthetic_sar_tile
from src.traceability.hasher import (
    compute_input_params_hash,
    compute_result_hash,
)

# ====================================================================
# Synthetic detection fixture (5 detections, deterministic content)
# ====================================================================


@pytest.fixture
def synthetic_detections() -> list[dict]:
    """Five synthetic vessel detections with stable lon/lat/confidence."""
    return [
        {"longitude": -5.50, "latitude": 36.00, "confidence": 0.81, "source": "cfar"},
        {"longitude": -5.30, "latitude": 36.20, "confidence": 0.62, "source": "yolo"},
        {"longitude": -5.10, "latitude": 35.80, "confidence": 0.91, "source": "fused"},
        {"longitude": -4.90, "latitude": 36.05, "confidence": 0.55, "source": "cfar"},
        {"longitude": -4.70, "latitude": 35.95, "confidence": 0.73, "source": "yolo"},
    ]


# ====================================================================
# Test 1 — same input -> same hash (deterministic)
# ====================================================================


def test_compute_result_hash_deterministic_same_input(
    synthetic_detections: list[dict],
) -> None:
    """Two consecutive calls on the same dict list must produce identical
    hashes. Re-runs implicitly under seed 42 (fixture is deterministic).
    """
    h1 = compute_result_hash(synthetic_detections)
    h2 = compute_result_hash(synthetic_detections)

    assert h1 == h2
    assert len(h1) == 64


# ====================================================================
# Test 2 — change one detection -> hash changes
# ====================================================================


def test_result_hash_changes_with_input(
    synthetic_detections: list[dict],
) -> None:
    """Mutating a single field of one detection must produce a different
    hash. This is the negative side of gate:reproducibility — without it
    the hash would be useless as tamper evidence.
    """
    baseline = compute_result_hash(synthetic_detections)

    mutated = [dict(d) for d in synthetic_detections]
    mutated[0]["confidence"] = 0.82  # was 0.81

    assert compute_result_hash(mutated) != baseline


# ====================================================================
# Test 3 — input_params_hash is order-stable but value-sensitive
# (reinforces I-TRACE-4)
# ====================================================================


def test_input_params_hash_stable_across_orderings() -> None:
    """Reordering keys must not change the hash; mutating a value must.

    Closes I-TRACE-4: the params hash must be a true content fingerprint,
    independent of how Settings serializes its dict.
    """
    params_a = {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "tile_size": 640,
        "model": "yolov8n-sar",
    }
    params_b = {
        "model": "yolov8n-sar",
        "tile_size": 640,
        "iou_threshold": 0.45,
        "confidence_threshold": 0.25,
    }
    params_c = {
        "confidence_threshold": 0.30,  # changed
        "iou_threshold": 0.45,
        "tile_size": 640,
        "model": "yolov8n-sar",
    }

    h_a = compute_input_params_hash(params_a)
    h_b = compute_input_params_hash(params_b)
    h_c = compute_input_params_hash(params_c)

    assert h_a == h_b, "key order should not affect the hash"
    assert h_a != h_c, "value change must produce a different hash"


# ====================================================================
# Test 4 — mini-pipeline reproducibility (CFAR + result hash)
# ====================================================================


@pytest.mark.invariant
def test_pipeline_synthetic_repro() -> None:
    """End-to-end mini-pipeline reproducibility under seed 42.

    Two independent runs of (synthetic SAR tile -> CFAR detect ->
    compute_result_hash) must yield the same output hash. This is the
    integration test for gate:reproducibility — if it fails, the D3
    bundle cannot be reproduced and the project loses Q3 evidence.
    """
    def run_once() -> str:
        image, _ = generate_synthetic_sar_tile(
            size=256, num_vessels=3, seed=42
        )
        # Adjust window sizes to fit a 256-px tile (defaults assume 640+).
        detector = CFARDetector(
            guard_size=3,
            training_size=15,
            pfa=1e-4,
            method="ca",
        )
        raw = detector.detect(image)
        # compute_result_hash sorts by lon/lat/confidence, so we map
        # CFAR pixel coords to those keys to give the hash a stable
        # ordering criterion that does not depend on detector
        # iteration order.
        detections = [
            {
                "longitude": float(d["x"]),
                "latitude": float(d["y"]),
                "confidence": float(d["snr"]),
                "method": d["method"],
            }
            for d in raw
        ]
        return compute_result_hash(detections)

    h1 = run_once()
    h2 = run_once()

    assert h1 == h2, "mini-pipeline must be reproducible under fixed seed"
    assert len(h1) == 64


# ====================================================================
# Test 5 — seed isolation sanity check
# ====================================================================


def test_seed_isolation() -> None:
    """Different seeds must produce different synthetic tiles, and
    therefore different result hashes through CFAR.

    Without this guarantee, ``seed=42`` would not really be controlling
    the experiment, defeating the whole reproducibility regime.
    """
    def run_with_seed(seed: int) -> str:
        image, _ = generate_synthetic_sar_tile(
            size=256, num_vessels=3, seed=seed
        )
        detector = CFARDetector(
            guard_size=3,
            training_size=15,
            pfa=1e-4,
            method="ca",
        )
        raw = detector.detect(image)
        detections = [
            {
                "longitude": float(d["x"]),
                "latitude": float(d["y"]),
                "confidence": float(d["snr"]),
            }
            for d in raw
        ]
        return compute_result_hash(detections)

    h_42 = run_with_seed(42)
    h_7 = run_with_seed(7)

    assert h_42 != h_7, (
        "different seeds must yield different result hashes — "
        "otherwise the seed is not really controlling the experiment"
    )
