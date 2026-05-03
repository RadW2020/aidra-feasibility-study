"""End-to-end reproducibility test (palanca L15).

Closes the **gate:reproducibility** check declared in CLAUDE.md §6:
*"mismo input → mismo output_hash"* across the full deterministic
pipeline path (preprocess → detect → fuse → hash).

Why no PostGIS / no full PipelineEngine?
----------------------------------------
The persistence layer is async + requires a running PostGIS, which
makes the test brittle and CPU-expensive in CI. The deterministic
chain that actually matters for evidence reproducibility is:

    raw sigma0  →  Lee filter  →  CFAR with clustering  →
    DetectionEngine fusion  →  cross-tile dedup  →  result_hash

These are all synchronous, dependency-free, fed from
:func:`generate_synthetic_sar_tile` with a fixed seed. We run it
twice from scratch and assert byte-identical output_hash.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.cfar import CFARDetector
from src.pipeline.detection import (
    Detection,
    DetectionEngine,
    DetectionResult,
)
from src.pipeline.preprocessing import (
    apply_lee_filter,
    generate_synthetic_sar_tile,
)
from src.traceability.hasher import (
    compute_input_params_hash,
    compute_result_hash,
)


def _detections_to_hashable(detections: list[Detection]) -> list[dict]:
    """Project a list of :class:`Detection` to the dict shape that
    :func:`compute_result_hash` expects (lon/lat/confidence/source +
    bbox)."""
    out: list[dict] = []
    for det in detections:
        center = det.center_geo or [0.0, 0.0]
        out.append({
            "longitude": float(center[0]),
            "latitude": float(center[1]),
            "confidence": round(float(det.confidence), 6),
            "source": det.source,
            "bbox": [float(x) for x in det.bbox_pixel],
            "tile_index": int(det.tile_index),
        })
    return out


class _FakeYOLO:
    """Deterministic stub of a YOLO detector. Uses the input mean as
    the entropy source so seed-controlled tiles produce repeatable
    detections without loading any real weight."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[dict]:
        self.calls += 1
        # Map deterministic statistics to two synthetic detections.
        h, w = image.shape[:2]
        v_mean = float(np.mean(image))
        cx1 = int(min(w - 5, max(5, h // 4)))
        cy1 = int(min(h - 5, max(5, w // 4)))
        return [
            {
                "bbox": [cx1 - 4, cy1 - 4, cx1 + 4, cy1 + 4],
                "confidence": round(0.4 + (v_mean % 0.3), 4),
                "class_name": "vessel",
            },
        ]

    def get_model_info(self) -> dict:
        return {"name": "fake-yolo-deterministic", "version": "test"}


def _build_pipeline_input(seed: int = 42) -> dict:
    """Single point of truth for the synthetic pipeline input."""
    return {
        "tile_size": 256,
        "num_vessels": 4,
        "seed": seed,
        "lee_window": 7,
        "edge_buffer_px": 16,
        "fusion_iou_threshold": 0.3,
    }


def _run_synthetic_pipeline(params: dict) -> tuple[DetectionResult, str]:
    """Execute the deterministic pipeline path end-to-end.

    Returns the :class:`DetectionResult` and the canonical
    ``output_hash`` derived from
    :func:`compute_result_hash`.
    """
    tile, _ground_truth = generate_synthetic_sar_tile(
        size=params["tile_size"],
        num_vessels=params["num_vessels"],
        seed=params["seed"],
    )
    # Lee filter to mimic preprocessing speckle suppression.
    filtered = apply_lee_filter(tile, window_size=params["lee_window"])

    cfar = CFARDetector(guard_size=3, training_size=15, pfa=1e-4)
    fake_yolo = _FakeYOLO()

    engine = DetectionEngine(
        fusion_iou_threshold=params["fusion_iou_threshold"],
        edge_buffer_px=params["edge_buffer_px"],
    )
    tiles = [
        {
            "data": filtered,
            "tile_index": 0,
            "row_offset": 0,
            "col_offset": 0,
        }
    ]
    result = engine.run(
        tiles=tiles,
        detector=fake_yolo,
        cfar=cfar,
        constraint_profile="ground",
        scene_shape=(params["tile_size"], params["tile_size"]),
    )
    output_hash = compute_result_hash(_detections_to_hashable(result.detections))
    return result, output_hash


@pytest.mark.invariant
class TestPipelineE2EReproducibility:
    """gate:reproducibility — full preprocess→detect→hash chain."""

    def test_same_input_yields_same_output_hash(self):
        params = _build_pipeline_input(seed=42)
        _, h1 = _run_synthetic_pipeline(params)
        _, h2 = _run_synthetic_pipeline(params)
        assert h1 == h2, (
            "Pipeline reproducibility broken: identical inputs and seed "
            "produced different output_hash values"
        )
        assert len(h1) == 64

    def test_different_seed_changes_output_hash(self):
        a = _build_pipeline_input(seed=42)
        b = _build_pipeline_input(seed=7)
        _, ha = _run_synthetic_pipeline(a)
        _, hb = _run_synthetic_pipeline(b)
        assert ha != hb, (
            "Different seeds must produce different output_hash; "
            "otherwise the seed has no observable effect on detections"
        )

    def test_input_params_hash_drives_change_signal(self):
        """If the input_params_hash changes, the output_hash must
        also change (a contract every D3 audit relies on)."""
        a = _build_pipeline_input(seed=42)
        b = _build_pipeline_input(seed=42)
        b["edge_buffer_px"] = 64  # tighten the edge filter
        ha = compute_input_params_hash(a)
        hb = compute_input_params_hash(b)
        assert ha != hb
        _, oa = _run_synthetic_pipeline(a)
        _, ob = _run_synthetic_pipeline(b)
        # The edge buffer is large enough to drop CFAR detections in
        # the corner; the output_hash must reflect that.
        assert oa != ob

    def test_detection_count_is_stable(self):
        """A weaker but cheaper invariant for CI dashboards."""
        params = _build_pipeline_input(seed=42)
        result_a, _ = _run_synthetic_pipeline(params)
        result_b, _ = _run_synthetic_pipeline(params)
        assert len(result_a.detections) == len(result_b.detections)
