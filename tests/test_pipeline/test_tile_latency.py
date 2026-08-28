"""I-MOD-2: per-tile latency percentiles measured by DetectionEngine."""

from __future__ import annotations

import time

import numpy as np

from src.pipeline.detection import DetectionEngine


class _SlowDet:
    """Sleeps a scripted amount per tile so percentiles are predictable."""

    def __init__(self, delays_ms):
        self._delays = list(delays_ms)

    def predict(self, image):
        time.sleep(self._delays.pop(0) / 1000.0)
        return []

    def get_model_info(self):
        return {"name": "stub", "hash": "0" * 64}


def _engine():
    return DetectionEngine(
        fusion_iou_threshold=0.3, edge_buffer_px=0, cfar_min_cluster_size=5,
        cfar_cluster_eps=1.5, cfar_min_mean_snr=2.0, fusion_yolo_weight=0.5,
        fusion_mode="center", fusion_center_tolerance_px=20.0, yolo_input="filtered",
    )


def test_percentiles_reflect_per_tile_timing():
    tiles = [
        {"data": np.full((16, 16), 0.5, dtype=np.float32), "tile_index": i,
         "row_offset": 0, "col_offset": 0}
        for i in range(5)
    ]
    res = _engine().run(tiles, detector=_SlowDet([5, 5, 5, 5, 60]), cfar=None)
    m = res.metrics
    assert m.num_tiles == 5
    assert 4.0 <= m.yolo_tile_ms_p50 < 30.0
    assert m.yolo_tile_ms_p95 > m.yolo_tile_ms_p50 * 2  # the 60 ms outlier shows in the tail
    # No CFAR -> combined per-tile equals YOLO per-tile.
    assert m.tile_ms_p50 == m.yolo_tile_ms_p50 and m.tile_ms_p95 == m.yolo_tile_ms_p95
    assert m.cfar_tile_ms_p50 == 0.0 and m.cfar_tile_ms_p95 == 0.0
    assert m.total_inference_ms >= sum([5, 5, 5, 5, 60])


def test_empty_run_has_zero_percentiles():
    res = _engine().run([], detector=_SlowDet([]), cfar=None)
    assert res.metrics.tile_ms_p50 == 0.0 and res.metrics.tile_ms_p95 == 0.0


def test_recorder_persists_percentiles():
    from pathlib import Path

    src = Path("src/traceability/recorder.py").read_text()
    assert "inference_p50_ms = COALESCE($20, inference_p50_ms)" in src
    assert "inference_p95_ms = COALESCE($21, inference_p95_ms)" in src
    engine_src = Path("src/pipeline/engine.py").read_text()
    assert engine_src.count("inference_p95_ms=detection_result.metrics.tile_ms_p95") == 2
    mig = Path("src/db/migrations/019_latency_percentiles_candidate_status_legacy_marks.sql").read_text()
    assert "inference_p50_ms REAL" in mig and "inference_p95_ms REAL" in mig
