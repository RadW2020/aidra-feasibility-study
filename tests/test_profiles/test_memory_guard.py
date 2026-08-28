"""Memory budget enforcement for constraint profiles (Settings.profile_memory_enforcement)."""

from __future__ import annotations

import numpy as np
import pytest

from src.profiles.memory_guard import MemoryBudgetExceeded, MemoryGuard
from src.profiles.metrics_collector import ResourceCollector


class _FakeCollector:
    def __init__(self, peaks):
        self._peaks = list(peaks)

    def peak_ram_mb_so_far(self):
        return self._peaks.pop(0) if self._peaks else 0.0


def test_guard_passes_under_budget_and_trips_above():
    guard = MemoryGuard(1024, _FakeCollector([500.0, 900.0, 1100.0]), profile_name="sat-low")
    guard.check()
    guard.check()
    with pytest.raises(MemoryBudgetExceeded, match="1100 MB > 1024 MB budget of profile 'sat-low'"):
        guard.check()
    assert guard.checks == 3 and guard.tripped_at_mb == 1100.0
    assert isinstance(MemoryBudgetExceeded("x"), MemoryError)  # engine maps it to OOM


def test_collector_exposes_running_peak():
    c = ResourceCollector(sample_interval_ms=10)
    assert c.peak_ram_mb_so_far() == 0.0
    c._ram_samples.extend([100.0, 250.5, 120.0])
    assert c.peak_ram_mb_so_far() == 250.5


def test_detection_engine_checks_guard_per_tile():
    from src.pipeline.detection import DetectionEngine

    class _Det:
        def predict(self, image):
            return []

        def get_model_info(self):
            return {"name": "stub", "hash": "0" * 64}

    tiles = [
        {"data": np.full((32, 32), 0.5, dtype=np.float32), "tile_index": i,
         "row_offset": 0, "col_offset": 0}
        for i in range(3)
    ]
    engine = DetectionEngine(
        fusion_iou_threshold=0.3, edge_buffer_px=0, cfar_min_cluster_size=5,
        cfar_cluster_eps=1.5, cfar_min_mean_snr=2.0, fusion_yolo_weight=0.5,
        fusion_mode="center", fusion_center_tolerance_px=20.0, yolo_input="filtered",
    )
    # Budget breached after the second tile.
    guard = MemoryGuard(512, _FakeCollector([100.0, 600.0, 700.0]), profile_name="sat-extreme")
    with pytest.raises(MemoryError):
        engine.run(tiles, detector=_Det(), cfar=None, memory_guard=guard)
    assert guard.checks == 2


def test_settings_default_is_abort():
    from src.config import Settings

    assert Settings(_env_file=None).profile_memory_enforcement == "abort"
