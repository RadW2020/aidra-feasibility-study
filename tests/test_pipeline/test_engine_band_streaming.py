"""R17: PipelineEngine detects band by band with a shared memory guard and pools results."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from src.config import Settings
from src.pipeline.detection import DetectionEngine
from src.pipeline.engine import PipelineEngine
from src.profiles.memory_guard import MemoryBudgetExceeded, MemoryGuard


class _Det:
    """YOLO stand-in: one box in tile 0 of every band it sees."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return [{"bbox": [10.0, 10.0, 40.0, 40.0], "confidence": 0.9, "class_name": "vessel"}] if self.calls == 1 else []

    def get_model_info(self):
        return {"name": "vesseltracker-sar-yolov8", "hash": "0" * 64}


class _FakeStream:
    """Two bands of 2 tiles each; band 2 duplicates band 1's first tile position (seam)."""

    def __init__(self, n_bands: int = 2, tiles_per_band: int = 2, size: int = 64) -> None:
        self.n_bands, self.tiles_per_band, self.size = n_bands, tiles_per_band, size
        self.metadata = {"num_tiles": n_bands * tiles_per_band, "original_shape": (size * 2, size * 2), "quality": "valid"}
        self.yielded = 0

    def bands(self):
        idx = 0
        for b in range(self.n_bands):
            band = []
            for t in range(self.tiles_per_band):
                band.append({
                    "array": np.full((self.size, self.size), 0.05, dtype=np.float32),
                    "yolo_input": np.full((self.size, self.size), 120, dtype=np.uint8),
                    "tile_index": idx,
                    "row_offset": 0 if b == 0 else self.size - 8,  # 8 px overlap between bands
                    "col_offset": t * self.size,
                    "geo_transform": (-5.8, 1e-4, 0.0, 36.2, 0.0, -1e-4),
                })
                idx += 1
            self.yielded += 1
            yield band


class _NoCfarManager:
    async def get_model(self, name, **kw):
        raise RuntimeError("no cfar in this test")


def _engine(settings: Settings) -> PipelineEngine:
    eng = object.__new__(PipelineEngine)
    eng.config = settings
    eng.detector_engine = DetectionEngine(
        fusion_iou_threshold=0.3, edge_buffer_px=0, cfar_min_cluster_size=5, cfar_cluster_eps=1.5,
        cfar_min_mean_snr=2.0, fusion_yolo_weight=0.5, fusion_mode="center",
        fusion_center_tolerance_px=20.0, yolo_input="unfiltered",
    )
    eng.model_manager = _NoCfarManager()
    from src.observability.loki_logger import StructuredLogger

    eng._log = StructuredLogger("aidra.test.engine")
    return eng


def test_detect_bands_pools_metrics_and_calls_band_hook():
    settings = Settings(_env_file=None)
    eng = _engine(settings)
    stream = _FakeStream()
    seen: list[int] = []
    res = eng._detect_bands(
        stream=stream, detector=_Det(), cfar=None, constraint_profile="ground", sensor="s1",
        scene_shape=(128, 128), on_band_result=lambda r, band: seen.append(len(band)),
    )
    assert stream.yielded == 2 and seen == [2, 2]
    assert res.metrics.num_tiles == 4
    assert len(res.metrics.tile_ms_samples) == 4  # pooled per-tile latencies
    assert res.metrics.tile_ms_p95 >= res.metrics.tile_ms_p50 >= 0.0
    assert len(res.detections) == 1 and res.detections[0].tile_index == 0
    assert "streamed:2 bands" in (res.notes or "")


def test_memory_guard_is_checked_across_bands():
    settings = Settings(_env_file=None)
    eng = _engine(settings)

    class _Coll:
        def __init__(self):
            self.peaks = [100.0, 100.0, 100.0, 5000.0]

        def peak_ram_mb_so_far(self):
            return self.peaks.pop(0) if self.peaks else 5000.0

    guard = MemoryGuard(1024, _Coll(), profile_name="sat-low")
    with pytest.raises(MemoryBudgetExceeded):
        eng._detect_bands(
            stream=_FakeStream(n_bands=3), detector=_Det(), cfar=None, constraint_profile="sat-low",
            sensor="s1", scene_shape=(128, 128), memory_guard=guard,
        )
    assert guard.checks == 4  # tripped on the 4th tile, i.e. inside the second band


def test_run_detection_ground_uses_stream():
    settings = Settings(_env_file=None)
    eng = _engine(settings)
    res = asyncio.run(
        eng._run_detection(tiles=None, profile="ground", detector=_Det(), sensor="s1", stream=_FakeStream())
    )
    assert res.metrics.num_tiles == 4 and len(res.detections) == 1


def test_prune_orphan_thumbnails(tmp_path: Path):
    from src.pipeline.detection import Detection

    eng = _engine(Settings(_env_file=None))
    keep = Detection(bbox_pixel=[0, 0, 1, 1], confidence=0.5, source="cfar")
    (tmp_path / f"{keep.id}.png").write_bytes(b"x")
    (tmp_path / "deadbeef-orphan.png").write_bytes(b"x")
    assert eng._prune_orphan_thumbnails(tmp_path, [keep]) == 1
    assert (tmp_path / f"{keep.id}.png").exists()


def test_settings_default_streams_in_bands():
    assert Settings(_env_file=None).tile_stream_band_rows == 4
