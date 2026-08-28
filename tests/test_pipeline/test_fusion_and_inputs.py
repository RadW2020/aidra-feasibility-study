"""R14 / R15 / CFAR-params: fusion by centre distance, unfiltered YOLO input,
and CFAR window geometry driven by Settings."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Settings
from src.models.cfar import CFARDetector
from src.pipeline.detection import DetectionEngine, _center_distance, sar_linear_to_uint8_gray


def _engine(**kw) -> DetectionEngine:
    base = dict(
        fusion_iou_threshold=0.3,
        edge_buffer_px=0,
        cfar_min_cluster_size=5,
        cfar_cluster_eps=1.5,
        cfar_min_mean_snr=2.0,
        fusion_yolo_weight=0.5,
        fusion_center_tolerance_px=20.0,
        yolo_input="filtered",
    )
    base.update(kw)
    return DetectionEngine(**base)


# A 4x4 CFAR cluster sitting inside a 40x40 YOLO box: IoU = 16/1600 = 0.01.
CFAR = [{"bbox": [118.0, 118.0, 122.0, 122.0], "mean_snr": 25.0}]
YOLO = [{"bbox": [100.0, 100.0, 140.0, 140.0], "confidence": 0.8}]


class TestFusionMode:
    def test_center_mode_fuses_point_cluster_inside_box(self):
        out = _engine(fusion_mode="center")._fuse_detections(CFAR, YOLO, 0)
        assert [d.source for d in out] == ["fused"]
        assert out[0].bbox_pixel == YOLO[0]["bbox"]  # YOLO box kept
        assert out[0].cfar_snr == 25.0 and out[0].yolo_score == 0.8
        assert 0.8 < out[0].confidence <= 1.0

    def test_iou_mode_reproduces_the_measured_failure(self):
        out = _engine(fusion_mode="iou")._fuse_detections(CFAR, YOLO, 0)
        assert sorted(d.source for d in out) == ["cfar", "yolo"]

    def test_center_mode_respects_tolerance(self):
        far_yolo = [{"bbox": [200.0, 200.0, 240.0, 240.0], "confidence": 0.8}]
        out = _engine(fusion_mode="center", fusion_center_tolerance_px=20.0)._fuse_detections(
            CFAR, far_yolo, 0
        )
        assert sorted(d.source for d in out) == ["cfar", "yolo"]
        assert _center_distance(CFAR[0]["bbox"], far_yolo[0]["bbox"]) > 20.0

    def test_center_mode_picks_nearest_box_once(self):
        two_yolo = [
            {"bbox": [100.0, 100.0, 140.0, 140.0], "confidence": 0.6},  # centre (120,120)
            {"bbox": [112.0, 112.0, 132.0, 132.0], "confidence": 0.9},  # centre (122,122), nearer
        ]
        out = _engine(fusion_mode="center")._fuse_detections(CFAR, two_yolo, 0)
        fused = [d for d in out if d.source == "fused"]
        assert len(fused) == 1 and fused[0].yolo_score == 0.9
        assert sum(d.source == "yolo" for d in out) == 1

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="fusion_mode"):
            _engine(fusion_mode="overlap")

    def test_defaults_come_from_settings(self):
        s = Settings(_env_file=None)
        eng = DetectionEngine(edge_buffer_px=0)
        assert eng.fusion_mode == s.fusion_mode == "center"
        assert eng.fusion_center_tolerance_px == s.fusion_center_tolerance_px
        assert eng.yolo_input == s.yolo_input == "unfiltered"


class _RecordingDetector:
    """YOLO stand-in that records what it was fed and returns nothing."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict(self, image):
        self.inputs.append(image)
        return []

    def get_model_info(self):
        return {"name": "rec", "hash": "0" * 64}


class TestYoloInput:
    def _tiles(self):
        filtered = np.full((64, 64), 0.5, dtype=np.float32)  # -3 dB
        yolo_gray = np.full((64, 64), 200, dtype=np.uint8)
        return [{"data": filtered, "yolo_input": yolo_gray, "tile_index": 0,
                 "row_offset": 0, "col_offset": 0}]

    def test_unfiltered_uses_prepared_gray(self):
        det = _RecordingDetector()
        res = _engine(yolo_input="unfiltered").run(self._tiles(), detector=det, cfar=None)
        assert det.inputs[0].shape == (64, 64, 3) and det.inputs[0].dtype == np.uint8
        assert int(det.inputs[0][0, 0, 0]) == 200
        assert res.notes is None

    def test_filtered_ignores_prepared_gray(self):
        det = _RecordingDetector()
        _engine(yolo_input="filtered").run(self._tiles(), detector=det, cfar=None)
        expected = sar_linear_to_uint8_gray(np.full((64, 64), 0.5, dtype=np.float32))[0, 0]
        assert int(det.inputs[0][0, 0, 0]) == int(expected) != 200

    def test_unfiltered_falls_back_and_records_it(self):
        det = _RecordingDetector()
        tiles = self._tiles()
        del tiles[0]["yolo_input"]
        res = _engine(yolo_input="unfiltered").run(tiles, detector=det, cfar=None)
        assert res.notes == "yolo_input:fallback_filtered=1/1"

    def test_gray_helper_matches_rgb_helper(self):
        from src.pipeline.detection import _sar_linear_to_uint8_rgb

        arr = np.random.default_rng(0).exponential(0.05, (16, 16)).astype(np.float32)
        gray = sar_linear_to_uint8_gray(arr)
        assert np.array_equal(_sar_linear_to_uint8_rgb(arr)[..., 0], gray)


class TestCfarFromSettings:
    def test_from_settings_drives_window_geometry(self):
        s = Settings(_env_file=None, cfar_guard_size=5, cfar_training_size=12, cfar_pfa=1e-4)
        cfar = CFARDetector.from_settings(s)
        assert (cfar.guard_size, cfar.training_size, cfar.pfa) == (5, 12, 1e-4)

    def test_defaults_match_what_production_always_ran(self):
        s = Settings(_env_file=None)
        assert (s.cfar_guard_size, s.cfar_training_size) == (8, 20)
        assert (CFARDetector().guard_size, CFARDetector().training_size) == (8, 20)

    def test_manager_and_harness_share_the_factory(self):
        from pathlib import Path

        for path in ("src/models/manager.py", "src/validation/harness.py"):
            src = Path(path).read_text()
            assert "CFARDetector.from_settings(" in src, path
            assert "CFARDetector()" not in src, f"{path} bypasses Settings (I-DET-4)"
