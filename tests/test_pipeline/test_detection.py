"""Tests for detection helpers — SAR-to-uint8 normalization, cross-tile dedup,
and the I-SAR-2 edge swath filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.detection import (
    Detection,
    _apply_edge_swath_filter,
    _dedup_geo_detections,
    _sar_linear_to_uint8_rgb,
)
from src.pipeline.engine import PipelineEngine


class TestSarLinearToUint8Rgb:
    """The helper feeds linear sigma0 to YOLO as uint8 RGB."""

    def test_output_shape_dtype(self):
        tile = np.full((32, 32), 0.1, dtype=np.float32)
        rgb = _sar_linear_to_uint8_rgb(tile)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_dynamic_range_is_used(self):
        """Bright + dark pixels should map to opposite ends of [0,255]."""
        tile = np.full((4, 4), 1e-3, dtype=np.float32)  # ~ -30 dB
        tile[0, 0] = 5.0  # ~ +7 dB → clipped to db_max
        rgb = _sar_linear_to_uint8_rgb(tile, db_min=-25.0, db_max=0.0)
        assert rgb[0, 0, 0] == 255
        # background near floor
        assert rgb[1, 1, 0] == 0

    def test_channels_replicated(self):
        tile = np.linspace(0.01, 1.0, 64, dtype=np.float32).reshape(8, 8)
        rgb = _sar_linear_to_uint8_rgb(tile)
        np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
        np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])


class TestDedupGeoDetections:
    """Cross-tile deduplication keeps the highest-confidence neighbour."""

    def _make(self, lon: float, lat: float, conf: float, source: str = "cfar") -> Detection:
        return Detection(
            bbox_pixel=[0, 0, 10, 10],
            center_geo=[lon, lat],
            confidence=conf,
            source=source,
        )

    def test_keeps_highest_confidence(self):
        a = self._make(-5.0, 36.0, 0.4)
        b = self._make(-5.00005, 36.00005, 0.9)  # ~5 m away, stronger
        c = self._make(-5.5, 36.0, 0.6)  # different vessel
        survivors = _dedup_geo_detections([a, b, c])
        assert len(survivors) == 2
        assert any(s.confidence == 0.9 for s in survivors)
        assert any(s.confidence == 0.6 for s in survivors)
        assert all(s.confidence != 0.4 for s in survivors)

    def test_preserves_detections_without_geo(self):
        a = Detection(bbox_pixel=[0, 0, 10, 10], confidence=0.5, source="cfar")
        b = self._make(-5.0, 36.0, 0.7)
        survivors = _dedup_geo_detections([a, b])
        assert len(survivors) == 2

    def test_empty_input(self):
        assert _dedup_geo_detections([]) == []


class TestDetectionQualityHelpers:
    """Quality labels keep raw detections but identify operational targets."""

    def test_quality_verdict_prioritizes_artifacts(self):
        assert (
            PipelineEngine._detection_quality_verdict(
                "fused", on_land=True, cluster_anomaly=False
            )
            == "land_artifact"
        )
        assert (
            PipelineEngine._detection_quality_verdict(
                "yolo", on_land=False, cluster_anomaly=True
            )
            == "cluster_artifact"
        )

    def test_quality_verdict_separates_targets_from_candidates(self):
        assert (
            PipelineEngine._detection_quality_verdict(
                "fused", on_land=False, cluster_anomaly=False
            )
            == "valid_sea_target"
        )
        assert (
            PipelineEngine._detection_quality_verdict(
                "yolo", on_land=False, cluster_anomaly=False
            )
            == "valid_sea_target"
        )
        assert (
            PipelineEngine._detection_quality_verdict(
                "cfar", on_land=False, cluster_anomaly=False
            )
            == "candidate"
        )

    def test_non_finite_detector_scores_are_dropped_before_db(self):
        assert PipelineEngine._finite_or_none(float("inf")) is None
        assert PipelineEngine._finite_or_none(float("-inf")) is None
        assert PipelineEngine._finite_or_none(float("nan")) is None
        assert PipelineEngine._finite_or_none(12.5) == 12.5


@pytest.mark.invariant
class TestEdgeSwathFilter:
    """I-SAR-2: detections within edge_buffer_px of any scene edge are dropped."""

    def _tile(self, idx: int, row_off: int, col_off: int, size: int = 100):
        return {
            "data": np.zeros((size, size), dtype=np.float32),
            "tile_index": idx,
            "row_offset": row_off,
            "col_offset": col_off,
        }

    def _det(self, tile_index: int, x_min: float, y_min: float, x_max: float, y_max: float, conf: float = 0.7):
        return Detection(
            bbox_pixel=[x_min, y_min, x_max, y_max],
            confidence=conf,
            source="cfar",
            tile_index=tile_index,
        )

    def test_buffer_zero_keeps_all(self):
        tiles = [self._tile(0, 0, 0, size=100)]
        dets = [self._det(0, 5, 5, 15, 15)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=0, scene_shape=(100, 100)
        )
        assert kept == dets
        assert dropped == 0

    def test_drops_detection_near_top_edge(self):
        tiles = [self._tile(0, 0, 0, size=100)]
        # Center at (5, 10) — top edge.
        dets = [self._det(0, 0, 0, 10, 20)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=32, scene_shape=(100, 100)
        )
        assert kept == []
        assert dropped == 1

    def test_drops_detection_near_right_edge(self):
        tiles = [self._tile(0, 0, 0, size=100)]
        # Center at (95, 50) — within 32 px of right edge (col=100).
        dets = [self._det(0, 90, 45, 100, 55)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=32, scene_shape=(100, 100)
        )
        assert kept == []
        assert dropped == 1

    def test_keeps_detection_in_safe_interior(self):
        tiles = [self._tile(0, 0, 0, size=100)]
        # Center at (50, 50) — well clear of any edge.
        dets = [self._det(0, 45, 45, 55, 55)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=32, scene_shape=(100, 100)
        )
        assert len(kept) == 1
        assert dropped == 0

    def test_per_tile_offset_recovers_scene_coords(self):
        # Tile #1 lives at (row=100, col=0) inside a 200x100 scene.
        # A detection at local (x=50, y=5) sits 105 px below scene top,
        # which is far from any edge — it must be kept.
        tiles = [
            self._tile(0, 0, 0, size=100),
            self._tile(1, 100, 0, size=100),
        ]
        dets = [self._det(1, 45, 0, 55, 10)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=32, scene_shape=(200, 100)
        )
        assert len(kept) == 1
        assert dropped == 0

    def test_scene_shape_inferred_from_tiles(self):
        tiles = [self._tile(0, 0, 0, size=100)]
        # Detection close to right edge under inferred 100×100.
        dets = [self._det(0, 90, 45, 100, 55)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=32, scene_shape=None
        )
        assert kept == []
        assert dropped == 1

    def test_unknown_tile_index_passthrough(self):
        # A detection whose tile_index is not in the tile list keeps:
        # we cannot compute scene coordinates, so we don't drop blindly.
        tiles = [self._tile(0, 0, 0, size=100)]
        dets = [self._det(99, 0, 0, 5, 5)]
        kept, dropped = _apply_edge_swath_filter(
            dets, tiles, edge_buffer_px=32, scene_shape=(100, 100)
        )
        assert len(kept) == 1
        assert dropped == 0


@pytest.mark.invariant
class TestEdgeFilterEndToEnd:
    """End-to-end: DetectionEngine.run drops near-edge detections when
    edge_buffer_px > 0."""

    def test_engine_run_applies_edge_filter(self):
        from src.pipeline.detection import DetectionEngine

        class _FakeDetector:
            def predict(self, _img):
                # Two YOLO detections: one in a safe interior, one on
                # the top-left edge.
                return [
                    {"bbox": [50, 50, 60, 60], "confidence": 0.9, "class_name": "vessel"},
                    {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_name": "vessel"},
                ]

            def get_model_info(self):
                return {"name": "fake-yolo", "version": "test"}

        engine = DetectionEngine(edge_buffer_px=32)
        tiles = [
            {
                "data": np.zeros((100, 100), dtype=np.float32),
                "tile_index": 0,
                "row_offset": 0,
                "col_offset": 0,
            }
        ]
        result = engine.run(
            tiles=tiles,
            detector=_FakeDetector(),
            cfar=None,
            constraint_profile="ground",
            scene_shape=(100, 100),
        )
        # Only the interior detection survives the I-SAR-2 filter.
        assert len(result.detections) == 1
        kept_bbox = result.detections[0].bbox_pixel
        assert kept_bbox == [50, 50, 60, 60]


# ====================================================================
# Sea-mask helper tests (CFAR land-bias engineering fix)
# ====================================================================


class TestSeaMaskHelper:
    """Tests for ``_build_sea_mask``: per-tile sea-only mask construction."""

    def test_sea_mask_returns_none_without_geocoding(self):
        """Tiles without lat/lon bounds must return None so CFAR runs
        unmasked rather than crashing."""
        from src.pipeline.detection import _build_sea_mask

        empty_bounds = {
            "lon_min": None,
            "lon_max": None,
            "lat_min": None,
            "lat_max": None,
        }
        assert _build_sea_mask(empty_bounds, (640, 640)) is None

    def test_sea_mask_open_ocean_is_all_sea(self):
        """A tile far from any coastline must yield an all-True mask
        (every pixel is over sea, CFAR may run on the whole tile)."""

        from src.pipeline.detection import _build_sea_mask, _get_globe

        if _get_globe() is None:
            import pytest as _p
            _p.skip("global-land-mask not installed in this environment")

        # Middle of the North Atlantic: ~40°N, -40°W, comfortably away
        # from the British Isles, the Azores and the US East Coast.
        bounds = {
            "lon_min": -40.05,
            "lon_max": -40.00,
            "lat_min": 40.00,
            "lat_max": 40.05,
        }
        mask = _build_sea_mask(bounds, (64, 64))
        assert mask is not None
        assert mask.shape == (64, 64)
        assert mask.all(), "mid-ocean tile must be all sea"

    def test_sea_mask_continental_interior_is_all_land(self):
        """A tile deep inside a continent must yield an all-False mask
        (CFAR would otherwise fire on terrain features)."""

        from src.pipeline.detection import _build_sea_mask, _get_globe

        if _get_globe() is None:
            import pytest as _p
            _p.skip("global-land-mask not installed in this environment")

        # Near Madrid, Spain — well inland.
        bounds = {
            "lon_min": -3.75,
            "lon_max": -3.70,
            "lat_min": 40.40,
            "lat_max": 40.45,
        }
        mask = _build_sea_mask(bounds, (64, 64))
        assert mask is not None
        assert mask.shape == (64, 64)
        assert not mask.any(), "continental-interior tile must be all land"

    def test_sea_mask_coastline_is_mixed(self):
        """A tile straddling a coastline must contain both sea and land,
        and the sea fraction must be reasonable (>1% and <99%)."""

        from src.pipeline.detection import _build_sea_mask, _get_globe

        if _get_globe() is None:
            import pytest as _p
            _p.skip("global-land-mask not installed in this environment")

        # Strait of Gibraltar — straddles the Spain/Morocco coastline
        # and the actual strait water.
        bounds = {
            "lon_min": -5.50,
            "lon_max": -5.20,
            "lat_min": 35.85,
            "lat_max": 36.10,
        }
        mask = _build_sea_mask(bounds, (128, 128))
        assert mask is not None
        sea_fraction = mask.mean()
        assert 0.05 < sea_fraction < 0.95, (
            f"Strait of Gibraltar tile should be mixed sea/land, "
            f"got sea_fraction={sea_fraction:.2%}"
        )
