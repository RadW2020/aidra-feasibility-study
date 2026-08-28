"""Tests for ``src.validation.harness`` (full-pipeline path) and the AP fix.

The full-pipeline path is exercised end-to-end on a small synthetic UTM
raster placed in open Adriatic water so the sea mask is all-sea and the
land flag is deterministically False. No model weight is loaded: YOLO is
a stub returning nothing, CFAR is the real detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.validation.harness import (
    PREDICTION_SETS,
    db_to_linear,
    lonlat_affine_from_raster,
    points_on_land,
    run_full_pipeline,
    settings_fingerprint,
    tile_indices,
)
from src.validation.metrics import ValidationReport, average_precision

rasterio = pytest.importorskip("rasterio")

# Open water in the central Adriatic (~42.5N, 15.5E, >60 km from any
# coast), UTM zone 33N. NB: 16.8E/43.0N is Korcula island — the land mask
# correctly suppressed a CFAR target there while this test was written.
_ADRIATIC_UTM = (541_000.0, 4_706_000.0)


class _NoYolo:
    """Stand-in detector: never fires, so every output comes from CFAR."""

    confidence_threshold = 0.25
    iou_threshold = 0.45

    def predict(self, image):  # noqa: D401 - interface method
        return []

    def get_model_info(self):
        return {"name": "stub", "hash": "0" * 64}


def _write_synthetic_scene(path: Path, size: int = 520, seed: int = 7) -> list[list[float]]:
    """Speckled sea in dB with a few bright point targets. Returns GT bboxes."""
    rng = np.random.default_rng(seed)
    sea_linear = rng.exponential(scale=0.02, size=(size, size)).astype(np.float32)
    targets = [(120, 140), (300, 260), (410, 90)]
    for r, c in targets:
        sea_linear[r - 2 : r + 3, c - 2 : c + 3] = 5.0  # ~24 dB above clutter
    db = 10.0 * np.log10(np.clip(sea_linear, 1e-6, None))
    transform = rasterio.transform.from_origin(_ADRIATIC_UTM[0], _ADRIATIC_UTM[1], 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=transform,
        nodata=-32768.0,
    ) as dst:
        dst.write(db.astype(np.float32), 1)
    return [[c - 3, r - 3, c + 3, r + 3] for r, c in targets]


class TestHelpers:
    def test_tile_indices_cover_whole_raster(self):
        wins = tile_indices(1000, 700, 256, 32)
        rows_max = max(r + h for r, _, h, _ in wins)
        cols_max = max(c + w for _, c, _, w in wins)
        assert rows_max == 1000 and cols_max == 700

    def test_db_to_linear_masks_nodata(self):
        arr = np.array([[-32768.0, 0.0, 10.0]], dtype=np.float32)
        lin = db_to_linear(arr)
        assert lin[0, 0] == 0.0
        assert lin[0, 1] == pytest.approx(1.0)
        assert lin[0, 2] == pytest.approx(10.0)

    def test_settings_fingerprint_excludes_secrets(self):
        from src.config import Settings

        a = Settings(_env_file=None, copernicus_password="x", database_url="postgresql://a")
        b = Settings(_env_file=None, copernicus_password="y", database_url="postgresql://b")
        c = Settings(_env_file=None, confidence_threshold=0.9)
        assert settings_fingerprint(a) == settings_fingerprint(b)
        assert settings_fingerprint(a) != settings_fingerprint(c)


class TestAffineAndLandMask:
    def test_affine_matches_pyproj_at_corners(self, tmp_path: Path):
        from pyproj import Transformer

        _write_synthetic_scene(tmp_path / "s_dB.tif", size=200)
        with rasterio.open(tmp_path / "s_dB.tif") as src:
            aff = lonlat_affine_from_raster(src)
            tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            for col, row in ((0, 0), (200, 0), (0, 200), (200, 200), (77, 133)):
                x, y = src.transform * (col, row)
                lon, lat = tr.transform(x, y)
                o_x, p_w, r_lon, o_y, r_lat, p_h = aff
                assert o_x + col * p_w + row * r_lon == pytest.approx(lon, abs=2e-5)
                assert o_y + col * r_lat + row * p_h == pytest.approx(lat, abs=2e-5)

    def test_points_on_land_open_sea_is_false(self, tmp_path: Path):
        pytest.importorskip("global_land_mask")
        _write_synthetic_scene(tmp_path / "s_dB.tif", size=200)
        with rasterio.open(tmp_path / "s_dB.tif") as src:
            aff = lonlat_affine_from_raster(src)
        flags = points_on_land(aff, np.array([[10.0, 10.0], [150.0, 190.0]]))
        assert flags.dtype == bool and not flags.any()
        assert points_on_land(aff, np.zeros((0, 2))).shape == (0,)


class TestFullPipelinePath:
    def test_four_sets_scene_coords_and_on_land(self, tmp_path: Path):
        from src.models.cfar import CFARDetector
        from src.pipeline.detection import DetectionEngine

        raster = tmp_path / "scene_VH_dB.tif"
        gt = _write_synthetic_scene(raster, size=520)
        engine = DetectionEngine(
            fusion_iou_threshold=0.3,
            edge_buffer_px=16,
            cfar_min_cluster_size=5,
            cfar_cluster_eps=1.5,
            cfar_min_mean_snr=2.0,
            fusion_yolo_weight=0.5,
        )
        scene = run_full_pipeline(
            raster,
            engine=engine,
            yolo=_NoYolo(),
            cfar=CFARDetector(),
            confidence_threshold=0.25,
            tile_size=256,
            tile_overlap=32,
            band_tile_rows=1,
        )

        assert set(scene.sets) == set(PREDICTION_SETS)
        assert scene.scene_shape == (520, 520)
        assert scene.stats["num_tiles"] == len(tile_indices(520, 520, 256, 32))
        assert scene.stats["num_bands"] > 1  # banding actually exercised
        assert scene.sets["yolo"] == [] and scene.sets["fused_only"] == []
        assert len(scene.sets["cfar"]) >= 3
        assert len(scene.sets["aidra"]) >= 3

        # Scene (not tile-local) pixel coordinates and the I-DET-2 flag.
        for pred in scene.sets["aidra"] + scene.sets["cfar"]:
            x0, y0, x1, y1 = pred["bbox"]
            assert 0 <= x0 <= x1 <= 520 and 0 <= y0 <= y1 <= 520
            assert pred["on_land"] is False
            assert pred["source"] in {"cfar", "yolo", "fused"}
            assert 0.0 <= pred["confidence"] <= 1.0

        # Every planted target is recovered within the xView3 20 px tolerance.
        from src.validation.metrics import match_predictions

        tp, _fp, fn, _ = match_predictions(
            scene.sets["aidra"], [{"bbox": b} for b in gt], 0.5,
            match_mode="center", center_tolerance_px=20.0,
        )
        assert tp == 3 and fn == 0
        assert scene.timings_s["total"] >= scene.timings_s["engine"] > 0


class TestFootprintClipping:
    def test_streaks_on_nodata_boundary_are_dropped(self, tmp_path: Path):
        """A target hugging the nodata edge is clipped (I-SAR-2/3); a far one survives."""
        from src.models.cfar import CFARDetector
        from src.pipeline.detection import DetectionEngine
        from src.validation.harness import _swath_edge_flags, valid_data_distance_map

        raster = tmp_path / "scene_VH_dB.tif"
        _write_synthetic_scene(raster, size=520)
        # Carve a nodata triangle in the top-left corner (rotated swath edge).
        with rasterio.open(raster, "r+") as dst:
            arr = dst.read(1)
            rr, cc = np.indices(arr.shape)
            arr[rr + cc < 300] = -32768.0
            # Bright target 12 px outside the nodata edge, inside the 32 px buffer.
            arr[158:163, 148:153] = 7.0
            dst.write(arr, 1)
        with rasterio.open(raster) as src:
            dist, ds = valid_data_distance_map(src, raster_in_db=True)
        assert dist.shape == (520 // ds, 520 // ds)
        # (120,140): row+col=260 < 300 -> inside nodata; (410,90) far from it.
        flags = _swath_edge_flags(dist, ds, np.array([[140.0, 120.0], [90.0, 410.0]]), 32)
        assert flags.tolist() == [True, False]

        engine = DetectionEngine(
            fusion_iou_threshold=0.3, edge_buffer_px=32, cfar_min_cluster_size=5,
            cfar_cluster_eps=1.5, cfar_min_mean_snr=2.0, fusion_yolo_weight=0.5,
        )
        scene = run_full_pipeline(
            raster, engine=engine, yolo=_NoYolo(), cfar=CFARDetector(),
            confidence_threshold=0.25, tile_size=256, tile_overlap=32, band_tile_rows=1,
        )
        assert scene.stats["swath_edge_buffer_px"] == 32
        assert scene.stats["swath_edge_dropped"]["cfar"] >= 1  # the planted edge target
        for p in scene.sets["aidra"]:
            cx, cy = (p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2
            assert cx + cy >= 300 + 32 - ds  # nothing survives inside the buffer
        # The two open-sea targets are still recovered.
        from src.validation.metrics import match_predictions

        tp, _, _, _ = match_predictions(
            scene.sets["aidra"], [{"bbox": [257, 297, 263, 303]}, {"bbox": [87, 407, 93, 413]}],
            0.5, match_mode="center", center_tolerance_px=20.0,
        )
        assert tp == 2


class TestAveragePrecisionEnvelope:
    def test_perfect_curve_is_one(self):
        curve = [
            {"confidence": 0.9, "precision": 1.0, "recall": 0.5},
            {"confidence": 0.8, "precision": 1.0, "recall": 1.0},
        ]
        assert average_precision(curve) == pytest.approx(1.0)

    def test_monotone_envelope_lifts_local_dips(self):
        # 4 GT. Sweep: TP, FP, TP, TP  -> precisions 1, .5, .667, .75 at
        # recalls .25, .25, .5, .75. Envelope: p(.25)=1 (max over r>=.25 is 1),
        # p(.5)=.75, p(.75)=.75. AP = .25*1 + .25*.75 + .25*.75 = 0.625.
        curve = [
            {"confidence": 0.9, "precision": 1.0, "recall": 0.25},
            {"confidence": 0.8, "precision": 0.5, "recall": 0.25},
            {"confidence": 0.7, "precision": 0.6667, "recall": 0.5},
            {"confidence": 0.6, "precision": 0.75, "recall": 0.75},
        ]
        assert average_precision(curve) == pytest.approx(0.625, abs=1e-3)
        # The previous (non-envelope) sum would have given .25*1 + 0*.5 +
        # .25*.6667 + .25*.75 = 0.604; the envelope must never be lower.
        raw = sum(
            (c["recall"] - p["recall"]) * c["precision"]
            for p, c in zip([{"recall": 0.0}] + curve[:-1], curve, strict=True)
        )
        assert average_precision(curve) >= raw

    def test_report_labels_ap_definition(self):
        r = ValidationReport(
            model_name="x", iou_threshold=0.5, confidence_threshold=0.25,
            num_scenes=1, num_ground_truth=1, num_predictions=1,
            true_positives=1, false_positives=0, false_negatives=0,
            total_area_km2=1.0, match_mode="center", center_tolerance_px=20.0,
            params={"pipeline_path": "full"}, provenance={"commit_sha": "abc1234"},
        )
        d = r.as_dict()
        assert d["ap_definition"].startswith("AP@center<=20px")
        assert d["params"]["pipeline_path"] == "full"
        assert d["provenance"]["commit_sha"] == "abc1234"
        assert d["f1"] == 1.0


class TestScriptsUseSharedModule:
    def test_no_duplicate_matcher_in_scripts(self):
        for path in ("scripts/run_validation.py", "scripts/validate_xview3_serial.py"):
            src = Path(path).read_text()
            assert "class ValidationReport" not in src, path
            assert "def _match_predictions" not in src, path
            assert "src.validation" in src, path
