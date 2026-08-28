"""Calibration-set builder of scripts/quantize_static_int8.py (no quantization run)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import quantize_static_int8 as qs  # noqa: E402


def _raster(path: Path, size: int = 400) -> None:
    rng = np.random.default_rng(3)
    db = 10 * np.log10(np.clip(rng.exponential(0.02, (size, size)), 1e-6, None)).astype(np.float32)
    db[:, : size // 4] = -32768.0  # nodata band on the left
    transform = rasterio.transform.from_origin(541_000.0, 4_706_000.0, 10.0, 10.0)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float32",
                       crs="EPSG:32633", transform=transform, nodata=-32768.0) as dst:
        dst.write(db, 1)


def test_builder_is_deterministic_uint8_rgb_and_skips_nodata(tmp_path: Path):
    _raster(tmp_path / "s_VH_dB.tif")
    centres = [(200, 300), (250, 320), (50, 20)]  # last one sits in the nodata band
    tiles, meta = qs.build_calibration_tiles(
        tmp_path / "s_VH_dB.tif", centres, n_vessel=3, n_sea=2, tile=64, seed=42
    )
    assert all(t.dtype == np.uint8 and t.shape == (64, 64, 3) for t in tiles)
    kinds = [m["kind"] for m in meta]
    assert kinds.count("vessel") == 2  # the nodata-centred tile was rejected
    assert kinds.count("sea") == 2
    sha1 = qs.calibration_sha256(tiles)
    tiles2, _ = qs.build_calibration_tiles(
        tmp_path / "s_VH_dB.tif", centres, n_vessel=3, n_sea=2, tile=64, seed=42
    )
    assert qs.calibration_sha256(tiles2) == sha1
    tiles3, _ = qs.build_calibration_tiles(
        tmp_path / "s_VH_dB.tif", centres, n_vessel=3, n_sea=2, tile=64, seed=43
    )
    assert qs.calibration_sha256(tiles3) != sha1


def test_vessel_centres_filter(tmp_path: Path):
    csv_path = tmp_path / "validation.csv"
    csv_path.write_text(
        "scene_id,is_vessel,confidence,detect_scene_row,detect_scene_column\n"
        "s1,True,HIGH,10,20\n"
        "s1,True,LOW,11,21\n"
        "s1,False,HIGH,12,22\n"
        "s2,True,MEDIUM,13,23\n"
        "s1,True,MEDIUM,14.0,24.0\n"
    )
    assert qs._load_vessel_centres(csv_path, "s1") == [(10, 20), (14, 24)]
