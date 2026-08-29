"""R17: banded PreprocessStream yields exactly the tiles preprocess_full returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from src.pipeline.preprocessing import (  # noqa: E402
    PreprocessStream,
    open_preprocess_stream,
    preprocess_full,
)


@pytest.fixture
def synthetic_product(tmp_path: Path) -> Path:
    """Minimal S1-like product: a georeferenced measurement TIFF, no LUT/GCPs."""
    prod = tmp_path / "S1X_SYNTH.SAFE"
    (prod / "measurement").mkdir(parents=True)
    rng = np.random.default_rng(11)
    dn = (rng.exponential(200.0, (700, 900))).astype(np.uint16)
    dn[300:305, 400:405] = 60000  # a bright target
    transform = rasterio.transform.from_origin(-5.8, 36.2, 0.0001, 0.0001)
    with rasterio.open(prod / "measurement" / "s1-vh.tiff", "w", driver="GTiff", height=700, width=900,
                       count=1, dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
        dst.write(dn, 1)
    return prod


def test_stream_matches_eager_preprocess(synthetic_product: Path):
    eager = preprocess_full(synthetic_product, tile_size=256, overlap=32)
    stream = open_preprocess_stream(synthetic_product, tile_size=256, overlap=32, band_tile_rows=2)
    assert stream.num_tiles == eager["metadata"]["num_tiles"] == len(eager["tiles"])
    assert stream.num_bands == 2  # 3 tile rows (0, 224, 448) -> bands of 2 rows

    streamed = [t for band in stream.bands() for t in band]
    assert len(streamed) == len(eager["tiles"])
    for i, (a, b) in enumerate(zip(eager["tiles"], streamed, strict=True)):
        assert a["tile_index"] == b["tile_index"] == i
        assert (a["row_offset"], a["col_offset"]) == (b["row_offset"], b["col_offset"])
        assert np.array_equal(a["array"], b["array"])
        assert np.array_equal(a["yolo_input"], b["yolo_input"])
        assert a["geo_bounds"] == b["geo_bounds"]
    # metadata known before any tile is read, including the I-SAR-1 gate
    for key in ("quality", "quality_reasons", "geo_transform", "tile_size", "overlap", "num_tiles"):
        assert stream.metadata[key] == eager["metadata"][key]
    assert stream.metadata["quality"] == "invalid"  # no LUT / GCPs in the synthetic product


def test_bands_are_lazy_and_bounded(synthetic_product: Path):
    stream = PreprocessStream(synthetic_product, tile_size=256, overlap=32, band_tile_rows=1)
    gen = stream.bands()
    first = next(gen)
    assert len(first) == len(stream._ctx["col_offsets"])  # one tile row per band
    assert all(t["row_offset"] == 0 for t in first)
    rest = list(gen)
    assert len(rest) == stream.num_bands - 1
