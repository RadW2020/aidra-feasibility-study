"""
Tests for the SAR preprocessing module.

Covers:
- Synthetic SAR tile generation (shape, vessel count)
- Lee speckle filter noise reduction
- Tiling: shape, overlap coverage, tile count
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.pipeline.preprocessing import (
    _build_pixel_to_geo_transform,
    _calibrate_tile_linear,
    _parse_calibration_lut,
    affine_geo_to_pixel,
    affine_pixel_to_geo,
    apply_lee_filter,
    create_tiles,
    generate_synthetic_sar_tile,
)

# ====================================================================
# SAR calibration LUT
# ====================================================================


class TestCalibrationLUT:
    """Tests for Sentinel-1 line/column calibration interpolation."""

    def test_parse_lut_interpolates_by_azimuth_line(self, tmp_path):
        xml = tmp_path / "calibration.xml"
        xml.write_text(
            """
<root>
  <calibrationVector>
    <line>0</line>
    <pixel>0 3</pixel>
    <sigmaNought>1 1</sigmaNought>
  </calibrationVector>
  <calibrationVector>
    <line>9</line>
    <pixel>0 3</pixel>
    <sigmaNought>2 2</sigmaNought>
  </calibrationVector>
</root>
""".strip()
        )

        lut = _parse_calibration_lut(xml, num_rows=10, num_cols=4)

        assert lut is not None
        window = lut.window(0, 10, 0, 4)
        assert window.shape == (10, 4)
        np.testing.assert_allclose(window[0], np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(window[-1], np.ones(4, dtype=np.float32) * 2)
        assert 1.0 < float(window[5, 0]) < 2.0

    def test_calibration_uses_2d_lut_not_first_row_only(self):
        dn = np.ones((2, 2), dtype=np.float32) * 4.0
        lut = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

        sigma0 = _calibrate_tile_linear(dn, lut)

        np.testing.assert_allclose(sigma0[0], [16.0, 16.0])
        np.testing.assert_allclose(sigma0[1], [4.0, 4.0])


# ====================================================================
# Synthetic tile generation
# ====================================================================


class TestGenerateSyntheticTile:
    """Tests for generate_synthetic_sar_tile."""

    def test_generate_synthetic_tile(self):
        """Output must have the requested shape and the correct number
        of ground-truth vessel entries.
        """
        size = 640
        num_vessels = 5
        image, ground_truth = generate_synthetic_sar_tile(
            size=size, num_vessels=num_vessels, seed=42
        )

        # Shape
        assert image.shape == (size, size)
        assert image.dtype == np.float32

        # Ground truth count
        assert len(ground_truth) == num_vessels

        # Each ground-truth entry must have bbox, center, width, height
        for gt in ground_truth:
            assert "bbox" in gt
            assert "center" in gt
            assert "width" in gt
            assert "height" in gt
            bbox = gt["bbox"]
            assert len(bbox) == 4
            assert bbox[0] < bbox[2]  # x_min < x_max
            assert bbox[1] < bbox[3]  # y_min < y_max

    def test_generate_deterministic(self):
        """Same seed must produce identical images."""
        img1, gt1 = generate_synthetic_sar_tile(size=128, num_vessels=3, seed=7)
        img2, gt2 = generate_synthetic_sar_tile(size=128, num_vessels=3, seed=7)

        np.testing.assert_array_equal(img1, img2)
        assert gt1 == gt2

    def test_vessels_brighter_than_background(self):
        """Vessel pixels must be significantly brighter than the background
        median, otherwise CFAR would never detect them.
        """
        image, ground_truth = generate_synthetic_sar_tile(
            size=256, num_vessels=3, vessel_amplitude=5.0, seed=10
        )
        bg_median = np.median(image)

        for gt in ground_truth:
            cx, cy = gt["center"]
            vessel_val = image[cy, cx]
            assert vessel_val > bg_median * 3, (
                f"Vessel at ({cx},{cy}) value {vessel_val:.2f} is not "
                f"much brighter than background median {bg_median:.2f}"
            )


# ====================================================================
# Lee speckle filter
# ====================================================================


class TestLeeFilter:
    """Tests for the Lee speckle filter."""

    def test_lee_filter_reduces_noise(self):
        """The variance of the filtered image must be lower than the
        original, demonstrating noise reduction.
        """
        rng = np.random.default_rng(42)
        noisy = rng.rayleigh(scale=0.5, size=(256, 256)).astype(np.float32)

        filtered = apply_lee_filter(noisy, window_size=7)

        var_original = float(np.var(noisy))
        var_filtered = float(np.var(filtered))

        assert var_filtered < var_original, (
            f"Filtered variance ({var_filtered:.4f}) should be less than "
            f"original variance ({var_original:.4f})"
        )

    def test_lee_filter_preserves_shape(self):
        """Output must have the same shape and dtype as the input."""
        image = np.ones((100, 100), dtype=np.float32) * 5.0
        filtered = apply_lee_filter(image, window_size=5)

        assert filtered.shape == image.shape
        assert filtered.dtype == np.float32

    def test_lee_filter_even_window_adjusted(self):
        """An even window_size must be silently adjusted to odd."""
        image = np.ones((64, 64), dtype=np.float32) * 3.0
        # Should not raise, even though 6 is even
        filtered = apply_lee_filter(image, window_size=6)
        assert filtered.shape == image.shape


# ====================================================================
# Tiling
# ====================================================================


class TestCreateTiles:
    """Tests for the create_tiles function."""

    def test_create_tiles_shape(self):
        """Every tile must have the exact (tile_size, tile_size) shape,
        including edge tiles which are zero-padded.
        """
        image = np.random.default_rng(0).random((1000, 1000)).astype(np.float32)
        tile_size = 640
        overlap = 64

        tiles = create_tiles(image, tile_size=tile_size, overlap=overlap)

        for tile_info in tiles:
            arr = tile_info["array"]
            assert arr.shape == (tile_size, tile_size), (
                f"Tile shape {arr.shape} != expected ({tile_size}, {tile_size})"
            )

    def test_create_tiles_overlap(self):
        """Overlapping tiles must cover the full image.

        We verify by checking that the union of all tile source regions
        (row_offset to row_offset+tile_size, col_offset to col_offset+tile_size)
        covers every row and column of the original image.
        """
        rows, cols = 500, 500
        image = np.ones((rows, cols), dtype=np.float32)
        tile_size = 256
        overlap = 64

        tiles = create_tiles(image, tile_size=tile_size, overlap=overlap)

        # Build a coverage mask
        coverage = np.zeros((rows, cols), dtype=bool)
        for tile_info in tiles:
            r = tile_info["row_offset"]
            c = tile_info["col_offset"]
            r_end = min(r + tile_size, rows)
            c_end = min(c + tile_size, cols)
            coverage[r:r_end, c:c_end] = True

        assert coverage.all(), "Not all pixels are covered by tiles"

    def test_create_tiles_count(self):
        """The number of tiles must match the expected formula:
        ceil(rows / step) * ceil(cols / step)  where step = tile_size - overlap.
        """
        rows, cols = 1280, 1280
        tile_size = 640
        overlap = 64
        step = tile_size - overlap

        image = np.zeros((rows, cols), dtype=np.float32)
        tiles = create_tiles(image, tile_size=tile_size, overlap=overlap)

        expected_rows = math.ceil(rows / step)
        expected_cols = math.ceil(cols / step)
        expected_count = expected_rows * expected_cols

        assert len(tiles) == expected_count, f"Expected {expected_count} tiles, got {len(tiles)}"

    def test_create_tiles_metadata(self):
        """Each tile dict must include row_offset, col_offset, and geo_bounds."""
        image = np.zeros((640, 640), dtype=np.float32)
        tiles = create_tiles(image, tile_size=640, overlap=0)

        assert len(tiles) == 1
        tile = tiles[0]
        assert tile["row_offset"] == 0
        assert tile["col_offset"] == 0
        assert "geo_bounds" in tile

    def test_create_tiles_overlap_too_large_raises(self):
        """overlap >= tile_size must raise ValueError."""
        image = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="overlap"):
            create_tiles(image, tile_size=64, overlap=64)


# ====================================================================
# Geo transform with rotation
# ====================================================================


class TestRotatedAffine:
    """Verify the GCP fit captures rotation and the helpers round-trip."""

    def test_gcp_fit_recovers_rotation(self):
        """A synthetic rotated affine recovered from GCPs round-trips."""
        # Truth: lon = -5.0 + 1e-4*col + 2e-5*row
        #        lat = 36.0 + 1e-5*col - 9e-5*row
        truth = (-5.0, 1e-4, 2e-5, 36.0, 1e-5, -9e-5)
        rng = np.random.default_rng(0)
        gcps = []
        for _ in range(64):
            col = float(rng.integers(0, 25000))
            line = float(rng.integers(0, 16000))
            lon = truth[0] + col * truth[1] + line * truth[2]
            lat = truth[3] + col * truth[4] + line * truth[5]
            gcps.append({"line": line, "pixel": col, "lat": lat, "lon": lon})

        gt = _build_pixel_to_geo_transform(gcps, 16000, 25000)
        assert gt is not None
        # Allow tiny numerical wobble
        assert math.isclose(gt[1], truth[1], rel_tol=1e-6)
        assert math.isclose(gt[2], truth[2], rel_tol=1e-6)
        assert math.isclose(gt[4], truth[4], rel_tol=1e-6)
        assert math.isclose(gt[5], truth[5], rel_tol=1e-6)

    def test_pixel_geo_round_trip(self):
        """affine_pixel_to_geo / affine_geo_to_pixel must invert exactly."""
        gt = (-5.0, 1e-4, 2e-5, 36.0, 1e-5, -9e-5)
        for col, row in [(0, 0), (12345, 6789), (25000, 16000)]:
            lon, lat = affine_pixel_to_geo(gt, col, row)
            c2, r2 = affine_geo_to_pixel(gt, lon, lat)
            assert math.isclose(c2, col, abs_tol=1e-6)
            assert math.isclose(r2, row, abs_tol=1e-6)
