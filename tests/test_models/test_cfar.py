"""
Tests for the CFAR (Constant False Alarm Rate) detector.

Covers:
- Detection of bright vessel-like signatures in synthetic SAR tiles
- Rejection of pure-noise images (no false positives)
- DBSCAN clustering of pixel detections into bounding boxes
- OS-CFAR variant functionality
- Effect of pfa parameter on detection count
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.cfar import CFARDetector

# ====================================================================
# Detection capability
# ====================================================================


class TestCFARDetection:
    """Tests verifying that CFAR finds real targets."""

    def test_cfar_detects_bright_points(self, sample_sar_tile):
        """CFAR must detect at least 3 of the 5 simulated vessels.

        The synthetic tile has 5 Gaussian bright-point vessels.  CFAR with
        standard parameters should pick up the majority even after DBSCAN
        clustering.
        """
        image, ground_truth = sample_sar_tile
        detector = CFARDetector(guard_size=8, training_size=20, pfa=1e-5)
        detections = detector.detect_with_clustering(image)

        assert len(detections) >= 3, (
            f"Expected at least 3 clustered detections from 5 vessels, "
            f"got {len(detections)}"
        )

        # Every detection must have the expected keys
        for det in detections:
            assert "bbox" in det
            assert "center" in det
            assert "num_pixels" in det
            assert "mean_intensity" in det
            assert "mean_snr" in det
            assert "method" in det
            assert det["method"] == "ca-cfar"

    def test_cfar_no_false_positives_on_noise(self):
        """CFAR should produce zero or at most 1 clustered detection on
        pure Rayleigh noise (no embedded targets).  Using a very low pfa
        (1e-6) to be conservative.
        """
        rng = np.random.default_rng(42)
        noise = rng.rayleigh(scale=0.3, size=(640, 640)).astype(np.float32)

        detector = CFARDetector(guard_size=3, training_size=15, pfa=1e-6)
        detections = detector.detect_with_clustering(noise)

        assert len(detections) <= 1, (
            f"Expected at most 1 false positive on pure noise, "
            f"got {len(detections)}"
        )


# ====================================================================
# Clustering
# ====================================================================


class TestCFARClustering:
    """Tests verifying DBSCAN groups pixel detections into bboxes."""

    def test_cfar_clustering(self, sample_sar_tile):
        """DBSCAN must group adjacent bright pixels into bounding boxes.

        Each clustered detection must have a valid 4-element bbox and
        at least min_cluster_size pixels.
        """
        image, _ = sample_sar_tile
        detector = CFARDetector(guard_size=8, training_size=20, pfa=1e-5)
        clustered = detector.detect_with_clustering(
            image, min_cluster_size=3, eps=2.0
        )

        for det in clustered:
            bbox = det["bbox"]
            assert len(bbox) == 4
            x_min, y_min, x_max, y_max = bbox
            assert x_min <= x_max
            assert y_min <= y_max
            assert det["num_pixels"] >= 3

    def test_pixel_detect_returns_raw_points(self, sample_sar_tile):
        """The raw detect() method must return individual pixel detections
        (not clustered), each with x, y, intensity, snr, and method.
        """
        image, _ = sample_sar_tile
        detector = CFARDetector(guard_size=8, training_size=20, pfa=1e-5)
        pixel_dets = detector.detect(image)

        assert len(pixel_dets) > 0
        for det in pixel_dets:
            assert "x" in det
            assert "y" in det
            assert "intensity" in det
            assert "snr" in det
            assert det["snr"] > 0


# ====================================================================
# OS-CFAR variant
# ====================================================================


class TestOSCFAR:
    """Tests for the Ordered Statistics CFAR variant."""

    def test_cfar_os_variant(self):
        """OS-CFAR must produce valid detections on a small synthetic tile.

        OS-CFAR is slower (generic_filter) so we use a small 128x128 tile
        to keep test time reasonable.
        """
        from src.pipeline.preprocessing import generate_synthetic_sar_tile

        image, _ = generate_synthetic_sar_tile(
            size=128, num_vessels=2, seed=99
        )

        detector = CFARDetector(
            guard_size=3,
            training_size=10,
            pfa=1e-4,
            method="os",
            os_percentile=0.75,
        )
        pixel_dets = detector.detect(image)

        # OS-CFAR should detect something on an image with bright targets
        assert len(pixel_dets) > 0
        for det in pixel_dets:
            assert det["method"] == "os-cfar"


# ====================================================================
# Parameter sensitivity
# ====================================================================


class TestCFARParameters:
    """Tests verifying that CFAR parameters influence detection behaviour."""

    def test_cfar_parameters(self, sample_sar_tile):
        """Lower pfa should produce fewer detections than higher pfa.

        pfa controls the probability of false alarm: a lower value sets
        a higher threshold, meaning fewer (but more reliable) detections.
        """
        image, _ = sample_sar_tile

        detector_strict = CFARDetector(
            guard_size=3, training_size=15, pfa=1e-7
        )
        detector_loose = CFARDetector(
            guard_size=3, training_size=15, pfa=1e-3
        )

        dets_strict = detector_strict.detect(image)
        dets_loose = detector_loose.detect(image)

        assert len(dets_strict) <= len(dets_loose), (
            f"Stricter pfa (1e-7) produced {len(dets_strict)} detections "
            f"but looser pfa (1e-3) produced only {len(dets_loose)}"
        )

    def test_invalid_method_raises(self):
        """Passing an invalid method name must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid CFAR method"):
            CFARDetector(method="invalid")

    def test_training_le_guard_raises(self):
        """training_size <= guard_size must raise ValueError."""
        with pytest.raises(ValueError, match="training_size"):
            CFARDetector(guard_size=10, training_size=5)


# ====================================================================
# SNR gate
# ====================================================================


class TestCFARMinSnr:
    """Tests for the min_mean_snr filter on detect_with_clustering."""

    def test_min_snr_filters_low_clusters(self, sample_sar_tile):
        """A high SNR threshold must drop more clusters than a low one."""
        image, _ = sample_sar_tile
        detector = CFARDetector(guard_size=8, training_size=20, pfa=1e-5)

        loose = detector.detect_with_clustering(
            image, min_cluster_size=3, eps=2.0, min_mean_snr=0.0
        )
        strict = detector.detect_with_clustering(
            image, min_cluster_size=3, eps=2.0, min_mean_snr=1e6
        )

        assert len(strict) <= len(loose)
        assert len(strict) == 0
