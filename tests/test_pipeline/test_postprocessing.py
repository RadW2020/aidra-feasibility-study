"""
Tests for the postprocessing module.

Covers:
- IoU computation (perfect, no overlap, partial)
- Non-Maximum Suppression (removes duplicates, preserves distinct)
- GeoJSON conversion
- Detection statistics aggregation
"""

from __future__ import annotations

import pytest

from src.pipeline.postprocessing import (
    apply_nms,
    compute_detection_stats,
    compute_iou,
    detections_to_geojson,
)

# ====================================================================
# IoU computation
# ====================================================================


class TestComputeIoU:
    """Tests for the compute_iou function."""

    def test_compute_iou_perfect_overlap(self):
        """Two identical boxes must have IoU = 1.0."""
        box = [10, 20, 50, 60]
        iou = compute_iou(box, box)
        assert iou == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self):
        """Two non-overlapping boxes must have IoU = 0.0."""
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 30, 30]
        iou = compute_iou(box_a, box_b)
        assert iou == pytest.approx(0.0)

    def test_compute_iou_partial(self):
        """Partially overlapping boxes must have 0 < IoU < 1."""
        box_a = [0, 0, 20, 20]
        box_b = [10, 10, 30, 30]

        iou = compute_iou(box_a, box_b)

        assert 0.0 < iou < 1.0

        # Manual calculation:
        #   intersection: [10,10,20,20] -> area = 10*10 = 100
        #   area_a = 20*20 = 400, area_b = 20*20 = 400
        #   union = 400 + 400 - 100 = 700
        #   IoU = 100 / 700 ~ 0.1429
        expected_iou = 100.0 / 700.0
        assert iou == pytest.approx(expected_iou, rel=1e-4)

    def test_compute_iou_touching_edges(self):
        """Boxes that share only an edge have zero area of overlap."""
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 20, 10]
        iou = compute_iou(box_a, box_b)
        assert iou == pytest.approx(0.0)

    def test_compute_iou_contained(self):
        """A box contained inside another has IoU = area_inner / area_outer."""
        outer = [0, 0, 100, 100]
        inner = [25, 25, 75, 75]
        iou = compute_iou(outer, inner)

        # inner area = 50*50 = 2500, outer = 100*100 = 10000
        # intersection = 2500, union = 10000
        expected = 2500.0 / 10000.0
        assert iou == pytest.approx(expected, rel=1e-4)


# ====================================================================
# Non-Maximum Suppression
# ====================================================================


class TestNMS:
    """Tests for apply_nms."""

    def test_nms_removes_duplicates(self):
        """NMS must suppress lower-confidence detections that overlap
        with higher-confidence ones.
        """
        detections = [
            {"bbox": [10, 10, 30, 30], "confidence": 0.9},
            {"bbox": [12, 12, 32, 32], "confidence": 0.7},
            {"bbox": [11, 11, 31, 31], "confidence": 0.6},
        ]

        result = apply_nms(detections, iou_threshold=0.5)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_nms_keeps_distinct(self):
        """NMS must preserve detections that do not overlap."""
        detections = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.8},
            {"bbox": [100, 100, 110, 110], "confidence": 0.75},
            {"bbox": [200, 200, 210, 210], "confidence": 0.9},
        ]

        result = apply_nms(detections, iou_threshold=0.5)

        assert len(result) == 3

    def test_nms_empty_input(self):
        """NMS on an empty list must return an empty list."""
        assert apply_nms([], iou_threshold=0.5) == []

    def test_nms_single_detection(self):
        """NMS on a single detection must return that detection."""
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.5}]
        result = apply_nms(dets, iou_threshold=0.5)
        assert len(result) == 1

    def test_nms_ordering(self):
        """NMS result must be ordered by confidence (descending)."""
        detections = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.5},
            {"bbox": [100, 100, 110, 110], "confidence": 0.9},
            {"bbox": [200, 200, 210, 210], "confidence": 0.7},
        ]

        result = apply_nms(detections, iou_threshold=0.5)

        confidences = [d["confidence"] for d in result]
        assert confidences == sorted(confidences, reverse=True)


# ====================================================================
# GeoJSON conversion
# ====================================================================


class TestDetectionsToGeoJSON:
    """Tests for detections_to_geojson."""

    def test_detections_to_geojson(self, sample_detections):
        """Output must be a valid GeoJSON FeatureCollection with one
        feature per detection.
        """
        # Add center_geo so the function can build geometry
        enriched = []
        for i, det in enumerate(sample_detections):
            d = dict(det)
            d["center_geo"] = [-5.5 + i * 0.01, 36.0 + i * 0.01]
            enriched.append(d)

        geojson = detections_to_geojson(enriched)

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == len(enriched)

        for feature in geojson["features"]:
            assert feature["type"] == "Feature"
            assert "geometry" in feature
            assert "properties" in feature
            geom = feature["geometry"]
            assert geom["type"] == "Point"
            assert len(geom["coordinates"]) == 2

    def test_geojson_from_bbox_geo(self):
        """When only bbox_geo is present, geometry should be a Polygon."""
        detections = [
            {
                "bbox_geo": [-5.5, 36.0, -5.4, 36.1],
                "confidence": 0.8,
                "source": "cfar",
            }
        ]

        geojson = detections_to_geojson(detections)
        feature = geojson["features"][0]
        assert feature["geometry"]["type"] == "Polygon"

    def test_geojson_empty_input(self):
        """Empty detections list must produce an empty FeatureCollection."""
        geojson = detections_to_geojson([])
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 0


# ====================================================================
# Detection statistics
# ====================================================================


class TestDetectionStats:
    """Tests for compute_detection_stats."""

    def test_detection_stats(self, sample_detections):
        """Stats must include total count, confidence aggregates, and
        by-source breakdown.
        """
        stats = compute_detection_stats(sample_detections)

        assert stats["total"] == 2
        assert 0.0 < stats["avg_confidence"] < 1.0
        assert stats["max_confidence"] == 0.85
        assert stats["min_confidence"] == 0.72

        # by_source should count yolo and fused
        assert stats["by_source"]["yolo"] == 1
        assert stats["by_source"]["fused"] == 1

    def test_detection_stats_empty(self):
        """Stats on empty detections must return zero-filled defaults."""
        stats = compute_detection_stats([])

        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["by_source"] == {"cfar": 0, "yolo": 0, "fused": 0}

    def test_detection_stats_spatial_extent(self):
        """When detections have bbox_geo, spatial_extent must be populated."""
        detections = [
            {
                "confidence": 0.8,
                "source": "cfar",
                "bbox_geo": [-5.5, 35.5, -5.4, 35.6],
            },
            {
                "confidence": 0.7,
                "source": "cfar",
                "bbox_geo": [-5.3, 35.7, -5.2, 35.8],
            },
        ]

        stats = compute_detection_stats(detections)

        assert len(stats["spatial_extent"]) == 4
        lon_min, lat_min, lon_max, lat_max = stats["spatial_extent"]
        assert lon_min == pytest.approx(-5.5)
        assert lat_max == pytest.approx(35.8)
