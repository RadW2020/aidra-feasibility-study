"""
Tests for src/tipcue/evaluator.py and src/tipcue/zones.py.

Coverage:
- Zone: contains_point, geometry auto-build, to_geojson_bbox
- get_zone / get_active_zones helpers
- TipEvaluator: no-match, match threshold, cooldown, priority computation,
  bbox computation, UUID coercion in triggering_detections
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

from src.tipcue.evaluator import TipEvaluator
from src.tipcue.zones import DEFAULT_ZONES, Zone, get_active_zones, get_zone

# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------


class TestZone:
    def _make_zone(self, **kwargs) -> Zone:
        defaults = dict(
            id="test_zone",
            name="Test Zone",
            bbox=[-6.0, 35.0, -5.0, 36.0],
            priority=0,
        )
        defaults.update(kwargs)
        return Zone(**defaults)

    def test_contains_point_inside(self) -> None:
        zone = self._make_zone()
        assert zone.contains_point(-5.5, 35.5)

    def test_contains_point_outside_lon(self) -> None:
        zone = self._make_zone()
        assert not zone.contains_point(-7.0, 35.5)

    def test_contains_point_outside_lat(self) -> None:
        zone = self._make_zone()
        assert not zone.contains_point(-5.5, 37.0)

    def test_contains_point_on_boundary(self) -> None:
        zone = self._make_zone()
        assert zone.contains_point(-6.0, 35.0)

    def test_geometry_auto_built(self) -> None:
        zone = self._make_zone()
        assert zone.geometry["type"] == "Polygon"
        ring = zone.geometry["coordinates"][0]
        assert ring[0] == ring[-1], "ring must be closed"

    def test_geometry_coordinates_match_bbox(self) -> None:
        zone = self._make_zone(bbox=[-6.0, 35.0, -5.0, 36.0])
        ring = zone.geometry["coordinates"][0]
        lons = [c[0] for c in ring[:-1]]
        lats = [c[1] for c in ring[:-1]]
        assert min(lons) == -6.0
        assert max(lons) == -5.0
        assert min(lats) == 35.0
        assert max(lats) == 36.0

    def test_to_geojson_bbox_parses(self) -> None:
        zone = self._make_zone()
        geojson_str = zone.to_geojson_bbox()
        parsed = json.loads(geojson_str)
        assert parsed["type"] == "Polygon"

    def test_inactive_zone(self) -> None:
        zone = self._make_zone(active=False)
        assert not zone.active


class TestZoneHelpers:
    def test_get_zone_found(self) -> None:
        zone = get_zone("gibraltar_strait")
        assert zone is not None
        assert zone.id == "gibraltar_strait"

    def test_get_zone_not_found(self) -> None:
        assert get_zone("nonexistent_zone") is None

    def test_get_active_zones_returns_all_by_default(self) -> None:
        active = get_active_zones()
        assert len(active) > 0
        for z in active:
            assert z.active

    def test_default_zones_have_valid_bboxes(self) -> None:
        for zone in DEFAULT_ZONES:
            lon_min, lat_min, lon_max, lat_max = zone.bbox
            assert lon_min < lon_max
            assert lat_min < lat_max


# ---------------------------------------------------------------------------
# TipEvaluator
# ---------------------------------------------------------------------------


def _det(lon: float, lat: float, conf: float = 0.9, det_id: UUID | None = None) -> dict:
    return {
        "id": det_id or uuid4(),
        "longitude": lon,
        "latitude": lat,
        "confidence": conf,
    }


class TestTipEvaluatorNoMatch:
    def test_empty_detections_returns_no_tips(self) -> None:
        ev = TipEvaluator()
        tips = ev.evaluate([], execution_id=uuid4())
        assert tips == []

    def test_detections_outside_all_zones_returns_no_tips(self) -> None:
        ev = TipEvaluator()
        dets = [_det(0.0, 0.0), _det(0.1, 0.1), _det(0.2, 0.2)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert tips == []

    def test_below_min_confidence_no_tip(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.8, min_detections=2)
        dets = [_det(-5.5, 35.5, conf=0.5), _det(-5.4, 35.6, conf=0.6)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert tips == []

    def test_below_min_detections_no_tip(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=3)
        dets = [_det(-5.5, 35.5, conf=0.9), _det(-5.4, 35.6, conf=0.9)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert tips == []

    def test_inactive_zone_skipped(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0], active=False)
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=1)
        dets = [_det(-5.5, 35.5, conf=0.9)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert tips == []


class TestTipEvaluatorMatch:
    def test_sufficient_detections_generates_tip(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=2)
        exec_id = uuid4()
        dets = [_det(-5.5, 35.5, conf=0.9), _det(-5.4, 35.6, conf=0.8)]
        tips = ev.evaluate(dets, execution_id=exec_id)
        assert len(tips) == 1
        assert tips[0].should_cue
        assert tips[0].zone_id == "z"
        assert tips[0].execution_id == exec_id

    def test_tip_target_bbox_encloses_detections(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=2)
        dets = [_det(-5.5, 35.5), _det(-5.4, 35.6)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        bbox = tips[0].target_bbox
        assert bbox[0] <= -5.5 <= bbox[2]
        assert bbox[1] <= 35.5 <= bbox[3]
        assert bbox[0] <= -5.4 <= bbox[2]
        assert bbox[1] <= 35.6 <= bbox[3]

    def test_triggering_detection_ids_captured(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=1)
        det_id = uuid4()
        dets = [_det(-5.5, 35.5, det_id=det_id)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert det_id in tips[0].triggering_detections

    def test_triggering_ids_accept_string_uuids(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=1)
        det_id = uuid4()
        dets = [{"id": str(det_id), "longitude": -5.5, "latitude": 35.5, "confidence": 0.9}]
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert UUID(str(det_id)) in tips[0].triggering_detections

    def test_multiple_zones_can_both_fire(self) -> None:
        z1 = Zone(id="z1", name="Z1", bbox=[-6.0, 35.0, -5.0, 36.0])
        z2 = Zone(id="z2", name="Z2", bbox=[10.0, 33.0, 16.0, 38.0])
        ev = TipEvaluator(zones_of_interest=[z1, z2], min_confidence=0.7, min_detections=1)
        dets = [_det(-5.5, 35.5), _det(13.0, 37.0)]
        tips = ev.evaluate(dets, execution_id=uuid4())
        zone_ids = {t.zone_id for t in tips}
        assert "z1" in zone_ids
        assert "z2" in zone_ids


class TestTipEvaluatorCooldown:
    def test_second_call_in_cooldown_skipped(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=1,
                          cooldown_minutes=60)
        dets = [_det(-5.5, 35.5)]
        exec_id = uuid4()
        tips1 = ev.evaluate(dets, execution_id=exec_id)
        assert len(tips1) == 1

        tips2 = ev.evaluate(dets, execution_id=exec_id)
        assert tips2 == []

    def test_reset_cooldowns_allows_re_fire(self) -> None:
        zone = Zone(id="z", name="Z", bbox=[-6.0, 35.0, -5.0, 36.0])
        ev = TipEvaluator(zones_of_interest=[zone], min_confidence=0.7, min_detections=1,
                          cooldown_minutes=60)
        dets = [_det(-5.5, 35.5)]
        ev.evaluate(dets, execution_id=uuid4())
        ev.reset_cooldowns()
        tips = ev.evaluate(dets, execution_id=uuid4())
        assert len(tips) == 1


class TestTipEvaluatorPriority:
    def _eval_priority(self, num: int, conf: float, zone_priority: int) -> int:
        return TipEvaluator._compute_priority(num, conf, zone_priority)

    def test_urgent_many_high_conf(self) -> None:
        assert self._eval_priority(5, 0.90, 0) == 2

    def test_high_priority_moderate_detections(self) -> None:
        assert self._eval_priority(3, 0.75, 0) == 1

    def test_high_priority_high_confidence(self) -> None:
        assert self._eval_priority(1, 0.85, 0) == 1

    def test_zone_priority_base(self) -> None:
        assert self._eval_priority(1, 0.5, 1) == 1

    def test_priority_capped_at_two(self) -> None:
        assert self._eval_priority(10, 0.99, 2) == 2

    def test_low_priority_baseline(self) -> None:
        assert self._eval_priority(1, 0.5, 0) == 0


class TestTipEvaluatorBbox:
    def test_single_detection_bbox_has_margin(self) -> None:
        bbox = TipEvaluator._compute_target_bbox([{"longitude": 0.0, "latitude": 0.0}])
        assert bbox[0] < 0.0
        assert bbox[1] < 0.0
        assert bbox[2] > 0.0
        assert bbox[3] > 0.0

    def test_two_detections_span_covered(self) -> None:
        dets = [{"longitude": -5.5, "latitude": 35.5},
                {"longitude": -5.4, "latitude": 35.6}]
        bbox = TipEvaluator._compute_target_bbox(dets)
        assert bbox[0] < -5.5
        assert bbox[2] > -5.4
        assert bbox[1] < 35.5
        assert bbox[3] > 35.6
