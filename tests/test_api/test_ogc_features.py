"""
Tests for OGC API - Features Part 1 endpoints (``/api/ogc/...``).

Tier 1 (mocked DB): validates conformance classes, FeatureCollection
structure, CRS84 geometries, and 404 handling for unknown collections.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tests.test_api.conftest import FakeRecord


@pytest.fixture
def fake_ogc_row() -> FakeRecord:
    """A fake row matching the OGC Features projection."""
    return FakeRecord(
        id=uuid4(),
        execution_id=uuid4(),
        created_at=datetime.now(tz=UTC),
        longitude=-5.5,
        latitude=36.0,
        center_geojson=json.dumps({"type": "Point", "coordinates": [-5.5, 36.0]}),
        bbox_geojson=json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-5.51, 35.99],
                        [-5.49, 35.99],
                        [-5.49, 36.01],
                        [-5.51, 36.01],
                        [-5.51, 35.99],
                    ]
                ],
            }
        ),
        bbox_pixel=[100.0, 200.0, 120.0, 220.0],
        confidence=0.85,
        source="fused",
        cfar_snr=12.5,
        yolo_score=0.85,
        class_name="vessel",
        on_land=False,
        cluster_anomaly=False,
        image_id="S1A_TEST_001",
        image_title="S1A_IW_GRDH_TEST",
        image_sensing_date=datetime(2026, 4, 1, tzinfo=UTC),
        model_name="yolov8n-sar",
        model_version="v1.0",
        model_hash="def456ghi789",
        constraint_profile="ground",
        incidence_angle=35.4,
        pipeline_version="1.0.0",
        commit_sha="0" * 40,
    )


# ------------------------------------------------------------------
# Landing + conformance
# ------------------------------------------------------------------


async def test_ogc_landing_page(client, mock_db):
    resp = await client.get("/api/ogc/")
    assert resp.status_code == 200
    body = resp.json()
    assert "links" in body
    rels = {link["rel"] for link in body["links"]}
    assert {"self", "service-desc", "conformance", "data"}.issubset(rels)


async def test_ogc_conformance_lists_three_classes(client, mock_db):
    resp = await client.get("/api/ogc/conformance")
    assert resp.status_code == 200
    body = resp.json()
    classes = body["conformsTo"]
    assert "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core" in classes
    assert "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30" in classes
    assert "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson" in classes


# ------------------------------------------------------------------
# Collections
# ------------------------------------------------------------------


async def test_ogc_collections_lists_detections(client, mock_db):
    resp = await client.get("/api/ogc/collections")
    assert resp.status_code == 200
    body = resp.json()
    ids = [c["id"] for c in body["collections"]]
    assert "detections" in ids
    det = next(c for c in body["collections"] if c["id"] == "detections")
    assert "http://www.opengis.net/def/crs/OGC/1.3/CRS84" in det["crs"]


async def test_ogc_collection_metadata(client, mock_db):
    resp = await client.get("/api/ogc/collections/detections")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "detections"
    assert body["itemType"] == "feature"
    assert body["storageCrs"] == "http://www.opengis.net/def/crs/OGC/1.3/CRS84"


async def test_ogc_unknown_collection_returns_404(client, mock_db):
    resp = await client.get("/api/ogc/collections/does-not-exist")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# Items
# ------------------------------------------------------------------


async def test_ogc_items_returns_feature_collection(
    client, mock_db, fake_ogc_row
):
    mock_db.fetch = AsyncMock(return_value=[fake_ogc_row])
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get("/api/ogc/collections/detections/items")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/geo+json")
    assert resp.headers.get("content-crs") == (
        "<http://www.opengis.net/def/crs/OGC/1.3/CRS84>"
    )

    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["numberMatched"] == 1
    assert body["numberReturned"] == 1
    assert len(body["features"]) == 1

    feat = body["features"][0]
    assert feat["type"] == "Feature"
    # Geometry is RFC 7946 (GeoJSON Polygon or Point in lon/lat order)
    assert feat["geometry"]["type"] in {"Point", "Polygon"}
    if feat["geometry"]["type"] == "Polygon":
        ring = feat["geometry"]["coordinates"][0]
        # First and last coord must coincide (closed ring)
        assert ring[0] == ring[-1]
        # Lon range check (sanity)
        for lon, lat in ring:
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90

    # Properties carry traceability fields.
    props = feat["properties"]
    for key in (
        "detection_id",
        "scene_id",
        "model_name",
        "model_hash",
        "pipeline_version",
        "commit_sha",
        "acquisition_time_utc",
        "incidence_angle",
        "on_land",
        "cluster_anomaly",
        "execution_id",
        "confidence",
    ):
        assert key in props, f"missing property {key}"


async def test_ogc_items_bbox_filter_passes_envelope(client, mock_db):
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get(
        "/api/ogc/collections/detections/items",
        params={"bbox": "-6,35,-5,37"},
    )
    assert resp.status_code == 200
    # The bbox geojson is the FIRST positional argument to db.fetch.
    call_args = mock_db.fetch.call_args[0]
    bbox_arg = call_args[1]
    assert bbox_arg is not None
    parsed = json.loads(bbox_arg)
    assert parsed["type"] == "Polygon"


async def test_ogc_items_invalid_bbox_returns_400(client, mock_db):
    resp = await client.get(
        "/api/ogc/collections/detections/items",
        params={"bbox": "1,2,3"},
    )
    assert resp.status_code == 400


async def test_ogc_items_invalid_datetime_returns_400(client, mock_db):
    resp = await client.get(
        "/api/ogc/collections/detections/items",
        params={"datetime": "not-a-date"},
    )
    assert resp.status_code == 400


async def test_ogc_items_unknown_collection_returns_404(client, mock_db):
    resp = await client.get("/api/ogc/collections/foo/items")
    assert resp.status_code == 404


async def test_ogc_single_item_returns_feature(client, mock_db, fake_ogc_row):
    mock_db.fetchrow = AsyncMock(return_value=fake_ogc_row)

    feat_id = fake_ogc_row["id"]
    resp = await client.get(f"/api/ogc/collections/detections/items/{feat_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "Feature"
    assert body["id"] == str(feat_id)
    assert body["properties"]["model_hash"] == "def456ghi789"


async def test_ogc_single_item_unknown_collection_returns_404(client, mock_db):
    resp = await client.get(f"/api/ogc/collections/foo/items/{uuid4()}")
    assert resp.status_code == 404


async def test_ogc_single_item_not_found_returns_404(client, mock_db):
    mock_db.fetchrow = AsyncMock(return_value=None)
    resp = await client.get(f"/api/ogc/collections/detections/items/{uuid4()}")
    assert resp.status_code == 404


async def test_ogc_items_pagination_links(client, mock_db, fake_ogc_row):
    mock_db.fetch = AsyncMock(return_value=[fake_ogc_row])
    mock_db.fetchval = AsyncMock(return_value=500)

    resp = await client.get(
        "/api/ogc/collections/detections/items",
        params={"limit": 10, "offset": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    rels = {link["rel"] for link in body["links"]}
    assert "next" in rels
    assert "prev" in rels
