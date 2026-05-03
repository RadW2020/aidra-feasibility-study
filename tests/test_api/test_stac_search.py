"""
Tests for STAC API - Item Search endpoints (POST/GET /api/stac/search)
and the conformsTo array on the catalog root.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tests.test_api.conftest import FakeRecord


@pytest.fixture
def fake_stac_item_row() -> FakeRecord:
    """Fake row matching the _SELECT_ITEMS_SEARCH projection."""
    return FakeRecord(
        id=uuid4(),
        created_at=datetime.now(tz=UTC),
        image_id="S1A_TEST_001",
        image_title="S1A_IW_GRDH_TEST",
        image_hash="abc123",
        image_bbox_geojson=json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-6.0, 35.0],
                        [-5.0, 35.0],
                        [-5.0, 37.0],
                        [-6.0, 37.0],
                        [-6.0, 35.0],
                    ]
                ],
            }
        ),
        image_sensing_date=datetime(2026, 4, 1, tzinfo=UTC),
        search_zone="gibraltar",
        model_name="yolov8n-sar",
        model_version="v1.0",
        constraint_profile="ground",
        num_detections=5,
        avg_confidence=0.78,
        output_hash="ghi789",
        input_params_hash="jkl012",
        commit_sha="0" * 40,
        incidence_angle=35.4,
        polarisation="VV+VH",
        orbit_direction="DESCENDING",
        relative_orbit=110,
        product_type="GRD",
        pixel_spacing=10.0,
        status="success",
    )


# ------------------------------------------------------------------
# conformsTo on catalog root
# ------------------------------------------------------------------


async def test_stac_catalog_has_conforms_to(client, mock_db):
    resp = await client.get("/api/stac/catalog.json")
    assert resp.status_code == 200
    body = resp.json()
    assert "conformsTo" in body
    classes = body["conformsTo"]
    assert "https://api.stacspec.org/v1.0.0/core" in classes
    assert "https://api.stacspec.org/v1.0.0/item-search" in classes
    assert "https://api.stacspec.org/v1.0.0/collections" in classes


# ------------------------------------------------------------------
# POST /search
# ------------------------------------------------------------------


async def test_stac_search_post_returns_feature_collection(
    client, mock_db, fake_stac_item_row
):
    mock_db.fetch = AsyncMock(return_value=[fake_stac_item_row])
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.post(
        "/api/stac/search",
        json={"limit": 50},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["numberMatched"] == 1
    assert body["numberReturned"] == 1
    assert len(body["features"]) == 1
    assert body["features"][0]["collection"] == "detections"


async def test_stac_search_post_bbox_filter(client, mock_db):
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.post(
        "/api/stac/search",
        json={"bbox": [-6.0, 35.0, -5.0, 37.0]},
    )
    assert resp.status_code == 200

    # First 4 positional args of fetch are lon_min,lat_min,lon_max,lat_max
    call_args = mock_db.fetch.call_args[0]
    assert call_args[1] == -6.0  # lon_min  (positional 1 because [0] is SQL str)
    assert call_args[2] == 35.0
    assert call_args[3] == -5.0
    assert call_args[4] == 37.0


async def test_stac_search_post_invalid_bbox_400(client, mock_db):
    resp = await client.post(
        "/api/stac/search",
        json={"bbox": [1.0, 2.0, 3.0]},
    )
    assert resp.status_code == 400


async def test_stac_search_post_invalid_collection_400(client, mock_db):
    resp = await client.post(
        "/api/stac/search",
        json={"collections": ["nonsense"]},
    )
    assert resp.status_code == 400


async def test_stac_search_post_inverted_bbox_400(client, mock_db):
    resp = await client.post(
        "/api/stac/search",
        json={"bbox": [10.0, 10.0, 5.0, 5.0]},
    )
    assert resp.status_code == 400


# ------------------------------------------------------------------
# GET /search (EODAG-style query string)
# ------------------------------------------------------------------


async def test_stac_search_get(client, mock_db, fake_stac_item_row):
    mock_db.fetch = AsyncMock(return_value=[fake_stac_item_row])
    mock_db.fetchval = AsyncMock(return_value=1)

    resp = await client.get(
        "/api/stac/search",
        params={"bbox": "-6,35,-5,37", "limit": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["numberReturned"] == 1


async def test_stac_search_get_invalid_bbox_400(client, mock_db):
    resp = await client.get(
        "/api/stac/search",
        params={"bbox": "not,a,bbox,xyz"},
    )
    assert resp.status_code == 400


async def test_stac_search_get_datetime_interval(client, mock_db):
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchval = AsyncMock(return_value=0)

    resp = await client.get(
        "/api/stac/search",
        params={"datetime": "2026-04-01T00:00:00Z/2026-04-30T23:59:59Z"},
    )
    assert resp.status_code == 200
    call_args = mock_db.fetch.call_args[0]
    # positions 5 and 6 are dt_from, dt_to
    assert call_args[5] is not None
    assert call_args[6] is not None
