"""Regression: bbox filters on the detection endpoints.

The bbox string-replace swaps ``$n::geometry`` for ``ST_GeomFromGeoJSON($n::text)``,
but the second use of the parameter inside ``ST_Intersects`` was an untyped ``$n``,
so every bbox query failed in production with ``function st_intersects(geometry,
text) is not unique``. Mocked-DB tests can't see SQL typing; these pin the query text.
"""

from __future__ import annotations

BBOX = "-5.8,35.7,-5.2,36.2"


def _sql(mock_db) -> str:
    return mock_db.fetch.call_args[0][0]


async def test_detections_list_casts_both_geometry_uses(client, mock_db):
    await client.get("/api/detections", params={"bbox": BBOX})
    assert "ST_Intersects(d.center_geo, ST_GeomFromGeoJSON($6::text))" in _sql(mock_db)
    assert "ST_Intersects(d.center_geo, ST_GeomFromGeoJSON($6::text))" in mock_db.fetchval.call_args[0][0]


async def test_detections_geojson_casts_both_geometry_uses(client, mock_db):
    await client.get("/api/detections.geojson", params={"bbox": BBOX})
    assert "ST_Intersects(d.center_geo, ST_GeomFromGeoJSON($6::text))" in _sql(mock_db)


async def test_ogc_items_cast_both_geometry_uses(client, mock_db):
    await client.get("/api/ogc/collections/detections/items", params={"bbox": BBOX})
    assert "ST_Intersects(d.center_geo, ST_GeomFromGeoJSON($1::text))" in _sql(mock_db)
