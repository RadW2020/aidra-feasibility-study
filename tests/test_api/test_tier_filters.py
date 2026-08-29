"""source / tier filters on GeoJSON and OGC Features (the fused high-precision tier)."""

from __future__ import annotations

import json

import pytest

from src.api.tiers import source_sql_clause, tier_of

pytestmark = pytest.mark.anyio


def test_tier_helpers():
    assert tier_of("fused") == "high" and tier_of("cfar") == "standard" and tier_of(None) == "standard"
    assert source_sql_clause("fused", None) == "AND d.source = 'fused'"
    assert source_sql_clause(None, "high") == "AND d.source = 'fused'"
    assert source_sql_clause(None, "standard") == "AND d.source <> 'fused'"
    assert source_sql_clause(None, None) == ""
    with pytest.raises(ValueError):
        source_sql_clause("fused' OR 1=1 --", None)


async def test_geojson_tier_filter_and_property(client, mock_db, fake_detection_row):
    fake_detection_row["source"] = "fused"
    fake_detection_row["center_geojson"] = '{"type": "Point", "coordinates": [-5.6, 36.0]}'
    fake_detection_row["bbox_geojson"] = None
    mock_db.fetch.return_value = [fake_detection_row]
    resp = await client.get("/api/detections.geojson?tier=high")
    assert resp.status_code == 200
    sql = mock_db.fetch.call_args[0][0]
    assert "AND d.source = 'fused'" in sql and sql.index("d.source = 'fused'") < sql.index("ORDER BY d.confidence DESC")
    fc = json.loads(resp.content)
    assert fc["features"][0]["properties"]["tier"] == "high"
    resp = await client.get("/api/detections.geojson?source=cfar")
    assert "AND d.source = 'cfar'" in mock_db.fetch.call_args[0][0]
    resp = await client.get("/api/detections.geojson?tier=standard")
    assert "AND d.source <> 'fused'" in mock_db.fetch.call_args[0][0]
    assert (await client.get("/api/detections.geojson?source=nope")).status_code == 422
    resp = await client.get("/api/detections.geojson")
    assert "d.source =" not in mock_db.fetch.call_args[0][0]


async def test_ogc_items_source_filter(client, mock_db):
    mock_db.fetch.return_value = []
    mock_db.fetchval.return_value = 0
    resp = await client.get("/api/ogc/collections/detections/items?source=fused")
    assert resp.status_code == 200, resp.text
    select_sql = mock_db.fetch.call_args[0][0]
    count_sql = mock_db.fetchval.call_args[0][0]
    assert "WHERE d.source = 'fused' AND" in select_sql and "WHERE d.source = 'fused' AND" in count_sql
    assert (await client.get("/api/ogc/collections/detections/items?tier=nope")).status_code == 422


def test_dashboard_has_source_selector_and_tier_panel():
    from pathlib import Path

    d = json.loads(Path("grafana/dashboards/01-map-detections.json").read_text())
    assert any(v["name"] == "source" for v in d["templating"]["list"])
    assert any("source = fused" in p["title"] for p in d["panels"])
    sqls = [t["rawSql"] for p in d["panels"] for t in p.get("targets", []) if "d.quality_verdict = '$quality'" in t.get("rawSql", "")]
    assert sqls and all("d.source = '$source'" in s for s in sqls if "FILTER (WHERE d.source = 'fused')" not in s)
