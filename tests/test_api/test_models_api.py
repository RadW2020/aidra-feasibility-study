"""
Integration tests for /api/profiles and /api/zones endpoints.

Tier 1: These endpoints return static data and do not hit the database,
so no mocking is needed beyond the base mock_db fixture.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# test_profiles_list
# ------------------------------------------------------------------


async def test_profiles_list(client):
    """GET /api/profiles returns exactly 5 constraint profiles."""
    resp = await client.get("/api/profiles")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 5


# ------------------------------------------------------------------
# test_profiles_names
# ------------------------------------------------------------------


async def test_profiles_names(client):
    """Profile names match the expected set from the spec."""
    resp = await client.get("/api/profiles")
    data = resp.json()

    names = {p["name"] for p in data}
    expected = {"ground", "sat-high", "sat-mid", "sat-low", "sat-extreme"}
    assert names == expected


# ------------------------------------------------------------------
# test_profiles_have_required_fields
# ------------------------------------------------------------------


async def test_profiles_have_required_fields(client):
    """Each profile entry contains cpu_limit and memory_limit_mb."""
    resp = await client.get("/api/profiles")
    data = resp.json()

    for profile in data:
        assert "cpu_limit" in profile, f"Missing cpu_limit in {profile['name']}"
        assert "memory_limit_mb" in profile, (
            f"Missing memory_limit_mb in {profile['name']}"
        )
        assert isinstance(profile["cpu_limit"], (int, float))
        assert isinstance(profile["memory_limit_mb"], int)


# ------------------------------------------------------------------
# test_profiles_display_name
# ------------------------------------------------------------------


async def test_profiles_display_name(client):
    """Each profile has a non-empty display_name string."""
    resp = await client.get("/api/profiles")
    data = resp.json()

    for profile in data:
        assert "display_name" in profile
        assert isinstance(profile["display_name"], str)
        assert len(profile["display_name"]) > 0


# ------------------------------------------------------------------
# test_zones_list
# ------------------------------------------------------------------


async def test_zones_list(client):
    """GET /api/zones returns exactly 5 search zones."""
    resp = await client.get("/api/zones")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 5


# ------------------------------------------------------------------
# test_zones_have_bbox
# ------------------------------------------------------------------


async def test_zones_have_bbox(client):
    """Each zone has a 4-element bbox: [lon_min, lat_min, lon_max, lat_max]."""
    resp = await client.get("/api/zones")
    data = resp.json()

    for zone in data:
        assert "bbox" in zone, f"Missing bbox in zone {zone.get('name')}"
        bbox = zone["bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4, (
            f"Zone {zone['name']} bbox has {len(bbox)} elements, expected 4"
        )
        for coord in bbox:
            assert isinstance(coord, (int, float)), (
                f"Zone {zone['name']} bbox contains non-numeric value: {coord}"
            )


# ------------------------------------------------------------------
# test_zones_names
# ------------------------------------------------------------------


async def test_zones_names(client):
    """Zone names match the expected set from SEARCH_ZONES."""
    resp = await client.get("/api/zones")
    data = resp.json()

    names = {z["name"] for z in data}
    expected = {
        "gibraltar",
        "mediterranean_west",
        "suez_approach",
        "english_channel",
        "north_adriatic",
    }
    assert names == expected


# ------------------------------------------------------------------
# test_zones_bbox_valid_coordinates
# ------------------------------------------------------------------


async def test_zones_bbox_valid_coordinates(client):
    """Each zone bbox has lon_min < lon_max and lat_min < lat_max."""
    resp = await client.get("/api/zones")
    data = resp.json()

    for zone in data:
        lon_min, lat_min, lon_max, lat_max = zone["bbox"]
        assert lon_min < lon_max, (
            f"Zone {zone['name']}: lon_min ({lon_min}) >= lon_max ({lon_max})"
        )
        assert lat_min < lat_max, (
            f"Zone {zone['name']}: lat_min ({lat_min}) >= lat_max ({lat_max})"
        )
