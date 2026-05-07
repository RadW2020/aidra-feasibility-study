"""
API test fixtures for AIDRA integration tests.

Provides two tiers of testing:
  - Tier 1 (always runs): Mock DB calls -- tests routing, validation, serialization.
  - Tier 2 (requires DB): Tests marked with @pytest.mark.db -- test actual SQL queries.

Key fixtures:
  - mock_db: patches src.db.connection.db with AsyncMock methods
  - test_app: FastAPI app wired with the real router but NO lifespan
  - client: httpx.AsyncClient using ASGITransport against test_app
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.router import router

# ====================================================================
# Tier 1: Mocked database fixtures (always available)
# ====================================================================


@pytest.fixture
def mock_db():
    """Monkey-patch methods on the global ``db`` singleton.

    The API modules import ``db`` via ``from src.db.connection import db``,
    which creates a direct reference to the singleton object.  Replacing
    the name in ``src.db.connection`` would not affect those references.
    Instead we patch the actual methods on the singleton so all call sites
    see the mocked behaviour.

    Original methods are restored automatically after the test.
    """
    from src.db.connection import db as real_db

    originals = {
        "fetch": real_db.fetch,
        "fetchrow": real_db.fetchrow,
        "fetchval": real_db.fetchval,
        "execute": real_db.execute,
    }

    real_db.fetch = AsyncMock(return_value=[])
    real_db.fetchrow = AsyncMock(return_value=None)
    real_db.fetchval = AsyncMock(return_value=0)
    real_db.execute = AsyncMock(return_value="INSERT 0 1")

    yield real_db

    # Restore originals
    for name, original in originals.items():
        setattr(real_db, name, original)


@pytest.fixture
def test_app(mock_db) -> FastAPI:
    """Create a lightweight FastAPI app with the real router but no lifespan.

    The lifespan is intentionally omitted to avoid requiring a running
    database, scheduler, or model files.
    """
    app = FastAPI(title="AIDRA Test")
    app.include_router(router)
    return app


@pytest.fixture
async def client(test_app) -> AsyncClient:
    """Async HTTP client bound to the test application via ASGITransport."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ====================================================================
# Helpers: fake asyncpg-like Record objects
# ====================================================================


class FakeRecord(dict):
    """Dict subclass that supports both key lookup and attribute-like .get().

    asyncpg Record objects expose data via ``row["column"]`` and
    ``row.get("column", default)``.  This class mimics that interface
    without requiring a real asyncpg connection.
    """

    def __getitem__(self, key):
        return super().__getitem__(key)

    def get(self, key, default=None):
        return super().get(key, default)


@pytest.fixture
def fake_detection_row() -> FakeRecord:
    """A single fake detection row as returned by the detections query."""
    return FakeRecord(
        id=uuid4(),
        execution_id=uuid4(),
        created_at=datetime.now(tz=UTC),
        longitude=-5.5,
        latitude=36.0,
        bbox_pixel=[100.0, 200.0, 120.0, 220.0],
        confidence=0.85,
        source="fused",
        cfar_snr=12.5,
        yolo_score=0.85,
        class_name="vessel",
        tile_index=0,
        constraint_profile="ground",
        model_name="yolov8n-sar",
        model_version="v1.0",
        image_id="S1A_TEST_001",
        quality_verdict="valid_sea_target",
    )


@pytest.fixture
def fake_execution_row() -> FakeRecord:
    """A single fake execution_log row with all columns."""
    return FakeRecord(
        id=uuid4(),
        created_at=datetime.now(tz=UTC),
        image_id="S1A_TEST_001",
        image_title="S1A_IW_GRDH_TEST",
        image_hash="abc123def456",
        image_bbox_geojson=None,
        image_sensing_date=datetime(2026, 4, 1, tzinfo=UTC),
        image_size_mb=800.0,
        search_zone="gibraltar",
        model_name="yolov8n-sar",
        model_version="v1.0",
        model_hash="def456ghi789",
        model_size_mb=6.2,
        model_format="pytorch",
        compression_technique="none",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        constraint_profile="ground",
        cpu_limit=4.0,
        memory_limit_mb=24576,
        tile_size=640,
        tile_overlap=64,
        num_detections=5,
        avg_confidence=0.78,
        max_confidence=0.92,
        min_confidence=0.55,
        total_duration_ms=3200.0,
        download_ms=1500.0,
        preprocessing_ms=500.0,
        inference_ms=150.0,
        postprocessing_ms=50.0,
        peak_ram_mb=512.0,
        avg_ram_mb=350.0,
        cpu_usage_pct=45.0,
        num_tiles=4,
        output_hash="ghi789jkl012",
        input_params_hash="jkl012mno345",
        commit_sha="0" * 40,
        status="success",
        error_message=None,
        trigger_type="manual",
        triggered_by=None,
        pipeline_version="1.0.0",
        hostname="test-host",
        notes=None,
    )


@pytest.fixture
def fake_tasking_row() -> FakeRecord:
    """A single fake tasking_queue row."""
    return FakeRecord(
        id=uuid4(),
        created_at=datetime.now(tz=UTC),
        trigger_type="cue",
        triggered_by=None,
        target_bbox_geojson='{"type":"Polygon","coordinates":[[[-5.8,35.7],[-5.2,35.7],[-5.2,36.2],[-5.8,36.2],[-5.8,35.7]]]}',
        target_zone="gibraltar",
        priority=1,
        reason="manual",
        status="pending",
        execution_id=None,
        result_status=None,
        confirmed_detections=None,
        attempts=0,
    )
