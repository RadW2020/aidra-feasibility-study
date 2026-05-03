"""
Shared pytest fixtures for the AIDRA test suite.

Provides:
- sample_sar_tile: 640x640 synthetic SAR tile with 5 vessels
- sample_detections: list of detection dicts for postprocessing tests
- mock_execution_record: dict with all execution_log fields
- settings: Settings instance with test defaults
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from src.config import Settings
from src.pipeline.preprocessing import generate_synthetic_sar_tile

# ====================================================================
# Synthetic SAR data
# ====================================================================


@pytest.fixture
def sample_sar_tile() -> tuple[np.ndarray, list[dict]]:
    """640x640 synthetic SAR tile with 5 simulated vessel signatures.

    Returns (image, ground_truth) where image is float32 and ground_truth
    is a list of dicts with bbox/center/width/height per vessel.
    """
    image, ground_truth = generate_synthetic_sar_tile(
        size=640, num_vessels=5, seed=42
    )
    return image, ground_truth


# ====================================================================
# Detection samples
# ====================================================================


@pytest.fixture
def sample_detections() -> list[dict]:
    """List of representative detection dicts for postprocessing tests."""
    return [
        {
            "bbox": [100, 200, 120, 220],
            "center": [110, 210],
            "confidence": 0.85,
            "source": "fused",
            "cfar_snr": 12.5,
            "yolo_score": 0.85,
            "tile_index": 0,
        },
        {
            "bbox": [300, 400, 310, 415],
            "center": [305, 407],
            "confidence": 0.72,
            "source": "yolo",
            "yolo_score": 0.72,
            "tile_index": 1,
        },
    ]


# ====================================================================
# Execution record
# ====================================================================


@pytest.fixture
def mock_execution_record() -> dict:
    """Dict with all execution_log fields for traceability tests."""
    return {
        "id": uuid4(),
        "created_at": datetime.now(tz=UTC),
        "image_id": "S1A_IW_GRDH_TEST_001",
        "image_title": "S1A_IW_GRDH_1SDV_20260401T000000_TEST",
        "image_hash": "abc123" * 10 + "abcd",
        "image_sensing_date": datetime(2026, 4, 1, tzinfo=UTC),
        "image_size_mb": 800.0,
        "search_zone": "gibraltar",
        "model_name": "yolov8n-sar",
        "model_version": "v1.0",
        "model_hash": "def456" * 10 + "defg",
        "model_size_mb": 6.2,
        "model_format": "pytorch",
        "compression_technique": "none",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "constraint_profile": "ground",
        "cpu_limit": 4.0,
        "memory_limit_mb": 24576,
        "tile_size": 640,
        "tile_overlap": 64,
        "num_detections": 5,
        "avg_confidence": 0.78,
        "max_confidence": 0.92,
        "min_confidence": 0.55,
        "total_duration_ms": 3200.0,
        "download_ms": 1500.0,
        "preprocessing_ms": 500.0,
        "inference_ms": 150.0,
        "postprocessing_ms": 50.0,
        "peak_ram_mb": 512.0,
        "avg_ram_mb": 350.0,
        "cpu_usage_pct": 45.0,
        "num_tiles": 4,
        "output_hash": "ghi789" * 10 + "ghij",
        "input_params_hash": "jkl012" * 10 + "jklm",
        "commit_sha": "0" * 40,
        "status": "success",
        "error_message": None,
        "trigger_type": "manual",
        "triggered_by": None,
        "pipeline_version": "1.0.0",
        "hostname": "test-host",
    }


# ====================================================================
# Settings
# ====================================================================


@pytest.fixture
def settings() -> Settings:
    """Settings instance with safe test defaults (no external connections)."""
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/aidra_test",
        copernicus_user="test_user",
        copernicus_password="test_pass",
        models_dir="/tmp/aidra_test/models",
        images_dir="/tmp/aidra_test/images",
        default_zone="gibraltar",
        default_model="yolov8n-sar",
        default_profile="ground",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        cfar_guard_size=3,
        cfar_training_size=15,
        cfar_pfa=1e-5,
        tile_size=640,
        tile_overlap=64,
        prometheus_enabled=False,
        scheduler_enabled=False,
        log_level="DEBUG",
    )
