"""Regression tests for explicit model variant resolution."""

from __future__ import annotations

from pathlib import Path

import pytest


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows

    async def fetch(self, query: str, *args):
        return self.rows

    async def fetchrow(self, query: str, *args):
        name, version = args
        for row in self.rows:
            if row["name"] == name and row["version"] == version:
                return row
        return None


@pytest.mark.asyncio
async def test_model_manager_rejects_ambiguous_base_model(tmp_path: Path):
    from src.models.manager import ModelManager

    mgr = ModelManager.__new__(ModelManager)
    mgr.db = _FakeDB(
        [
            {"name": "vesseltracker-sar-yolov8", "version": "v1.0"},
            {"name": "vesseltracker-sar-yolov8", "version": "int8-dynamic"},
        ]
    )
    mgr.models_dir = tmp_path
    mgr.max_cached_models = 2
    mgr._cache = {}
    mgr._load_order = []

    with pytest.raises(ValueError, match="Ambiguous model"):
        await mgr.get_model("vesseltracker-sar-yolov8")


def test_sar_pipeline_rejects_coco_classes():
    from src.pipeline.engine import PipelineEngine

    with pytest.raises(ValueError, match="not approved for Sentinel-1 SAR"):
        PipelineEngine._validate_model_for_sensor(
            {
                "name": "yolov8n",
                "classes": ["person", "car", "boat"],
            },
            "s1",
        )


def test_sar_pipeline_accepts_ship_class():
    from src.pipeline.engine import PipelineEngine

    PipelineEngine._validate_model_for_sensor(
        {
            "name": "vesseltracker-sar-yolov8",
            "classes": ["ship"],
        },
        "s1",
    )
