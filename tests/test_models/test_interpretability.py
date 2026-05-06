"""
Tests for src/models/interpretability.py orchestrator.

Locks in the manifest schema with separate execution-model and
gradcam-model identities, so future readers (auditors, the AI Act
declaration) can distinguish the *subject* of the explanation from the
*renderer* of the heatmap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from PIL import Image


def _png_bytes_for(tmp_path: Path, name: str) -> Path:
    """Create a tiny valid PNG so PIL can load it."""
    p = tmp_path / name
    arr = (np.random.default_rng(0).random((32, 32)) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(p)
    return p


@pytest.fixture
def fake_db():
    db = MagicMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock()
    return db


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Models dir with one FP32 PT and one INT8 ONNX present.

    The orchestrator must pick the FP32 PT for Grad-CAM, even though the
    source execution ran the INT8 variant.
    """
    d = tmp_path / "models"
    d.mkdir()
    (d / "vesseltracker-sar-yolov8.pt").write_bytes(b"fake-fp32-pt-bytes" * 100)
    (d / "vesseltracker-sar-yolov8-int8-dynamic.onnx").write_bytes(b"fake-int8")
    return d


async def test_manifest_records_both_model_identities(
    fake_db, models_dir: Path, tmp_path: Path
):
    """The manifest must include execution_model_* AND gradcam_model_*."""
    from src.models.interpretability import run_interpretability_for_execution

    exec_id = uuid4()
    # Production case: execution_log records the BASE model name plus the
    # hash of the *actual* file used (here INT8). The orchestrator must
    # surface both: base name + INT8 hash on the execution side, FP32 PT
    # name + hash on the gradcam side.
    fake_db.fetchrow.return_value = {
        "model_name": "vesseltracker-sar-yolov8",
        "model_hash": "ea0ee6da" + "0" * 56,
    }

    thumb = _png_bytes_for(tmp_path, "thumb.png")
    fake_db.fetch.return_value = [
        {
            "id": uuid4(),
            "thumbnail_path": str(thumb),
            "confidence": 0.9,
            "source": "fused",
        }
    ]

    out_root = tmp_path / "out"
    fake_yolo = MagicMock()

    with patch(
        "src.models.interpretability.gradcam_yolov8",
        return_value=np.zeros((32, 32), dtype=np.float32),
    ), patch(
        "src.models.interpretability.cfar_score_map",
        return_value=np.zeros((32, 32), dtype=np.float32),
    ), patch("ultralytics.YOLO", return_value=fake_yolo):
        result = await run_interpretability_for_execution(
            db=fake_db,
            models_dir=models_dir,
            out_root=out_root,
            execution_id=exec_id,
            n_samples=1,
        )

    # Return dict carries both identities.
    assert result["execution_model_name"] == "vesseltracker-sar-yolov8"
    assert result["execution_model_hash"].startswith("ea0ee6da")
    assert result["gradcam_model_name"] == "vesseltracker-sar-yolov8"
    expected_pt_hash = hashlib.sha256(
        (models_dir / "vesseltracker-sar-yolov8.pt").read_bytes()
    ).hexdigest()
    assert result["gradcam_model_hash"] == expected_pt_hash

    # And so does the on-disk manifest.
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["execution_model_name"] == "vesseltracker-sar-yolov8"
    assert manifest["execution_model_hash"].startswith("ea0ee6da")
    assert manifest["gradcam_model_name"] == "vesseltracker-sar-yolov8"
    assert manifest["gradcam_model_hash"] == expected_pt_hash
    # Crucial: execution_model_hash and gradcam_model_hash must differ
    # — that is the whole point of separating them.
    assert manifest["execution_model_hash"] != manifest["gradcam_model_hash"]


async def test_orchestrator_skips_int8_pt_filter(
    fake_db, tmp_path: Path
):
    """When only INT8/pruned PTs exist, RuntimeError is raised — the
    filter that excludes them is the I-AIA-2 fix and must hold."""
    from src.models.interpretability import run_interpretability_for_execution

    md = tmp_path / "models"
    md.mkdir()
    # Only INT8 / pruned variants on disk — no usable FP32 baseline.
    (md / "vesseltracker-sar-yolov8-int8-dynamic.pt").write_bytes(b"int8")
    (md / "vesseltracker-sar-yolov8-pruned30.pt").write_bytes(b"pruned")

    exec_id = uuid4()
    fake_db.fetchrow.return_value = {
        "model_name": "vesseltracker-sar-yolov8",
        "model_hash": "h",
    }
    thumb = _png_bytes_for(tmp_path, "thumb.png")
    fake_db.fetch.return_value = [
        {"id": uuid4(), "thumbnail_path": str(thumb), "confidence": 0.9, "source": "fused"}
    ]

    with pytest.raises(RuntimeError, match="No .pt baseline"):
        await run_interpretability_for_execution(
            db=fake_db,
            models_dir=md,
            out_root=tmp_path / "out",
            execution_id=exec_id,
            n_samples=1,
        )


async def test_no_thumbnails_raises(fake_db, models_dir: Path, tmp_path: Path):
    from src.models.interpretability import run_interpretability_for_execution

    fake_db.fetchrow.return_value = {
        "model_name": "vesseltracker-sar-yolov8",
        "model_hash": "h",
    }
    fake_db.fetch.return_value = []

    with pytest.raises(RuntimeError, match="No detections with thumbnails"):
        await run_interpretability_for_execution(
            db=fake_db,
            models_dir=models_dir,
            out_root=tmp_path / "out",
            execution_id=UUID(int=0),
            n_samples=5,
        )
