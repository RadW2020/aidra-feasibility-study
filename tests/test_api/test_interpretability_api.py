"""
Tests for /api/interpretability endpoints.

Tier 1 (mocked orchestrator): validates routing, request shape, and
error mapping. Heavy lifting (DB queries, YOLO load, Grad-CAM) is
covered by the unit tests in tests/test_models/test_interpretability.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4


async def test_post_run_returns_summary(client, mock_db, tmp_path):
    fake_result = {
        "run_id": "abc_interp_deadbeef",
        "execution_id": str(uuid4()),
        "manifest_path": str(tmp_path / "manifest.json"),
        "n_samples": 20,
        "gradcam_ok": 20,
        "cfar_ok": 20,
        "execution_model_name": "vesseltracker-sar-yolov8-int8-dynamic",
        "execution_model_hash": "ea0ee6da" + "0" * 56,
        "gradcam_model_name": "vesseltracker-sar-yolov8",
        "gradcam_model_hash": "f" * 64,
    }

    with patch(
        "src.models.interpretability.run_interpretability_for_execution",
        new=AsyncMock(return_value=fake_result),
    ):
        resp = await client.post(
            "/api/interpretability/run",
            json={"n_samples": 20, "out_dir": str(tmp_path)},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "abc_interp_deadbeef"
    assert body["gradcam_ok"] == 20
    # Both model identities must surface — subject of explanation vs renderer.
    assert body["execution_model_hash"].startswith("ea0ee6da")
    assert body["gradcam_model_name"] == "vesseltracker-sar-yolov8"
    assert body["gradcam_model_hash"] == "f" * 64


async def test_post_run_propagates_runtime_error_as_400(client, mock_db, tmp_path):
    with patch(
        "src.models.interpretability.run_interpretability_for_execution",
        new=AsyncMock(side_effect=RuntimeError("No detections with thumbnails available.")),
    ):
        resp = await client.post(
            "/api/interpretability/run",
            json={"out_dir": str(tmp_path)},
        )

    assert resp.status_code == 400
    assert "thumbnails" in resp.json()["detail"]


async def test_post_run_unexpected_exception_returns_500(client, mock_db, tmp_path):
    with patch(
        "src.models.interpretability.run_interpretability_for_execution",
        new=AsyncMock(side_effect=ValueError("boom")),
    ):
        resp = await client.post(
            "/api/interpretability/run",
            json={"out_dir": str(tmp_path)},
        )

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


async def test_post_run_passes_filters_through(client, mock_db, tmp_path):
    exec_id = uuid4()
    fake_result = {
        "run_id": "x",
        "execution_id": str(exec_id),
        "manifest_path": "x",
        "n_samples": 5,
        "gradcam_ok": 5,
        "cfar_ok": 5,
        "execution_model_name": "m",
        "execution_model_hash": "h",
        "gradcam_model_name": "m",
        "gradcam_model_hash": "h",
    }
    target = AsyncMock(return_value=fake_result)

    with patch(
        "src.models.interpretability.run_interpretability_for_execution",
        new=target,
    ):
        resp = await client.post(
            "/api/interpretability/run",
            json={
                "execution_id": str(exec_id),
                "n_samples": 5,
                "model": "vesseltracker-sar-yolov8",
                "out_dir": str(tmp_path),
            },
        )

    assert resp.status_code == 200
    kwargs = target.call_args.kwargs
    assert kwargs["n_samples"] == 5
    assert kwargs["model_name"] == "vesseltracker-sar-yolov8"
    assert str(kwargs["execution_id"]) == str(exec_id)
