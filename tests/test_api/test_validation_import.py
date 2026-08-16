"""Tests del endpoint POST /api/validation/import (tier 1, mock_db).

El harness pesado de xView3 corre offline en una workstation; produccion
solo persiste el resultado. Las metricas derivadas se recalculan en el
servidor a partir de los conteos y la curva PR: nunca se confia en las
cifras del cliente.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.anyio


VALID_BODY = {
    "model_name": "vesseltracker-sar-yolov8",
    "iou_threshold": 0.5,
    "confidence_threshold": 0.25,
    "num_scenes": 11,
    "num_ground_truth": 1997,
    "num_predictions": 2191,
    "true_positives": 286,
    "false_positives": 1905,
    "false_negatives": 1711,
    "total_area_km2": 468575.2,
    "match_mode": "center",
    "center_tolerance_px": 20.0,
    "dataset": "xview3-sar/validation/mediterranean",
    "model_version": "v1.0",
}


async def test_import_persists_and_recomputes_metrics(client, mock_db):
    run_id = uuid4()
    mock_db.fetchrow.return_value = {"id": run_id}

    resp = await client.post("/api/validation/import", json=VALID_BODY)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["validation_run_id"] == str(run_id)
    # Recalculadas en servidor desde TP/FP/FN, no aceptadas del cliente:
    assert body["precision"] == round(286 / (286 + 1905), 4)
    assert body["pd_recall"] == round(286 / (286 + 1711), 4)
    assert body["far_per_km2"] == round(1905 / 468575.2, 4)


async def test_import_rejects_bad_match_mode(client, mock_db):
    bad = dict(VALID_BODY, match_mode="teleport")
    resp = await client.post("/api/validation/import", json=bad)
    assert resp.status_code == 422


async def test_import_requires_dataset(client, mock_db):
    bad = {k: v for k, v in VALID_BODY.items() if k != "dataset"}
    resp = await client.post("/api/validation/import", json=bad)
    assert resp.status_code == 422


async def test_import_client_metrics_are_ignored(client, mock_db):
    """Un cliente que envie precision/mAP inflados no puede colarlos."""
    mock_db.fetchrow.return_value = {"id": uuid4()}
    inflated = dict(VALID_BODY, precision=0.99, map_at_iou=0.95, pd_recall=0.99)
    resp = await client.post("/api/validation/import", json=inflated)
    assert resp.status_code == 200
    body = resp.json()
    assert body["precision"] == round(286 / (286 + 1905), 4)
    assert body["map_at_iou"] == 0.0  # sin pr_curve no hay mAP que declarar
