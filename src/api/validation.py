"""Validation endpoints (mAP / Pd / FAR persistence).

Closes audit finding C1 (2026-05-08): the dashboards
``03-compression-bench`` and ``10-evaluator-evidence`` rendered
``'NEEDS_DB_METRIC: mAP/Pd/FAR'`` because validation results were
JSON-on-disk only. Two endpoints:

* ``POST /api/validation/synthetic`` — runs a deterministic
  synthetic-GT validation in-process (seed-controlled, 5 scenes ×
  8 vessels by default) and persists the resulting report.
* ``GET /api/validation/runs`` — list rows from ``validation_runs``
  for use by dashboards and external consumers.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.validation.persistence import list_validation_runs, persist_report
from src.validation.synthetic import run_synthetic_validation

logger = logging.getLogger("aidra.api.validation")

router = APIRouter(prefix="/validation", tags=["validation"])


class SyntheticValidationRequest(BaseModel):
    num_scenes: int = Field(5, ge=1, le=50)
    num_vessels: int = Field(8, ge=1, le=64)
    tile_size: int = Field(640, ge=128, le=2048)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    model: str | None = Field(
        default=None,
        description=(
            "Model name to validate. Defaults to settings.default_model "
            "when omitted."
        ),
    )


@router.post("/synthetic")
async def run_synthetic(
    request: SyntheticValidationRequest,
) -> dict[str, Any]:
    """Run synthetic-GT validation end-to-end and persist the result.

    Internally generates ``num_scenes`` synthetic SAR tiles with known
    vessel positions (controlled by ``seed``), runs CFAR + the
    requested YOLO model against each, computes mAP / Pd / FAR /
    precision exactly as ``scripts/run_validation.py`` would on a real
    manifest, and inserts a ``validation_runs`` row with
    ``dataset='synthetic-seed-<seed>'``. Returns the persisted row's
    summary so callers can echo it directly.
    """
    from src.config import Settings
    from src.main import get_engine

    engine = get_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline engine not available — model files missing or "
                "lifespan failed to start."
            ),
        )

    settings = Settings()
    model_name = request.model or settings.default_model
    try:
        detector = await engine.model_manager.get_model(
            name=model_name,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    info = detector.get_model_info()

    report = await run_synthetic_validation(
        yolo_detector=detector,
        detection_engine=engine.detector_engine,
        num_scenes=request.num_scenes,
        num_vessels=request.num_vessels,
        tile_size=request.tile_size,
        seed=request.seed,
        iou_threshold=request.iou_threshold,
        confidence_threshold=request.confidence_threshold,
    )

    dataset = f"synthetic-seed-{request.seed}"
    new_id = await persist_report(
        report,
        dataset=dataset,
        model_version=str(info.get("version", "unknown")),
        model_hash=info.get("hash"),
        compression_technique=str(info.get("compression_technique", "none")),
        notes=(
            f"synthetic ground truth: num_scenes={request.num_scenes}, "
            f"num_vessels={request.num_vessels}, tile_size={request.tile_size}, "
            f"seed={request.seed}"
        ),
    )

    summary = report.as_dict()
    summary["validation_run_id"] = str(new_id)
    summary["dataset"] = dataset
    return summary


@router.get("/runs")
async def list_runs(
    model_name: str | None = None,
    compression_technique: str | None = None,
    dataset: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List validation_runs rows ordered by created_at DESC."""
    if not 1 <= limit <= 500:
        raise HTTPException(
            status_code=400, detail="limit must be in [1, 500]"
        )
    rows = await list_validation_runs(
        model_name=model_name,
        compression_technique=compression_technique,
        dataset=dataset,
        limit=limit,
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        rec: dict[str, Any] = dict(r)
        if rec.get("created_at") is not None:
            rec["created_at"] = rec["created_at"].isoformat()
        out.append(rec)
    return out
