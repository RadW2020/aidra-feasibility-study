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
    match_mode: str = Field("center", pattern="^(iou|center)$")
    center_tolerance_px: float = Field(20.0, ge=0.0, le=512.0)
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    model: str | None = Field(
        default=None,
        description=("Model name to validate. Defaults to settings.default_model when omitted."),
    )
    model_version: str | None = Field(
        default=None,
        description=(
            "Exact model version/variant. Defaults to "
            "settings.default_model_version for the default model."
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
                "Pipeline engine not available — model files missing or lifespan failed to start."
            ),
        )

    settings = Settings()
    model_name = request.model or settings.default_model
    model_version = request.model_version
    if model_name == settings.default_model and model_version is None:
        model_version = settings.default_model_version
    try:
        detector = await engine.model_manager.get_model(
            name=model_name,
            version=model_version,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    info = detector.get_model_info()
    try:
        engine._validate_model_for_sensor(info, "s1")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report = await run_synthetic_validation(
        yolo_detector=detector,
        detection_engine=engine.detector_engine,
        num_scenes=request.num_scenes,
        num_vessels=request.num_vessels,
        tile_size=request.tile_size,
        seed=request.seed,
        iou_threshold=request.iou_threshold,
        match_mode=request.match_mode,
        center_tolerance_px=request.center_tolerance_px,
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
        raise HTTPException(status_code=400, detail="limit must be in [1, 500]")
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


class ImportValidationRequest(BaseModel):
    """Externally computed validation report to persist.

    Mirror of ``src.validation.metrics.ValidationReport`` plus provenance
    fields. Covers the case where the heavy harness runs offline (e.g.
    ``scripts/validate_xview3_serial.py`` against real xView3-SAR scenes
    on a workstation) and production only needs the outcome persisted so
    dashboards stop falling back to synthetic-only metrics.
    Derived metrics (precision/Pd/FAR/mAP) are recomputed server-side
    from the counts and PR curve, never trusted from the client.
    """

    model_name: str
    iou_threshold: float = Field(ge=0.0, le=1.0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    num_scenes: int = Field(ge=1)
    num_ground_truth: int = Field(ge=0)
    num_predictions: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    total_area_km2: float = Field(ge=0.0)
    pr_curve: list[dict[str, float]] = Field(default_factory=list)
    match_mode: str = Field("center", pattern="^(iou|center)$")
    center_tolerance_px: float = Field(20.0, ge=0.0, le=512.0)
    dataset: str = Field(min_length=1)
    dataset_split: str | None = None
    model_version: str = "unknown"
    model_hash: str | None = None
    compression_technique: str = "none"
    notes: str | None = None


@router.post("/import")
async def import_report(request: ImportValidationRequest) -> dict[str, Any]:
    """Persist an externally computed validation report.

    Protected by the same bearer-token middleware as every state-changing
    ``/api`` route. Returns the persisted row id plus the server-side
    recomputed metrics.
    """
    from src.validation.metrics import ValidationReport

    report = ValidationReport(
        model_name=request.model_name,
        iou_threshold=request.iou_threshold,
        confidence_threshold=request.confidence_threshold,
        num_scenes=request.num_scenes,
        num_ground_truth=request.num_ground_truth,
        num_predictions=request.num_predictions,
        true_positives=request.true_positives,
        false_positives=request.false_positives,
        false_negatives=request.false_negatives,
        total_area_km2=request.total_area_km2,
        pr_curve=request.pr_curve,
        match_mode=request.match_mode,
        center_tolerance_px=request.center_tolerance_px,
    )
    new_id = await persist_report(
        report,
        dataset=request.dataset,
        dataset_split=request.dataset_split,
        model_version=request.model_version,
        model_hash=request.model_hash,
        compression_technique=request.compression_technique,
        notes=request.notes,
    )
    logger.info(
        "Imported external validation report",
        extra={
            "validation_run_id": str(new_id),
            "dataset": request.dataset,
            "model": request.model_name,
        },
    )
    return {
        "validation_run_id": str(new_id),
        "dataset": request.dataset,
        "model_name": request.model_name,
        "precision": round(report.precision, 4),
        "pd_recall": round(report.pd_recall, 4),
        "far_per_km2": round(report.far_per_km2, 4),
        "map_at_iou": round(report.map_at_iou, 4),
    }
