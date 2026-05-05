"""
Interpretability endpoints (D4 annex).

Exposes ``POST /api/interpretability/run`` so the Grad-CAM + CFAR
heatmap pipeline can be triggered without SSH'ing into the container.
The handler delegates to
``src.models.interpretability.run_interpretability_for_execution``,
the same function the CLI script ``scripts/run_interpretability.py``
calls — single source of truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import Settings
from src.db.connection import db

logger = logging.getLogger("aidra.api.interpretability")

router = APIRouter(prefix="/interpretability", tags=["interpretability"])


class InterpretabilityRunRequest(BaseModel):
    execution_id: UUID | None = None
    n_samples: int = 20
    model: str | None = None
    out_dir: str = "/data/interpretability"


@router.post("/run")
async def run_interpretability(req: InterpretabilityRunRequest) -> dict:
    """Trigger a Grad-CAM + CFAR run for the given execution.

    If ``execution_id`` is omitted, picks the most recent successful
    execution that has detections. If ``model`` is omitted, uses the
    model name recorded on the execution row.

    Returns the run_id, manifest path, and OK counts so the caller can
    verify completeness without listing the output directory.
    """
    from src.models.interpretability import run_interpretability_for_execution

    settings = Settings()
    try:
        result = await run_interpretability_for_execution(
            db=db,
            models_dir=Path(settings.models_dir),
            out_root=Path(req.out_dir),
            execution_id=req.execution_id,
            n_samples=req.n_samples,
            model_name=req.model,
        )
    except RuntimeError as exc:
        # Caller-recoverable: missing execution / no PT model on disk.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Interpretability run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
