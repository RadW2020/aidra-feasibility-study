"""The real AIDRA API with the heavy pipeline replaced, for agent evals only.

``uvicorn evals.api_app:app`` serves ``src.main.app`` unchanged — same routes,
middleware, auth scopes, audit log, preflight and database — plus:

* a :class:`StubEngine` in place of ``PipelineEngine`` when
  ``EVAL_ENGINE=stub``: it writes the pending -> running -> success rows through
  the real ``ExecutionRecorder`` but downloads nothing and runs no model. The
  evals test the agent interface, not SAR inference;
* ``POST /__eval__/state`` to switch the engine (stub | none) and clear
  in-process state between scenarios. Never mounted by the product.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import Request

import src.main as aidra_main
from src.config import Settings
from src.db.connection import db
from src.models.manager import ModelManager
from src.pipeline.engine import PipelineRequest, PipelineResult
from src.traceability.recorder import ExecutionRecorder

app = aidra_main.app


class StubEngine:
    """Stands in for ``PipelineEngine``: same bookkeeping, no scene, no inference."""

    def __init__(self, settings: Settings) -> None:
        self.config = settings
        self.model_manager = ModelManager(models_dir=Path(settings.models_dir), db=db)
        self.recorder = ExecutionRecorder(db)
        self.detector_engine = None

    async def run(self, request: PipelineRequest, execution_id: UUID | None = None) -> PipelineResult:
        start = time.monotonic()
        name = request.model or self.config.default_model
        version = request.model_version or (
            self.config.default_model_version if name == self.config.default_model else None
        )
        row = await db.fetchrow(
            "SELECT file_hash, size_mb, format, compression_technique FROM models_registry "
            "WHERE name = $1 AND ($2::text IS NULL OR version = $2) ORDER BY version LIMIT 1",
            name, version,
        )
        execution_id = await self.recorder.create_pending(
            image_id=request.image_id or f"S1A_EVAL_{request.zone.upper()}_LATEST",
            image_hash="pending",
            model_name=name,
            model_version=version or "v1.0",
            model_hash=row["file_hash"] if row else "eval-stub",
            model_size_mb=float(row["size_mb"]) if row else 0.0,
            model_format=row["format"] if row else "pytorch",
            compression_technique=(row["compression_technique"] if row else "none") or "none",
            search_zone=request.zone,
            confidence_threshold=request.confidence_threshold or self.config.confidence_threshold,
            iou_threshold=request.iou_threshold or self.config.iou_threshold,
            constraint_profile=request.profile,
            trigger_type=request.trigger_type,
            input_params_hash="eval-stub",
            commit_sha="eval",
            execution_id=execution_id,
        )
        await self.recorder.update_status(execution_id, "running")
        await asyncio.sleep(0.2)
        await self.recorder.update(
            execution_id=execution_id,
            status="success",
            num_detections=0,
            total_duration_ms=(time.monotonic() - start) * 1000.0,
            notes="eval stub engine: no Copernicus download, no inference",
        )
        return PipelineResult(execution_id=execution_id, status="success")


@app.post("/__eval__/state", include_in_schema=False)
async def eval_state(request: Request) -> dict[str, Any]:
    from src.api import pipeline as pipeline_api
    from src.api.auth import reset_rate_limits

    body = await request.json()
    mode = body.get("engine", "stub")
    aidra_main._engine = StubEngine(Settings()) if mode == "stub" else None
    pipeline_api._pipeline_state.update(running=False, current_profile=None, progress=None, current_execution_id=None)
    pipeline_api._EARLY_FAILURES.clear()
    reset_rate_limits()
    return {"engine": mode}
