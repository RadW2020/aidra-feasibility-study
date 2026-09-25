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

import json
import logging
import re
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import Settings
from src.db.connection import db

logger = logging.getLogger("aidra.api.interpretability")

router = APIRouter(prefix="/interpretability", tags=["interpretability"])


class InterpretabilityRunRequest(BaseModel):
    execution_id: UUID | None = None
    n_samples: int = 20
    model: str | None = None
    # Confined to Settings.interpretability_dir; None = that directory.
    out_dir: str | None = None


@router.post("/run")
async def run_interpretability(req: InterpretabilityRunRequest) -> dict:
    """Trigger a Grad-CAM + CFAR run for the given execution.

    If ``execution_id`` is omitted, picks the most recent successful
    execution that has detections. If ``model`` is omitted, uses the
    model name recorded on the execution row.

    Returns the run_id, manifest path, and OK counts so the caller can
    verify completeness without listing the output directory.
    """
    from src.api.auth import confined_dir
    from src.models.interpretability import run_interpretability_for_execution

    settings = Settings()
    out_root = confined_dir(req.out_dir, settings.interpretability_dir)
    try:
        result = await run_interpretability_for_execution(
            db=db,
            models_dir=Path(settings.models_dir),
            out_root=out_root,
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


# ---------------------------------------------------------------------------
# Read-only access to the generated runs (mirror evidence without SSH)
# ---------------------------------------------------------------------------

_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{8,120}$")
_FILE = re.compile(r"^[A-Za-z0-9_.-]{1,160}\.(png|json)$")


def _runs_root() -> Path:
    return Path(Settings().interpretability_dir)


def _run_dir(run_id: str) -> Path:
    if not _RUN_ID.match(run_id):
        raise HTTPException(status_code=422, detail="invalid run_id")
    d = _runs_root() / run_id
    if not d.is_dir() or not (d / "manifest.json").exists():
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return d


@router.get("/runs")
async def list_runs(limit: int = 50) -> list[dict]:
    """Runs under the interpretability root, newest first, with manifest headline fields."""
    root = _runs_root()
    if not root.is_dir():
        return []
    out: list[dict] = []
    for d in sorted((p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()),
                    key=lambda p: p.stat().st_mtime, reverse=True)[: max(1, min(limit, 500))]:
        try:
            m = json.loads((d / "manifest.json").read_text())
        except Exception:
            continue
        out.append({
            "run_id": d.name,
            "created_at": m.get("created_at"),
            "execution_id": m.get("execution_id"),
            "commit_sha": m.get("commit_sha"),
            "n_samples": m.get("n_samples"),
            "gradcam_layer": m.get("gradcam_layer"),
            "gradcam_target": m.get("gradcam_target"),
            "sampling": (m.get("sampling") or {}).get("strategy"),
            "files": sum(1 for _ in d.glob("*.png")),
        })
    return out


@router.get("/runs/{run_id}/manifest")
async def get_manifest(run_id: str) -> dict:
    return json.loads((_run_dir(run_id) / "manifest.json").read_text())


@router.get("/runs/{run_id}/files/{name}")
async def get_run_file(run_id: str, name: str) -> FileResponse:
    if not _FILE.match(name):
        raise HTTPException(status_code=422, detail="invalid file name")
    f = _run_dir(run_id) / name
    if not f.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, media_type="image/png" if name.endswith(".png") else "application/json")
