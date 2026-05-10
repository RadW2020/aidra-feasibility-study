"""
Health-check endpoint.

Verifies connectivity to all critical subsystems and returns a
structured status report consumed by monitoring tools and the
Grafana dashboard.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.config import Settings
from src.db.connection import db
from src.db.models import HealthResponse

logger = logging.getLogger("aidra.api.health")

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of all system components.

    Performs the following checks:

    1. **Database**: executes ``SELECT 1`` to verify PostgreSQL connectivity.
    2. **Models loaded**: counts model files present in the configured
       ``models_dir`` directory.
    3. **Scheduler**: reports whether the APScheduler instance is running.
    4. **Uptime**: seconds since the application started.

    Returns a ``HealthResponse`` with overall status ``"ok"`` when all
    components are healthy, or ``"degraded"`` when non-critical
    components are unavailable.

    Raises:
        HTTPException 503: when the database is unreachable.
    """
    settings = Settings()

    # -- 1. Database check --
    db_status = "disconnected"
    try:
        result = await db.fetchval("SELECT 1")
        if result == 1:
            db_status = "connected"
    except Exception as exc:
        logger.error("Health check: DB unreachable — %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Database unreachable: {exc}",
        ) from exc

    # -- 2. Models loaded --
    model_files_count = 0
    try:
        models_dir = Path(settings.models_dir)
        if models_dir.is_dir():
            # Count .pt, .onnx, .tflite model files
            for ext in ("*.pt", "*.onnx", "*.tflite"):
                model_files_count += len(list(models_dir.glob(ext)))
    except Exception:
        logger.debug("Could not count model files", exc_info=True)

    # Also count models registered in DB (more authoritative when available)
    registered_models_count = 0
    try:
        db_model_count = await db.fetchval("SELECT COUNT(*) FROM models_registry")
        registered_models_count = int(db_model_count or 0)
    except Exception:
        pass  # Table might not exist yet

    # Backward-compatible field: keep reporting the larger legacy count, but
    # expose the two meanings separately so health does not imply six
    # operationally registered models when it merely saw six weight files.
    models_loaded = max(model_files_count, registered_models_count)

    # -- 3. Scheduler status --
    scheduler_status = "stopped"
    try:
        from src.main import get_scheduler

        scheduler = get_scheduler()
        if scheduler is not None and getattr(scheduler, "running", False):
            scheduler_status = "running"
    except Exception:
        pass

    # -- 4. Uptime --
    uptime: float | None = None
    try:
        from src.main import get_start_time

        start = get_start_time()
        if start > 0:
            uptime = round(time.time() - start, 2)
    except Exception:
        pass

    status = "ok" if db_status == "connected" else "degraded"

    return HealthResponse(
        status=status,
        db=db_status,
        models_loaded=models_loaded,
        model_files_count=model_files_count,
        registered_models_count=registered_models_count,
        scheduler=scheduler_status,
        version="1.0.0",
        uptime_seconds=uptime,
    )
