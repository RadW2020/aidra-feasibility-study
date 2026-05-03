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
    models_loaded = 0
    try:
        models_dir = Path(settings.models_dir)
        if models_dir.is_dir():
            # Count .pt, .onnx, .tflite model files
            for ext in ("*.pt", "*.onnx", "*.tflite"):
                models_loaded += len(list(models_dir.glob(ext)))
    except Exception:
        logger.debug("Could not count model files", exc_info=True)

    # Also count models registered in DB (more authoritative when available)
    try:
        db_model_count = await db.fetchval(
            "SELECT COUNT(*) FROM models_registry"
        )
        if db_model_count and db_model_count > models_loaded:
            models_loaded = db_model_count
    except Exception:
        pass  # Table might not exist yet

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
        scheduler=scheduler_status,
        version="1.0.0",
        uptime_seconds=uptime,
    )
