"""
Entrypoint de la aplicacion FastAPI.

Lifespan:
- startup: conectar a DB, migraciones, logging, engine singleton, scheduler
- shutdown: cerrar conexiones, parar scheduler

Middleware:
- CORS (origenes permitidos: localhost, todos en dev)
- Request timing (header X-Process-Time)
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api import errors
from src.api.audit import audit_middleware
from src.api.auth import auth_middleware
from src.api.router import router as api_router
from src.config import Settings
from src.db.connection import db
from src.observability.loki_logger import setup_logging
from src.observability.prometheus_metrics import SYSTEM_INFO

logger = logging.getLogger("aidra.main")

# ---------------------------------------------------------------------------
# Global singletons — accessible from API handlers via get_*() functions
# ---------------------------------------------------------------------------

_start_time: float = 0.0
_scheduler: object | None = None
_engine: object | None = None  # PipelineEngine singleton


def get_start_time() -> float:
    """Return the application start timestamp (epoch seconds)."""
    return _start_time


def get_scheduler() -> object | None:
    """Return the APScheduler instance (or None if not started)."""
    return _scheduler


def get_engine():
    """Return the PipelineEngine singleton (or None if not initialized)."""
    return _engine


# ---------------------------------------------------------------------------
# Engine factory — builds the full dependency tree
# ---------------------------------------------------------------------------


def _build_engine(settings: Settings):
    """Build PipelineEngine with all dependencies.

    Returns None if any critical dependency is missing (e.g. model files,
    missing packages). The system remains operational without the engine —
    API endpoints that need it will return 503.
    """
    try:
        # CFARDetector / YOLODetector are imported transitively by
        # ModelManager._instantiate_local; we don't need them here.
        from src.pipeline.detection import DetectionEngine
        from src.pipeline.engine import PipelineEngine
        from src.pipeline.ingestion import CopernicusAuth, ImageIngester
        from src.profiles.manager import ProfileManager
        from src.traceability.recorder import ExecutionRecorder

        # Optional: Tip & Cue
        tip_evaluator = None
        if settings.tipcue_enabled:
            try:
                from src.tipcue.evaluator import TipEvaluator
                from src.tipcue.zones import get_active_zones
                tip_evaluator = TipEvaluator(
                    zones_of_interest=get_active_zones(),
                    min_confidence=settings.tipcue_min_confidence,
                    min_detections=settings.tipcue_min_detections,
                    cooldown_minutes=settings.tipcue_cooldown_minutes,
                )
                logger.info("Tip & Cue evaluator initialized")
            except Exception:
                logger.warning("Tip & Cue not available", exc_info=True)

        # Build components
        auth = CopernicusAuth(
            username=settings.copernicus_user,
            password=settings.copernicus_password,
        )
        ingester = ImageIngester(
            auth=auth,
            images_dir=Path(settings.images_dir),
            rate_limit_mbps=settings.download_rate_limit_mbps,
        )

        # Find a model file to load
        models_dir = Path(settings.models_dir)
        model_file = None
        for pattern in [f"{settings.default_model}.pt", "yolov8n.pt", "*.pt"]:
            matches = list(models_dir.glob(pattern))
            if matches:
                model_file = matches[0]
                break

        if not model_file:
            logger.warning(
                "No model files found in %s — engine disabled. "
                "Run: ./scripts/download-models.sh",
                models_dir,
            )
            return None

        from src.models.manager import ModelManager

        detector_engine = DetectionEngine(
            fusion_iou_threshold=settings.fusion_iou_threshold,
            fusion_yolo_weight=settings.fusion_yolo_weight,
            fusion_mode=settings.fusion_mode,
            fusion_center_tolerance_px=settings.fusion_center_tolerance_px,
            yolo_input=settings.yolo_input,
            edge_buffer_px=settings.edge_buffer_px,
            cfar_min_cluster_size=settings.cfar_min_cluster_size,
            cfar_cluster_eps=settings.cfar_cluster_eps,
            cfar_min_mean_snr=settings.cfar_min_mean_snr,
        )
        model_manager_instance = ModelManager(
            models_dir=Path(settings.models_dir),
            db=db,
        )
        recorder = ExecutionRecorder(db)
        profile_manager = ProfileManager()

        engine = PipelineEngine(
            ingester=ingester,
            detector_engine=detector_engine,
            model_manager=model_manager_instance,
            recorder=recorder,
            profile_manager=profile_manager,
            tip_evaluator=tip_evaluator,
            config=settings,
        )

        logger.info(
            "PipelineEngine initialized (model=%s, tip_cue=%s)",
            model_file.name,
            tip_evaluator is not None,
        )
        return engine

    except ImportError as e:
        logger.warning("PipelineEngine not available: %s", e)
        return None
    except Exception:
        logger.exception("Failed to build PipelineEngine")
        return None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle."""
    global _start_time, _scheduler, _engine

    settings = Settings()

    # --- 1. Logging ---
    setup_logging(settings)
    logger.info("AIDRA starting up...")

    # --- 2. Database pool ---
    await db.connect(settings)

    # --- 3. Migrations ---
    migrations_dir = Path(__file__).resolve().parent / "db" / "migrations"
    if migrations_dir.is_dir():
        await db.run_migrations(migrations_dir)

    # --- 3b. Cards from git -> models volume (I-MOD-4 / I-AIA-1) ---
    try:
        from src.models.cards_sync import sync_model_cards

        sync_model_cards(settings.model_cards_dist_dir, Path(settings.models_dir) / "cards")
    except Exception:
        logger.warning("Model card sync failed", exc_info=True)

    # --- 4. Scan and register models ---
    try:
        from src.models.manager import ModelManager
        model_manager = ModelManager(
            models_dir=Path(settings.models_dir),
            db=db,
        )
        registered = await model_manager.scan_and_register()
        logger.info("Model registry: %d models registered", len(registered))
    except ImportError:
        logger.debug("ModelManager not available")
    except Exception:
        logger.warning("Model scan failed", exc_info=True)

    # --- 5. Build engine singleton ---
    _engine = _build_engine(settings)

    # --- 5. APScheduler ---
    scheduler = None
    if settings.scheduler_enabled:
        try:
            from src.pipeline.scheduler_jobs import configure_scheduler
            scheduler = configure_scheduler(engine=_engine, config=settings)
            scheduler.start()
            _scheduler = scheduler
            logger.info("APScheduler started with %d jobs", len(scheduler.get_jobs()))
        except ImportError:
            logger.warning("APScheduler or scheduler_jobs not available")
        except Exception:
            logger.exception("Failed to start APScheduler")

    # --- 6. Prometheus system info ---
    _start_time = time.time()
    try:
        SYSTEM_INFO.info({
            "version": "1.0.0",
            "python": platform.python_version(),
            "arch": platform.machine(),
            "os": platform.system(),
            "hostname": platform.node(),
        })
    except Exception:
        logger.exception("Failed to register Prometheus system info")

    logger.info("AIDRA startup complete")

    yield

    # --- Shutdown ---
    logger.info("AIDRA shutting down...")
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            logger.exception("Error stopping APScheduler")
    _scheduler = None
    _engine = None

    await db.disconnect()
    logger.info("AIDRA shutdown complete")


# ====================================================================
# Application
# ====================================================================

app = FastAPI(
    title="AIDRA \u2014 AI-Enabled On-Board Data Processing Assessment",
    description=(
        "Sentinel-1 SAR vessel detection under simulated on-board hardware "
        "constraints, with full provenance (SHA256 of image, model and output) "
        "for every run.\n\n"
        "**Start here:** `GET /api/catalog` (zones, profiles, models), "
        "`GET /api/vocabulary` (what every status and verdict means), "
        "`GET /api/pipeline/status` (what is running), `GET /api/executions` "
        "(run history with outcomes).\n\n"
        "**Errors** carry `error.code` (stable), `hint`, `valid_values` and "
        "`request_id`. **Writes** need a bearer token with scope `run` or "
        "`admin`; `POST /api/pipeline/preview` is a free dry run of a trigger. "
        "See AGENTS.md in the repository."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
errors.install(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


# Middleware added last runs first. Effective order, outermost first:
#   request id -> audit -> auth (scopes, rate limit) -> timing -> CORS -> routes
# Audit sits outside auth so refused calls are recorded too; auth sets
# request.state.principal before refusing, so the audit row names the caller.
app.middleware("http")(auth_middleware)
app.middleware("http")(audit_middleware)
app.middleware("http")(errors.request_id_middleware)


app.include_router(api_router)
