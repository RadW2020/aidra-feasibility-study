"""
Jobs programados con APScheduler (in-process, sin broker externo).

Jobs definidos:
1. scheduled_scan    -- Ejecuta pipeline en zonas predefinidas periodicamente
2. process_pending_cues -- Procesa cues pendientes en tasking_queue
3. cleanup_old_images -- Limpia imagenes temporales antiguas
4. health_probe      -- Verifica conectividad con Copernicus y DB

Dependencias:
- APScheduler >= 3.10
- PipelineEngine (src.pipeline.engine)
- Settings (src.config)

Usage:
    from src.pipeline.scheduler_jobs import configure_scheduler

    scheduler = configure_scheduler(engine=engine, config=settings)
    scheduler.start()
    # ... on shutdown:
    scheduler.shutdown(wait=False)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import Settings
from src.db.connection import db
from src.db.queries import SELECT_PENDING_CUES, UPDATE_CUE_STATUS
from src.observability.prometheus_metrics import ACTIVE_CUES, CUES_EXECUTED_TOTAL
from src.pipeline.engine import PipelineEngine, PipelineRequest

if TYPE_CHECKING:
    pass

logger = logging.getLogger("aidra.scheduler")


# ====================================================================
# Scheduler configuration
# ====================================================================


def configure_scheduler(
    engine: PipelineEngine | None,
    config: Settings,
) -> AsyncIOScheduler:
    """Create and configure the APScheduler with all recurring jobs.

    Does **not** start the scheduler -- the caller must invoke
    ``scheduler.start()`` after the event loop is running.

    If ``engine`` is None (e.g. missing model files at startup),
    pipeline-dependent jobs (scan, cues) are skipped. Cleanup and
    health probe jobs are always registered.

    Parameters
    ----------
    engine:
        The pipeline engine instance, or None if not available.
    config:
        Application settings controlling intervals and feature flags.

    Returns
    -------
    AsyncIOScheduler
        Configured scheduler (not yet started).
    """
    scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "misfire_grace_time": 3600,
        },
    )

    # Job 1: Scheduled zone scan (requires engine)
    if engine is not None:
        scheduler.add_job(
            scheduled_scan,
            trigger=IntervalTrigger(hours=config.scheduler_interval_hours),
            kwargs={"engine": engine, "zone": config.default_zone},
            id="scheduled_scan",
            name="Scheduled zone scan",
            max_instances=1,
            misfire_grace_time=3600,
        )
    else:
        logger.warning("Scheduled scan job skipped — engine not available")

    # Job 2: Tip & Cue processor (requires engine)
    if config.tipcue_enabled and engine is not None:
        scheduler.add_job(
            process_pending_cues,
            trigger=IntervalTrigger(minutes=15),
            kwargs={"engine": engine},
            id="cue_processor",
            name="Tip & Cue processor",
            max_instances=1,
        )
    elif config.tipcue_enabled:
        logger.warning("Cue processor job skipped — engine not available")

    # Job 3: Cleanup old images
    # Borra imagenes descargadas hace mas de 24h (ejecuta a las 3:00 AM)
    scheduler.add_job(
        cleanup_old_images,
        trigger=CronTrigger(hour=3, minute=0),
        kwargs={"images_dir": config.images_dir, "max_age_hours": 24},
        id="cleanup_images",
        name="Image cleanup",
    )

    # Job 4: Health probe
    # Verifica Copernicus + DB cada 30 minutos
    scheduler.add_job(
        health_probe,
        trigger=IntervalTrigger(minutes=30),
        kwargs={"config": config},
        id="health_probe",
        name="System health probe",
    )

    logger.info(
        "Scheduler configured with %d jobs",
        len(scheduler.get_jobs()),
    )
    return scheduler


# ====================================================================
# Job functions
# ====================================================================


async def scheduled_scan(engine: PipelineEngine, zone: str) -> None:
    """Execute the detection pipeline on the specified zone.

    This job is triggered periodically by the scheduler.  It creates
    a :class:`PipelineRequest` with ``trigger_type="scheduled"`` and
    runs the full pipeline.

    Parameters
    ----------
    engine:
        Pipeline engine to use for the scan.
    zone:
        Search zone key (e.g. ``"gibraltar"``).
    """
    logger.info("Scheduled scan starting for zone '%s'", zone)
    start = time.monotonic()

    try:
        request = PipelineRequest(
            zone=zone,
            model=engine.config.default_model,
            profile=engine.config.default_profile,
            trigger_type="scheduled",
        )
        result = await engine.run(request)
        elapsed = time.monotonic() - start
        logger.info(
            "Scheduled scan completed: zone='%s', detections=%d, %.1fs",
            zone,
            result.num_detections,
            elapsed,
        )
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception(
            "Scheduled scan failed: zone='%s', %.1fs elapsed",
            zone,
            elapsed,
        )


async def process_pending_cues(engine: PipelineEngine) -> None:
    """Process pending cues from the tasking_queue, ordered by priority.

    Steps for each cue:
    1. Fetch pending cues from the database (priority DESC).
    2. For each cue, run the pipeline with ``trigger_type="cue"``
       targeting the cue's bounding box.
    3. Update the cue record with the execution result.

    Parameters
    ----------
    engine:
        Pipeline engine to use for cue execution.
    """
    logger.info("Processing pending cues")

    try:
        rows = await db.fetch(SELECT_PENDING_CUES, 10)
    except Exception:
        logger.exception("Failed to fetch pending cues from database")
        return

    if not rows:
        logger.debug("No pending cues to process")
        ACTIVE_CUES.set(0)
        return

    ACTIVE_CUES.set(len(rows))
    logger.info("Found %d pending cues to process", len(rows))

    for row in rows:
        cue_id: UUID = row["id"]
        target_zone: str | None = row.get("target_zone")
        triggered_by: UUID | None = row.get("triggered_by")
        priority: int = row.get("priority", 0)

        # Parse target bbox from GeoJSON
        target_bbox_geojson: str | None = row.get("target_bbox_geojson")
        aoi_bbox: list[float] | None = None
        if target_bbox_geojson:
            try:
                import json
                geojson = json.loads(target_bbox_geojson)
                coords = geojson.get("coordinates", [[]])[0]
                if coords and len(coords) >= 4:
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    aoi_bbox = [min(lons), min(lats), max(lons), max(lats)]
            except (json.JSONDecodeError, KeyError, IndexError):
                logger.warning(
                    "Failed to parse target_bbox GeoJSON for cue %s",
                    cue_id,
                )

        logger.info(
            "Processing cue %s (priority=%d, zone=%s)",
            cue_id,
            priority,
            target_zone,
        )

        try:
            request = PipelineRequest(
                zone=target_zone or engine.config.default_zone,
                model=engine.config.default_model,
                profile=engine.config.default_profile,
                aoi_bbox=aoi_bbox,
                trigger_type="cue",
                triggered_by=triggered_by,
            )
            result = await engine.run(request)

            # Update cue with result
            await db.execute(
                UPDATE_CUE_STATUS,
                cue_id,
                "completed",
                result.execution_id,
                result.status,
                result.num_detections,
            )

            CUES_EXECUTED_TOTAL.labels(status="confirmed").inc()
            logger.info(
                "Cue %s completed: detections=%d",
                cue_id,
                result.num_detections,
            )

        except Exception:
            # Mark cue as failed but keep it for retry
            try:
                await db.execute(
                    UPDATE_CUE_STATUS,
                    cue_id,
                    "pending",
                    None,
                    "error",
                    None,
                )
            except Exception:
                logger.exception(
                    "Failed to update cue %s status after error", cue_id
                )

            CUES_EXECUTED_TOTAL.labels(status="discarded").inc()
            logger.exception("Cue %s processing failed", cue_id)


async def cleanup_old_images(
    images_dir: str,
    max_age_hours: int = 24,
) -> None:
    """Delete downloaded satellite images older than max_age_hours.

    Scans the images directory for ``.zip`` files and extracted
    directories, removing any that have not been modified within
    the retention window.

    Parameters
    ----------
    images_dir:
        Path to the directory containing downloaded images.
    max_age_hours:
        Maximum age in hours before a file is eligible for deletion.
    """
    import shutil

    images_path = Path(images_dir)
    if not images_path.exists():
        logger.debug("Images directory does not exist: %s", images_dir)
        return

    now = time.time()
    max_age_seconds = max_age_hours * 3600
    removed_count = 0
    freed_mb = 0.0

    for item in images_path.iterdir():
        try:
            mtime = item.stat().st_mtime
            age_seconds = now - mtime

            if age_seconds < max_age_seconds:
                continue

            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                item.unlink()
                freed_mb += size_mb
                removed_count += 1
            elif item.is_dir():
                # Compute directory size before removal
                dir_size = sum(
                    f.stat().st_size
                    for f in item.rglob("*")
                    if f.is_file()
                )
                shutil.rmtree(item)
                freed_mb += dir_size / (1024 * 1024)
                removed_count += 1

        except OSError:
            logger.warning(
                "Failed to remove old image: %s", item, exc_info=True
            )

    logger.info(
        "Image cleanup completed: removed %d items, freed %.1f MB",
        removed_count,
        freed_mb,
    )


async def health_probe(config: Settings) -> None:
    """Verify connectivity with external services.

    Checks:
    1. Copernicus Data Space API -- HTTP GET to the OData catalogue.
    2. PostgreSQL database -- simple query via the connection pool.

    Results are logged at INFO level.  Failures are logged as
    warnings but do not raise exceptions (the scheduler should
    continue running).

    Parameters
    ----------
    config:
        Application settings (used for service URLs).
    """
    results: dict[str, str] = {}

    # Check 1: Copernicus API reachability
    copernicus_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(copernicus_url)
            if resp.status_code < 500:
                results["copernicus"] = "ok"
            else:
                results["copernicus"] = f"degraded (HTTP {resp.status_code})"
    except httpx.HTTPError as exc:
        results["copernicus"] = f"unreachable ({type(exc).__name__})"
    except Exception as exc:
        results["copernicus"] = f"error ({type(exc).__name__})"

    # Check 2: Database connectivity
    try:
        val = await db.fetchval("SELECT 1")
        results["database"] = "ok" if val == 1 else "unexpected"
    except RuntimeError:
        results["database"] = "not_connected"
    except Exception as exc:
        results["database"] = f"error ({type(exc).__name__})"

    # Log results
    all_ok = all(v == "ok" for v in results.values())
    level = logging.INFO if all_ok else logging.WARNING
    logger.log(
        level,
        "Health probe: %s",
        ", ".join(f"{k}={v}" for k, v in results.items()),
    )
