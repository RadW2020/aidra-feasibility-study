"""
Pipeline trigger and status endpoints.

Uses the PipelineEngine singleton from main.py. The engine manages
execution IDs, recording, and cleanup internally — the API layer
only translates HTTP requests into PipelineRequest objects and
delegates to the engine.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.db.connection import db
from src.db.models import (
    PipelineStatusResponse,
    PipelineTriggerRequest,
    PipelineTriggerResponse,
)

logger = logging.getLogger("aidra.api.pipeline")

router = APIRouter(tags=["pipeline"])

# ---------------------------------------------------------------------------
# In-memory pipeline state (single-instance concurrency control)
# ---------------------------------------------------------------------------

_pipeline_lock = asyncio.Lock()

_pipeline_state: dict = {
    "running": False,
    "current_profile": None,
    "progress": None,
    "current_execution_id": None,
}

_ALL_PROFILES = ["ground", "sat-high", "sat-mid", "sat-low", "sat-extreme"]


def _get_engine():
    """Get the PipelineEngine singleton. Raises 503 if not available."""
    from src.main import get_engine

    engine = get_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline engine not available. Possible causes: "
                "no model files in models/ directory, missing dependencies, "
                "or Copernicus credentials not configured. "
                "Check logs for details."
            ),
        )
    return engine


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------


async def _run_pipeline_background(
    request: PipelineTriggerRequest,
    execution_id: UUID,
) -> None:
    """Execute a single pipeline run in the background.

    The engine receives the pre-created *execution_id* so the API
    can return it immediately.  This function manages the in-memory
    state for the ``/status`` endpoint.
    """
    global _pipeline_state

    engine = None
    try:
        from src.main import get_engine

        engine = get_engine()
        if engine is None:
            logger.error("Engine not available for background pipeline run")
            return

        from src.pipeline.engine import PipelineRequest

        pipeline_request = PipelineRequest(
            zone=request.zone,
            model=request.model,
            model_version=request.model_version,
            profile=request.profile,
            sensor=request.sensor,
            image_id=request.image_id,
            aoi_bbox=request.aoi_bbox,
            confidence_threshold=request.confidence_threshold,
            trigger_type="manual",
        )

        _pipeline_state["progress"] = 0.1

        result = await engine.run(pipeline_request, execution_id=execution_id)

        _pipeline_state["current_execution_id"] = str(result.execution_id)
        _pipeline_state["progress"] = 1.0

        logger.info(
            "Pipeline completed: execution_id=%s, detections=%d",
            result.execution_id,
            result.num_detections,
        )

    except Exception as exc:
        logger.error("Pipeline background task failed: %s", exc, exc_info=True)
    finally:
        _pipeline_state.update(
            running=False,
            current_profile=None,
            progress=None,
        )


async def _run_all_profiles_background(
    request: PipelineTriggerRequest,
    execution_ids: dict[str, UUID],
) -> None:
    """Execute pipeline with all profiles in the background.

    Uses engine.run_all_profiles() which downloads and preprocesses
    the image once, then runs detection under each profile.

    Parameters
    ----------
    request:
        The trigger request from the API.
    execution_ids:
        Pre-created execution IDs keyed by profile name so the API
        can return the mapping immediately.
    """
    global _pipeline_state

    try:
        from src.main import get_engine

        engine = get_engine()
        if engine is None:
            logger.error("Engine not available for all-profiles run")
            return

        from src.pipeline.engine import PipelineRequest

        pipeline_request = PipelineRequest(
            zone=request.zone,
            model=request.model,
            model_version=request.model_version,
            profile="ground",  # base profile, run_all_profiles overrides
            image_id=request.image_id,
            aoi_bbox=request.aoi_bbox,
            confidence_threshold=request.confidence_threshold,
            trigger_type="manual",
        )

        _pipeline_state["progress"] = 0.0

        results = await engine.run_all_profiles(
            pipeline_request,
            execution_ids=execution_ids,
        )

        logger.info(
            "All-profiles run completed: %d profiles, %d total detections",
            len(results),
            sum(r.num_detections for r in results.values() if r.status == "success"),
        )

    except Exception as exc:
        logger.error("All-profiles background task failed: %s", exc, exc_info=True)
    finally:
        _pipeline_state.update(
            running=False,
            current_profile=None,
            progress=None,
            current_execution_id=None,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/pipeline/trigger", response_model=PipelineTriggerResponse)
async def trigger_pipeline(
    request: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
) -> PipelineTriggerResponse:
    """Launch the detection pipeline in the background.

    Creates an ``execution_id`` upfront and returns it immediately so
    the caller can track progress.  Uses an asyncio lock to prevent
    two concurrent requests from both passing the running check.

    Raises:
        HTTPException 400: if the profile is unknown.
        HTTPException 409: if a pipeline is already running.
        HTTPException 503: if the engine is not available.
    """
    # Validate engine is available
    _get_engine()

    # Validate profile
    if request.profile not in _ALL_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile '{request.profile}'. Valid: {_ALL_PROFILES}",
        )

    async with _pipeline_lock:
        # Check concurrency (inside lock to prevent race condition)
        if _pipeline_state["running"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A pipeline is already running "
                    f"(profile={_pipeline_state['current_profile']}). "
                    "Wait for it to finish or check GET /pipeline/status."
                ),
            )

        # Create execution_id upfront so we can return it immediately
        execution_id = uuid4()

        # Mark running BEFORE enqueueing background task (atomically with the lock)
        _pipeline_state.update(
            running=True,
            current_profile=request.profile,
            progress=0.0,
            current_execution_id=str(execution_id),
        )

    background_tasks.add_task(_run_pipeline_background, request, execution_id)

    return PipelineTriggerResponse(
        execution_id=execution_id,
        status="started",
    )


@router.post("/pipeline/trigger-all-profiles")
async def trigger_all_profiles(
    request: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Execute the same image with ALL constraint profiles.

    Downloads the image once, then runs detection under each of the
    five profiles sequentially. Uses engine.run_all_profiles() for
    efficiency -- no redundant downloads.

    Returns a map of ``profile -> execution_id`` so the caller can
    track each profile's execution independently.

    Raises:
        HTTPException 409: if a pipeline is already running.
        HTTPException 503: if the engine is not available.
    """
    _get_engine()

    async with _pipeline_lock:
        if _pipeline_state["running"]:
            raise HTTPException(
                status_code=409,
                detail="A pipeline is already running. Wait for it to finish.",
            )

        # Create execution IDs upfront for every profile
        execution_ids: dict[str, UUID] = {profile: uuid4() for profile in _ALL_PROFILES}

        _pipeline_state.update(
            running=True,
            current_profile="all",
            progress=0.0,
            current_execution_id=None,
        )

    background_tasks.add_task(
        _run_all_profiles_background,
        request,
        execution_ids,
    )

    return {
        "status": "started",
        "profiles": _ALL_PROFILES,
        "executions": {p: str(eid) for p, eid in execution_ids.items()},
    }


@router.post("/pipeline/reset")
async def reset_pipeline_state() -> dict:
    """Release ``_pipeline_state`` when it disagrees with ``execution_log``.

    The in-memory ``_pipeline_state`` flag and the ``execution_log`` rows are
    updated by independent code paths: the background task that runs the
    pipeline owns the flag, while ``reap_orphan_executions`` only touches
    the database. If the API process is restarted mid-run, or the orphan
    reaper closes a row that the background task is still chasing, the
    flag can stay ``running=True`` indefinitely and every new
    ``/api/pipeline/trigger*`` call returns 409.

    This endpoint reconciles the two sources of truth without an SSH or
    container restart: it inspects the current ``execution_id`` (and any
    recent rows when the run was ``trigger-all-profiles``), and only
    clears the flag when no row remains in ``pending`` or ``running``.
    Returns 409 with the offending execution ids when the run is
    genuinely still active.

    Auditable by design: every call is bearer-token authenticated through
    the global write middleware and writes an INFO log line with the
    state being cleared.
    """
    if not _pipeline_state["running"]:
        return {"status": "already_idle", "cleared": False}

    eid = _pipeline_state.get("current_execution_id")
    profile = _pipeline_state.get("current_profile")

    blocking_ids: list[str] = []

    if eid:
        row = await db.fetchrow(
            "SELECT status FROM execution_log WHERE id = $1::uuid",
            eid,
        )
        if row and row["status"] in ("pending", "running"):
            blocking_ids.append(eid)
    elif profile == "all":
        # trigger-all-profiles does not pin current_execution_id; reconcile
        # by looking at the most recent batch (last 6 hours is a safe
        # upper bound — sat-extreme tops out near 2.5h, so anything older
        # is either complete or already reaped).
        rows = await db.fetch(
            "SELECT id FROM execution_log "
            "WHERE status IN ('pending', 'running') "
            "AND created_at > NOW() - INTERVAL '6 hours'",
        )
        blocking_ids.extend(str(r["id"]) for r in rows)

    if blocking_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Pipeline state is consistent with execution_log; not clearing.",
                "blocking_execution_ids": blocking_ids,
            },
        )

    prior = dict(_pipeline_state)
    _pipeline_state.update(
        running=False,
        current_profile=None,
        progress=None,
        current_execution_id=None,
    )
    logger.info("Pipeline state reset; prior state was %s", prior)
    return {"status": "cleared", "cleared": True, "prior_state": prior}


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status() -> PipelineStatusResponse:
    """Return the status of the currently running pipeline.

    When no pipeline is active, ``running`` is ``false`` and all other
    fields are ``null``.
    """
    eid = _pipeline_state.get("current_execution_id")

    return PipelineStatusResponse(
        running=_pipeline_state["running"],
        current_profile=_pipeline_state.get("current_profile"),
        progress=_pipeline_state.get("progress"),
        current_execution_id=UUID(eid) if eid else None,
    )
