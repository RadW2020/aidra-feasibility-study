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
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.api import idempotency
from src.api.audit import note_resource
from src.api.errors import ApiError
from src.config import Settings
from src.db.connection import db
from src.db.models import (
    PipelineStatusResponse,
    PipelineTriggerRequest,
    PipelineTriggerResponse,
)
from src.pipeline import preflight as pf

logger = logging.getLogger("aidra.api.pipeline")

router = APIRouter(tags=["pipeline"])

# ---------------------------------------------------------------------------
# Runs that died before their execution_log row existed
# ---------------------------------------------------------------------------
#
# The trigger hands out an execution_id before the background task runs. If
# the engine fails before ``create_pending`` (a model removed between
# preflight and run, say) there is no row to explain it, so the reason is kept
# here and GET /api/executions/{id} reports it instead of a bare 404.
# Process-local and bounded: it only has to outlive a client's polling window.

_EARLY_FAILURES: OrderedDict[str, dict[str, Any]] = OrderedDict()
_EARLY_FAILURES_MAX = 200


def record_early_failure(execution_id: UUID, error: str) -> None:
    _EARLY_FAILURES[str(execution_id)] = {
        "error": error,
        "failed_at": datetime.now(tz=UTC).isoformat(),
    }
    while len(_EARLY_FAILURES) > _EARLY_FAILURES_MAX:
        _EARLY_FAILURES.popitem(last=False)


def early_failure(execution_id: UUID) -> dict[str, Any] | None:
    return _EARLY_FAILURES.get(str(execution_id))


async def _note_if_never_recorded(execution_id: UUID, exc: Exception) -> None:
    try:
        exists = await db.fetchval("SELECT 1 FROM execution_log WHERE id = $1", execution_id)
    except Exception:  # noqa: BLE001
        exists = None
    if not exists:
        record_early_failure(execution_id, f"{type(exc).__name__}: {exc}")

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
        await _note_if_never_recorded(execution_id, exc)
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
        for eid in execution_ids.values():
            await _note_if_never_recorded(eid, exc)
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


def _raise_blocking(issues: list[pf.Issue], report: dict[str, Any]) -> None:
    """Turn preflight blocking issues into one structured error response."""
    first = issues[0]
    raise ApiError(
        pf.http_status_for(issues),
        first.code,
        "; ".join(i.message for i in issues),
        hint=first.hint,
        valid_values=first.valid_values,
        retryable=first.code in ("pipeline_busy",),
        blocking_issues=[i.as_dict() for i in issues],
        warnings=report["warnings"],
        in_flight=report["in_flight"],
    )


@router.post("/pipeline/preview")
async def preview_pipeline(request: PipelineTriggerRequest) -> dict[str, Any]:
    """Dry run of ``POST /pipeline/trigger``: nothing is started or written.

    Resolves what would actually run (model version, thresholds from
    ``Settings``), lists ``blocking_issues`` (unknown zone/model/profile,
    ambiguous model, non-SAR model, missing card, engine unavailable, another
    run in flight) and ``warnings`` (rejected or candidate variant, an
    equivalent run in the last 24 h), and estimates the duration from past
    runs. Public: it reads, it does not mutate.
    """
    from src.main import get_engine

    report, _issues = await pf.preflight(
        request, engine=get_engine(), settings=Settings(), api_state=_pipeline_state,
    )
    report["next_step"] = (
        "POST /api/pipeline/trigger with the same body (scope 'run'), ideally with an Idempotency-Key header."
        if report["ok"]
        else "Resolve blocking_issues first; nothing can start while any remain."
    )
    return report


@router.post("/pipeline/trigger", response_model=PipelineTriggerResponse)
async def trigger_pipeline(
    request: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> PipelineTriggerResponse | Any:
    """Launch the detection pipeline in the background.

    Runs the same preflight as ``POST /pipeline/preview`` first, so an
    ``execution_id`` is only issued for a request that can start. Refuses
    while *any* run is in flight — scheduled and Tip & Cue runs included, not
    only runs started through this API. Send ``Idempotency-Key`` to make a
    retry return the original run instead of starting another.

    Raises:
        HTTPException 400: if the profile is unknown.
        HTTPException 409: if a pipeline is already running (``pipeline_busy``).
        HTTPException 422: for any other blocking preflight issue.
        HTTPException 503: if the engine is not available.
    """
    engine = _get_engine()
    claim = await idempotency.begin(http_request, request.model_dump())
    if claim.replay is not None:
        return claim.replay

    try:
        async with _pipeline_lock:
            # Preflight inside the lock: the in-flight check and the state
            # update below must be atomic across concurrent requests.
            report, issues = await pf.preflight(
                request, engine=engine, settings=Settings(), api_state=_pipeline_state,
            )
            if issues:
                _raise_blocking(issues, report)

            execution_id = uuid4()
            _pipeline_state.update(
                running=True,
                current_profile=request.profile,
                progress=0.0,
                current_execution_id=str(execution_id),
            )
    except HTTPException:
        await idempotency.release(claim)
        raise

    # The engine gets the request exactly as sent: it resolves the same model
    # version itself, and passing the resolved value would change
    # input_params_hash for a request identical to historical ones (I-TRACE-4).
    resolved = report["resolved_request"]
    background_tasks.add_task(_run_pipeline_background, request, execution_id)
    note_resource(http_request, "execution", execution_id)

    response = PipelineTriggerResponse(
        execution_id=execution_id,
        status="started",
        resolved_request=resolved,
        warnings=report["warnings"],
        poll={
            "url": f"/api/executions/{execution_id}",
            "after_seconds": 30,
            "expected_duration_minutes": (report["estimate"] or {}).get("duration_minutes_p50"),
        },
    )
    await idempotency.complete(claim, 200, response.model_dump(mode="json"))
    return response


@router.post("/pipeline/trigger-all-profiles")
async def trigger_all_profiles(
    request: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
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
    engine = _get_engine()

    async with _pipeline_lock:
        report, issues = await pf.preflight(
            request.model_copy(update={"profile": "ground"}),
            engine=engine, settings=Settings(), api_state=_pipeline_state,
        )
        if issues:
            _raise_blocking(issues, report)

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
    note_resource(http_request, "execution_batch", ",".join(str(e) for e in execution_ids.values()))

    return {
        "status": "started",
        "profiles": _ALL_PROFILES,
        "executions": {p: str(eid) for p, eid in execution_ids.items()},
        "warnings": report["warnings"],
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
                "code": "pipeline_active",
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


def _scheduler_summary() -> dict[str, Any] | None:
    try:
        from src.main import get_scheduler

        scheduler = get_scheduler()
    except Exception:  # noqa: BLE001
        return None
    if scheduler is None:
        return {"running": False, "jobs": []}
    jobs = []
    for job in getattr(scheduler, "get_jobs", list)():
        nxt = getattr(job, "next_run_time", None)
        jobs.append({"id": job.id, "name": job.name, "next_run_time": nxt.isoformat() if nxt else None})
    return {"running": bool(getattr(scheduler, "running", False)), "jobs": jobs}


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status() -> PipelineStatusResponse:
    """What is running on the box right now.

    ``running`` / ``current_profile`` / ``progress`` describe runs started
    through this API only (in-process state; ``progress`` is coarse: 0, 0.1,
    1.0). ``in_flight`` comes from ``execution_log`` and includes scheduled
    and Tip & Cue runs; ``busy`` is true when either says a run is active —
    it is what ``POST /pipeline/trigger`` checks before refusing with 409.
    """
    from src.main import get_engine

    eid = _pipeline_state.get("current_execution_id")
    in_flight = await pf.in_flight_executions(Settings())

    return PipelineStatusResponse(
        running=_pipeline_state["running"],
        current_profile=_pipeline_state.get("current_profile"),
        progress=_pipeline_state.get("progress"),
        current_execution_id=UUID(eid) if eid else None,
        busy=bool(_pipeline_state["running"] or in_flight),
        in_flight=in_flight,
        engine_available=get_engine() is not None,
        scheduler=_scheduler_summary(),
    )
