"""Preflight for a pipeline run: resolve the request and find what would stop it.

``POST /api/pipeline/preview`` returns this report as a dry run and
``POST /api/pipeline/trigger`` refuses to start when it has blocking issues.
Both use the same checks, and the checks call the code that gates a real run
(``ModelManager`` file and card lookups, ``PipelineEngine``'s SAR gate), so
the preview can't drift from what the engine would do.

Before this existed the trigger returned ``status: started`` and a fresh
``execution_id`` *before* validating zone or model. A bad request then failed
inside the background task before ``create_pending``, so no row ever
appeared and the id 404'd forever.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import Settings
from src.db.connection import db
from src.db.queries import SELECT_IN_FLIGHT_EXECUTIONS
from src.pipeline.ingestion import SEARCH_ZONES
from src.profiles.definitions import PROFILES

logger = logging.getLogger("aidra.pipeline.preflight")

SENSORS = ("s1", "s2")

_SELECT_MODEL_ROWS = """
    SELECT name, version, format, file_path, compression_technique, classes,
           status, rejection_reason
    FROM models_registry
    WHERE name = $1
    ORDER BY version
"""

_SELECT_MODEL_NAMES = "SELECT DISTINCT name FROM models_registry ORDER BY name"

_SELECT_DURATION_HISTORY = """
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_duration_ms) AS p50_ms,
           MAX(total_duration_ms) AS max_ms,
           COUNT(*) AS runs
    FROM execution_log
    WHERE status = 'success'
      AND model_name = $1 AND model_version = $2 AND constraint_profile = $3
      AND total_duration_ms IS NOT NULL
      AND created_at > NOW() - INTERVAL '120 days'
"""

_SELECT_LAST_TERMINAL_ON_PROFILE = """
    SELECT id, created_at, status, peak_ram_mb, error_message
    FROM execution_log
    WHERE model_name = $1 AND model_version = $2 AND constraint_profile = $3
      AND status IN ('success', 'error', 'failed', 'invalid')
    ORDER BY created_at DESC
    LIMIT 1
"""

_SELECT_RECENT_EQUIVALENT = """
    SELECT id, created_at, num_detections, image_id
    FROM execution_log
    WHERE status = 'success'
      AND search_zone = $1 AND model_name = $2 AND model_version = $3
      AND constraint_profile = $4
      AND created_at > NOW() - INTERVAL '24 hours'
    ORDER BY created_at DESC
    LIMIT 1
"""


@dataclass
class Issue:
    code: str
    message: str
    hint: str | None = None
    valid_values: list[Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# Which HTTP status the trigger answers with, per blocking code. The first two
# keep the historical codes of POST /api/pipeline/trigger.
STATUS_BY_CODE = {
    "engine_unavailable": 503,
    "pipeline_busy": 409,
    "unknown_profile": 400,
}


def http_status_for(issues: list[Issue]) -> int:
    for code in ("engine_unavailable", "pipeline_busy", "unknown_profile"):
        if any(i.code == code for i in issues):
            return STATUS_BY_CODE[code]
    return 422


def _tipcue_zone_hint(zone: str) -> str | None:
    try:
        from src.tipcue.zones import get_zone

        z = get_zone(zone)
    except Exception:  # noqa: BLE001
        return None
    if z is None:
        return None
    return f"'{zone}' is a Tip & Cue zone; the pipeline searches the '{z.search_zone}' search zone for it."


def _check_bbox(bbox: list[float] | None, issues: list[Issue]) -> None:
    if bbox is None:
        return
    fmt = "aoi_bbox is [lon_min, lat_min, lon_max, lat_max] in WGS-84 degrees"
    if len(bbox) != 4:
        issues.append(Issue("invalid_bbox", f"aoi_bbox must have 4 values, got {len(bbox)}", hint=fmt))
        return
    lon_min, lat_min, lon_max, lat_max = bbox
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180 and -90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        issues.append(Issue("invalid_bbox", "aoi_bbox is outside lon [-180, 180] / lat [-90, 90]", hint=fmt))
    elif lon_min >= lon_max or lat_min >= lat_max:
        issues.append(Issue("invalid_bbox", "aoi_bbox needs lon_min < lon_max and lat_min < lat_max", hint=fmt))


async def in_flight_executions(settings: Settings) -> list[dict[str, Any]]:
    """pending/running rows that this process may still own (scheduled and cue runs included).

    Rows older than the process are orphans of a previous container (the
    reaper will close them); counting them would block every trigger for up
    to ``orphan_reaper_threshold_minutes`` after a restart.
    """
    window = settings.orphan_reaper_threshold_minutes
    try:
        import time

        from src.main import get_start_time

        started = get_start_time()
        if started > 0:
            window = min(window, max(1, int((time.time() - started) / 60) + 1))
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = await db.fetch(SELECT_IN_FLIGHT_EXECUTIONS, window)
    except Exception:  # noqa: BLE001 - no DB answer: fall back to the in-memory flag
        logger.warning("In-flight lookup failed", exc_info=True)
        return []
    return [
        {
            "execution_id": str(r["id"]),
            "status": r["status"],
            "trigger_type": r["trigger_type"],
            "profile": r["constraint_profile"],
            "model": r["model_name"],
            "model_version": r["model_version"],
            "zone": r["search_zone"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def preflight(
    request: Any,
    *,
    engine: Any,
    settings: Settings,
    api_state: dict[str, Any],
) -> tuple[dict[str, Any], list[Issue]]:
    """Resolve ``request`` (a ``PipelineTriggerRequest``) and list what would stop the run.

    Returns the report published by ``POST /api/pipeline/preview`` and the
    blocking issues the trigger turns into an error response.
    """
    blocking: list[Issue] = []
    warnings: list[Issue] = []

    if engine is None:
        blocking.append(Issue(
            "engine_unavailable",
            "Pipeline engine not available: no model files, missing dependencies or failed startup",
            hint="Not fixable by the caller; GET /api/health and the service logs say why.",
        ))

    profile = PROFILES.get(request.profile)
    if profile is None:
        blocking.append(Issue(
            "unknown_profile", f"Unknown profile '{request.profile}'",
            valid_values=list(PROFILES),
        ))

    if request.image_id is None and request.zone not in SEARCH_ZONES:
        blocking.append(Issue(
            "unknown_zone", f"Unknown search zone '{request.zone}'",
            hint=_tipcue_zone_hint(request.zone) or "Pass a search zone or an explicit Copernicus image_id.",
            valid_values=list(SEARCH_ZONES),
        ))

    if request.sensor not in SENSORS:
        blocking.append(Issue("unknown_sensor", f"Unknown sensor '{request.sensor}'", valid_values=list(SENSORS)))
    elif request.sensor != "s1":
        warnings.append(Issue(
            "sensor_out_of_scope",
            "Only Sentinel-1 SAR ('s1') is part of the evaluated scope; 's2' results are not evidence.",
        ))

    _check_bbox(request.aoi_bbox, blocking)

    confidence = request.confidence_threshold
    if confidence is None:
        confidence = settings.confidence_threshold

    model_name = request.model or settings.default_model
    model_version = request.model_version
    if model_version is None and model_name == settings.default_model:
        model_version = settings.default_model_version
    model_report = await _check_model(model_name, model_version, request.sensor, engine, blocking, warnings)
    if model_report and model_report.get("version"):
        model_version = model_report["version"]

    in_flight = await in_flight_executions(settings)
    if api_state.get("running") or in_flight:
        ids = [r["execution_id"] for r in in_flight]
        current = api_state.get("current_execution_id")
        if current and current not in ids:
            ids.append(current)
        blocking.append(Issue(
            "pipeline_busy",
            "A pipeline is already running"
            + (f" ({', '.join(ids)})" if ids else f" (profile={api_state.get('current_profile')})"),
            hint="One run at a time on this host. Poll GET /api/executions/{id} for the blocking run and retry when it is terminal.",
        ))

    estimate = None
    recent = None
    if model_version and profile is not None:
        try:
            h = await db.fetchrow(_SELECT_DURATION_HISTORY, model_name, model_version, request.profile)
            if h is not None and h["runs"]:
                estimate = {
                    "duration_minutes_p50": round(float(h["p50_ms"]) / 60000.0, 1),
                    "duration_minutes_max": round(float(h["max_ms"]) / 60000.0, 1),
                    "based_on_runs": int(h["runs"]),
                    "basis": f"successful {model_name}:{model_version} runs on '{request.profile}' in the last 120 days",
                }
            last = await db.fetchrow(_SELECT_LAST_TERMINAL_ON_PROFILE, model_name, model_version, request.profile)
            if last is not None and "memory budget exceeded" in (last["error_message"] or ""):
                warnings.append(Issue(
                    "previous_run_exceeded_memory_budget",
                    f"The last {model_name}:{model_version} run on '{request.profile}' ({last['id']}) aborted on "
                    f"the profile's memory budget (peak {last['peak_ram_mb'] or '?'} MB vs "
                    f"{profile.memory_limit_mb} MB); this run is likely to abort the same way.",
                    hint="Pick a larger profile or a lower-footprint variant, or confirm the run is meant to re-measure the limit.",
                ))
            if request.image_id is None:
                r = await db.fetchrow(_SELECT_RECENT_EQUIVALENT, request.zone, model_name, model_version, request.profile)
                if r is not None:
                    recent = {
                        "execution_id": str(r["id"]),
                        "created_at": r["created_at"],
                        "num_detections": r["num_detections"],
                        "image_id": r["image_id"],
                    }
                    warnings.append(Issue(
                        "recent_equivalent_run",
                        f"Execution {r['id']} ran the same zone, model and profile in the last 24 h "
                        f"({r['num_detections']} detections).",
                        hint="A manual run always executes (no dedup); reuse that result unless a new scene is expected.",
                    ))
        except Exception:  # noqa: BLE001 - history is advisory
            logger.debug("Preflight history lookup failed", exc_info=True)

    report = {
        "ok": not blocking,
        "resolved_request": {
            "zone": request.zone,
            "image_id": request.image_id,
            "sensor": request.sensor,
            "model": model_name,
            "model_version": model_version,
            "profile": request.profile,
            "confidence_threshold": confidence,
            "iou_threshold": settings.iou_threshold,
            "aoi_bbox": request.aoi_bbox,
            "thresholds_from": "request" if request.confidence_threshold is not None else "Settings (I-DET-4)",
        },
        "model": model_report,
        "profile": (
            {
                "name": profile.name,
                "cpu_limit": profile.cpu_limit,
                "memory_limit_mb": profile.memory_limit_mb,
                "simulates": profile.simulates,
            }
            if profile
            else None
        ),
        "blocking_issues": [i.as_dict() for i in blocking],
        "warnings": [i.as_dict() for i in warnings],
        "in_flight": in_flight,
        "estimate": estimate,
        "recent_equivalent_run": recent,
        "cost": (
            "Searches Copernicus for the newest scene of the zone (last 7 days), downloads it "
            "(~1.7 GB, capped at download_rate_limit_mbps) and holds the host's detection capacity "
            "for the whole run; results are persisted to execution_log and detections."
        ),
    }
    return report, blocking


async def _check_model(
    name: str,
    version: str | None,
    sensor: str,
    engine: Any,
    blocking: list[Issue],
    warnings: list[Issue],
) -> dict[str, Any] | None:
    if name.startswith("cfar"):
        return {"name": name, "version": version, "builtin": True, "registry_status": None,
                "sar_compatible": True, "has_model_card": True}

    try:
        rows = [dict(r) for r in await db.fetch(_SELECT_MODEL_ROWS, name)]
    except Exception:  # noqa: BLE001
        logger.warning("Model registry lookup failed", exc_info=True)
        rows = []
    manager = getattr(engine, "model_manager", None)

    if not rows:
        path = manager._find_model_file(name, version) if manager is not None else None
        if path is None:
            try:
                names = [r["name"] for r in await db.fetch(_SELECT_MODEL_NAMES)]
            except Exception:  # noqa: BLE001
                names = []
            blocking.append(Issue(
                "unknown_model", f"Model '{name}' is not registered",
                hint="GET /api/catalog lists registered models with their versions and status.",
                valid_values=names + ["cfar-default"],
            ))
            return None
        warnings.append(Issue(
            "model_not_registered",
            f"'{name}' has a weight file but no registry row; the run records it without registry status.",
        ))
        has_card = _card_ok(manager, name, Path(str(path)), blocking)
        return {"name": name, "version": version, "registry_status": None, "has_model_card": has_card}

    versions = [r["version"] for r in rows]
    if version is None:
        if len(rows) > 1:
            blocking.append(Issue(
                "ambiguous_model",
                f"Model '{name}' has several versions; pass model_version",
                valid_values=versions,
            ))
            return {"name": name, "version": None, "available_versions": versions}
        row = rows[0]
    else:
        row = next((r for r in rows if r["version"] == version), None)
        if row is None:
            blocking.append(Issue(
                "unknown_model_version", f"Model '{name}' has no version '{version}'",
                valid_values=versions,
            ))
            return {"name": name, "version": version, "available_versions": versions}

    status = row.get("status") or "active"
    if status == "rejected":
        warnings.append(Issue(
            "model_rejected",
            f"{name}:{row['version']} is rejected (I-MOD-3): {row.get('rejection_reason')}",
            hint="Running it is allowed for re-evaluation, but its results are not evidence for the variant.",
        ))
    elif status == "candidate":
        warnings.append(Issue(
            "model_candidate",
            f"{name}:{row['version']} is a candidate: its compression triplet has not been graded yet (I-MOD-3).",
        ))
    elif status == "retired":
        warnings.append(Issue("model_retired", f"{name}:{row['version']} is retired."))

    classes = list(row.get("classes") or [])
    sar_ok = True
    try:
        from src.pipeline.engine import PipelineEngine

        PipelineEngine._validate_model_for_sensor({"name": name, "classes": classes}, sensor)
    except ValueError as exc:
        sar_ok = False
        blocking.append(Issue(
            "model_not_sar_compatible", str(exc),
            hint="Use a model whose classes include 'ship' or 'vessel' (see GET /api/catalog).",
        ))

    has_card = _card_ok(manager, name, Path(str(row.get("file_path") or f"{name}.pt")), blocking) if manager else None
    return {
        "name": name,
        "version": row["version"],
        "format": row.get("format"),
        "compression_technique": row.get("compression_technique"),
        "registry_status": status,
        "rejection_reason": row.get("rejection_reason"),
        "sar_compatible": sar_ok,
        "has_model_card": has_card,
    }


def _card_ok(manager: Any, name: str, path: Path, blocking: list[Issue]) -> bool:
    try:
        manager._require_model_card(name, path)
        return True
    except FileNotFoundError as exc:
        blocking.append(Issue(
            "model_card_missing", str(exc),
            hint="I-AIA-1: a model without MODEL_CARD.md never enters the evaluation pipeline.",
        ))
        return False
