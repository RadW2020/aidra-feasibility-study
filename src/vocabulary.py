"""Closed vocabularies of the AIDRA domain, with what each value means.

Single source of truth for the enumerations that the API exposes as plain
strings (execution status, quality verdict, detection source, ...). The API
validates filters against these tuples and ``GET /api/vocabulary`` publishes
them with their meaning, so a client — human or agent — never has to infer
from prose or from Grafana SQL what ``skipped`` or ``land_artifact`` implies.

``tests/test_api/test_agent_vocabulary.py`` checks that every status literal
the pipeline writes appears here: a new status without a meaning fails CI.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# execution_log.status
# ---------------------------------------------------------------------------

EXECUTION_STATUSES: dict[str, dict[str, Any]] = {
    "pending": {
        "terminal": False,
        "meaning": "Row written before the run starts (I-TRACE-2); the model is resolved, the scene is not yet downloaded.",
    },
    "running": {
        "terminal": False,
        "meaning": "Searching, downloading, preprocessing or detecting. Poll again; a full scene takes 6-52 min depending on model and profile.",
    },
    "success": {
        "terminal": True,
        "meaning": "Detections persisted with image/model/output hashes. num_valid_targets counts sea targets only (I-DET-2).",
    },
    "skipped": {
        "terminal": True,
        "meaning": "Nothing new to do: the same scene, model and parameters already succeeded, or a cue found no scene sensed after the one that raised it. Not a failure; notes says which.",
    },
    "invalid": {
        "terminal": True,
        "meaning": "Scene failed the I-SAR-1 preprocessing quality gate; no inference was run on it. error_message lists the reasons.",
    },
    "error": {
        "terminal": True,
        "meaning": "The run failed. error_message carries the exception, e.g. a profile memory-budget abort or a Copernicus failure.",
    },
    "failed": {
        "terminal": True,
        "meaning": "Reaped orphan: stuck in pending/running longer than the reaper threshold (process restarted or died mid-run).",
    },
}

NON_TERMINAL_STATUSES: tuple[str, ...] = tuple(
    s for s, v in EXECUTION_STATUSES.items() if not v["terminal"]
)

# ---------------------------------------------------------------------------
# detections.quality_verdict (migration 008)
# ---------------------------------------------------------------------------

QUALITY_VERDICTS: dict[str, dict[str, Any]] = {
    "valid_sea_target": {
        "counts_as_sea_target": True,
        "meaning": "At sea, inside the valid footprint, not a density anomaly, confirmed by YOLO (source yolo or fused).",
    },
    "candidate": {
        "counts_as_sea_target": False,
        "meaning": "At sea but CFAR-only: a bright point reflector without CNN confirmation.",
    },
    "land_artifact": {
        "counts_as_sea_target": False,
        "meaning": "Centre falls on land (global-land-mask). Kept for audit, excluded from sea metrics (I-DET-2).",
    },
    "cluster_artifact": {
        "counts_as_sea_target": False,
        "meaning": "Part of an anomalous density cluster (I-DET-3): likely swath-edge or speckle artefact.",
    },
    "outside_footprint": {
        "counts_as_sea_target": False,
        "meaning": "Historical rows only (migration 008 backfill): centre outside the scene footprint. The live pipeline clips these before persistence (I-SAR-3).",
    },
}

# ---------------------------------------------------------------------------
# detections.source and the derived tier (src/api/tiers.py)
# ---------------------------------------------------------------------------

DETECTION_SOURCES: dict[str, dict[str, Any]] = {
    "cfar": {"tier": "standard", "meaning": "CA-CFAR cluster only (point reflector, no CNN agreement)."},
    "yolo": {"tier": "standard", "meaning": "YOLOv8 box only (no CFAR cluster within the fusion tolerance)."},
    "fused": {
        "tier": "high",
        "meaning": "CFAR and YOLO agreed (centre distance <= fusion tolerance). On xView3: precision 0.31, Pd 0.12.",
    },
}

TIERS: dict[str, str] = {
    "high": "source=fused. A high-precision operating point, never a default filter.",
    "standard": "Everything that is not fused.",
}

# ---------------------------------------------------------------------------
# execution_log.trigger_type, tasking_queue.status, models_registry.status
# ---------------------------------------------------------------------------

TRIGGER_TYPES: dict[str, str] = {
    "manual": "Requested through POST /api/pipeline/trigger. Always runs (no dedup).",
    "scheduled": "APScheduler zone scan every scheduler_interval_hours. Deduplicated against earlier successes.",
    "cue": "Tip & Cue re-observation from tasking_queue. Deduplicated; only runs on a scene sensed after its parent's, and never raises cues of its own.",
}

CUE_STATUSES: dict[str, dict[str, Any]] = {
    "pending": {"terminal": False, "meaning": "Queued; the cue processor picks pending cues every 15 minutes, highest priority first. result_status 'awaiting_new_scene' means no scene sensed after the triggering one is published yet: the cue is re-checked every tipcue_recheck_minutes."},
    "processing": {"terminal": False, "meaning": "Picked by the cue processor; a pipeline run is in progress."},
    "completed": {"terminal": True, "meaning": "A run on a later scene finished; result_status and execution_id say how. Cues closed before 2026-11-17 may carry result_status 'skipped' or point at their own scene: those did not re-observe."},
    "expired": {"terminal": True, "meaning": "No scene sensed after the triggering one appeared within tipcue_max_wait_hours; nothing was re-observed (result_status 'no_new_scene')."},
    "failed": {"terminal": True, "meaning": "Retries exhausted (attempts >= max_attempts); last_error says why."},
}

MODEL_STATUSES: dict[str, str] = {
    "active": "Evaluated and usable as evidence.",
    "candidate": "Compressed variant whose triplet {baseline, variant, profile} has not been graded yet (I-MOD-3).",
    "rejected": "Exceeded the declared degradation budget; kept with rejection_reason, never deleted (I-MOD-3).",
    "retired": "No longer used for new evaluations.",
}


def as_document() -> dict[str, Any]:
    """Vocabulary as published by ``GET /api/vocabulary``."""

    def _entries(table: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for value, spec in table.items():
            if isinstance(spec, str):
                out.append({"value": value, "meaning": spec})
            else:
                out.append({"value": value, **spec})
        return out

    return {
        "execution_statuses": _entries(EXECUTION_STATUSES),
        "quality_verdicts": _entries(QUALITY_VERDICTS),
        "detection_sources": _entries(DETECTION_SOURCES),
        "tiers": _entries(TIERS),
        "trigger_types": _entries(TRIGGER_TYPES),
        "cue_statuses": _entries(CUE_STATUSES),
        "model_statuses": _entries(MODEL_STATUSES),
    }
