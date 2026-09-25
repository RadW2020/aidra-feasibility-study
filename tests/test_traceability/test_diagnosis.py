"""Outcome categories and provenance checks for execution_log rows (src/traceability/diagnosis.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.traceability.diagnosis import diagnose, memory_budget_mb, provenance

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def _row(**kw):
    base = {
        "id": uuid4(),
        "created_at": NOW - timedelta(minutes=30),
        "status": "success",
        "constraint_profile": "ground",
        "num_detections": 12,
        "num_valid_targets": 5,
        "error_message": None,
        "notes": None,
        "image_hash": "a" * 64,
        "model_hash": "b" * 64,
        "output_hash": "c" * 64,
        "input_params_hash": "d" * 64,
        "commit_sha": "e" * 40,
    }
    base.update(kw)
    return base


def _cat(row):
    return diagnose(row, reaper_threshold_minutes=240, now=NOW)["category"]


def test_success_with_and_without_detections():
    assert _cat(_row()) == "succeeded"
    assert _cat(_row(num_detections=0)) == "succeeded_empty"


def test_memory_budget_abort_extracts_numbers():
    msg = (
        "MemoryError: memory budget exceeded: peak RSS 2650 MB > 2048 MB budget of profile "
        "'sat-mid' (enforcement=abort)"
    )
    out = diagnose(_row(status="error", constraint_profile="sat-mid", error_message=msg),
                   reaper_threshold_minutes=240, now=NOW)
    assert out["category"] == "memory_budget_exceeded"
    assert out["evidence"]["peak_ram_mb"] == 2650
    assert out["evidence"]["memory_budget_mb"] == 2048
    assert "2650" in out["summary"] and "2048" in out["summary"]


def test_reaped_orphan_and_stalled_run():
    assert _cat(_row(status="failed", error_message="reaped: stuck in running for >240 minutes")) == "reaped_orphan"
    old = _row(status="running", created_at=NOW - timedelta(minutes=300))
    assert _cat(old) == "stalled"
    assert _cat(_row(status="running")) == "in_progress"


def test_skips_are_not_failures():
    eid = uuid4()
    dup = _row(status="skipped", notes=f"skipped: scene X already processed by execution {eid} with the same model and parameters (40 detections)")
    out = diagnose(dup, reaper_threshold_minutes=240, now=NOW)
    assert out["category"] == "skipped_duplicate"
    assert out["evidence"]["previous_execution_id"] == str(eid)
    assert _cat(_row(status="skipped", notes="skipped: cue would reprocess its own scene (P1)")) == "skipped_own_scene"


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("IngestionError: Copernicus search failed: 503", "ingestion_failed"),
        ("FileNotFoundError: Model not found: foo:None", "model_unavailable"),
        ("ValueError: Model is not approved for Sentinel-1 SAR vessel detection", "model_unavailable"),
        ("TimeoutError: preprocessing timed out", "timeout"),
        ("RuntimeError: something else", "failed_other"),
        # Real production annotations (May 2026).
        ("killed by container redeploy", "interrupted_by_restart"),
        ("Orphaned by container restart at 18:33 UTC during throttle bug investigation", "interrupted_by_restart"),
    ],
)
def test_error_categories(message, category):
    assert _cat(_row(status="error", error_message=message)) == category


def test_invalid_scene():
    assert _cat(_row(status="invalid", error_message="quality_invalid: no_valid_pixels")) == "scene_invalid"


def test_memory_budget_falls_back_to_profile_definition():
    assert memory_budget_mb({"constraint_profile": "sat-low"}) == 1024
    assert memory_budget_mb({"constraint_profile": "sat-low", "memory_limit_mb": 999}) == 999
    assert memory_budget_mb({"constraint_profile": "nope"}) is None


def test_provenance_complete_only_for_success():
    ok = provenance(_row())
    assert ok["complete"] is True and ok["verdict"] == "complete"
    missing = provenance(_row(commit_sha="unknown", image_hash="pending"))
    assert missing["complete"] is False
    assert "commit_sha" in missing["verdict"] and "image_hash" in missing["verdict"]
    failed = provenance(_row(status="error"))
    assert failed["applicable"] is False and failed["complete"] is False
