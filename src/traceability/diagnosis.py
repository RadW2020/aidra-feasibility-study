"""Explain an ``execution_log`` row in domain terms.

The facts that say why a run ended the way it did are all in the row
(``status``, ``error_message``, ``notes``, ``peak_ram_mb``, the profile), but
reading them correctly takes project knowledge: that ``skipped`` is not a
failure, that the reaper writes ``reaped: stuck in ...``, that a memory-budget
abort surfaces as ``status='error'`` with the budget in the message. This
module turns the row into an outcome category with the evidence behind it and
checks provenance completeness (I-TRACE-1/4). Pure functions, no DB: the API
layer fetches, this module judges.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from src.profiles.definitions import PROFILES
from src.vocabulary import EXECUTION_STATUSES

_BUDGET_RE = re.compile(
    r"peak RSS (?P<peak>[\d.]+) MB > (?P<limit>[\d.]+) MB budget of profile '(?P<profile>[^']+)'"
)
_DUP_RE = re.compile(r"already processed by execution (?P<eid>[0-9a-f-]{36})")

# Fields a successful run must carry to count as evidence (I-TRACE-1, I-TRACE-4).
_PROVENANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("image_hash", "I-TRACE-1: SHA256 of the input scene"),
    ("model_hash", "I-TRACE-1: SHA256 of the model weights"),
    ("output_hash", "I-TRACE-1: SHA256 of the serialized detections"),
    ("input_params_hash", "I-TRACE-4: hash of request + Settings"),
    ("commit_sha", "I-TRACE-4: code version that produced the run"),
)
_PLACEHOLDERS = {"", "pending", "unknown", None}


def _age_minutes(created_at: datetime | None, now: datetime) -> float | None:
    if created_at is None:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return round((now - created_at).total_seconds() / 60.0, 1)


def memory_budget_mb(row: dict[str, Any]) -> int | None:
    """The profile's RAM budget: the column when recorded, else the profile definition."""
    if row.get("memory_limit_mb"):
        return int(row["memory_limit_mb"])
    profile = PROFILES.get(row.get("constraint_profile") or "")
    return profile.memory_limit_mb if profile else None


def diagnose(
    row: dict[str, Any],
    *,
    reaper_threshold_minutes: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Outcome category + one-sentence summary + evidence + next steps."""
    now = now or datetime.now(tz=UTC)
    status = row.get("status") or "unknown"
    error = row.get("error_message") or ""
    notes = row.get("notes") or ""
    profile = row.get("constraint_profile")
    peak = row.get("peak_ram_mb")
    budget = memory_budget_mb(row)
    age = _age_minutes(row.get("created_at"), now)
    evidence: dict[str, Any] = {"status": status}
    steps: list[str] = []

    def out(category: str, summary: str) -> dict[str, Any]:
        return {
            "category": category,
            "summary": summary,
            "evidence": evidence,
            "next_steps": steps,
        }

    if status in ("pending", "running"):
        evidence["age_minutes"] = age
        evidence["reaper_threshold_minutes"] = reaper_threshold_minutes
        if age is not None and age >= reaper_threshold_minutes:
            steps.append("The orphan reaper will mark it 'failed'; do not wait on it.")
            return out("stalled", f"Still '{status}' after {age:.0f} min, past the {reaper_threshold_minutes} min reaper threshold: the process that owned it is gone.")
        steps.append("Poll again later; runs take 6-52 min depending on model and profile.")
        return out("in_progress", f"'{status}' for {age if age is not None else '?'} min.")

    if status == "success":
        n = row.get("num_detections") or 0
        valid = row.get("num_valid_targets")
        evidence.update(num_detections=n, num_valid_targets=valid, peak_ram_mb=peak, memory_budget_mb=budget)
        if n == 0:
            steps.append("Zero detections is a result, not an error: check the scene footprint and the sea mask coverage in notes.")
            return out("succeeded_empty", "Completed with 0 detections.")
        sea = f", {valid} of them valid sea targets" if valid is not None else ""
        return out("succeeded", f"Completed with {n} detections{sea}.")

    if status == "skipped":
        evidence["notes"] = notes
        dup = _DUP_RE.search(notes)
        if dup:
            evidence["previous_execution_id"] = dup.group("eid")
            steps.append(f"Read the earlier result instead: get_execution({dup.group('eid')}).")
            return out("skipped_duplicate", "Not run: the same scene, model and parameters had already succeeded (same output_hash by construction, I-TRACE-4).")
        if "own scene" in notes:
            return out("skipped_own_scene", "Not run: the Tip & Cue search returned the scene its parent execution had already processed.")
        return out("skipped", f"Not run: {notes or 'no reason recorded'}.")

    if status == "invalid":
        evidence["error_message"] = error
        steps.append("The scene failed preprocessing quality (I-SAR-1); pick another scene.")
        return out("scene_invalid", f"No inference: scene quality gate failed ({error.removeprefix('quality_invalid: ') or 'no reasons recorded'}).")

    if status == "failed" and "reaped:" in error:
        evidence.update(error_message=error, age_minutes=age)
        steps.append("A container restart or crash interrupted it. Re-run if the result is still needed.")
        return out("reaped_orphan", "Marked failed by the orphan reaper: it never reached a terminal status.")

    if status in ("error", "failed"):
        evidence["error_message"] = error
        m = _BUDGET_RE.search(error)
        if m or "MemoryError" in error or "OOM under profile" in error:
            peak_mb = float(m.group("peak")) if m else peak
            limit_mb = float(m.group("limit")) if m else budget
            evidence.update(peak_ram_mb=peak_mb, memory_budget_mb=limit_mb, profile=profile)
            steps.append(
                "Needs a profile whose memory_limit_mb exceeds the peak, or a lower-footprint variant; "
                "list_executions(image_id=...) shows how other profiles fared on this scene."
            )
            detail = f" (peak RSS {peak_mb:.0f} MB > {limit_mb:.0f} MB)" if peak_mb and limit_mb else ""
            return out("memory_budget_exceeded", f"Aborted: exceeded the '{profile}' memory budget{detail}, emulating an on-board OOM kill.")
        if any(s in error for s in ("IngestionError", "Copernicus", "AuthenticationError", "No images found")):
            steps.append("Upstream data problem (Copernicus search, download or credentials); retry later or pick another scene.")
            return out("ingestion_failed", "Failed while finding or downloading the scene.")
        if any(s in error for s in ("Model not found", "AI Act gate", "not approved for Sentinel-1", "Ambiguous model")):
            steps.append("Check get_catalog for registered models, versions and card status.")
            return out("model_unavailable", "Failed to load the requested model.")
        if "imeout" in error or "timed out" in error:
            return out("timeout", "Exceeded a pipeline timeout.")
        return out("failed_other", f"Failed: {error[:200] or 'no error message recorded'}.")

    return out("unknown", f"Unrecognised status '{status}'.")


def provenance(row: dict[str, Any]) -> dict[str, Any]:
    """Which traceability fields are present; ``complete`` only matters for successes."""
    checks = []
    for field, invariant in _PROVENANCE_FIELDS:
        value = row.get(field)
        checks.append({
            "field": field,
            "present": value not in _PLACEHOLDERS,
            "invariant": invariant,
        })
    applicable = row.get("status") == "success"
    complete = all(c["present"] for c in checks)
    missing = [c["field"] for c in checks if not c["present"]]
    if not applicable:
        verdict = "not_applicable: only a successful run is evidence"
    elif complete:
        verdict = "complete"
    else:
        verdict = f"incomplete: missing {', '.join(missing)}"
    return {
        "applicable": applicable,
        "complete": applicable and complete,
        "verdict": verdict,
        "checks": checks,
        "image_hash": row.get("image_hash"),
        "model_hash": row.get("model_hash"),
        "output_hash": row.get("output_hash"),
        "input_params_hash": row.get("input_params_hash"),
        "commit_sha": row.get("commit_sha"),
    }


def status_meaning(status: str | None) -> dict[str, Any]:
    spec = EXECUTION_STATUSES.get(status or "")
    if spec is None:
        return {"terminal": True, "meaning": f"Unrecognised status '{status}'."}
    return {"terminal": spec["terminal"], "meaning": spec["meaning"]}
