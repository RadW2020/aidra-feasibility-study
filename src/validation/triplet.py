"""Grade a compression triplet {baseline FP32, variant, hardware profile}.

I-MOD-1: no compression claim without the triplet. I-MOD-2: each leg reports
mAP@0.5, Pd, FAR/km², latency p50/p95, peak RAM and size on disk. I-MOD-3:
the variant is graded against a degradation budget declared *before* the run
(``Settings.triplet_max_delta_map_pts``, default 5 AP points).

The quality legs come from ``validation_runs``; they are only comparable when
both were computed under the same pipeline settings, so the baseline is paired
with the variant by ``provenance_json.settings_hash`` when the reports carry
one. The hardware leg comes from ``execution_log``. Pure functions: the API
route fetches the rows, this module judges them.
"""

from __future__ import annotations

from typing import Any

WITHIN = "within_budget"
EXCEEDS = "exceeds_budget"
INSUFFICIENT = "insufficient_evidence"


def settings_hash_of(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    prov = row.get("provenance_json") or {}
    if isinstance(prov, str):
        import json

        try:
            prov = json.loads(prov)
        except ValueError:
            return None
    return prov.get("settings_hash") if isinstance(prov, dict) else None


def pair_quality_legs(
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Pick the (baseline, variant) pair to compare and say how it was chosen.

    Rows arrive newest first. The newest variant leg is kept; the baseline is
    the newest one on the same dataset and pipeline path with the same
    ``settings_hash``, else the newest on the same dataset and path.
    """
    if not variant_rows:
        return (baseline_rows[0] if baseline_rows else None), None, "no_variant_leg"
    variant = variant_rows[0]
    same_setup = [
        b for b in baseline_rows
        if b.get("dataset") == variant.get("dataset")
        and b.get("pipeline_path") == variant.get("pipeline_path")
    ]
    if not same_setup:
        return None, variant, "no_baseline_on_same_dataset_and_path"
    vh = settings_hash_of(variant)
    if vh:
        for b in same_setup:
            if settings_hash_of(b) == vh:
                return b, variant, "settings_hash"
    return same_setup[0], variant, "latest_same_dataset_and_path"


def _leg(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "validation_run_id": str(row.get("id")),
        "created_at": row.get("created_at"),
        "dataset": row.get("dataset"),
        "pipeline_path": row.get("pipeline_path"),
        "commit_sha": row.get("commit_sha"),
        "settings_hash": settings_hash_of(row),
        "num_scenes": row.get("num_scenes"),
        "num_ground_truth": row.get("num_ground_truth"),
        "num_predictions": row.get("num_predictions"),
        "map_at_0_5": row.get("map_at_iou"),
        "pd_recall": row.get("pd_recall"),
        "far_per_km2": row.get("far_per_km2"),
        "precision": row.get("precision"),
        "notes": row.get("notes"),
    }


def _pts(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a) * 100.0, 2)


def grade(
    *,
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    hardware_rows: list[dict[str, Any]],
    baseline_version: str,
    variant_version: str,
    max_delta_map_pts: float,
) -> dict[str, Any]:
    """Verdict + the numbers behind it + what evidence is missing."""
    baseline, variant, pairing = pair_quality_legs(baseline_rows, variant_rows)
    missing: list[str] = []
    if baseline is None:
        missing.append(f"quality leg for baseline {baseline_version} (validation_runs)")
    if variant is None:
        missing.append(f"quality leg for variant {variant_version} (validation_runs)")

    quality: dict[str, Any] = {
        "baseline": _leg(baseline),
        "variant": _leg(variant),
        "pairing": pairing,
        "comparable": pairing == "settings_hash",
    }
    if baseline is not None and variant is not None:
        quality["delta"] = {
            "map_pts": _pts(baseline.get("map_at_iou"), variant.get("map_at_iou")),
            "pd_pts": _pts(baseline.get("pd_recall"), variant.get("pd_recall")),
            "precision_pts": _pts(baseline.get("precision"), variant.get("precision")),
            "far_per_km2": (
                round(variant["far_per_km2"] - baseline["far_per_km2"], 5)
                if variant.get("far_per_km2") is not None and baseline.get("far_per_km2") is not None
                else None
            ),
        }
        if pairing != "settings_hash":
            quality["warning"] = (
                "Legs were not matched by settings_hash: they may have been computed under "
                "different pipeline settings, so the delta mixes model and pipeline effects."
            )

    by_profile: dict[str, dict[str, Any]] = {}
    for h in hardware_rows:
        entry = by_profile.setdefault(h["constraint_profile"], {"profile": h["constraint_profile"]})
        side = "baseline" if h["model_version"] == baseline_version else "variant"
        entry[side] = {
            "successful_runs": int(h.get("runs") or 0),
            "failed_runs": int(h.get("failed_runs") or 0),
            "memory_budget_aborts": int(h.get("budget_aborts") or 0),
            "tile_latency_p50_ms": h.get("tile_p50_ms"),
            "tile_latency_p95_ms": h.get("tile_p95_ms"),
            "run_duration_p50_min": round(h["run_p50_ms"] / 60000.0, 1) if h.get("run_p50_ms") else None,
            "peak_ram_mb": h.get("peak_ram_mb"),
            "model_size_mb": h.get("model_size_mb"),
        }
    hardware = []
    for entry in by_profile.values():
        b, v = entry.get("baseline"), entry.get("variant")
        if b and v and b.get("tile_latency_p50_ms") and v.get("tile_latency_p50_ms"):
            entry["speedup_tile_p50"] = round(b["tile_latency_p50_ms"] / v["tile_latency_p50_ms"], 2)
        if b and v and b.get("peak_ram_mb") is not None and v.get("peak_ram_mb") is not None:
            entry["peak_ram_delta_mb"] = round(v["peak_ram_mb"] - b["peak_ram_mb"], 1)
        hardware.append(entry)
    if not any(e.get("variant", {}).get("successful_runs") for e in hardware):
        missing.append(f"hardware leg: no successful run of {variant_version} on the requested profile(s)")

    delta = (quality.get("delta") or {}).get("map_pts")
    if baseline is None or variant is None or delta is None:
        verdict = INSUFFICIENT
        reason = "A quality leg is missing, so no degradation can be computed; latency alone is not a verdict."
    elif -delta > max_delta_map_pts:
        verdict = EXCEEDS
        reason = f"AP drops {-delta:.2f} pts, above the declared {max_delta_map_pts:g}-pt budget (I-MOD-3): mark the variant rejected with this justification."
    else:
        verdict = WITHIN
        reason = f"AP changes by {delta:+.2f} pts, within the declared {max_delta_map_pts:g}-pt budget."
        if not quality["comparable"]:
            reason += " Legs not matched by settings_hash; confirm they share pipeline settings."

    return {
        "verdict": verdict,
        "verdict_reason": reason,
        # I-MOD-1: a verdict without every leg is not a triplet.
        "complete_triplet": not missing,
        "declared_budget": {
            "max_delta_map_pts": max_delta_map_pts,
            "source": "Settings.triplet_max_delta_map_pts (I-MOD-3)",
        },
        "quality": quality,
        "hardware": sorted(hardware, key=lambda e: e["profile"]),
        "missing_evidence": missing,
    }
