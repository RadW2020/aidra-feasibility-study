"""Compression-triplet grading (I-MOD-1/2/3), src/validation/triplet.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.validation.triplet import EXCEEDS, INSUFFICIENT, WITHIN, grade, pair_quality_legs

T0 = datetime(2026, 8, 29, 7, 42, tzinfo=UTC)


def _leg(version, ap, settings_hash, minutes=0, dataset="xview3-sar/validation/adriatic", path="full"):
    return {
        "id": f"{version}-{settings_hash}",
        "created_at": T0 + timedelta(minutes=minutes),
        "model_version": version,
        "dataset": dataset,
        "pipeline_path": path,
        "map_at_iou": ap,
        "pd_recall": 0.14,
        "far_per_km2": 0.004,
        "precision": 0.13,
        "num_predictions": 2000,
        "provenance_json": {"settings_hash": settings_hash},
    }


def _hw(version, profile="ground", p50=250.0):
    return {"constraint_profile": profile, "model_version": version, "runs": 2, "failed_runs": 0,
            "budget_aborts": 0, "tile_p50_ms": p50, "tile_p95_ms": p50 * 1.5,
            "run_p50_ms": 380000.0, "peak_ram_mb": 2600.0, "model_size_mb": 26.5}


def test_baseline_is_paired_by_settings_hash_not_recency():
    # The newest FP32 leg was computed under older settings; the matching one must win.
    old_settings = _leg("v1.0", 0.0148, "old", minutes=5)
    same_settings = _leg("v1.0", 0.0256, "new", minutes=1)
    variant = _leg("int8-static", 0.0317, "new")
    b, v, how = pair_quality_legs([old_settings, same_settings], [variant])
    assert b is same_settings and v is variant and how == "settings_hash"


def test_within_budget_with_speedup():
    out = grade(
        baseline_rows=[_leg("v1.0", 0.0256, "h")],
        variant_rows=[_leg("int8-static", 0.0317, "h")],
        hardware_rows=[_hw("v1.0", p50=2333.0), _hw("int8-static", p50=252.0)],
        baseline_version="v1.0", variant_version="int8-static", max_delta_map_pts=5.0,
    )
    assert out["verdict"] == WITHIN
    assert out["quality"]["delta"]["map_pts"] == 0.61
    assert out["quality"]["comparable"] is True
    assert out["hardware"][0]["speedup_tile_p50"] == 9.26
    assert out["complete_triplet"] is True


def test_exceeds_budget():
    out = grade(
        baseline_rows=[_leg("v1.0", 0.30, "h")],
        variant_rows=[_leg("int8-dynamic", 0.20, "h")],
        hardware_rows=[_hw("v1.0"), _hw("int8-dynamic")],
        baseline_version="v1.0", variant_version="int8-dynamic", max_delta_map_pts=5.0,
    )
    assert out["verdict"] == EXCEEDS
    assert "rejected" in out["verdict_reason"]


def test_missing_quality_leg_is_insufficient_even_with_latency():
    out = grade(
        baseline_rows=[_leg("v1.0", 0.03, "h")],
        variant_rows=[],
        hardware_rows=[_hw("v1.0"), _hw("int8-static")],
        baseline_version="v1.0", variant_version="int8-static", max_delta_map_pts=5.0,
    )
    assert out["verdict"] == INSUFFICIENT
    assert any("variant" in m for m in out["missing_evidence"])
    assert out["complete_triplet"] is False


def test_unmatched_settings_are_flagged():
    out = grade(
        baseline_rows=[_leg("v1.0", 0.03, "a")],
        variant_rows=[_leg("int8-static", 0.03, "b")],
        hardware_rows=[_hw("int8-static")],
        baseline_version="v1.0", variant_version="int8-static", max_delta_map_pts=5.0,
    )
    assert out["quality"]["comparable"] is False
    assert "warning" in out["quality"]
