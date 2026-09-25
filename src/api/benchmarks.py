"""
Benchmark endpoints.

Provides aggregated performance metrics (latency, RAM, CPU, detections)
grouped by model variant and constraint profile, as well as a
comparison matrix for side-by-side analysis.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from src.db.connection import db
from src.db.models import BenchmarkResult
from src.db.queries import SELECT_BENCHMARKS_BY_MODEL, SELECT_PROFILE_COMPARISON

logger = logging.getLogger("aidra.api.benchmarks")

router = APIRouter(tags=["benchmarks"])


def _row_to_benchmark(row) -> BenchmarkResult:  # type: ignore[no-untyped-def]
    """Convert an asyncpg Record to a BenchmarkResult model."""
    return BenchmarkResult(
        model_name=row["model_name"],
        model_version=row["model_version"],
        model_size_mb=row["model_size_mb"] or 0.0,
        compression_technique=row["compression_technique"] or "none",
        constraint_profile=row["constraint_profile"],
        runs=row["runs"],
        avg_inference_ms=row["avg_inference_ms"] or 0.0,
        p50_inference_ms=row.get("p50_inference_ms"),
        p95_inference_ms=row.get("p95_inference_ms"),
        avg_peak_ram_mb=row["avg_peak_ram_mb"] or 0.0,
        avg_cpu_pct=row["avg_cpu_pct"] or 0.0,
        avg_detections=row["avg_detections"] or 0.0,
        avg_confidence=row.get("avg_confidence"),
    )


@router.get("/benchmarks", response_model=list[BenchmarkResult])
async def list_benchmarks(
    model: str | None = Query(
        None, description="Filter by model name (e.g. vesseltracker-sar-yolov8, cfar-default)"
    ),
    profile: str | None = Query(
        None, description="Filter by constraint profile (e.g. ground, sat-high)"
    ),
) -> list[BenchmarkResult]:
    """Return aggregated benchmark results.

    Groups execution records by model variant and constraint profile,
    computing statistics: mean, P50, and P95 inference latency,
    average peak RAM, CPU usage, detection count, and confidence.

    Only executions with ``status='success'`` are included.
    """
    try:
        rows = await db.fetch(SELECT_BENCHMARKS_BY_MODEL, model, profile)
        return [_row_to_benchmark(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to fetch benchmarks: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query benchmarks: {exc}",
        ) from exc


_SELECT_VALIDATION_LEGS = """
    SELECT id, created_at, model_name, model_version, dataset, pipeline_path,
           commit_sha, num_scenes, num_ground_truth, num_predictions,
           map_at_iou, pd_recall, far_per_km2, precision, notes, provenance_json
    FROM validation_runs
    WHERE model_name = $1 AND model_version = $2
      AND ($3::text IS NULL OR dataset = $3)
      AND ($4::text IS NULL OR pipeline_path = $4)
    ORDER BY created_at DESC
    LIMIT 50
"""

_SELECT_HARDWARE_LEGS = """
    SELECT constraint_profile, model_version,
           COUNT(*) FILTER (WHERE status = 'success') AS runs,
           COUNT(*) FILTER (WHERE status IN ('error', 'failed')) AS failed_runs,
           COUNT(*) FILTER (WHERE error_message LIKE '%memory budget exceeded%') AS budget_aborts,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inference_p50_ms)
               FILTER (WHERE status = 'success') AS tile_p50_ms,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inference_p95_ms)
               FILTER (WHERE status = 'success') AS tile_p95_ms,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_duration_ms)
               FILTER (WHERE status = 'success') AS run_p50_ms,
           MAX(peak_ram_mb) FILTER (WHERE status = 'success') AS peak_ram_mb,
           MAX(model_size_mb) AS model_size_mb
    FROM execution_log
    WHERE model_name = $1 AND model_version = ANY($2::text[])
      AND ($3::text IS NULL OR constraint_profile = $3)
    GROUP BY constraint_profile, model_version
"""

_SELECT_REGISTRY_VERSIONS = """
    SELECT version, status, rejection_reason, compression_technique, size_mb
    FROM models_registry WHERE name = $1 ORDER BY version
"""


@router.get("/benchmarks/triplet")
async def compression_triplet(
    model: str = Query(..., description="Base model name, e.g. vesseltracker-sar-yolov8."),
    variant_version: str = Query(..., description="Compressed variant version, e.g. int8-static."),
    baseline_version: str = Query("v1.0", description="FP32 baseline version."),
    profile: str | None = Query(None, description="Constraint profile for the hardware leg; omit for all profiles."),
    dataset: str | None = Query(None, description="Validation dataset, e.g. xview3-sar/validation/adriatic; omit for the variant's latest."),
    pipeline_path: str | None = Query(None, pattern="^(detector|full|synthetic)$", description="detector (model alone) or full (production pipeline)."),
) -> dict:
    """Grade a compression triplet {baseline, variant, profile} (I-MOD-1/2/3).

    Quality deltas come from ``validation_runs`` (legs paired by
    ``settings_hash`` so both were computed under the same pipeline settings),
    latency / RAM / size from ``execution_log``. The verdict is
    ``within_budget``, ``exceeds_budget`` or ``insufficient_evidence``
    against ``Settings.triplet_max_delta_map_pts``; ``missing_evidence`` names
    the legs still to produce.
    """
    from src.api.errors import ApiError
    from src.config import Settings
    from src.profiles.definitions import PROFILES
    from src.validation.triplet import grade

    if profile is not None and profile not in PROFILES:
        raise ApiError(422, "unknown_profile", f"Unknown profile '{profile}'", valid_values=list(PROFILES))
    registry = {r["version"]: dict(r) for r in await db.fetch(_SELECT_REGISTRY_VERSIONS, model)}
    baseline_rows = [dict(r) for r in await db.fetch(_SELECT_VALIDATION_LEGS, model, baseline_version, dataset, pipeline_path)]
    variant_rows = [dict(r) for r in await db.fetch(_SELECT_VALIDATION_LEGS, model, variant_version, dataset, pipeline_path)]
    for version, rows in ((baseline_version, baseline_rows), (variant_version, variant_rows)):
        if version not in registry and not rows:
            raise ApiError(
                422, "unknown_model_version",
                f"'{model}' has no registered version or validation run '{version}'",
                valid_values=sorted(registry),
                hint="GET /api/catalog lists registered models and versions.",
            )
    hardware_rows = [
        dict(r) for r in await db.fetch(_SELECT_HARDWARE_LEGS, model, [baseline_version, variant_version], profile)
    ]
    result = grade(
        baseline_rows=baseline_rows,
        variant_rows=variant_rows,
        hardware_rows=hardware_rows,
        baseline_version=baseline_version,
        variant_version=variant_version,
        max_delta_map_pts=Settings().triplet_max_delta_map_pts,
    )
    return {
        "model": model,
        "baseline_version": baseline_version,
        "variant_version": variant_version,
        "profile": profile,
        "registry": {
            "baseline": registry.get(baseline_version),
            "variant": registry.get(variant_version),
        },
        **result,
    }


@router.get("/benchmarks/compare")
async def compare_benchmarks(
    models: str | None = Query(
        None,
        description="Comma-separated model names to compare (e.g. vesseltracker-sar-yolov8,cfar-default)",
    ),
    profiles: str | None = Query(
        None,
        description="Comma-separated profiles (e.g. ground,sat-high,sat-low)",
    ),
    image_id: str | None = Query(
        None,
        description="Compare only executions on this specific image",
    ),
) -> dict:
    """Generate a comparison matrix: model x profile.

    Returns a dictionary with:
        - ``models``: list of model names included
        - ``profiles``: list of profile names included
        - ``matrix``: list of benchmark rows matching the filters
        - ``comparison``: per-image comparison when *image_id* is provided

    If *image_id* is specified, uses ``SELECT_PROFILE_COMPARISON`` for
    per-image comparison across profiles.
    """
    try:
        result: dict = {
            "models": [],
            "profiles": [],
            "matrix": [],
            "comparison": [],
        }

        # Parse comma-separated lists
        model_list = (
            [m.strip() for m in models.split(",") if m.strip()] if models else None
        )
        profile_list = (
            [p.strip() for p in profiles.split(",") if p.strip()]
            if profiles
            else None
        )

        # Fetch benchmark rows for each requested model (or all)
        if model_list:
            all_rows = []
            for m in model_list:
                rows = await db.fetch(SELECT_BENCHMARKS_BY_MODEL, m, None)
                all_rows.extend(rows)
        else:
            all_rows = await db.fetch(SELECT_BENCHMARKS_BY_MODEL, None, None)

        benchmarks = [_row_to_benchmark(r) for r in all_rows]

        # Filter by requested profiles
        if profile_list:
            benchmarks = [
                b for b in benchmarks if b.constraint_profile in profile_list
            ]

        result["matrix"] = [b.model_dump() for b in benchmarks]
        result["models"] = sorted({b.model_name for b in benchmarks})
        result["profiles"] = sorted({b.constraint_profile for b in benchmarks})

        # Per-image comparison
        if image_id:
            for m in result["models"]:
                try:
                    comp_rows = await db.fetch(
                        SELECT_PROFILE_COMPARISON, image_id, m
                    )
                    for row in comp_rows:
                        result["comparison"].append(dict(row))
                except Exception:
                    logger.debug(
                        "Comparison query failed for model %s", m, exc_info=True
                    )

        return result

    except Exception as exc:
        logger.error("Failed to compare benchmarks: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare benchmarks: {exc}",
        ) from exc
