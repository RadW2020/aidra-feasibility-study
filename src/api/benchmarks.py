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
        None, description="Filter by model name (e.g. yolov8n-sar)"
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


@router.get("/benchmarks/compare")
async def compare_benchmarks(
    models: str | None = Query(
        None,
        description="Comma-separated model names to compare (e.g. yolov8n-sar,yolov8n-sar-int8)",
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
