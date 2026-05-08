"""Persistence layer for ValidationReport rows.

Closes audit finding C1 (2026-05-08): the dashboards
``03-compression-bench`` and ``10-evaluator-evidence`` displayed a
literal ``'NEEDS_DB_METRIC: mAP/Pd/FAR'`` because validation results
existed only as JSON files in ``reports/``. Migration 011 added the
``validation_runs`` table; this module wires the report dataclass to
the table.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from src.db.connection import db
from src.db.queries import INSERT_VALIDATION_RUN, SELECT_VALIDATION_RUNS
from src.validation.metrics import ValidationReport

logger = logging.getLogger(__name__)


async def persist_report(
    report: ValidationReport,
    *,
    dataset: str,
    model_version: str = "unknown",
    model_hash: str | None = None,
    compression_technique: str = "none",
    execution_id: UUID | None = None,
    dataset_split: str | None = None,
    notes: str | None = None,
) -> UUID:
    """Insert a ValidationReport into ``validation_runs``.

    The report only knows about model_name and matcher config; the
    caller supplies the dataset label and (if available) the
    execution_id of the pipeline run whose detections were validated.

    Returns the new row's ``id`` so callers can echo it in API
    responses or use it to chain follow-up writes.
    """
    pr_payload = json.dumps(report.pr_curve) if report.pr_curve else None

    row = await db.fetchrow(
        INSERT_VALIDATION_RUN,
        execution_id,                     # $1
        report.model_name,                # $2
        model_version,                    # $3
        model_hash,                       # $4
        compression_technique,            # $5
        dataset,                          # $6
        dataset_split,                    # $7
        report.match_mode,                # $8
        float(report.iou_threshold),      # $9
        float(report.center_tolerance_px),  # $10
        float(report.confidence_threshold),  # $11
        int(report.num_scenes),           # $12
        int(report.num_ground_truth),     # $13
        int(report.num_predictions),      # $14
        int(report.true_positives),       # $15
        int(report.false_positives),      # $16
        int(report.false_negatives),      # $17
        float(report.total_area_km2),     # $18
        float(report.map_at_iou),         # $19
        float(report.pd_recall),          # $20
        float(report.far_per_km2),        # $21
        float(report.precision),          # $22
        pr_payload,                       # $23
        notes,                            # $24
    )

    new_id: UUID = row["id"]
    logger.info(
        "Persisted validation_run %s: model=%s dataset=%s mAP=%.4f Pd=%.4f FAR=%.4f",
        new_id,
        report.model_name,
        dataset,
        report.map_at_iou,
        report.pd_recall,
        report.far_per_km2,
    )
    return new_id


async def list_validation_runs(
    *,
    model_name: str | None = None,
    compression_technique: str | None = None,
    dataset: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read-only listing for the API and dashboards."""
    rows = await db.fetch(
        SELECT_VALIDATION_RUNS,
        model_name,
        compression_technique,
        dataset,
        limit,
    )
    return [dict(r) for r in rows]
