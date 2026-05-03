"""
Operational metrics over AIDRA production detections.

These are NOT a substitute for mAP@0.5 / Pd / FAR on a labelled split;
they are diagnostic proxies derived from cross-detector agreement and
the I-DET-3 cluster_anomaly flag. They are the most defensible numbers
we can produce **before** acquiring a labelled Strait-of-Gibraltar split.

Outputs:
  - per-source counts and confidence stats
  - detector-agreement rate (fused fraction)
  - cluster-anomaly rate (FAR proxy)
  - on-land filter recovery (I-DET-2 effectiveness)
  - per-execution summary

Usage:
    docker exec aidra-app python /app/scripts/operational_metrics.py
    docker exec aidra-app python /app/scripts/operational_metrics.py --json
    docker exec aidra-app python /app/scripts/operational_metrics.py --execution-id <UUID>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any
from uuid import UUID

_PER_SOURCE = """
    SELECT
        d.source,
        COUNT(*)                                        AS total,
        COUNT(*) FILTER (WHERE d.on_land)               AS on_land,
        COUNT(*) FILTER (WHERE d.cluster_anomaly)       AS cluster_anomaly,
        COUNT(*) FILTER (WHERE d.on_land AND d.cluster_anomaly) AS both,
        COUNT(*) FILTER (WHERE NOT d.on_land AND NOT d.cluster_anomaly) AS clean,
        ROUND(AVG(d.confidence)::numeric, 4)            AS avg_conf,
        ROUND(MIN(d.confidence)::numeric, 4)            AS min_conf,
        ROUND(MAX(d.confidence)::numeric, 4)            AS max_conf
    FROM detections d
    {where_clause}
    GROUP BY d.source
    ORDER BY total DESC
"""

_AGREEMENT = """
    SELECT
        COUNT(*) FILTER (WHERE source='fused') AS fused,
        COUNT(*) FILTER (WHERE source='yolo')  AS yolo_only,
        COUNT(*) FILTER (WHERE source='cfar')  AS cfar_only,
        COUNT(*)                                AS total
    FROM detections d
    {where_clause}
"""

_FAR_PROXY = """
    SELECT
        COUNT(*) FILTER (WHERE cluster_anomaly) * 1.0 / NULLIF(COUNT(*),0) AS anomaly_rate,
        COUNT(*) FILTER (WHERE on_land)         * 1.0 / NULLIF(COUNT(*),0) AS on_land_rate
    FROM detections d
    {where_clause}
"""


async def collect(execution_id: UUID | None) -> dict[str, Any]:
    from src.config import Settings
    from src.db.connection import db

    settings = Settings()
    await db.connect(settings)

    where_clause = ""
    args: list[Any] = []
    if execution_id is not None:
        where_clause = "WHERE d.execution_id = $1"
        args = [execution_id]

    try:
        per_src_rows = await db.fetch(
            _PER_SOURCE.format(where_clause=where_clause), *args
        )
        agree_row = await db.fetchrow(
            _AGREEMENT.format(where_clause=where_clause), *args
        )
        far_row = await db.fetchrow(
            _FAR_PROXY.format(where_clause=where_clause), *args
        )

        per_source = []
        for r in per_src_rows:
            row = dict(r)
            # Coerce Decimal → float for JSON serialisability.
            from decimal import Decimal
            for k, v in list(row.items()):
                if isinstance(v, Decimal):
                    row[k] = float(v)
            per_source.append(row)

        total = int(agree_row["total"] or 0)
        fused = int(agree_row["fused"] or 0)
        yolo = int(agree_row["yolo_only"] or 0)
        cfar = int(agree_row["cfar_only"] or 0)

        agreement = {
            "total_detections": total,
            "fused": fused,
            "yolo_only": yolo,
            "cfar_only": cfar,
            "agreement_rate_pct": (
                round(100.0 * fused / total, 2) if total else 0.0
            ),
            "interpretation": (
                "Fraction where YOLO and CFAR fired on the same bbox after "
                "post-processing fusion. Low values mean the two detectors "
                "are largely independent — a feature for ensemble robustness, "
                "not a bug."
            ),
        }

        anomaly_rate = float(far_row["anomaly_rate"] or 0)
        on_land_rate = float(far_row["on_land_rate"] or 0)
        far_proxy = {
            "cluster_anomaly_rate_pct": round(100.0 * anomaly_rate, 2),
            "on_land_rate_pct": round(100.0 * on_land_rate, 2),
            "interpretation": (
                "cluster_anomaly_rate is an upper bound on false-alarm rate "
                "due to dense-clutter artefacts (I-DET-3). on_land_rate "
                "shows how much I-DET-2 recovered from the absence of "
                "global-land-mask in early runs (now backfilled)."
            ),
        }

        return {
            "execution_id": str(execution_id) if execution_id else "ALL_RUNS",
            "per_source": per_source,
            "detector_agreement": agreement,
            "far_proxy": far_proxy,
            "disclaimer": (
                "Operational diagnostics, NOT validated mAP/Pd/FAR. "
                "Validation against a labelled Strait-of-Gibraltar split "
                "(or xView3-SAR val) is required for AI Act conformity "
                "and is tracked as a deliverable for D2."
            ),
        }
    finally:
        await db.disconnect()


def render_text(d: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"AIDRA — Operational metrics ({d['execution_id']})")
    lines.append("=" * 70)
    lines.append("")
    lines.append("PER SOURCE:")
    lines.append(
        f"{'source':<8} {'total':>7} {'on_land':>7} {'anomaly':>7} "
        f"{'clean':>7} {'avg_conf':>9} {'min':>7} {'max':>7}"
    )
    for r in d["per_source"]:
        lines.append(
            f"{r['source']:<8} {r['total']:>7} {r['on_land']:>7} "
            f"{r['cluster_anomaly']:>7} {r['clean']:>7} "
            f"{r['avg_conf']:>9} {r['min_conf']:>7} {r['max_conf']:>7}"
        )
    lines.append("")
    lines.append("DETECTOR AGREEMENT (YOLO ∩ CFAR via post-processing fusion):")
    a = d["detector_agreement"]
    lines.append(f"  total={a['total_detections']}  fused={a['fused']}  "
                 f"yolo-only={a['yolo_only']}  cfar-only={a['cfar_only']}")
    lines.append(f"  agreement_rate = {a['agreement_rate_pct']}%")
    lines.append("")
    lines.append("FAR PROXY:")
    f = d["far_proxy"]
    lines.append(f"  cluster_anomaly_rate = {f['cluster_anomaly_rate_pct']}%")
    lines.append(f"  on_land_rate         = {f['on_land_rate_pct']}%")
    lines.append("")
    lines.append("DISCLAIMER: " + d["disclaimer"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-id", type=UUID, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = asyncio.run(collect(args.execution_id))
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render_text(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
