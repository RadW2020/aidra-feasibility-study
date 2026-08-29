"""Detection tiers derived from ``detections.source``.

``source='fused'`` means CFAR and YOLO agreed on the same object (centre
distance <= Settings.fusion_center_tolerance_px). On xView3 (11 Adriatic
scenes, 2026-08-29) that subset had precision 0.31 vs 0.16 for the full
output and FAR 0.0011 vs 0.0083/km2, at Pd 0.12 vs 0.36 — a high-precision
operating point, not a replacement for the full output (I-DET-2: nothing is
dropped; this only labels and filters).
"""

from __future__ import annotations

import re

SOURCES: tuple[str, ...] = ("cfar", "yolo", "fused")
TIER_HIGH = "high"
TIER_STANDARD = "standard"
_SAFE = re.compile(r"^(cfar|yolo|fused)$")


def tier_of(source: str | None) -> str:
    return TIER_HIGH if source == "fused" else TIER_STANDARD


def source_sql_clause(source: str | None, tier: str | None, alias: str = "d") -> str:
    """Inline SQL predicate for the optional ``source`` / ``tier`` filters.

    Values are validated against a closed enum before being interpolated, so
    the clause is safe to splice into a positional query without shifting
    its parameters. ``tier=high`` is ``source=fused``; ``tier=standard`` is
    everything else.
    """
    if source is not None:
        if not _SAFE.match(source):
            raise ValueError(f"invalid source {source!r}")
        return f"AND {alias}.source = '{source}'"
    if tier == TIER_HIGH:
        return f"AND {alias}.source = 'fused'"
    if tier == TIER_STANDARD:
        return f"AND {alias}.source <> 'fused'"
    return ""
