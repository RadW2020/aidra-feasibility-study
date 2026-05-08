"""AIDRA validation module: mAP / Pd / FAR computation and persistence.

Single source of truth for the matching logic and the report structure
used by:

* ``scripts/run_validation.py`` — CLI harness consuming a labels manifest.
* ``src/api/validation.py`` — HTTP endpoint that runs synthetic
  validation in-process and persists the result.

Splitting these helpers out of ``scripts/run_validation.py`` lets both
entry points share identical matcher code so a synthetic validation
and a real-dataset run produce comparable numbers.
"""

from src.validation.metrics import (
    ValidationReport,
    bbox_center_distance,
    bbox_iou,
    match_predictions,
    pr_curve_from_scored,
)
from src.validation.persistence import persist_report

__all__ = [
    "ValidationReport",
    "bbox_iou",
    "bbox_center_distance",
    "match_predictions",
    "pr_curve_from_scored",
    "persist_report",
]
