"""Ground-truth-anchored sampling and fidelity metrics for the D4 annex.

The production interpretability run (``src.models.interpretability``) has no
ground truth: it explains detections of a live scene. The xView3 validation
dumps (``scripts/validate_xview3_serial.py --dump-predictions``) do have it,
so the D4 annex can also show *true positives at high and low confidence,
false positives and missed vessels* and measure whether the Grad-CAM heat
actually points at the vessel (pointing game, heat mass inside the box).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from src.validation.metrics import bbox_center_distance

STRATA: tuple[str, ...] = ("tp_high", "tp_low", "fp", "fn")


def _centre(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def build_pool(
    preds: list[dict[str, Any]], gt: list[dict[str, Any]], tol_px: float = 20.0
) -> dict[str, list[dict[str, Any]]]:
    """Split one scene into TP (with matched GT), FP and FN (missed GT).

    Greedy centre matching by descending confidence, same convention as the
    validation metrics. Predictions and GT flagged ``on_land`` are skipped:
    they are excluded from the sea metrics (I-DET-2) and would only add
    land clutter to the annex.
    """
    preds = [p for p in preds if not p.get("on_land", False)]
    gt = [g for g in gt if not g.get("on_land", False)]
    order = sorted(range(len(preds)), key=lambda i: -float(preds[i].get("confidence", 0.0)))
    matched: dict[int, int] = {}
    pool: dict[str, list[dict[str, Any]]] = {"tp": [], "fp": [], "fn": []}
    for pi in order:
        best, best_d = None, tol_px
        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            d = bbox_center_distance(preds[pi]["bbox"], g["bbox"])
            if d <= best_d:
                best, best_d = gi, d
        if best is None:
            pool["fp"].append({**preds[pi], "kind": "fp"})
        else:
            matched[best] = pi
            pool["tp"].append({**preds[pi], "kind": "tp", "gt_bbox": gt[best]["bbox"],
                               "gt_vessel_length_m": gt[best].get("vessel_length_m")})
    for gi, g in enumerate(gt):
        if gi not in matched:
            pool["fn"].append({"bbox": g["bbox"], "gt_bbox": g["bbox"], "confidence": None,
                               "source": None, "kind": "fn",
                               "gt_vessel_length_m": g.get("vessel_length_m")})
    return pool


def stratify(
    pool: dict[str, list[dict[str, Any]]], n_per_stratum: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seeded sample of ``n_per_stratum`` items from tp_high / tp_low / fp / fn.

    TP are split at the median TP confidence. Every picked item carries
    ``stratum``. Returns the samples and a summary of pool and picked sizes.
    """
    rng = random.Random(seed)
    tps = sorted(pool["tp"], key=lambda p: float(p["confidence"]))
    half = len(tps) // 2
    groups = {
        "tp_low": tps[:half],
        "tp_high": tps[half:],
        "fp": list(pool["fp"]),
        "fn": list(pool["fn"]),
    }
    picked: list[dict[str, Any]] = []
    info: dict[str, Any] = {"seed": seed, "n_per_stratum": n_per_stratum, "strata": {}}
    for name in STRATA:
        items = list(groups[name])
        rng.shuffle(items)
        take = items[:n_per_stratum]
        for it in take:
            picked.append({**it, "stratum": name})
        info["strata"][name] = {"pool": len(groups[name]), "picked": len(take)}
    if tps:
        info["tp_confidence_median"] = float(tps[half]["confidence"]) if half < len(tps) else None
    return picked, info


def chip_window(
    centre_xy: tuple[float, float], chip: int, height: int, width: int
) -> tuple[int, int, int, int]:
    """``(row0, col0, rows, cols)`` of a ``chip``-sized window around a centre, clamped."""
    cx, cy = centre_xy
    r0 = int(round(cy)) - chip // 2
    c0 = int(round(cx)) - chip // 2
    r0 = max(0, min(r0, max(0, height - chip)))
    c0 = max(0, min(c0, max(0, width - chip)))
    return r0, c0, min(chip, height), min(chip, width)


def pointing_game(
    cam: np.ndarray, box_local: list[float], tol_px: float = 20.0
) -> dict[str, float | bool]:
    """Fidelity of a heatmap w.r.t. a box in chip-local pixel coordinates.

    - ``hit``: the CAM argmax falls inside the box grown by ``tol_px``.
    - ``mass_in_box``: fraction of total heat inside the (grown) box.
    - ``peak_distance_px``: distance from the argmax to the box centre.
    """
    cam = np.asarray(cam, dtype=np.float64)
    if cam.size == 0 or float(cam.sum()) <= 0.0:
        return {"hit": False, "mass_in_box": 0.0, "peak_distance_px": float("inf")}
    h, w = cam.shape
    x0, y0, x1, y1 = box_local
    gx0, gy0 = max(0, int(np.floor(x0 - tol_px))), max(0, int(np.floor(y0 - tol_px)))
    gx1, gy1 = min(w, int(np.ceil(x1 + tol_px))), min(h, int(np.ceil(y1 + tol_px)))
    r, c = np.unravel_index(int(np.argmax(cam)), cam.shape)
    hit = gy0 <= r < gy1 and gx0 <= c < gx1
    mass = float(cam[gy0:gy1, gx0:gx1].sum() / cam.sum()) if gx1 > gx0 and gy1 > gy0 else 0.0
    bx, by = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return {
        "hit": bool(hit),
        "mass_in_box": round(mass, 4),
        "peak_distance_px": round(float(((c - bx) ** 2 + (r - by) ** 2) ** 0.5), 1),
    }
