"""What-if on saved xView3 prediction dumps: CFAR-only detections near land.

Measures, without re-running inference, what dropping CFAR-only detections
within D km of land would do to Pd / precision / FAR, and how often those
detections match a ground-truth vessel by distance band. It is the evidence
behind ``Settings.coastal_buffer_km`` and the ``coastal_candidate`` verdict.

    python scripts/coastal_buffer_whatif.py \
        reports/predictions/xview3_adriatic_full_vessels_r14r15 468575.2

Arguments: the dump directory written by the validation harness, and the
total scene area in km2 (``total_area_km2`` of the matching report). With a
buffer of 0 the output reproduces that report's TP / FP / FN.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from global_land_mask import globe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.metrics import match_predictions  # noqa: E402

BUFFERS_KM = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
BANDS_KM = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, math.inf)]
MAX_KM = 3.0
STEP_KM = 0.25
BEARINGS = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
MATCH = {"iou_threshold": 0.5, "match_mode": "center", "center_tolerance_px": 20.0}


def lonlat(affine: list[float], bbox: list[float]) -> tuple[float, float]:
    """Centre of a pixel bbox through the scene's rotation-aware affine."""
    col = (bbox[0] + bbox[2]) / 2.0
    row = (bbox[1] + bbox[3]) / 2.0
    origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h = affine[:6]
    return origin_x + col * pixel_w + row * rot_lon, origin_y + col * rot_lat + row * pixel_h


def distance_to_land_km(lon: float, lat: float) -> float:
    """Smallest sampled radius with land on it; inf beyond ``MAX_KM``."""
    if globe.is_land(lat, lon):
        return 0.0
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    for radius in np.arange(STEP_KM, MAX_KM + 1e-9, STEP_KM):
        lats = lat + radius / 111.32 * np.sin(BEARINGS)
        lons = lon + radius / km_per_deg_lon * np.cos(BEARINGS)
        if np.any(globe.is_land(lats, lons)):
            return float(radius)
    return math.inf


def main(dump_dir: Path, area_km2: float) -> None:
    scenes = [json.loads(p.read_text()) for p in sorted(dump_dir.glob("*.json"))]
    for scene in scenes:
        for pred in scene["sets"]["aidra"]:
            pred["_km"] = (
                distance_to_land_km(*lonlat(scene["lonlat_affine"], pred["bbox"]))
                if pred["source"] == "cfar"
                else math.inf
            )
    print(f"scenes={len(scenes)} ground_truth={sum(len(s['ground_truth']) for s in scenes)}")

    print("\nDropping CFAR-only detections within the buffer:")
    print(f"{'buffer_km':>9} {'preds':>6} {'dropped':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'Pd':>7} {'prec':>7} {'FAR/km2':>8}")
    for buffer_km in BUFFERS_KM:
        tp = fp = fn = kept_n = dropped = 0
        for scene in scenes:
            preds = scene["sets"]["aidra"]
            kept = [p for p in preds if not (buffer_km > 0 and p["source"] == "cfar" and p["_km"] <= buffer_km)]
            dropped += len(preds) - len(kept)
            kept_n += len(kept)
            s_tp, s_fp, s_fn, _ = match_predictions(kept, scene["ground_truth"], **MATCH)
            tp += s_tp
            fp += s_fp
            fn += s_fn
        print(
            f"{buffer_km:>9} {kept_n:>6} {dropped:>8} {tp:>5} {fp:>5} {fn:>5} "
            f"{tp / (tp + fn):>7.4f} {tp / max(tp + fp, 1):>7.4f} {fp / area_km2:>8.5f}"
        )

    print("\nCFAR-only detections by distance to land, and how many match a vessel:")
    for low, high in BANDS_KM:
        total = hits = 0
        for scene in scenes:
            preds = scene["sets"]["aidra"]
            _, _, _, scored = match_predictions(preds, scene["ground_truth"], **MATCH)
            ordered = sorted(preds, key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
            for pred, (_, is_tp) in zip(ordered, scored, strict=True):
                if pred["source"] == "cfar" and low < pred["_km"] <= high:
                    total += 1
                    hits += int(is_tp)
        label = f"{low}-{high} km" if math.isfinite(high) else f">{low} km"
        print(f"  {label:>12}: {total:>5} detections, {hits:>4} matched ({100 * hits / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]))
