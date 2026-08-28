"""
Offline analysis of ``--dump-predictions`` output from the xView3 validator.

Answers, from the dumped prediction sets and ground truth of every scene,
the questions the aggregate metrics cannot:

1. **Why does the CFAR ∩ YOLO fusion (IoU >= 0.3) never fire?**
   Box-size statistics per source and, for every YOLO box, the distance
   to the nearest CFAR box and their IoU. A high "within 20 px" rate with
   a near-zero "IoU >= fusion threshold" rate is the signature of a
   geometric mismatch (point-like CFAR clusters vs. YOLO extents), not of
   the detectors disagreeing about where vessels are.
2. **Complementarity.** Venn split of the ground truth: recovered by CFAR
   only, YOLO only, both, or neither (centre matching, xView3 tolerance).
3. **D4 sampling pool.** Per set, every TP / FP / FN with scene id, bbox,
   confidence and ``on_land`` so the interpretability run can draw
   stratified samples instead of top-confidence-only ones.

Run::

    python -m scripts.analyze_prediction_dumps \
        --dumps reports/predictions/xview3_adriatic_full \
        --output reports/analysis_xview3_adriatic_full.json

Writes ``<output>.json`` plus a Markdown summary next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("aidra.analyze_dumps")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.validation.metrics import bbox_center_distance, bbox_iou  # noqa: E402


def _size_stats(boxes: list[list[float]]) -> dict[str, float | int]:
    if not boxes:
        return {"n": 0}
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    areas = [w * h for w, h in zip(widths, heights, strict=True)]
    return {
        "n": len(boxes),
        "width_px_median": round(statistics.median(widths), 1),
        "height_px_median": round(statistics.median(heights), 1),
        "area_px2_median": round(statistics.median(areas), 1),
        "area_px2_p90": round(sorted(areas)[int(0.9 * (len(areas) - 1))], 1),
    }


def _nearest(box: list[float], candidates: list[list[float]]) -> tuple[float, float]:
    """Return ``(min_center_distance, iou_with_that_candidate)``."""
    best_d = float("inf")
    best_iou = 0.0
    for c in candidates:
        d = bbox_center_distance(box, c)
        if d < best_d:
            best_d = d
            best_iou = bbox_iou(box, c)
    return best_d, best_iou


def _match_sets(
    preds: list[dict[str, Any]], gt: list[dict[str, Any]], tol_px: float
) -> tuple[set[int], list[int], list[int]]:
    """Greedy centre matching. Returns (matched_gt_idx, tp_pred_idx, fp_pred_idx)."""
    order = sorted(range(len(preds)), key=lambda i: -float(preds[i].get("confidence", 0.0)))
    matched: set[int] = set()
    tp_idx: list[int] = []
    fp_idx: list[int] = []
    for pi in order:
        best = None
        best_d = tol_px
        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            d = bbox_center_distance(preds[pi]["bbox"], g["bbox"])
            if d <= best_d:
                best_d = d
                best = gi
        if best is None:
            fp_idx.append(pi)
        else:
            matched.add(best)
            tp_idx.append(pi)
    return matched, tp_idx, fp_idx


def analyze(dump_dir: Path, *, tol_px: float, fusion_iou: float) -> dict[str, Any]:
    files = sorted(dump_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No dumps under {dump_dir}")

    cfar_boxes: list[list[float]] = []
    yolo_boxes: list[list[float]] = []
    yolo_to_cfar: list[tuple[float, float]] = []
    venn = {"both": 0, "cfar_only": 0, "yolo_only": 0, "neither": 0}
    gt_total = 0
    pool: dict[str, dict[str, list[dict[str, Any]]]] = {}
    per_scene: list[dict[str, Any]] = []

    for f in files:
        d = json.loads(f.read_text())
        sid = d["scene_id"]
        gt = d["ground_truth"]
        sets: dict[str, list[dict[str, Any]]] = d["sets"]
        gt_total += len(gt)

        cfar = sets.get("cfar", [])
        yolo = sets.get("yolo", [])
        cfar_boxes.extend(p["bbox"] for p in cfar)
        yolo_boxes.extend(p["bbox"] for p in yolo)
        cfar_bb = [p["bbox"] for p in cfar]
        for p in yolo:
            yolo_to_cfar.append(_nearest(p["bbox"], cfar_bb) if cfar_bb else (float("inf"), 0.0))

        m_cfar, _, _ = _match_sets(cfar, gt, tol_px)
        m_yolo, _, _ = _match_sets(yolo, gt, tol_px)
        scene_venn = {
            "both": len(m_cfar & m_yolo),
            "cfar_only": len(m_cfar - m_yolo),
            "yolo_only": len(m_yolo - m_cfar),
            "neither": len(gt) - len(m_cfar | m_yolo),
        }
        for k, v in scene_venn.items():
            venn[k] += v

        scene_pool: dict[str, dict[str, int]] = {}
        for name, preds in sets.items():
            matched, tp_idx, fp_idx = _match_sets(preds, gt, tol_px)
            bucket = pool.setdefault(name, {"tp": [], "fp": [], "fn": []})
            for i in tp_idx:
                bucket["tp"].append({"scene_id": sid, **preds[i]})
            for i in fp_idx:
                bucket["fp"].append({"scene_id": sid, **preds[i]})
            for gi, g in enumerate(gt):
                if gi not in matched:
                    bucket["fn"].append({"scene_id": sid, **g})
            scene_pool[name] = {"tp": len(tp_idx), "fp": len(fp_idx), "fn": len(gt) - len(matched)}

        per_scene.append({"scene_id": sid, "n_gt": len(gt), "venn_cfar_yolo": scene_venn, "sets": scene_pool})

    n_yolo = len(yolo_to_cfar)
    within = sum(1 for d, _ in yolo_to_cfar if d <= tol_px)
    iou_hits = sum(1 for _, i in yolo_to_cfar if i >= fusion_iou)
    ious_within = [i for d, i in yolo_to_cfar if d <= tol_px]

    return {
        "dump_dir": str(dump_dir),
        "num_scenes": len(files),
        "num_ground_truth": gt_total,
        "center_tolerance_px": tol_px,
        "fusion_iou_threshold": fusion_iou,
        "box_sizes": {"cfar": _size_stats(cfar_boxes), "yolo": _size_stats(yolo_boxes)},
        "yolo_vs_nearest_cfar": {
            "n_yolo": n_yolo,
            "within_tolerance": within,
            "within_tolerance_rate": round(within / n_yolo, 4) if n_yolo else None,
            "iou_ge_fusion_threshold": iou_hits,
            "iou_ge_fusion_threshold_rate": round(iou_hits / n_yolo, 4) if n_yolo else None,
            "iou_median_when_within_tolerance": (
                round(statistics.median(ious_within), 4) if ious_within else None
            ),
            "iou_max_when_within_tolerance": round(max(ious_within), 4) if ious_within else None,
        },
        "gt_venn_cfar_yolo": {
            **venn,
            "recovered_by_any_rate": round((gt_total - venn["neither"]) / gt_total, 4) if gt_total else None,
        },
        "pool_sizes": {
            name: {k: len(v) for k, v in buckets.items()} for name, buckets in pool.items()
        },
        "per_scene": per_scene,
        "pool": pool,
    }


def _markdown(a: dict[str, Any]) -> str:
    y = a["yolo_vs_nearest_cfar"]
    v = a["gt_venn_cfar_yolo"]
    bs = a["box_sizes"]
    lines = [
        f"## Análisis de volcados — {a['num_scenes']} escenas, {a['num_ground_truth']} GT\n",
        "### Geometría de las cajas",
        f"- CFAR: n={bs['cfar'].get('n')}, mediana {bs['cfar'].get('width_px_median')}×"
        f"{bs['cfar'].get('height_px_median')} px (área mediana {bs['cfar'].get('area_px2_median')} px²)",
        f"- YOLO: n={bs['yolo'].get('n')}, mediana {bs['yolo'].get('width_px_median')}×"
        f"{bs['yolo'].get('height_px_median')} px (área mediana {bs['yolo'].get('area_px2_median')} px²)",
        "",
        f"### ¿Por qué no dispara la fusión (IoU ≥ {a['fusion_iou_threshold']})?",
        f"- Cajas YOLO con un CFAR a ≤ {a['center_tolerance_px']:.0f} px: "
        f"{y['within_tolerance']}/{y['n_yolo']} ({(y['within_tolerance_rate'] or 0) * 100:.1f} %)",
        f"- Cajas YOLO con IoU ≥ umbral frente al CFAR más cercano: "
        f"{y['iou_ge_fusion_threshold']}/{y['n_yolo']} ({(y['iou_ge_fusion_threshold_rate'] or 0) * 100:.2f} %)",
        f"- IoU mediana / máxima cuando están co-localizadas: "
        f"{y['iou_median_when_within_tolerance']} / {y['iou_max_when_within_tolerance']}",
        "",
        "### Complementariedad sobre el GT (matching por centro)",
        f"- Ambos: {v['both']} · solo CFAR: {v['cfar_only']} · solo YOLO: {v['yolo_only']} · "
        f"ninguno: {v['neither']} · recuperado por alguno: {(v['recovered_by_any_rate'] or 0) * 100:.1f} %",
        "",
        "### Tamaño del pool para muestreo D4 (TP / FP / FN por conjunto)",
    ]
    for name, sizes in a["pool_sizes"].items():
        lines.append(f"- `{name}`: TP {sizes['tp']} · FP {sizes['fp']} · FN {sizes['fn']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center-tolerance-px", type=float, default=20.0)
    parser.add_argument(
        "--fusion-iou",
        type=float,
        default=None,
        help="Defaults to Settings.fusion_iou_threshold.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.fusion_iou is None:
        from src.config import Settings

        args.fusion_iou = Settings().fusion_iou_threshold

    result = analyze(args.dumps, tol_px=args.center_tolerance_px, fusion_iou=args.fusion_iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    md = _markdown(result)
    args.output.with_suffix(".md").write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
