"""
D4 with ground truth: Grad-CAM + CFAR heatmaps on stratified xView3 samples.

Takes one scene's prediction dump (``validate_xview3_serial --dump-predictions``)
and its raster, draws a seeded, stratified sample — true positives at high
and low confidence, false positives, missed vessels — and renders for each:

    <idx>_input.png       unfiltered SAR chip (production YOLO stretch)
    <idx>_gradcam.png     Grad-CAM of the FP32 YOLO baseline (renderer)
    <idx>_cfar_score.png  CFAR test statistic on the Lee-filtered chip

plus ``manifest.json`` with per-sample provenance (scene, bbox, GT bbox,
stratum, source, confidence, PNG SHA256) and the fidelity metrics the
production run cannot compute: pointing-game hit rate and heat mass inside
the vessel box, per stratum. ``summary.md`` is ready to paste into
``D4_INTERPRETABILITY_ANNEX.md``.

Run::

    python -m scripts.run_interpretability_xview3 \\
        --dump reports/predictions/xview3_adriatic_full_vessels_r14r15/264ed833a13b7f2av.json \\
        --tar data/xview3/scenes/264ed833a13b7f2av.tar.gz \\
        --report reports/validation_xview3_adriatic_full_vessels_r14r15_aidra.json \\
        --set aidra --n-per-stratum 5 --chip 256 --seed 42 \\
        --out reports/interpretability/xview3_264ed833a13b7f2av_r14r15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.validation.d4_sampling import (  # noqa: E402
    STRATA,
    build_pool,
    chip_window,
    pointing_game,
    stratify,
)

logger = logging.getLogger("aidra.d4_xview3")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None, help="Validation report whose provenance to copy.")
    parser.add_argument("--set", default="aidra", help="Prediction set in the dump to explain.")
    parser.add_argument("--band", default="VH_dB.tif")
    parser.add_argument("--n-per-stratum", type=int, default=5)
    parser.add_argument("--chip", type=int, default=256)
    parser.add_argument("--tol-px", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--yolo-pt", default="vesseltracker-sar-yolov8", help="FP32 .pt used as Grad-CAM renderer.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("data/xview3/scratch_d4"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-raster", action="store_true")
    parser.add_argument("--gradcam-layer", default="model.model.15", help="Hooked C2f layer (P3=15, P4=18, P5=21).")
    parser.add_argument(
        "--gradcam-mode",
        choices=["targeted", "global"],
        default="targeted",
        help="targeted = backprop the class score of the anchor nearest the box centre; global = mean |output|.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import rasterio
    import rasterio.windows

    from src.config import Settings
    from src.models.interpretability import (
        cfar_score_map,
        gradcam_yolov8,
        save_grayscale_png,
        save_heatmap_png,
    )
    from src.pipeline.detection import sar_linear_to_uint8_gray
    from src.pipeline.preprocessing import apply_lee_filter
    from src.traceability.hasher import compute_sha256, get_commit_sha
    from src.validation.harness import db_to_linear, seed_everything

    settings = Settings()
    seed_info = seed_everything(args.seed)
    dump = json.loads(args.dump.read_text())
    scene_id = dump["scene_id"]
    preds = dump["sets"][args.set]
    gt = dump["ground_truth"]

    pool = build_pool(preds, gt, tol_px=args.tol_px)
    samples, sampling = stratify(pool, args.n_per_stratum, args.seed)
    logger.info("Scene %s set=%s pool tp=%d fp=%d fn=%d -> %d samples", scene_id, args.set,
                len(pool["tp"]), len(pool["fp"]), len(pool["fn"]), len(samples))

    # Raster
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    member = f"{scene_id}/{args.band}"
    raster = args.tmp_dir / member
    if not raster.exists():
        with tarfile.open(args.tar, "r:gz") as tf:
            tf.extract(tf.getmember(member), args.tmp_dir)

    # Renderer: FP32 .pt (the only variant with autograd)
    from ultralytics import YOLO

    pt_path = args.models_dir / f"{args.yolo_pt}.pt"
    yolo = YOLO(str(pt_path))
    renderer_hash = compute_sha256(pt_path)

    args.out.mkdir(parents=True, exist_ok=True)
    report_prov: dict[str, Any] = {}
    report_params: dict[str, Any] = {}
    if args.report and args.report.exists():
        rep = json.loads(args.report.read_text())
        report_prov = rep.get("provenance", {})
        report_params = rep.get("params", {})

    records: list[dict[str, Any]] = []
    with rasterio.open(raster) as src:
        H, W = int(src.height), int(src.width)
        for idx, s in enumerate(samples):
            box = s["gt_bbox"] if s["kind"] in ("tp", "fn") else s["bbox"]
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            r0, c0, rows, cols = chip_window((cx, cy), args.chip, H, W)
            arr = src.read(1, window=rasterio.windows.Window(c0, r0, cols, rows)).astype(np.float32)
            linear = db_to_linear(arr)
            gray = sar_linear_to_uint8_gray(linear)  # production YOLO input (unfiltered)
            filtered = apply_lee_filter(linear, window_size=7)  # production CFAR input
            prefix = f"{idx:03d}_{s['stratum']}"
            in_png, cam_png, cfar_png = (args.out / f"{prefix}_input.png", args.out / f"{prefix}_gradcam.png",
                                         args.out / f"{prefix}_cfar_score.png")
            save_grayscale_png(gray, in_png)
            rec: dict[str, Any] = {
                "idx": idx, "stratum": s["stratum"], "kind": s["kind"], "source": s.get("source"),
                "confidence": s.get("confidence"), "scene_id": scene_id,
                "bbox_scene": s.get("bbox"), "gt_bbox_scene": s.get("gt_bbox"),
                "gt_vessel_length_m": s.get("gt_vessel_length_m"),
                "chip_window": {"row0": r0, "col0": c0, "rows": rows, "cols": cols},
                "input_png": in_png.name, "input_sha256": _sha(in_png),
            }
            box_local = [box[0] - c0, box[1] - r0, box[2] - c0, box[3] - r0]
            rec["box_local"] = [round(v, 1) for v in box_local]
            try:
                target = ((box_local[0] + box_local[2]) / 2.0, (box_local[1] + box_local[3]) / 2.0)
                cam = gradcam_yolov8(
                    yolo, gray,
                    target_layer_name=args.gradcam_layer,
                    target_xy=target if args.gradcam_mode == "targeted" else None,
                    target_tol_px=args.tol_px,
                )
                save_heatmap_png(gray, cam, cam_png)
                rec.update({"gradcam_png": cam_png.name, "gradcam_sha256": _sha(cam_png),
                            "gradcam": pointing_game(cam, box_local, args.tol_px)})
            except Exception as exc:  # pragma: no cover - model/backend dependent
                logger.warning("Grad-CAM failed on sample %d: %s", idx, exc)
                rec.update({"gradcam_png": None, "gradcam_sha256": None, "gradcam": None})
            try:
                score = cfar_score_map(filtered, guard_size=settings.cfar_guard_size,
                                       training_size=settings.cfar_training_size)
                save_heatmap_png(gray, score, cfar_png)
                rec.update({"cfar_png": cfar_png.name, "cfar_sha256": _sha(cfar_png),
                            "cfar": pointing_game(score, box_local, args.tol_px)})
            except Exception as exc:  # pragma: no cover
                logger.warning("CFAR map failed on sample %d: %s", idx, exc)
                rec.update({"cfar_png": None, "cfar_sha256": None, "cfar": None})
            records.append(rec)
            logger.info("  %s %s conf=%s gradcam_hit=%s cfar_hit=%s", prefix, s.get("source"), s.get("confidence"),
                        (rec.get("gradcam") or {}).get("hit"), (rec.get("cfar") or {}).get("hit"))

    # Per-stratum fidelity summary
    summary: dict[str, Any] = {}
    for name in STRATA:
        recs = [r for r in records if r["stratum"] == name]
        for kind in ("gradcam", "cfar"):
            vals = [r[kind] for r in recs if r.get(kind)]
            summary.setdefault(name, {})[kind] = {
                "n": len(vals),
                "pointing_hit_rate": round(sum(v["hit"] for v in vals) / len(vals), 3) if vals else None,
                "mass_in_box_mean": round(float(np.mean([v["mass_in_box"] for v in vals])), 3) if vals else None,
            }

    manifest = {
        "run_id": f"d4_xview3_{scene_id}_{args.set}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "commit_sha": get_commit_sha(),
        "scene_id": scene_id,
        "tar_sha256": dump.get("tar_sha256"),
        "dump": str(args.dump),
        "prediction_set": args.set,
        "execution_model_name": (
            Path(report_prov["model_path"]).stem if report_prov.get("model_path") else report_params.get("yolo_model")
        ),
        "gradcam_mode": args.gradcam_mode,
        "gradcam_layer": args.gradcam_layer,
        "execution_model_hash": report_prov.get("model_hash"),
        "gradcam_model_name": args.yolo_pt,
        "gradcam_model_hash": renderer_hash,
        "cfar_params": {"guard_size": settings.cfar_guard_size, "training_size": settings.cfar_training_size},
        "pipeline_params": {k: report_params.get(k) for k in ("fusion_mode", "yolo_input", "confidence_threshold", "edge_buffer_px")},
        "seed": seed_info,
        "chip_px": args.chip,
        "tolerance_px": args.tol_px,
        "sampling": {"strategy": "stratified_tp_high/tp_low/fp/fn", **sampling},
        "fidelity_summary": summary,
        "samples": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    lines = [f"## D4 con ground truth — escena `{scene_id}`, conjunto `{args.set}` ({len(records)} muestras)\n",
             "| Estrato | Pool | Muestras | Grad-CAM pointing-game | Grad-CAM masa en caja | CFAR pointing-game | CFAR masa en caja |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in STRATA:
        st = sampling["strata"][name]
        g, c = summary[name]["gradcam"], summary[name]["cfar"]
        lines.append(f"| `{name}` | {st['pool']} | {st['picked']} | {g['pointing_hit_rate']} | {g['mass_in_box_mean']} | {c['pointing_hit_rate']} | {c['mass_in_box_mean']} |")
    lines.append("")
    lines.append(f"Grad-CAM `{args.gradcam_mode}` @ `{args.gradcam_layer}`; renderer: `{args.yolo_pt}` (`{renderer_hash[:12]}…`); sujeto: `{manifest['execution_model_name']}` "
                 f"(`{str(report_prov.get('model_hash') or '')[:12]}…`); CFAR guard/training {settings.cfar_guard_size}/{settings.cfar_training_size}; "
                 f"tolerancia {args.tol_px:g} px; seed {args.seed}; commit `{manifest['commit_sha'][:12]}`.")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")

    if not args.keep_raster:
        shutil.rmtree(raster.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
