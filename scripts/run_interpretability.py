"""
Run YOLOv8 Grad-CAM + CFAR score map on detection thumbnails (D4 annex).

Direct version: bypasses the run_interpretability orchestrator and calls
gradcam_yolov8 + cfar_score_map per sample, with full tracebacks on
failure. Closes I-AIA-2 and the AI_ACT_DECLARATION.md §5 promise.

Output:
    /data/interpretability/<run_id>/
        000_input.png           (SAR thumbnail, log-stretched)
        000_gradcam.png         (Grad-CAM overlay on YOLOv8 last C2f)
        000_cfar_score.png      (CFAR Pfa pre-threshold heatmap)
        ...
        manifest.json           (commit_sha, model_hash, per-sample SHA256)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("aidra.interpretability.cli")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


async def main_async(
    execution_id: UUID | None,
    n_samples: int,
    model_name: str | None,
    out_root: Path,
) -> int:
    from PIL import Image

    from src.config import Settings
    from src.db.connection import db
    from src.models.interpretability import (
        cfar_score_map,
        gradcam_yolov8,
        save_grayscale_png,
        save_heatmap_png,
    )
    from src.traceability.hasher import get_commit_sha

    settings = Settings()
    await db.connect(settings)
    try:
        if execution_id is None:
            row = await db.fetchrow(
                "SELECT id FROM execution_log WHERE status='success' "
                "AND num_detections > 0 ORDER BY created_at DESC LIMIT 1"
            )
            if row is None:
                logger.error("No successful execution found.")
                return 2
            execution_id = row["id"]

        meta = await db.fetchrow(
            "SELECT model_name, model_hash FROM execution_log WHERE id=$1",
            execution_id,
        )
        picked_model = model_name or meta["model_name"]
        model_hash = meta["model_hash"]
        logger.info("execution=%s model=%s", execution_id, picked_model)

        # Sample N detections with thumbnails (high-conf sea targets).
        rows = await db.fetch(
            "SELECT id, thumbnail_path, confidence, source "
            "FROM detections WHERE execution_id=$1 "
            "  AND thumbnail_path IS NOT NULL "
            "  AND on_land = false AND cluster_anomaly = false "
            "ORDER BY confidence DESC LIMIT $2",
            execution_id,
            n_samples * 3,
        )
        candidates = [dict(r) for r in rows]
        if not candidates:
            logger.error("No detections with thumbnails.")
            return 2
        random.seed(42)
        picked = random.sample(candidates, min(n_samples, len(candidates)))

        # Grad-CAM requires a PyTorch model (autograd). Always load the PT
        # baseline directly — bypassing the registry which may return the
        # latest ONNX variant (INT8). Architecture is identical; heatmaps
        # are transferable (documented in INT8 MODEL_CARD §Interpretabilidad).
        from ultralytics import YOLO as _YOLO

        models_dir = Path(settings.models_dir)
        pt_candidates = sorted(
            [
                p for p in models_dir.glob(f"{picked_model}*.pt")
                if "int8" not in p.name and "pruned" not in p.name
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not pt_candidates:
            logger.error("No .pt model found for %s — Grad-CAM impossible.", picked_model)
            return 2
        pt_path = pt_candidates[0]
        logger.info("Grad-CAM: loading PT model directly: %s", pt_path)
        yolo = _YOLO(str(pt_path))
        logger.info("Loaded YOLO for Grad-CAM: %s", pt_path.name)

        # Output dir.
        run_id = f"{execution_id}_interp_{uuid4().hex[:8]}"
        out_dir = out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id": run_id,
            "execution_id": str(execution_id),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "commit_sha": get_commit_sha(),
            "model_name": picked_model,
            "model_hash": model_hash,
            "n_samples": len(picked),
            "samples": [],
        }

        n_cam_ok = 0
        n_cfar_ok = 0
        for idx, det in enumerate(picked):
            prefix = f"{idx:03d}"
            in_path = out_dir / f"{prefix}_input.png"
            cam_path = out_dir / f"{prefix}_gradcam.png"
            cfar_path = out_dir / f"{prefix}_cfar_score.png"

            try:
                tile = np.asarray(Image.open(det["thumbnail_path"]))
            except Exception:
                logger.warning("Could not load %s", det["thumbnail_path"])
                continue

            save_grayscale_png(tile, in_path)

            cam_ok = False
            try:
                cam = gradcam_yolov8(yolo, tile)
                save_heatmap_png(tile, cam, cam_path)
                cam_ok = True
                n_cam_ok += 1
            except Exception as exc:
                print(
                    f"Grad-CAM FAIL sample {idx}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)

            cfar_ok = False
            try:
                score = cfar_score_map(tile)
                save_heatmap_png(tile, score, cfar_path)
                cfar_ok = True
                n_cfar_ok += 1
            except Exception as exc:
                print(
                    f"CFAR FAIL sample {idx}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

            manifest["samples"].append({
                "idx": idx,
                "detection_id": str(det["id"]),
                "confidence": float(det["confidence"]),
                "source": det["source"],
                "thumbnail_path": det["thumbnail_path"],
                "input_png": in_path.name,
                "input_sha256": _sha256_file(in_path),
                "gradcam_png": cam_path.name if cam_ok else None,
                "gradcam_sha256": _sha256_file(cam_path) if cam_ok else None,
                "cfar_png": cfar_path.name if cfar_ok else None,
                "cfar_sha256": _sha256_file(cfar_path) if cfar_ok else None,
            })

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(
            "DONE run=%s | gradcam_ok=%d/%d cfar_ok=%d/%d -> %s",
            run_id, n_cam_ok, len(picked), n_cfar_ok, len(picked),
            manifest_path,
        )
        sys.stdout.write(str(manifest_path) + "\n")
        return 0
    finally:
        await db.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-id", type=UUID, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="/data/interpretability")
    args = parser.parse_args(argv)
    return asyncio.run(
        main_async(
            execution_id=args.execution_id,
            n_samples=args.n,
            model_name=args.model,
            out_root=Path(args.out),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
