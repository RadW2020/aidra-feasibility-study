"""
Run YOLOv8 Grad-CAM + CFAR score map on detection thumbnails (D4 annex).

Thin CLI wrapper around
``src.models.interpretability.run_interpretability_for_execution`` so the
same orchestration logic is invokable from the API
(``POST /api/interpretability/run``) and from the command line.

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
import logging
import sys
from pathlib import Path
from uuid import UUID

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("aidra.interpretability.cli")


async def main_async(
    execution_id: UUID | None,
    n_samples: int,
    model_name: str | None,
    out_root: Path,
) -> int:
    from src.config import Settings
    from src.db.connection import db
    from src.models.interpretability import run_interpretability_for_execution

    settings = Settings()
    await db.connect(settings)
    try:
        try:
            result = await run_interpretability_for_execution(
                db=db,
                models_dir=Path(settings.models_dir),
                out_root=out_root,
                execution_id=execution_id,
                n_samples=n_samples,
                model_name=model_name,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 2

        logger.info(
            "DONE run=%s | gradcam_ok=%d/%d cfar_ok=%d/%d -> %s",
            result["run_id"],
            result["gradcam_ok"],
            result["n_samples"],
            result["cfar_ok"],
            result["n_samples"],
            result["manifest_path"],
        )
        sys.stdout.write(result["manifest_path"] + "\n")
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
