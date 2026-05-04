#!/usr/bin/env python3
"""Convert a PyTorch YOLOv8 weight to dynamic-INT8 ONNX.

Produces the {baseline FP32, INT8 variant, profile} triplet that AIDRA
invariant I-MOD-1 requires for any compression run.  Outputs:

    <models_dir>/<basename>.onnx                   FP32 intermediate (ONNX)
    <models_dir>/<basename>-int8-dynamic.onnx      INT8 quantized
    <cards_dir>/<basename>-int8-dynamic.MODEL_CARD.md

The model is registered at the next AIDRA startup because
:class:`src.models.manager.ModelManager` auto-discovers ``.pt`` /
``.onnx`` files in the models directory and parses the suffix
(see ``_COMPRESSION_SUFFIXES``).  No DB writes are performed by this
script; it is a pure file producer.

Usage::

    python scripts/quantize_to_int8.py \\
        --pt-source models/vesseltracker-sar-yolov8.pt \\
        --models-dir models \\
        --cards-dir models/cards

Notes
-----
- Dynamic quantization computes scales at inference time from the
  observed activation ranges; no calibration dataset required.  This
  is appropriate for a feasibility study but a production system
  would prefer static quantization with a representative SAR
  calibration set.
- The intermediate FP32 ONNX is kept on disk; it is the input to
  quantize_dynamic and also useful as a portable baseline for
  cross-framework benchmarking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("quantize_to_int8")

_CARD_TEMPLATE = """# Model Card — {model_id} (INT8 dynamic)

## Identity

| Field | Value |
|---|---|
| Model ID | `{model_id}` |
| Base model | [`{base_name}`](./{base_name}.MODEL_CARD.md) |
| Variant tag | `int8-dynamic` |
| Format | ONNX |
| File | `{onnx_filename}` |
| File size | {size_mb:.2f} MB ({reduction_pct:.0f}% smaller than FP32 baseline) |
| SHA256 | `{file_hash}` |
| Created | {created_at} |
| Created by | `scripts/quantize_to_int8.py` |

## Compression method

- **Technique:** Dynamic INT8 quantization via `onnxruntime.quantization.quantize_dynamic`.
- **Calibration:** None (dynamic quantization computes activation
  scales at inference time from observed ranges).
- **Operators quantized:** MatMul / Conv weights converted to INT8;
  activations remain FP32 and are dynamically quantized per-batch.
- **Reproducibility:** Re-running the script on the same FP32 ONNX
  produces a byte-identical INT8 file (no randomness involved).

## Source artifacts (provenance chain)

- FP32 PyTorch checkpoint: `{pt_filename}`
  - SHA256: `{pt_hash}`
  - Size: {pt_size_mb:.2f} MB
- FP32 ONNX intermediate: `{onnx_fp32_filename}`
  - SHA256: `{onnx_fp32_hash}`
  - Size: {onnx_fp32_size_mb:.2f} MB

## Expected behaviour

Dynamic INT8 typically loses 1–3 mAP points versus FP32 on detection
tasks while running 1.5–3× faster on CPU and using ~75% less weight
memory.  Concrete deltas for AIDRA must be measured empirically and
recorded in the corresponding ``execution_log`` rows
(``compression_technique = 'dynamic_int8'``).

The terna {{baseline FP32, INT8 variant, hardware profile}} required
by invariant I-MOD-1 is satisfied once this variant has been
benchmarked under at least one constraint profile.

## Limitations and known issues

- Dynamic quantization quantizes activations on the fly, which is
  slower than static quantization for the same accuracy band.  A
  follow-up static-INT8 variant with SAR calibration data is the
  natural next step for the D3 evidence package.
- The terna baseline (`{base_name}` FP32) MUST be benchmarked under
  the same scene + tile_size + iou_threshold settings to make the
  comparison meaningful.  See ``run-triplet`` slash command.

## AI Act references

- Regulation (EU) 2024/1689 Annex IV — technical documentation:
  this card plus the persisted ``execution_log`` rows constitute the
  complete model documentation for this variant.
- See ``../{base_name}.MODEL_CARD.md`` for dataset, intended use,
  performance scope and bias considerations carried over from the
  baseline; only the compression method differs here.
"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def export_pt_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    """Convert a PyTorch YOLO checkpoint to FP32 ONNX."""
    from ultralytics import YOLO

    logger.info("Loading PyTorch checkpoint: %s", pt_path)
    yolo = YOLO(str(pt_path))

    logger.info("Exporting to ONNX (FP32) ...")
    # YOLO.export writes the file in the same directory as the .pt
    # (with the same stem) by default; we rename it afterwards.
    exported = yolo.export(format="onnx", opset=13, dynamic=False, simplify=True)
    exported_path = Path(exported)

    if exported_path != onnx_path:
        if onnx_path.exists():
            onnx_path.unlink()
        exported_path.rename(onnx_path)
    logger.info("FP32 ONNX written: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)


def quantize_dynamic_int8(fp32_path: Path, int8_path: Path) -> None:
    """Apply dynamic INT8 quantization to a FP32 ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info("Quantizing %s → %s (dynamic INT8) ...", fp32_path.name, int8_path.name)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("INT8 ONNX written: %s (%.2f MB)", int8_path, int8_path.stat().st_size / 1e6)


def write_model_card(
    *,
    cards_dir: Path,
    base_name: str,
    pt_path: Path,
    onnx_fp32_path: Path,
    int8_path: Path,
) -> Path:
    """Render the variant MODEL_CARD.md from the template."""
    cards_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"{base_name}-int8-dynamic"
    int8_size = int8_path.stat().st_size / 1e6
    fp32_size = onnx_fp32_path.stat().st_size / 1e6
    pt_size = pt_path.stat().st_size / 1e6
    reduction = (1 - int8_size / fp32_size) * 100 if fp32_size else 0
    body = _CARD_TEMPLATE.format(
        model_id=model_id,
        base_name=base_name,
        onnx_filename=int8_path.name,
        size_mb=int8_size,
        reduction_pct=reduction,
        file_hash=sha256_file(int8_path),
        created_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        pt_filename=pt_path.name,
        pt_hash=sha256_file(pt_path),
        pt_size_mb=pt_size,
        onnx_fp32_filename=onnx_fp32_path.name,
        onnx_fp32_hash=sha256_file(onnx_fp32_path),
        onnx_fp32_size_mb=fp32_size,
    )
    card_path = cards_dir / f"{model_id}.MODEL_CARD.md"
    card_path.write_text(body)
    logger.info("Model card written: %s", card_path)
    return card_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--pt-source",
        required=True,
        type=Path,
        help="Path to the FP32 PyTorch checkpoint (.pt) to quantize.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where the resulting .onnx files are written.",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=Path("models/cards"),
        help="Directory where the variant MODEL_CARD.md is written.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable summary.",
    )
    args = parser.parse_args()

    pt_path: Path = args.pt_source.resolve()
    if not pt_path.exists():
        logger.error("PyTorch source not found: %s", pt_path)
        return 1
    if pt_path.suffix.lower() != ".pt":
        logger.error("Expected a .pt file; got %s", pt_path.suffix)
        return 1

    models_dir: Path = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    base_name = pt_path.stem  # e.g. "vesseltracker-sar-yolov8"
    onnx_fp32_path = models_dir / f"{base_name}.onnx"
    int8_path = models_dir / f"{base_name}-int8-dynamic.onnx"

    logger.info("Triplet target: baseline=%s.pt → fp32=%s → int8=%s",
                base_name, onnx_fp32_path.name, int8_path.name)

    export_pt_to_onnx(pt_path, onnx_fp32_path)
    quantize_dynamic_int8(onnx_fp32_path, int8_path)
    card_path = write_model_card(
        cards_dir=args.cards_dir.resolve(),
        base_name=base_name,
        pt_path=pt_path,
        onnx_fp32_path=onnx_fp32_path,
        int8_path=int8_path,
    )

    summary = {
        "base_model": base_name,
        "pt_source": {
            "path": str(pt_path),
            "size_mb": round(pt_path.stat().st_size / 1e6, 2),
            "sha256": sha256_file(pt_path),
        },
        "fp32_onnx": {
            "path": str(onnx_fp32_path),
            "size_mb": round(onnx_fp32_path.stat().st_size / 1e6, 2),
            "sha256": sha256_file(onnx_fp32_path),
        },
        "int8_onnx": {
            "path": str(int8_path),
            "size_mb": round(int8_path.stat().st_size / 1e6, 2),
            "sha256": sha256_file(int8_path),
        },
        "model_card": str(card_path),
        "size_reduction_pct": round(
            (1 - int8_path.stat().st_size / onnx_fp32_path.stat().st_size) * 100, 1
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2))
        logger.info("Summary JSON: %s", args.summary_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
