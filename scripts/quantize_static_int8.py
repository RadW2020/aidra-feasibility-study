#!/usr/bin/env python3
"""Static INT8 quantization of a YOLOv8 weight with an xView3-SAR calibration set.

Replaces the dynamic-INT8 variant (rejected under I-MOD-3: non-deterministic
detection counts, +24.6 % RAM). Static quantization fixes the activation
scales offline from a representative calibration set, so inference is
deterministic run-to-run and needs no runtime dequantization buffers.

Outputs::

    <models_dir>/<base>.onnx                      FP32 ONNX intermediate (reused if present)
    <models_dir>/<base>-int8-static.onnx          static INT8 (QDQ) model
    <cards_dir>/<base>-int8-static.MODEL_CARD.md  card with calibration provenance
    <summary_json>                                machine-readable provenance

Calibration set (deterministic, seeded): ``--n-vessel-tiles`` 640-px tiles
centred on xView3 vessels (``validation.csv``, ``is_vessel=True``,
HIGH/MEDIUM) plus ``--n-sea-tiles`` random open-sea tiles from the same
scene, converted with the production YOLO input stretch (unfiltered
sigma0 → dB → uint8, ``Settings.yolo_input == "unfiltered"``). The card
records the scene id, the tar SHA256, the tile centres, the seed and the
SHA256 of the calibration tensor so the exact set can be rebuilt.

Usage::

    python scripts/quantize_static_int8.py \\
        --pt-source models/vesseltracker-sar-yolov8.pt \\
        --calibration-tar data/xview3/scenes/264ed833a13b7f2av.tar.gz \\
        --validation-csv x-view-us-data/validation.csv \\
        --models-dir models --cards-dir models/cards --seed 42
"""

from __future__ import annotations

import argparse
import csv
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

logger = logging.getLogger("quantize_static_int8")

_CARD_TEMPLATE = """---
model_id: {model_id}
version: int8-static
created_at: {created_date}
base_model: {base_name} v1.0 (FP32 pytorch)
compression_technique: static_int8
format: onnx
license: Apache-2.0
authors: ["AIDRA project — quantization by RadW2020"]
status: candidate
---

# Propósito

Variante INT8 **estática** (ONNX Runtime, formato QDQ, pesos QInt8 /
activaciones QUInt8) del modelo base `{base_name}` v1.0. Sustituye a la
variante dinámica, rechazada bajo I-MOD-3 (conteos de detección no
deterministas, +24.6 % RAM). Estado `candidate` hasta completar la terna
{{baseline FP32, esta variante, perfil}} con la degradación máxima
declarada antes del run (ΔmAP ≤ 5 pts, `Settings`).

# Identidad

| Campo | Valor |
|---|---|
| Fichero | `{onnx_filename}` |
| Tamaño | {size_mb:.2f} MB ({reduction_pct:.0f} % menor que el FP32 ONNX, {pt_reduction_pct:.0f} % menor que el `.pt`) |
| SHA256 | `{file_hash}` |
| Creado | {created_at} por `scripts/quantize_static_int8.py` (commit `{commit_sha}`) |

# Método de compresión

- `onnxruntime.quantization.quantize_static` (QDQ, `weight_type=QInt8`,
  `activation_type=QUInt8`, **`op_types_to_quantize=["Conv"]`**,
  `per_channel=True`, calibración **{calibrate_method}**), vía
  `ModelQuantizer.quantize_static_onnx`.
- **Por qué solo Conv:** con la configuración por defecto de ORT (todos los
  tipos de op) el detector quedaba a 0 detecciones incluso a conf 0.01
  (FP32: 100 en las mismas teselas): los Sigmoid / Softmax / Div / Mul de
  la cabeza DFL en UINT8 colapsan las puntuaciones de clase.
- **Por qué Percentile:** MinMax es frágil — una tesela de mar con un
  reflector brillante fija el rango de activación y el acuerdo con FP32
  sobre 44 teselas con barcos osciló entre 15/55 y 45/55 según qué 16
  teselas de mar entraran en la calibración. Barrido 2026-08-29 (44
  teselas con barcos, conf 0.25, FP32 = 55 detecciones): **Percentile con
  32 teselas de barcos, sin mar: 53 recuperadas / 12 extra / 2 perdidas**;
  Percentile-24 barcos 51 / 9 / 4; Percentile 20 barcos + 4 mar 32 / 1 / 23
  (las teselas de mar, sin objetivos, comprimen el rango de puntuaciones);
  MinMax 15 / 1 / 40; MinMax+media móvil 16 / 0 / 39; Entropy(8) ~1.
  60 imágenes con Percentile agotan 16 GB de RAM (ORT guarda todas las
  activaciones para el histograma). El chequeo es in-sample; la prueba
  fuera de muestra es la terna sobre las 11 escenas xView3.
- Acuerdo con FP32 sobre las {agree_tiles} teselas de calibración con
  barcos (centro ≤ 20 px, conf ≥ 0.25): **{agree_matched}/{agree_fp32}**
  detecciones FP32 recuperadas, {agree_extra} extra, {agree_missed}
  perdidas. INT8 determinista (dos `predict` idénticos): {agree_det}.
- Escalas fijadas offline → salida determinista run a run; mismo FP32 ONNX
  + mismo set de calibración (seed) → mismo fichero INT8.

# Set de calibración (provenance)

| Campo | Valor |
|---|---|
| Escena xView3 | `{scene_id}` (split validation, Adriático) |
| Tar SHA256 | `{tar_sha256}` |
| Ground truth | `{validation_csv}` (`is_vessel=True`, confianza HIGH/MEDIUM) |
| Teselas | {n_vessel} centradas en barcos + {n_sea} de mar abierto aleatorias = {n_total} × 640×640 px |
| Conversión | σ⁰ dB → lineal → uint8 (stretch −25..0 dB), **sin** filtro Lee (= `Settings.yolo_input="unfiltered"`, R15) |
| Seed | {seed} |
| SHA256 del tensor de calibración | `{calib_sha256}` |
| Centros (fila, col) | ver `{summary_json}` |

# Cadena de origen

- FP32 PyTorch: `{pt_filename}` — SHA256 `{pt_hash}` — {pt_size_mb:.2f} MB
- FP32 ONNX intermedio: `{onnx_fp32_filename}` — SHA256 `{onnx_fp32_hash}` — {onnx_fp32_size_mb:.2f} MB

# Métricas

Pendientes de la terna sobre las mismas 11 escenas xView3 que el baseline
(`scripts/validate_xview3_serial.py --pipeline-path full --model all
--yolo-model {model_id}`): AP, Pd, FAR/km², precisión, latencia p50/p95 por
tesela, RAM pico, tamaño. Se anotan aquí al ejecutarla; hasta entonces esta
variante **no** entra en evaluación (I-MOD-1/3).

# Datos de entrenamiento

Idénticos al modelo base — esta variante **no fue re-entrenada**; solo se
fijaron las escalas de activación con el set de calibración descrito
arriba. Ver `{base_name}.MODEL_CARD.md` para dataset, licencia y cobertura
geográfica.

# Sesgos

- Mismos sesgos y domain shift que el baseline (`{base_name}.MODEL_CARD.md`).
- Calibración con una única escena/track del Adriático (VH): las escalas de
  activación pueden no cubrir mares con mayor clutter (Gibraltar, Canal) ni
  otras polarizaciones.

# Limitaciones

- Cuantización de 8 bits en las Conv: las detecciones de confianza media
  son las primeras en perderse (in-sample: 2 de 55 perdidas, 12 extra).
- Reproducibilidad del artefacto condicionada al FP32 ONNX intermedio: dos
  exportaciones de ultralytics del mismo `.pt` no son byte-idénticas (metadatos),
  pero dado el mismo FP32 ONNX + mismo set de calibración el INT8 sí lo es
  (verificado 2026-08-29: SHA256 idéntico en dos cuantizaciones).

# Interpretabilidad

Grad-CAM no es aplicable al grafo ONNX cuantizado (sin autograd). El anexo D4
usa el baseline FP32 `.pt` como *renderer* del mapa de calor y esta variante
como *sujeto* de la explicación (misma convención que la variante dinámica,
`EVIDENCE.md` § D4); el manifest registra ambos hashes por separado.

# Trazabilidad

- `models_registry`: nombre `{base_name}`, versión `int8-static`,
  `compression_technique=static_int8`, `status=candidate`, SHA256 del fichero.
- Cada run persiste `model_hash` (SHA256 de este `.onnx`), `commit_sha`,
  `input_params_hash` e `inference_p50_ms` / `inference_p95_ms` (I-MOD-2).
- Provenance del artefacto: `{summary_json}` (config de cuantización,
  hashes del `.pt` / FP32 ONNX / INT8, set de calibración con centros y SHA256).

# Conformidad AI Act

Reglamento (UE) 2024/1689: sistema de propósito limitado, no Anexo III
(`AI_ACT_DECLARATION.md`).
Documentación técnica (Anexo IV) = esta ficha + `models_registry` +
`execution_log` + reportes de validación (`reports/`). Supervisión humana:
las detecciones son una capa georreferenciada con confianza y
`quality_verdict`; ninguna decisión automatizada sobre personas.
"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _load_vessel_centres(validation_csv: Path, scene_id: str) -> list[tuple[int, int]]:
    """(row, col) of HIGH/MEDIUM vessels of ``scene_id`` in ``validation.csv``."""
    centres: list[tuple[int, int]] = []
    with validation_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["scene_id"] != scene_id:
                continue
            if row.get("is_vessel") != "True" or (row.get("confidence") or "").upper() not in {"HIGH", "MEDIUM"}:
                continue
            try:
                centres.append((int(float(row["detect_scene_row"])), int(float(row["detect_scene_column"]))))
            except (KeyError, ValueError):
                continue
    return centres


def build_calibration_tiles(
    raster_path: Path,
    vessel_centres: list[tuple[int, int]],
    *,
    n_vessel: int,
    n_sea: int,
    tile: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Deterministic calibration tiles as uint8 RGB using the production YOLO stretch."""
    import rasterio
    import rasterio.windows

    from src.pipeline.detection import sar_linear_to_uint8_gray
    from src.validation.harness import db_to_linear

    rng = np.random.default_rng(seed)
    tiles: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    with rasterio.open(raster_path) as src:
        h, w = int(src.height), int(src.width)
        half = tile // 2

        def _read(r0: int, c0: int) -> np.ndarray | None:
            r0 = int(min(max(0, r0), h - tile))
            c0 = int(min(max(0, c0), w - tile))
            arr = src.read(1, window=rasterio.windows.Window(c0, r0, tile, tile)).astype(np.float32)
            valid = (arr > -1000.0).mean()
            if valid < 0.95:
                return None
            gray = sar_linear_to_uint8_gray(db_to_linear(arr))
            return np.stack([gray, gray, gray], axis=-1)

        picked = list(vessel_centres)
        rng.shuffle(picked)
        for r, c in picked:
            if len([m for m in meta if m["kind"] == "vessel"]) >= n_vessel:
                break
            img = _read(r - half, c - half)
            if img is None:
                continue
            tiles.append(img)
            meta.append({"kind": "vessel", "row": int(r), "col": int(c)})

        attempts = 0
        while len([m for m in meta if m["kind"] == "sea"]) < n_sea and attempts < n_sea * 20:
            attempts += 1
            r0 = int(rng.integers(0, max(1, h - tile)))
            c0 = int(rng.integers(0, max(1, w - tile)))
            img = _read(r0, c0)
            if img is None:
                continue
            tiles.append(img)
            meta.append({"kind": "sea", "row": r0 + half, "col": c0 + half})
    return tiles, meta


def calibration_sha256(tiles: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for t in tiles:
        h.update(np.ascontiguousarray(t).tobytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pt-source", required=True, type=Path)
    parser.add_argument("--calibration-tar", required=True, type=Path, help="xView3 scene tar.gz")
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--band", default="VH_dB.tif")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--cards-dir", type=Path, default=Path("models/cards"))
    parser.add_argument("--tmp-dir", type=Path, default=Path("data/xview3/scratch_calib"))
    # 32 vessel-centred tiles, no open-sea tiles: Percentile calibration keeps
    # every activation in RAM (60 tiles -> >16 GB, OOM) and sea tiles hurt
    # (20 vessel + 4 sea: 32/55 FP32 detections recovered on the 44-tile
    # check; 24 vessel-only: 51/55; 32 vessel-only: 53/55, 12 extra, 2 missed).
    parser.add_argument("--n-vessel-tiles", type=int, default=32)
    parser.add_argument("--n-sea-tiles", type=int, default=0)
    parser.add_argument("--calibrate-method", default="Percentile", choices=["Percentile", "MinMax", "Entropy"])
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--keep-raster", action="store_true")
    args = parser.parse_args(argv)

    from src.models.compression.quantization import ModelQuantizer
    from src.traceability.hasher import get_commit_sha

    pt_path = args.pt_source.resolve()
    if not pt_path.exists() or pt_path.suffix.lower() != ".pt":
        logger.error("Expected an existing .pt: %s", pt_path)
        return 1
    base_name = pt_path.stem
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp32 = models_dir / f"{base_name}.onnx"
    int8_path = models_dir / f"{base_name}-int8-static.onnx"
    summary_json = args.summary_json or (models_dir / f"{base_name}-int8-static.calibration.json")

    # 1. Calibration set from the xView3 scene.
    scene_id = args.calibration_tar.name.removesuffix(".tar.gz")
    tar_sha = sha256_file(args.calibration_tar)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    member = f"{scene_id}/{args.band}"
    with tarfile.open(args.calibration_tar, "r:gz") as tf:
        tf.extract(tf.getmember(member), args.tmp_dir)
    raster = args.tmp_dir / member
    centres = _load_vessel_centres(args.validation_csv, scene_id)
    logger.info("Scene %s: %d vessel centres available", scene_id, len(centres))
    tiles, meta = build_calibration_tiles(
        raster, centres, n_vessel=args.n_vessel_tiles, n_sea=args.n_sea_tiles,
        tile=args.tile_size, seed=args.seed,
    )
    calib_sha = calibration_sha256(tiles)
    logger.info("Calibration set: %d tiles, sha256 %s", len(tiles), calib_sha[:16])
    if not args.keep_raster:
        shutil.rmtree(raster.parent, ignore_errors=True)

    # 2. FP32 ONNX intermediate (reuse the dynamic script's export if present).
    if not onnx_fp32.exists():
        from ultralytics import YOLO

        exported = Path(YOLO(str(pt_path)).export(format="onnx", opset=13, dynamic=False, simplify=True))
        if exported != onnx_fp32:
            exported.rename(onnx_fp32)
        logger.info("FP32 ONNX exported: %s", onnx_fp32)

    # 3. Static quantization.
    quant_cfg = {
        "quant_format": "QDQ",
        "op_types_to_quantize": ["Conv"],
        "per_channel": True,
        "reduce_range": False,
        "calibrate_method": args.calibrate_method,
        "weight_type": "QInt8",
        "activation_type": "QUInt8",
    }
    result = ModelQuantizer(onnx_fp32).quantize_static_onnx(
        int8_path, tiles, quant_format="QDQ",
        op_types_to_quantize=quant_cfg["op_types_to_quantize"], per_channel=True,
        reduce_range=False, calibrate_method=args.calibrate_method,
    )
    logger.info("INT8 static written: %s (%.2f MB)", int8_path, int8_path.stat().st_size / 1e6)

    # Sanity + fidelity check against FP32 on the vessel calibration tiles
    # (centre matching <= 20 px, conf >= Settings.confidence_threshold).
    from src.config import Settings
    from src.models.yolo import YOLODetector
    from src.validation.metrics import match_predictions

    conf = Settings().confidence_threshold
    det8 = YOLODetector(int8_path, conf, Settings().iou_threshold)
    det32 = YOLODetector(pt_path, conf, Settings().iou_threshold)
    agree = {"conf": conf, "tiles": 0, "fp32_dets": 0, "int8_dets": 0, "matched": 0, "extra": 0, "missed": 0, "int8_deterministic": True}
    for t, m in zip(tiles, meta, strict=True):
        if m["kind"] != "vessel":
            continue
        o8 = det8.predict(t)
        o8b = det8.predict(t)
        o32 = det32.predict(t)
        if [(d["bbox"], d["confidence"]) for d in o8] != [(d["bbox"], d["confidence"]) for d in o8b]:
            agree["int8_deterministic"] = False
        tp, fp, fn, _ = match_predictions(
            [{"bbox": d["bbox"], "confidence": d["confidence"]} for d in o8],
            [{"bbox": d["bbox"]} for d in o32], 0.5, match_mode="center", center_tolerance_px=20.0,
        )
        agree["tiles"] += 1
        agree["fp32_dets"] += len(o32)
        agree["int8_dets"] += len(o8)
        agree["matched"] += tp
        agree["extra"] += fp
        agree["missed"] += fn
    logger.info("FP32 agreement on calibration tiles: %s", agree)

    # 4. Card + summary.
    created = datetime.now(tz=UTC)
    int8_mb = int8_path.stat().st_size / 1e6
    fp32_mb = onnx_fp32.stat().st_size / 1e6
    pt_mb = pt_path.stat().st_size / 1e6
    args.cards_dir.mkdir(parents=True, exist_ok=True)
    card = args.cards_dir / f"{base_name}-int8-static.MODEL_CARD.md"
    card.write_text(_CARD_TEMPLATE.format(
        model_id=f"{base_name}-int8-static", base_name=base_name,
        created_date=created.strftime("%Y-%m-%d"), created_at=created.strftime("%Y-%m-%d %H:%M UTC"),
        commit_sha=get_commit_sha()[:12], onnx_filename=int8_path.name, size_mb=int8_mb,
        calibrate_method=args.calibrate_method,
        reduction_pct=(1 - int8_mb / fp32_mb) * 100 if fp32_mb else 0,
        pt_reduction_pct=(1 - int8_mb / pt_mb) * 100 if pt_mb else 0,
        file_hash=sha256_file(int8_path), scene_id=scene_id, tar_sha256=tar_sha,
        validation_csv=str(args.validation_csv), n_vessel=sum(m["kind"] == "vessel" for m in meta),
        n_sea=sum(m["kind"] == "sea" for m in meta), n_total=len(meta), seed=args.seed,
        calib_sha256=calib_sha, summary_json=str(summary_json), pt_filename=pt_path.name,
        agree_matched=agree["matched"], agree_fp32=agree["fp32_dets"], agree_extra=agree["extra"],
        agree_missed=agree["missed"], agree_tiles=agree["tiles"], agree_det=agree["int8_deterministic"],
        pt_hash=sha256_file(pt_path), pt_size_mb=pt_mb, onnx_fp32_filename=onnx_fp32.name,
        onnx_fp32_hash=sha256_file(onnx_fp32), onnx_fp32_size_mb=fp32_mb,
    ))
    summary = {
        "model_id": f"{base_name}-int8-static",
        "created_at": created.isoformat(),
        "commit_sha": get_commit_sha(),
        "int8": {"path": str(int8_path), "sha256": sha256_file(int8_path), "size_mb": round(int8_mb, 2)},
        "fp32_onnx": {"path": str(onnx_fp32), "sha256": sha256_file(onnx_fp32), "size_mb": round(fp32_mb, 2)},
        "pt": {"path": str(pt_path), "sha256": sha256_file(pt_path), "size_mb": round(pt_mb, 2)},
        "quantization": result.model_dump() if hasattr(result, "model_dump") else str(result),
        "quantization_config": quant_cfg,
        "fp32_agreement_on_calibration_tiles": agree,
        "calibration": {
            "scene_id": scene_id, "tar_sha256": tar_sha, "member": member,
            "validation_csv": str(args.validation_csv), "seed": args.seed,
            "tile_size": args.tile_size, "n_tiles": len(meta), "tiles": meta,
            "tensor_sha256": calib_sha, "stretch": "sigma0_linear->dB[-25,0]->uint8, no Lee",
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Card: %s | summary: %s", card, summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
