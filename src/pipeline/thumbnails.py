"""
Generacion de thumbnails SAR por deteccion (wow effect #1).

Para cada deteccion, recorta una ventana de ``+- padding`` pixeles
alrededor del bbox_pixel desde el tile SAR ya calibrado y filtrado,
normaliza la dinamica (log + percentile clip) y guarda un PNG pequeno
(~10-20 KB).

Salida:
    /data/thumbnails/<execution_id>/<detection_id>.png

Cierra criterio:
  - Q3 GEOINT: evidencia visual por deteccion (lo que el operador
    pide siempre que ve un numero de confianza).
  - AI Act / D4: explainability — ver el barco junto al numero.
  - Trazabilidad: el PNG entra al bundle D3 con SHA256 manifestado.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================
# Public API
# ====================================================================


def generate_thumbnails(
    detections: list[Any],
    tiles: list[dict[str, Any]],
    execution_id: UUID | str,
    out_root: Path,
    padding: int = 32,
    min_size: int = 64,
) -> int:
    """Genera un PNG por deteccion con un crop SAR alrededor del bbox.

    Modifica ``detections`` in-place asignando ``thumbnail_path`` a las
    detecciones que se hayan podido recortar (ruta absoluta).

    Parameters
    ----------
    detections:
        Lista de objetos ``Detection`` (con ``id``, ``tile_index``,
        ``bbox_pixel``, ``thumbnail_path``).
    tiles:
        Lista de dicts ``{"array": np.ndarray, "row_offset", ...}``
        producida por ``preprocess_full``.
    execution_id:
        UUID del run (sirve para agrupar los PNG).
    out_root:
        Directorio raiz donde colgar el subdirectorio
        ``<execution_id>/``.
    padding:
        Pixeles extra alrededor del bbox (default 32 px → ~320 m a 10 m
        de pixel spacing en GRD).
    min_size:
        Tamano minimo del crop en pixeles. Detecciones diminutas se
        expanden hasta este tamano para que la imagen sea legible.

    Returns
    -------
    int
        Numero de thumbnails efectivamente escritos.
    """
    out_root = Path(out_root)
    out_dir = out_root / str(execution_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    by_tile: dict[int, np.ndarray] = {}

    # Cache tile arrays by tile_index → row/col offsets for cropping.
    tile_meta: dict[int, dict[str, Any]] = {}
    for idx, t in enumerate(tiles):
        if "array" not in t:
            continue
        by_tile[idx] = np.asarray(t["array"])
        tile_meta[idx] = {
            "row_offset": int(t.get("row_offset", 0)),
            "col_offset": int(t.get("col_offset", 0)),
        }

    if not by_tile:
        logger.info("generate_thumbnails: no tiles with array — nothing to crop")
        return 0

    for det in detections:
        try:
            tile_idx = int(getattr(det, "tile_index", 0))
            arr = by_tile.get(tile_idx)
            if arr is None:
                continue
            bbox = list(getattr(det, "bbox_pixel", []))
            if len(bbox) != 4:
                continue

            x_min, y_min, x_max, y_max = (float(v) for v in bbox)
            # bbox is in tile-local coords (some pipelines use absolute);
            # detect heuristically: if anything > tile shape, treat as
            # absolute and translate using offsets.
            tile_h, tile_w = arr.shape[:2]
            if x_max > tile_w * 1.5 or y_max > tile_h * 1.5:
                meta = tile_meta.get(tile_idx, {})
                col_off = float(meta.get("col_offset", 0))
                row_off = float(meta.get("row_offset", 0))
                x_min -= col_off
                x_max -= col_off
                y_min -= row_off
                y_max -= row_off

            # Padding + ensure min size around the centre.
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            half_w = max((x_max - x_min) / 2.0 + padding, min_size / 2.0)
            half_h = max((y_max - y_min) / 2.0 + padding, min_size / 2.0)

            r0 = int(max(0, np.floor(cy - half_h)))
            r1 = int(min(tile_h, np.ceil(cy + half_h)))
            c0 = int(max(0, np.floor(cx - half_w)))
            c1 = int(min(tile_w, np.ceil(cx + half_w)))

            if r1 - r0 < 4 or c1 - c0 < 4:
                continue  # crop demasiado pequeno tras clamping

            crop = arr[r0:r1, c0:c1]
            if crop.ndim == 3:
                # SAR a veces tiene dimension de canal — colapsar.
                crop = crop.mean(axis=-1)

            png = _normalize_for_png(crop)
            out_path = out_dir / f"{det.id}.png"
            _write_png(png, out_path)
            det.thumbnail_path = str(out_path.resolve())
            saved += 1
        except Exception as exc:
            logger.debug(
                "Thumbnail failed for detection: %s",
                exc,
                exc_info=False,
            )
            continue

    logger.info(
        "Thumbnails generated: %d / %d detections (exec=%s)",
        saved,
        len(detections),
        execution_id,
    )
    return saved


# ====================================================================
# Helpers
# ====================================================================


def _normalize_for_png(arr: np.ndarray) -> np.ndarray:
    """Convierte un crop SAR a uint8 con buena dinamica visual.

    Pasos:
      1. Cast a float32, abs (SAR magnitude).
      2. log1p para comprimir backscatter (rango dinamico tipico:
         3-4 ordenes de magnitud).
      3. Clip a percentiles 2-98 para esquivar outliers.
      4. Min-max → [0, 255] uint8.
    """
    a = np.asarray(arr, dtype=np.float32)
    a = np.abs(a)
    a = np.log1p(a)
    if a.size == 0 or not np.isfinite(a).any():
        return np.zeros(a.shape, dtype=np.uint8)
    p2, p98 = np.percentile(a[np.isfinite(a)], (2.0, 98.0))
    if p98 <= p2:
        p98 = p2 + 1e-6
    a = np.clip(a, p2, p98)
    a = (a - p2) / (p98 - p2)
    return (a * 255.0).astype(np.uint8)


def _write_png(arr_u8: np.ndarray, out_path: Path) -> None:
    """Escribe un array uint8 (HxW) como PNG en escala de grises.

    Lazy import de Pillow para no obligar la dependencia en arranque.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Pillow no esta instalado: thumbnails desactivados"
        ) from exc

    img = Image.fromarray(arr_u8, mode="L")
    img.save(out_path, format="PNG", optimize=True)
