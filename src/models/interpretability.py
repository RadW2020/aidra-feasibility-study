"""
Interpretabilidad sobre el detector YOLOv8 + CFAR (entregable D4).

Implementa:
  - Grad-CAM sobre el ultimo bloque convolucional de la cabeza YOLOv8.
  - Heatmap del score CFAR pre-threshold (Pfa map) sobre la misma escena.

Salida por muestra:
  outputs/<run_id>/<sample_idx>_input.png         — tile SAR de entrada (escala log)
  outputs/<run_id>/<sample_idx>_gradcam.png       — superposicion Grad-CAM
  outputs/<run_id>/<sample_idx>_cfar_score.png    — heatmap CFAR
  outputs/<run_id>/manifest.json                  — meta + SHA256 de cada PNG

Cierra criterio AI Act / D4: explainability sobre muestreo del eval set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.traceability.hasher import compute_sha256, get_commit_sha

logger = logging.getLogger(__name__)


# =====================================================================
# Manifest
# =====================================================================


@dataclass
class InterpretabilityManifest:
    """Meta del run de interpretabilidad (volcado a manifest.json)."""

    run_id: str
    created_at: str
    commit_sha: str
    model_name: str
    model_hash: str
    n_samples: int
    samples: list[dict[str, Any]] = field(default_factory=list)


# =====================================================================
# Grad-CAM YOLOv8
# =====================================================================


def gradcam_yolov8(
    model: Any,
    image: np.ndarray,
    target_layer_name: str = "model.model.21",
) -> np.ndarray:
    """Calcula un heatmap Grad-CAM sobre la cabeza YOLOv8.

    Parameters
    ----------
    model:
        Instancia ``ultralytics.YOLO``.
    image:
        Tile SAR ``(H, W)`` o ``(H, W, 3)`` en uint8 / float32.
    target_layer_name:
        Nombre del modulo a hookear. Por defecto la ultima C2f
        ``model.model.21`` justo antes de la cabeza Detect — produce
        un feature map (B,C,H,W) sobre el que Grad-CAM aplica.

    Returns
    -------
    np.ndarray
        Heatmap normalizado a ``[0, 1]``, shape ``(H, W)``.

    Notes
    -----
    Implementacion lazy import: Torch solo se importa cuando se invoca
    esta funcion. Esto permite que el resto del paquete (incluido el
    bundler D3) se importe sin tener Torch instalado.
    """
    import torch
    import torch.nn.functional as F

    # Fail fast for ONNX models — they have no PyTorch autograd graph.
    if hasattr(model, "model") and isinstance(model.model, str):
        raise ValueError(
            "gradcam_yolov8 requires a PyTorch YOLO model. "
            f"Got ONNX session at '{model.model}'. "
            "Load the .pt baseline instead."
        )

    image_rgb = (
        np.repeat(image[..., None], 3, axis=-1) if image.ndim == 2 else image
    )

    img_t = torch.as_tensor(image_rgb, dtype=torch.float32)
    if img_t.max() > 1.0:
        img_t = img_t / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # YOLOv8 needs spatial dims divisible by 32. Resize up to 320x320
    # (smallest "letterbox" size that preserves resolution for small SAR
    # thumbnails). For Grad-CAM the absolute resolution is irrelevant —
    # we'll bilinear-upsample the heatmap back to the original shape.
    target_size = 320
    img_t = F.interpolate(
        img_t, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )
    img_t.requires_grad_(True)

    # `model` is an `ultralytics.YOLO`. The Detect head sits at
    # `model.model.model[22]` for YOLOv8n/s. The default target_layer_name
    # is "model.model.22" relative to the YOLO root.
    target_layer = _resolve_module(model, target_layer_name)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, output):
        activations["v"] = output.detach() if not isinstance(output, list | tuple) else output[0].detach()

    def bwd_hook(_m, _gin, gout):
        gradients["v"] = gout[0].detach() if isinstance(gout, tuple) else gout.detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.model.eval()
        out = model.model(img_t)
        # Use scalar of summed objectness as backprop target.
        if isinstance(out, list | tuple):
            scalar = sum(o.abs().mean() for o in out if isinstance(o, torch.Tensor))
        else:
            scalar = out.abs().mean()
        scalar.backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    act = activations.get("v")
    grad = gradients.get("v")
    if act is None or grad is None:
        logger.warning("Grad-CAM: missing activations/gradients — fallback to zeros")
        return np.zeros(image_rgb.shape[:2], dtype=np.float32)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    cam_np = cam.squeeze().cpu().numpy()
    cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
    if cam_max > cam_min:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    return cam_np.astype(np.float32)


def _resolve_module(root: Any, dotted: str) -> Any:
    """Resuelve ``model.model.22`` -> nn.Module por atributos / indices."""
    obj = root
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


# =====================================================================
# CFAR pre-threshold heatmap
# =====================================================================


def cfar_score_map(
    sar_tile: np.ndarray,
    guard_size: int = 3,
    training_size: int = 15,
) -> np.ndarray:
    """Heatmap del score CFAR (test statistic) pre-threshold.

    Calcula el ratio ``(value - mean_clutter) / std_clutter`` en una
    ventana 2D usando la misma geometria que ``src.models.cfar``
    (guard ring + training ring) pero sin aplicar el threshold de Pfa.

    Returns
    -------
    np.ndarray
        Heatmap ``(H, W)`` con scores normalizados a ``[0, 1]``.
    """
    from scipy.ndimage import uniform_filter

    arr = sar_tile.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)

    win_outer = 2 * (guard_size + training_size) + 1
    win_inner = 2 * guard_size + 1

    sum_outer = uniform_filter(
        arr, size=win_outer, mode="reflect"
    ) * (win_outer * win_outer)
    sum_inner = uniform_filter(
        arr, size=win_inner, mode="reflect"
    ) * (win_inner * win_inner)
    n_train = (win_outer * win_outer) - (win_inner * win_inner)
    mean_clutter = (sum_outer - sum_inner) / max(n_train, 1)

    sq = arr * arr
    sq_outer = uniform_filter(sq, size=win_outer, mode="reflect") * (
        win_outer * win_outer
    )
    sq_inner = uniform_filter(sq, size=win_inner, mode="reflect") * (
        win_inner * win_inner
    )
    var_clutter = ((sq_outer - sq_inner) / max(n_train, 1)) - (
        mean_clutter * mean_clutter
    )
    std_clutter = np.sqrt(np.maximum(var_clutter, 1e-6))

    score = (arr - mean_clutter) / std_clutter
    score = np.clip(score, 0.0, None)
    smin, smax = float(score.min()), float(score.max())
    if smax > smin:
        score = (score - smin) / (smax - smin)
    return score.astype(np.float32)


# =====================================================================
# Rendering helpers
# =====================================================================


def save_heatmap_png(
    background: np.ndarray,
    heatmap: np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
) -> None:
    """Guarda un PNG del heatmap superpuesto sobre el fondo SAR.

    Lazy import de matplotlib para que el resto del paquete no
    arrastre dependencia grafica.
    """
    import matplotlib  # type: ignore[import-untyped]

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    bg = background.astype(np.float32)
    if bg.ndim == 3:
        bg = bg.mean(axis=-1)
    bg_min, bg_max = float(bg.min()), float(bg.max())
    if bg_max > bg_min:
        bg = (bg - bg_min) / (bg_max - bg_min)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(bg, cmap="gray", interpolation="nearest")
    ax.imshow(heatmap, cmap="jet", alpha=alpha, interpolation="bilinear")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_grayscale_png(arr: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    a = arr.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    a = np.log1p(np.maximum(a, 0))
    amin, amax = float(a.min()), float(a.max())
    if amax > amin:
        a = (a - amin) / (amax - amin)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(a, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =====================================================================
# Run
# =====================================================================


def run_interpretability(
    samples: list[np.ndarray],
    yolo_model: Any,
    out_dir: Path,
    run_id: str,
    model_name: str,
    model_hash: str,
    cfar_guard: int = 3,
    cfar_training: int = 15,
) -> Path:
    """Genera Grad-CAM YOLOv8 + heatmap CFAR para una lista de tiles.

    Parameters
    ----------
    samples:
        Lista de tiles SAR (numpy arrays HxW o HxWx3).
    yolo_model:
        Instancia ``ultralytics.YOLO`` ya cargada (sin gate AI-Act
        adicional: el caller la trae registrada).
    out_dir:
        Directorio de salida (se crea ``<out_dir>/<run_id>/``).
    run_id:
        UUID o etiqueta del run, propagado al manifest.
    model_name / model_hash:
        Para el manifest (anclar interpretabilidad al modelo).

    Returns
    -------
    Path
        Ruta al ``manifest.json`` generado.
    """
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = InterpretabilityManifest(
        run_id=run_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        commit_sha=get_commit_sha(),
        model_name=model_name,
        model_hash=model_hash,
        n_samples=len(samples),
    )

    for idx, tile in enumerate(samples):
        prefix = f"{idx:03d}"
        in_path = out_dir / f"{prefix}_input.png"
        cam_path = out_dir / f"{prefix}_gradcam.png"
        cfar_path = out_dir / f"{prefix}_cfar_score.png"

        save_grayscale_png(tile, in_path)

        try:
            cam = gradcam_yolov8(yolo_model, tile)
            save_heatmap_png(tile, cam, cam_path)
            cam_ok = True
        except Exception as exc:
            logger.warning(
                "Grad-CAM failed for sample %d: %s", idx, exc, exc_info=True
            )
            cam_ok = False

        try:
            score = cfar_score_map(
                tile, guard_size=cfar_guard, training_size=cfar_training
            )
            save_heatmap_png(tile, score, cfar_path)
            cfar_ok = True
        except Exception as exc:
            logger.warning("CFAR heatmap failed for sample %d: %s", idx, exc)
            cfar_ok = False

        sample_record = {
            "idx": idx,
            "input_png": in_path.name,
            "input_sha256": compute_sha256(in_path),
            "gradcam_png": cam_path.name if cam_ok else None,
            "gradcam_sha256": compute_sha256(cam_path) if cam_ok else None,
            "cfar_png": cfar_path.name if cfar_ok else None,
            "cfar_sha256": compute_sha256(cfar_path) if cfar_ok else None,
        }
        manifest.samples.append(sample_record)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "created_at": manifest.created_at,
                "commit_sha": manifest.commit_sha,
                "model_name": manifest.model_name,
                "model_hash": manifest.model_hash,
                "n_samples": manifest.n_samples,
                "samples": manifest.samples,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Interpretability run %s: %d samples -> %s",
        run_id,
        len(samples),
        manifest_path,
    )
    return manifest_path
