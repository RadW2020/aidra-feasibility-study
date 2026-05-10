"""
Wrapper para modelo YOLOv8 de deteccion de barcos.

Modelos soportados:
- YOLOv8n (nano): 3.2M params, 6.2 MB, ~50 ms/imagen en CPU ARM
- YOLOv8s (small): 11.2M params, 22.5 MB, ~100 ms/imagen en CPU ARM
- YOLOv8m (medium): 25.9M params, 52 MB, ~200 ms/imagen en CPU ARM

Formatos de exportacion soportados:
- PyTorch (.pt): formato nativo
- ONNX (.onnx): para quantizacion y deployment
- OpenVINO: optimizado para Intel (referencia)

Dependencias:
- ultralytics
- torch (viene con ultralytics)
- onnxruntime (para inferencia ONNX)
- psutil (medicion de recursos)
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray
from ultralytics import YOLO

from src.models.base import BaseDetector

logger = logging.getLogger(__name__)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class YOLODetector(BaseDetector):
    """Wrapper around Ultralytics YOLOv8 for vessel detection.

    Parameters
    ----------
    model_path:
        Path to the weights file (``.pt`` or ``.onnx``).
    confidence_threshold:
        Minimum confidence for a detection to be kept.
    iou_threshold:
        IoU threshold used in YOLOv8 built-in NMS.
    device:
        Inference device (``"cpu"`` recommended for OCI ARM Free Tier).
    """

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Derived metadata
        suffix = self.model_path.suffix.lower()
        self.model_format: str = "onnx" if suffix == ".onnx" else "pytorch"
        self.model_name: str = self.model_path.stem
        self.model_version: str = "v1.0"
        self.model_size_mb: float = self.model_path.stat().st_size / (1024 * 1024)
        self.model_hash: str = _sha256_file(self.model_path)
        # Set by ModelManager after construction from models_registry row
        # or from the filename version tag.  Default ``"none"`` for raw
        # baselines.  Surfaced through :meth:`get_model_info` so the
        # engine can persist it on every execution_log row.
        self.compression_technique: str = "none"

        logger.info(
            "Loading YOLO model: %s (%.1f MB, format=%s, hash=%s…)",
            self.model_name,
            self.model_size_mb,
            self.model_format,
            self.model_hash[:12],
        )

        # Load model via ultralytics
        self._model = YOLO(str(self.model_path))

        logger.info("YOLO model loaded successfully on device=%s", self.device)

    # ------------------------------------------------------------------
    # Model weight access (for resilience/bit-flip testing)
    # ------------------------------------------------------------------

    def get_torch_model(self):
        """Return the underlying PyTorch nn.Module for weight access.

        Used by the resilience module (BitFlipSimulator) to access and
        manipulate model weights for radiation tolerance testing.
        """
        return self._model.model

    def get_weights_dict(self) -> dict[str, Any]:
        """Return a copy of model weights as {name: numpy_array} dict.

        Each tensor is detached, moved to CPU, and converted to a
        contiguous numpy array copy (safe for mutation).
        """
        weights = {}
        for name, param in self._model.model.named_parameters():
            weights[name] = param.detach().cpu().numpy().copy()
        return weights

    def load_weights_dict(self, weights: dict[str, Any]) -> None:
        """Load weights from a {name: numpy_array} dict back into the model.

        Used after bit-flip injection to test inference with corrupted weights.
        """
        import torch

        state = self._model.model.state_dict()
        for name, arr in weights.items():
            if name in state:
                state[name] = torch.from_numpy(arr)
        self._model.model.load_state_dict(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Run inference on a single image (tile).

        Parameters
        ----------
        image:
            Image array of shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        List of detections, each containing ``bbox`` ``[x_min, y_min,
        x_max, y_max]``, ``confidence``, ``class_id``, and
        ``class_name``.
        """
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    {
                        "bbox": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )

        logger.debug(
            "YOLO predict: %d detections (conf>=%.2f)",
            len(detections),
            self.confidence_threshold,
        )
        return detections

    def predict_batch(self, tiles: list[NDArray[np.uint8]]) -> list[list[dict[str, Any]]]:
        """Run batch inference on multiple tiles.

        Parameters
        ----------
        tiles:
            List of image arrays.

        Returns
        -------
        List of detection lists, one per input tile.
        """
        if not tiles:
            return []

        results = self._model.predict(
            source=tiles,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        batch_detections: list[list[dict[str, Any]]] = []
        for result in results:
            tile_dets: list[dict[str, Any]] = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                    tile_dets.append(
                        {
                            "bbox": [
                                float(xyxy[0]),
                                float(xyxy[1]),
                                float(xyxy[2]),
                                float(xyxy[3]),
                            ],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name,
                        }
                    )
            batch_detections.append(tile_dets)

        logger.info(
            "YOLO batch predict: %d tiles, %d total detections",
            len(tiles),
            sum(len(d) for d in batch_detections),
        )
        return batch_detections

    def export_onnx(self, output_path: Path | str, opset: int = 13) -> Path:
        """Export a PyTorch model to ONNX format.

        Parameters
        ----------
        output_path:
            Destination path for the ``.onnx`` file.
        opset:
            ONNX opset version.

        Returns
        -------
        Resolved path of the exported ``.onnx`` file.

        Raises
        ------
        RuntimeError
            If the loaded model is not in PyTorch format.
        """
        if self.model_format != "pytorch":
            raise RuntimeError(
                f"Cannot export to ONNX: model is already in {self.model_format} format"
            )

        output_path = Path(output_path)
        logger.info("Exporting YOLO model to ONNX: %s (opset=%d)", output_path, opset)

        # ultralytics .export() returns the path to the exported file
        exported = self._model.export(format="onnx", opset=opset)
        exported_path = Path(exported)

        # Move the exported file to the requested output location if needed
        if exported_path != output_path:
            import shutil

            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported_path), str(output_path))

        logger.info("ONNX export complete: %s", output_path)
        return output_path.resolve()

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model.

        Returns
        -------
        Dictionary with ``name``, ``format``, ``size_mb``, ``hash``,
        ``num_params``, ``num_layers``, ``input_size``, and ``classes``.
        """
        # Extract parameter/layer counts from the underlying torch model
        num_params: int | None = None
        num_layers: int | None = None
        input_size: list[int] = [640, 640]

        try:
            model_obj = self._model.model
            if hasattr(model_obj, "parameters"):
                num_params = sum(p.numel() for p in model_obj.parameters())
            if hasattr(model_obj, "model"):
                # YOLOv8 sequential layers
                num_layers = len(list(model_obj.model.modules()))
        except Exception:
            logger.debug("Could not extract num_params/num_layers", exc_info=True)

        # Classes
        classes: list[str] = []
        try:
            names = self._model.names
            if isinstance(names, dict):
                classes = list(names.values())
            elif isinstance(names, (list, tuple)):
                classes = list(names)
        except Exception:
            classes = ["vessel"]

        return {
            "name": self.model_name,
            "version": self.model_version,
            "format": self.model_format,
            "size_mb": round(self.model_size_mb, 2),
            "hash": self.model_hash,
            "num_params": num_params,
            "num_layers": num_layers,
            "input_size": input_size,
            "classes": classes,
            "compression_technique": self.compression_technique,
        }

    def benchmark(
        self,
        image: NDArray[np.uint8],
        num_runs: int = 100,
    ) -> dict[str, float]:
        """Run *num_runs* inferences and collect timing / resource statistics.

        Parameters
        ----------
        image:
            Image array to use for benchmarking.
        num_runs:
            Number of inference iterations.

        Returns
        -------
        Dictionary with ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
        ``p50_ms``, ``p95_ms``, ``p99_ms``, ``peak_ram_mb``, and
        ``cpu_percent``.
        """
        logger.info("Benchmarking YOLO: %d runs", num_runs)

        process = psutil.Process()
        # Reset CPU measurement
        process.cpu_percent(interval=None)

        ram_samples: list[float] = []
        latencies: list[float] = []

        for _i in range(num_runs):
            mem_before = process.memory_info().rss / (1024 * 1024)

            t0 = time.perf_counter()
            self._model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)
            mem_after = process.memory_info().rss / (1024 * 1024)
            ram_samples.append(max(mem_before, mem_after))

        cpu_percent = process.cpu_percent(interval=None)

        arr = np.array(latencies)
        stats: dict[str, float] = {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "peak_ram_mb": float(max(ram_samples)) if ram_samples else 0.0,
            "cpu_percent": float(cpu_percent) if cpu_percent is not None else 0.0,
        }

        logger.info(
            "Benchmark complete: mean=%.1f ms, p95=%.1f ms, peak_ram=%.0f MB",
            stats["mean_ms"],
            stats["p95_ms"],
            stats["peak_ram_mb"],
        )
        return stats
