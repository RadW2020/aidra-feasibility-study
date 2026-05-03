"""
Quantizacion de modelos: reduce la precision de los pesos.

Tecnicas implementadas:
1. Dynamic Quantization (PyTorch): FP32 -> INT8, post-entrenamiento
2. Static Quantization (ONNX Runtime): FP32 -> INT8, con calibracion
3. FP16 Half-precision: FP32 -> FP16 (intermedio)

Cada tecnica produce un nuevo archivo de modelo (.onnx o .pt) que
se registra en el sistema con su propio hash SHA256.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.quantization
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result model
# ------------------------------------------------------------------


class QuantizationResult(BaseModel):
    """Result of a model quantization operation."""

    original_path: Path
    quantized_path: Path
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float = Field(
        description="original_size / quantized_size"
    )
    technique: str
    original_hash: str
    quantized_hash: str

    class Config:
        arbitrary_types_allowed = True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file.

    Parameters
    ----------
    path:
        Path to the file.
    chunk_size:
        Read buffer size in bytes.

    Returns
    -------
    Hexadecimal SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class ModelQuantizer:
    """Quantization engine for YOLO models.

    Supports dynamic INT8 (PyTorch), static INT8 (ONNX Runtime),
    and FP16 half-precision conversion.

    Parameters
    ----------
    model_path:
        Path to the original model file (``.pt`` or ``.onnx``).
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )
        self.original_hash = _sha256_file(self.model_path)
        self.original_size_mb = _file_size_mb(self.model_path)
        logger.info(
            "ModelQuantizer initialised: %s (%.2f MB, hash=%s...)",
            self.model_path.name,
            self.original_size_mb,
            self.original_hash[:12],
        )

    # ------------------------------------------------------------------
    # 1. Dynamic INT8 quantization (PyTorch)
    # ------------------------------------------------------------------

    def quantize_dynamic_pytorch(self, output_path: Path | str) -> QuantizationResult:
        """Apply dynamic INT8 quantization using PyTorch.

        Converts ``Linear`` and ``Conv2d`` layers to INT8 dynamically
        at inference time. Does not require calibration data.

        Parameters
        ----------
        output_path:
            Destination path for the quantized ``.pt`` file.

        Returns
        -------
        QuantizationResult with original and quantized metadata.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting dynamic INT8 quantization: %s", self.model_path.name)

        # Load model via ultralytics to get the underlying torch model
        yolo = YOLO(str(self.model_path))
        torch_model = yolo.model

        # Apply dynamic quantization on Linear and Conv2d layers
        quantized_model = torch.quantization.quantize_dynamic(
            torch_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )

        # Save the quantized state dict wrapped in the YOLO checkpoint format
        # to maintain compatibility with Ultralytics loading
        ckpt: dict[str, Any] = {
            "model": quantized_model,
            "optimizer": None,
            "train_args": getattr(yolo, "overrides", {}),
            "date": None,
            "version": getattr(yolo, "__version__", "8.0.0"),
        }
        torch.save(ckpt, str(output_path))

        quantized_hash = _sha256_file(output_path)
        quantized_size = _file_size_mb(output_path)
        ratio = self.original_size_mb / quantized_size if quantized_size > 0 else 0.0

        logger.info(
            "Dynamic INT8 quantization complete: %.2f MB -> %.2f MB (ratio=%.2fx)",
            self.original_size_mb,
            quantized_size,
            ratio,
        )

        return QuantizationResult(
            original_path=self.model_path,
            quantized_path=output_path,
            original_size_mb=self.original_size_mb,
            quantized_size_mb=quantized_size,
            compression_ratio=round(ratio, 2),
            technique="dynamic_int8_pytorch",
            original_hash=self.original_hash,
            quantized_hash=quantized_hash,
        )

    # ------------------------------------------------------------------
    # 2. Static INT8 quantization (ONNX Runtime)
    # ------------------------------------------------------------------

    def quantize_static_onnx(
        self,
        output_path: Path | str,
        calibration_data: list[NDArray[np.uint8]],
        quant_format: str = "QDQ",
    ) -> QuantizationResult:
        """Apply static INT8 quantization using ONNX Runtime.

        Requires calibration data (50-100 representative images) to
        determine activation ranges. Produces the smallest and fastest
        model variant.

        Parameters
        ----------
        output_path:
            Destination path for the quantized ``.onnx`` file.
        calibration_data:
            List of image arrays for calibration (shape ``(H, W, C)``
            or ``(H, W)``).
        quant_format:
            Quantization format: ``"QDQ"`` (default) or ``"QOperator"``.

        Returns
        -------
        QuantizationResult with original and quantized metadata.
        """
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting static INT8 ONNX quantization: %s "
            "(calibration images=%d, format=%s)",
            self.model_path.name,
            len(calibration_data),
            quant_format,
        )

        # If source is .pt, export to ONNX first
        onnx_source = self.model_path
        tmp_onnx: Path | None = None
        if self.model_path.suffix.lower() == ".pt":
            yolo = YOLO(str(self.model_path))
            exported = yolo.export(format="onnx", opset=13)
            tmp_onnx = Path(exported)
            onnx_source = tmp_onnx
            logger.info("Exported temporary ONNX for static quantization: %s", onnx_source)

        # Build calibration data reader
        class _CalibrationReader(CalibrationDataReader):
            """Feeds calibration images to ONNX Runtime quantizer."""

            def __init__(self, images: list[NDArray[np.uint8]], input_size: int = 640) -> None:
                self._images = images
                self._input_size = input_size
                self._idx = 0
                # Determine the input name from the model
                import onnxruntime as ort

                session = ort.InferenceSession(str(onnx_source))
                self._input_name = session.get_inputs()[0].name
                del session

            def get_next(self) -> dict[str, NDArray[np.float32]] | None:
                if self._idx >= len(self._images):
                    return None
                img = self._images[self._idx]
                self._idx += 1

                # Preprocess: resize, normalize, CHW, add batch dim
                import cv2

                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                resized = cv2.resize(img, (self._input_size, self._input_size))
                blob = resized.astype(np.float32) / 255.0
                blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
                blob = np.expand_dims(blob, axis=0)  # add batch
                return {self._input_name: blob}

        calibration_reader = _CalibrationReader(calibration_data)

        # Determine quantization format enum
        fmt = QuantFormat.QDQ if quant_format == "QDQ" else QuantFormat.QOperator

        # Run static quantization
        quantize_static(
            model_input=str(onnx_source),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=fmt,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )

        # Clean up temporary ONNX if we created one
        if tmp_onnx is not None and tmp_onnx.exists():
            tmp_onnx.unlink()
            logger.debug("Removed temporary ONNX file: %s", tmp_onnx)

        quantized_hash = _sha256_file(output_path)
        quantized_size = _file_size_mb(output_path)
        ratio = self.original_size_mb / quantized_size if quantized_size > 0 else 0.0

        logger.info(
            "Static INT8 ONNX quantization complete: %.2f MB -> %.2f MB (ratio=%.2fx)",
            self.original_size_mb,
            quantized_size,
            ratio,
        )

        return QuantizationResult(
            original_path=self.model_path,
            quantized_path=output_path,
            original_size_mb=self.original_size_mb,
            quantized_size_mb=quantized_size,
            compression_ratio=round(ratio, 2),
            technique=f"static_int8_onnx_{quant_format.lower()}",
            original_hash=self.original_hash,
            quantized_hash=quantized_hash,
        )

    # ------------------------------------------------------------------
    # 3. FP16 half-precision
    # ------------------------------------------------------------------

    def quantize_fp16(self, output_path: Path | str) -> QuantizationResult:
        """Convert model weights from FP32 to FP16 half-precision.

        Reduces model size by approximately 50% with minimal accuracy
        loss. Works by exporting through ONNX with half-precision
        conversion.

        Parameters
        ----------
        output_path:
            Destination path for the FP16 ``.onnx`` file.

        Returns
        -------
        QuantizationResult with original and quantized metadata.
        """
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting FP16 conversion: %s", self.model_path.name)

        # If source is .pt, export to ONNX first
        onnx_source = self.model_path
        tmp_onnx: Path | None = None
        if self.model_path.suffix.lower() == ".pt":
            yolo = YOLO(str(self.model_path))
            exported = yolo.export(format="onnx", opset=13)
            tmp_onnx = Path(exported)
            onnx_source = tmp_onnx
            logger.info("Exported temporary ONNX for FP16 conversion: %s", onnx_source)

        # Load ONNX model and convert to FP16
        model_fp32 = onnx.load(str(onnx_source))
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, str(output_path))

        # Clean up temporary ONNX if we created one
        if tmp_onnx is not None and tmp_onnx.exists():
            tmp_onnx.unlink()
            logger.debug("Removed temporary ONNX file: %s", tmp_onnx)

        quantized_hash = _sha256_file(output_path)
        quantized_size = _file_size_mb(output_path)
        ratio = self.original_size_mb / quantized_size if quantized_size > 0 else 0.0

        logger.info(
            "FP16 conversion complete: %.2f MB -> %.2f MB (ratio=%.2fx)",
            self.original_size_mb,
            quantized_size,
            ratio,
        )

        return QuantizationResult(
            original_path=self.model_path,
            quantized_path=output_path,
            original_size_mb=self.original_size_mb,
            quantized_size_mb=quantized_size,
            compression_ratio=round(ratio, 2),
            technique="fp16_half_precision",
            original_hash=self.original_hash,
            quantized_hash=quantized_hash,
        )
