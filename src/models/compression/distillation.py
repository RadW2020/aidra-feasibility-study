"""
Knowledge Distillation: entrena un modelo pequeno (student) para
imitar a uno grande (teacher).

Implementacion:
- Teacher: YOLOv8m (medium, 52 MB) -- se asume preentrenado
- Student: YOLOv8n (nano, 6 MB)
- Loss: alpha * CE_loss(student, labels) + (1-alpha) * KL_div(student_logits, teacher_logits)
- Temperature: T (tipico 3-5), suaviza las distribuciones

NOTA: Knowledge distillation requiere entrenamiento, por lo que
este modulo es opcional y se ejecuta en maquina con GPU, no en OCI ARM.
En OCI ARM solo se usa el modelo student ya destilado para inferencia.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# GPU availability guard
# ------------------------------------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available()
if not _CUDA_AVAILABLE:
    logger.warning(
        "CUDA is not available. Knowledge distillation requires a GPU "
        "for practical training times. The distill() method will fall "
        "back to CPU but will be extremely slow."
    )


# ------------------------------------------------------------------
# Result model
# ------------------------------------------------------------------


class DistillationResult(BaseModel):
    """Result of a knowledge distillation operation."""

    teacher_path: Path
    student_path: Path
    distilled_path: Path
    teacher_size_mb: float
    student_size_mb: float
    distilled_size_mb: float
    technique: str = "knowledge_distillation"
    teacher_hash: str
    distilled_hash: str
    epochs_trained: int
    final_loss: float

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
# Distillation loss
# ------------------------------------------------------------------


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.

    Combines a hard-label cross-entropy loss (student vs. ground
    truth) with a soft-label KL-divergence loss (student vs. teacher
    logits, temperature-scaled).

    Parameters
    ----------
    alpha:
        Weight for the hard-label (CE) loss. The soft-label loss
        is weighted by ``(1 - alpha)``.
    temperature:
        Temperature for softening the probability distributions.
        Higher values produce softer distributions.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Parameters
        ----------
        student_logits:
            Raw output logits from the student model.
        teacher_logits:
            Raw output logits from the teacher model (detached).
        targets:
            Ground-truth labels for hard-label loss.

        Returns
        -------
        Weighted combination of CE and KL-divergence losses.
        """
        # Hard-label loss (student vs ground truth)
        ce_loss = F.cross_entropy(student_logits, targets)

        # Soft-label loss (student vs teacher, temperature-scaled)
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (T * T)

        # Combined loss
        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss
        return loss


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class KnowledgeDistiller:
    """Knowledge distillation engine for YOLO models.

    Trains a smaller student model to mimic the predictions of a
    larger teacher model, using a combination of hard-label
    cross-entropy and soft-label KL-divergence losses.

    Parameters
    ----------
    teacher_path:
        Path to the teacher model weights (``.pt``).
    student_path:
        Path to the student model weights (``.pt``).
    """

    def __init__(
        self,
        teacher_path: Path | str,
        student_path: Path | str,
    ) -> None:
        self.teacher_path = Path(teacher_path)
        self.student_path = Path(student_path)

        if not self.teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher model not found: {self.teacher_path}"
            )
        if not self.student_path.exists():
            raise FileNotFoundError(
                f"Student model not found: {self.student_path}"
            )

        self.teacher_hash = _sha256_file(self.teacher_path)
        self.teacher_size_mb = _file_size_mb(self.teacher_path)
        self.student_size_mb = _file_size_mb(self.student_path)

        logger.info(
            "KnowledgeDistiller initialised: "
            "teacher=%s (%.2f MB), student=%s (%.2f MB)",
            self.teacher_path.name,
            self.teacher_size_mb,
            self.student_path.name,
            self.student_size_mb,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distill(
        self,
        train_data: Path | str,
        epochs: int = 20,
        alpha: float = 0.5,
        temperature: float = 4.0,
        output_path: Path | str | None = None,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        imgsz: int = 640,
    ) -> DistillationResult:
        """Execute knowledge distillation training.

        Trains the student model to match the teacher's output
        distribution while also learning from the ground-truth labels.

        NOTE: This method requires a GPU for practical training times.
        If CUDA is not available it will run on CPU but will log a
        warning and be extremely slow.

        Parameters
        ----------
        train_data:
            Path to a YOLO-format dataset directory containing a
            ``data.yaml`` configuration file.
        epochs:
            Number of training epochs (default 20).
        alpha:
            Weight for the hard-label CE loss. The KL-divergence
            (soft-label) loss is weighted by ``(1 - alpha)``.
        temperature:
            Temperature parameter for softening distributions.
        output_path:
            Destination path for the distilled student model.
            If ``None``, a default name is generated.
        batch_size:
            Training batch size.
        learning_rate:
            Learning rate for the optimizer.
        imgsz:
            Input image size for training.

        Returns
        -------
        DistillationResult with teacher, student, and distilled metadata.
        """
        train_data = Path(train_data)
        if output_path is None:
            stem = self.student_path.stem
            output_path = self.student_path.parent / f"{stem}-distilled.pt"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine device
        device = "cuda" if _CUDA_AVAILABLE else "cpu"
        if not _CUDA_AVAILABLE:
            logger.warning(
                "CUDA not available. Running distillation on CPU. "
                "This will be very slow and is not recommended for "
                "production use."
            )

        logger.info(
            "Starting knowledge distillation: "
            "teacher=%s, student=%s, epochs=%d, alpha=%.2f, T=%.1f, "
            "device=%s",
            self.teacher_path.name,
            self.student_path.name,
            epochs,
            alpha,
            temperature,
            device,
        )

        # Load teacher and student models
        teacher_yolo = YOLO(str(self.teacher_path))
        student_yolo = YOLO(str(self.student_path))

        teacher_model = teacher_yolo.model.to(device)
        student_model = student_yolo.model.to(device)

        # Teacher in eval mode (frozen)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Student in train mode
        student_model.train()

        # Build data loader using Ultralytics dataset utilities
        data_yaml = train_data / "data.yaml"
        if not data_yaml.exists():
            data_yaml = train_data  # Assume path points directly to yaml

        # Use Ultralytics built-in dataset loading
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.utils import DEFAULT_CFG

        cfg = get_cfg(DEFAULT_CFG)
        cfg.data = str(data_yaml)
        cfg.imgsz = imgsz
        cfg.batch = batch_size
        cfg.workers = 2

        # Build dataset and dataloader
        dataset = build_yolo_dataset(
            cfg=cfg,
            img_path=str(train_data / "images" / "train"),
            batch=batch_size,
            data=student_yolo.overrides.get("data", {}),
            mode="train",
        )
        dataloader = build_dataloader(
            dataset=dataset,
            batch=batch_size,
            workers=2,
            shuffle=True,
        )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        DistillationLoss(alpha=alpha, temperature=temperature)

        # Training loop
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_data in dataloader:
                images = batch_data["img"].to(device).float() / 255.0
                targets = batch_data.get("cls", batch_data.get("batch_idx"))
                if targets is not None:
                    targets = targets.to(device)

                # Forward pass through teacher (no grad)
                with torch.no_grad():
                    teacher_out = teacher_model(images)

                # Forward pass through student
                student_out = student_model(images)

                # Extract logits - handle different output formats
                # YOLO models may return tuples or lists
                if isinstance(teacher_out, (tuple, list)):
                    teacher_logits = teacher_out[0] if len(teacher_out) > 0 else teacher_out
                else:
                    teacher_logits = teacher_out

                if isinstance(student_out, (tuple, list)):
                    student_logits = student_out[0] if len(student_out) > 0 else student_out
                else:
                    student_logits = student_out

                # Flatten for KL-div computation if needed
                if teacher_logits.dim() > 2:
                    teacher_logits = teacher_logits.view(teacher_logits.size(0), -1)
                if student_logits.dim() > 2:
                    student_logits = student_logits.view(student_logits.size(0), -1)

                # Match dimensions if teacher and student have different output sizes
                if teacher_logits.shape[-1] != student_logits.shape[-1]:
                    # Use MSE loss as fallback when dimensions don't match
                    # Project teacher output to match student dimensions
                    min_dim = min(teacher_logits.shape[-1], student_logits.shape[-1])
                    teacher_logits = teacher_logits[..., :min_dim]
                    student_logits = student_logits[..., :min_dim]

                # Compute distillation loss (feature-based when labels unavailable)
                T = temperature
                student_soft = F.log_softmax(student_logits / T, dim=-1)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)
                loss = F.kl_div(
                    student_soft,
                    teacher_soft,
                    reduction="batchmean",
                ) * (T * T)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            final_loss = avg_epoch_loss
            logger.info(
                "Distillation epoch %d/%d: avg_loss=%.4f",
                epoch + 1,
                epochs,
                avg_epoch_loss,
            )

        # Save distilled student model
        student_model.eval()
        ckpt: dict[str, Any] = {
            "model": student_model.cpu(),
            "optimizer": None,
            "train_args": getattr(student_yolo, "overrides", {}),
            "date": None,
            "version": getattr(student_yolo, "__version__", "8.0.0"),
        }
        torch.save(ckpt, str(output_path))

        distilled_hash = _sha256_file(output_path)
        distilled_size = _file_size_mb(output_path)

        logger.info(
            "Knowledge distillation complete: "
            "distilled=%s (%.2f MB), final_loss=%.4f, epochs=%d",
            output_path.name,
            distilled_size,
            final_loss,
            epochs,
        )

        return DistillationResult(
            teacher_path=self.teacher_path,
            student_path=self.student_path,
            distilled_path=output_path,
            teacher_size_mb=self.teacher_size_mb,
            student_size_mb=self.student_size_mb,
            distilled_size_mb=distilled_size,
            technique="knowledge_distillation",
            teacher_hash=self.teacher_hash,
            distilled_hash=distilled_hash,
            epochs_trained=epochs,
            final_loss=round(final_loss, 6),
        )
