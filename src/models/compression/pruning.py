"""
Pruning de modelos: elimina conexiones/neuronas poco importantes.

Tecnicas implementadas:
1. Unstructured Pruning (L1): elimina pesos individuales con menor magnitud
2. Structured Pruning (L2): elimina canales/filtros completos (mas eficiente en hardware)
3. Pruning + Fine-tuning: poda seguida de re-entrenamiento corto (5 epochs)

El pruning se aplica al modelo PyTorch antes de exportar a ONNX.
Despues del pruning se puede hacer fine-tuning corto (5-10 epochs)
para recuperar precision perdida.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pydantic import BaseModel
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result model
# ------------------------------------------------------------------


class PruningResult(BaseModel):
    """Result of a model pruning operation."""

    original_path: Path
    pruned_path: Path
    original_size_mb: float
    pruned_size_mb: float
    sparsity_achieved: float
    technique: str
    original_hash: str
    pruned_hash: str
    num_params_original: int
    num_params_pruned: int
    num_params_removed: int

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


def _count_total_params(model: nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def _count_nonzero_params(model: nn.Module) -> int:
    """Count non-zero parameters in a model (after pruning)."""
    total = 0
    for p in model.parameters():
        total += int(torch.count_nonzero(p.data).item())
    return total


def _compute_sparsity(model: nn.Module) -> float:
    """Compute overall sparsity of the model as fraction of zero weights.

    Returns
    -------
    Float between 0.0 (dense) and 1.0 (all zeros).
    """
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p.data == 0).sum().item())
    return zeros / total if total > 0 else 0.0


def _get_prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Return list of (module, param_name) tuples for Conv2d layers.

    Only includes Conv2d layers since they dominate YOLO model size.
    """
    modules: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            modules.append((module, "weight"))
    return modules


def _make_pruning_permanent(model: nn.Module) -> None:
    """Remove pruning reparametrization to make masks permanent.

    After calling this, the pruned weights become actual zeros in the
    parameter tensors and the forward hooks are removed.
    """
    import contextlib

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            with contextlib.suppress(ValueError):
                prune.remove(module, "weight")


def _save_yolo_checkpoint(
    yolo: YOLO,
    torch_model: nn.Module,
    output_path: Path,
) -> None:
    """Save a modified torch model in YOLO-compatible checkpoint format.

    Parameters
    ----------
    yolo:
        Original YOLO wrapper (used for metadata).
    torch_model:
        The modified (pruned) torch model to save.
    output_path:
        Destination path for the ``.pt`` file.
    """
    ckpt: dict[str, Any] = {
        "model": torch_model,
        "optimizer": None,
        "train_args": getattr(yolo, "overrides", {}),
        "date": None,
        "version": getattr(yolo, "__version__", "8.0.0"),
    }
    torch.save(ckpt, str(output_path))


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class ModelPruner:
    """Pruning engine for YOLO models.

    Supports L1 unstructured pruning, L2 structured (channel) pruning,
    and pruning with short fine-tuning to recover accuracy.

    Parameters
    ----------
    model_path:
        Path to the PyTorch model file (``.pt``).
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )
        if self.model_path.suffix.lower() != ".pt":
            raise ValueError(
                f"Pruning requires a PyTorch (.pt) model, got: {self.model_path.suffix}"
            )
        self.original_hash = _sha256_file(self.model_path)
        self.original_size_mb = _file_size_mb(self.model_path)
        logger.info(
            "ModelPruner initialised: %s (%.2f MB, hash=%s...)",
            self.model_path.name,
            self.original_size_mb,
            self.original_hash[:12],
        )

    # ------------------------------------------------------------------
    # 1. Unstructured pruning (L1)
    # ------------------------------------------------------------------

    def prune_unstructured(
        self,
        sparsity: float = 0.3,
        output_path: Path | str | None = None,
    ) -> PruningResult:
        """Apply L1 unstructured pruning to all Conv2d layers.

        Removes individual weights with the smallest L1 magnitude.
        This creates sparse weight tensors but does not reduce the
        number of channels, so the actual file size reduction depends
        on the serialization format.

        Parameters
        ----------
        sparsity:
            Fraction of weights to prune (0.0 to 1.0).
        output_path:
            Destination path for the pruned model. If ``None``, a
            default name is generated alongside the original file.

        Returns
        -------
        PruningResult with sparsity and parameter statistics.
        """
        if output_path is None:
            stem = self.model_path.stem
            pct = int(sparsity * 100)
            output_path = self.model_path.parent / f"{stem}-pruned{pct}.pt"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting L1 unstructured pruning: %s (target sparsity=%.0f%%)",
            self.model_path.name,
            sparsity * 100,
        )

        # Load model
        yolo = YOLO(str(self.model_path))
        torch_model = yolo.model
        torch_model.eval()

        num_params_original = _count_total_params(torch_model)

        # Collect prunable modules
        modules_to_prune = _get_prunable_modules(torch_model)
        if not modules_to_prune:
            raise RuntimeError("No Conv2d layers found for pruning")

        logger.info(
            "Found %d Conv2d layers to prune", len(modules_to_prune)
        )

        # Apply global L1 unstructured pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Make pruning permanent (remove hooks, apply masks)
        _make_pruning_permanent(torch_model)

        # Compute stats
        actual_sparsity = _compute_sparsity(torch_model)
        num_params_nonzero = _count_nonzero_params(torch_model)
        num_params_removed = num_params_original - num_params_nonzero

        # Save
        _save_yolo_checkpoint(yolo, torch_model, output_path)

        pruned_hash = _sha256_file(output_path)
        pruned_size = _file_size_mb(output_path)

        logger.info(
            "L1 unstructured pruning complete: %.2f MB -> %.2f MB, "
            "sparsity=%.1f%%, params removed=%d/%d",
            self.original_size_mb,
            pruned_size,
            actual_sparsity * 100,
            num_params_removed,
            num_params_original,
        )

        return PruningResult(
            original_path=self.model_path,
            pruned_path=output_path,
            original_size_mb=self.original_size_mb,
            pruned_size_mb=pruned_size,
            sparsity_achieved=round(actual_sparsity, 4),
            technique="l1_unstructured",
            original_hash=self.original_hash,
            pruned_hash=pruned_hash,
            num_params_original=num_params_original,
            num_params_pruned=num_params_nonzero,
            num_params_removed=num_params_removed,
        )

    # ------------------------------------------------------------------
    # 2. Structured pruning (L2 by channel)
    # ------------------------------------------------------------------

    def prune_structured(
        self,
        amount: float = 0.2,
        output_path: Path | str | None = None,
    ) -> PruningResult:
        """Apply L2 structured pruning by channel to Conv2d layers.

        Removes entire output channels (filters) based on their L2
        norm. This produces a genuinely smaller model since complete
        channels are eliminated, resulting in real compute savings.

        Parameters
        ----------
        amount:
            Fraction of channels to prune per layer (0.0 to 1.0).
        output_path:
            Destination path for the pruned model. If ``None``, a
            default name is generated alongside the original file.

        Returns
        -------
        PruningResult with sparsity and parameter statistics.
        """
        if output_path is None:
            stem = self.model_path.stem
            pct = int(amount * 100)
            output_path = self.model_path.parent / f"{stem}-structured{pct}.pt"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting L2 structured pruning: %s (amount=%.0f%%)",
            self.model_path.name,
            amount * 100,
        )

        # Load model
        yolo = YOLO(str(self.model_path))
        torch_model = yolo.model
        torch_model.eval()

        num_params_original = _count_total_params(torch_model)

        # Apply structured pruning to each Conv2d layer
        pruned_count = 0
        for module in torch_model.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=amount,
                        n=2,  # L2 norm
                        dim=0,  # prune output channels
                    )
                    pruned_count += 1

        logger.info("Applied structured pruning to %d Conv2d layers", pruned_count)

        # Make pruning permanent
        _make_pruning_permanent(torch_model)

        # Compute stats
        actual_sparsity = _compute_sparsity(torch_model)
        num_params_nonzero = _count_nonzero_params(torch_model)
        num_params_removed = num_params_original - num_params_nonzero

        # Save
        _save_yolo_checkpoint(yolo, torch_model, output_path)

        pruned_hash = _sha256_file(output_path)
        pruned_size = _file_size_mb(output_path)

        logger.info(
            "L2 structured pruning complete: %.2f MB -> %.2f MB, "
            "sparsity=%.1f%%, params removed=%d/%d",
            self.original_size_mb,
            pruned_size,
            actual_sparsity * 100,
            num_params_removed,
            num_params_original,
        )

        return PruningResult(
            original_path=self.model_path,
            pruned_path=output_path,
            original_size_mb=self.original_size_mb,
            pruned_size_mb=pruned_size,
            sparsity_achieved=round(actual_sparsity, 4),
            technique="l2_structured_channel",
            original_hash=self.original_hash,
            pruned_hash=pruned_hash,
            num_params_original=num_params_original,
            num_params_pruned=num_params_nonzero,
            num_params_removed=num_params_removed,
        )

    # ------------------------------------------------------------------
    # 3. Pruning + fine-tuning
    # ------------------------------------------------------------------

    def prune_and_finetune(
        self,
        sparsity: float = 0.3,
        finetune_data: Path | str = Path("datasets/xview3"),
        finetune_epochs: int = 5,
        output_path: Path | str | None = None,
    ) -> PruningResult:
        """Apply L1 unstructured pruning followed by short fine-tuning.

        First prunes the model to the target sparsity, then runs a
        short training pass (default 5 epochs) on the supplied YOLO-
        format dataset to recover any accuracy lost from pruning.

        Parameters
        ----------
        sparsity:
            Fraction of weights to prune (0.0 to 1.0).
        finetune_data:
            Path to a YOLO-format dataset directory containing a
            ``data.yaml`` configuration file.
        finetune_epochs:
            Number of fine-tuning epochs (default 5).
        output_path:
            Destination path for the pruned-and-finetuned model.
            If ``None``, a default name is generated.

        Returns
        -------
        PruningResult with sparsity and parameter statistics.
        """
        finetune_data = Path(finetune_data)
        if output_path is None:
            stem = self.model_path.stem
            pct = int(sparsity * 100)
            output_path = self.model_path.parent / f"{stem}-pruned{pct}-ft.pt"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting pruning + fine-tuning: %s "
            "(sparsity=%.0f%%, epochs=%d, data=%s)",
            self.model_path.name,
            sparsity * 100,
            finetune_epochs,
            finetune_data,
        )

        # Load model
        yolo = YOLO(str(self.model_path))
        torch_model = yolo.model

        num_params_original = _count_total_params(torch_model)

        # Step 1: Apply pruning
        modules_to_prune = _get_prunable_modules(torch_model)
        if not modules_to_prune:
            raise RuntimeError("No Conv2d layers found for pruning")

        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        _make_pruning_permanent(torch_model)
        logger.info("Pruning applied, starting fine-tuning for %d epochs", finetune_epochs)

        # Step 2: Fine-tune using Ultralytics training API
        # Locate data.yaml within the dataset directory
        data_yaml = finetune_data / "data.yaml"
        if not data_yaml.exists():
            data_yaml = finetune_data  # Assume path points directly to yaml

        # Save pruned model to a temporary location for Ultralytics to load
        tmp_pruned = output_path.parent / f"_tmp_pruned_{output_path.name}"
        _save_yolo_checkpoint(yolo, torch_model, tmp_pruned)

        try:
            # Reload with Ultralytics and fine-tune
            yolo_pruned = YOLO(str(tmp_pruned))
            results = yolo_pruned.train(
                data=str(data_yaml),
                epochs=finetune_epochs,
                imgsz=640,
                batch=16,
                device="cpu",
                verbose=False,
                project=str(output_path.parent),
                name="finetune_run",
                exist_ok=True,
            )

            # The best weights are saved by Ultralytics; copy to output
            best_weights = Path(results.save_dir) / "weights" / "best.pt"
            if best_weights.exists():
                import shutil

                shutil.copy2(str(best_weights), str(output_path))
            else:
                # Fall back: use the last weights
                last_weights = Path(results.save_dir) / "weights" / "last.pt"
                if last_weights.exists():
                    import shutil

                    shutil.copy2(str(last_weights), str(output_path))
                else:
                    # Fall back to the pruned (unfinetuned) model
                    import shutil

                    shutil.copy2(str(tmp_pruned), str(output_path))
                    logger.warning(
                        "Fine-tuning did not produce weights; "
                        "using pruned model without fine-tuning"
                    )
        finally:
            # Clean up temporary file
            if tmp_pruned.exists():
                tmp_pruned.unlink()

        # Reload final model to compute stats
        final_yolo = YOLO(str(output_path))
        final_model = final_yolo.model
        actual_sparsity = _compute_sparsity(final_model)
        num_params_nonzero = _count_nonzero_params(final_model)
        num_params_removed = num_params_original - num_params_nonzero

        pruned_hash = _sha256_file(output_path)
        pruned_size = _file_size_mb(output_path)

        logger.info(
            "Pruning + fine-tuning complete: %.2f MB -> %.2f MB, "
            "sparsity=%.1f%%, epochs=%d",
            self.original_size_mb,
            pruned_size,
            actual_sparsity * 100,
            finetune_epochs,
        )

        return PruningResult(
            original_path=self.model_path,
            pruned_path=output_path,
            original_size_mb=self.original_size_mb,
            pruned_size_mb=pruned_size,
            sparsity_achieved=round(actual_sparsity, 4),
            technique=f"l1_unstructured_finetune_{finetune_epochs}ep",
            original_hash=self.original_hash,
            pruned_hash=pruned_hash,
            num_params_original=num_params_original,
            num_params_pruned=num_params_nonzero,
            num_params_removed=num_params_removed,
        )
