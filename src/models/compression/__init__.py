"""
Model compression utilities for AIDRA.

Provides quantization, pruning, and knowledge distillation techniques
to produce optimised model variants for on-board satellite processing.
"""

from src.models.compression.distillation import (
    DistillationResult,
    KnowledgeDistiller,
)
from src.models.compression.pruning import ModelPruner, PruningResult
from src.models.compression.quantization import (
    ModelQuantizer,
    QuantizationResult,
)

__all__ = [
    "ModelQuantizer",
    "QuantizationResult",
    "ModelPruner",
    "PruningResult",
    "KnowledgeDistiller",
    "DistillationResult",
]
