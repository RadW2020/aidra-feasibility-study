"""Base detector interface for AIDRA models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseDetector(ABC):
    """Abstract base class for all detection models in AIDRA."""

    @abstractmethod
    def predict(self, image: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Run inference on a single image (tile)."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
