"""Weights loaded back into the detector must be those of the model that infers.

Produccion 2026-11-18: el barrido de bit-flip leia los pesos antes de la
primera inferencia, Ultralytics fusionaba Conv+BN al inferir, y recargar los
pesos originales (sin ningun flip) ya cambiaba las detecciones de 5 a 21.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.yolo import YOLODetector


def _detector(module: torch.nn.Module) -> YOLODetector:
    det = object.__new__(YOLODetector)
    det._model = SimpleNamespace(model=module)
    return det


def test_weights_round_trip_unchanged() -> None:
    det = _detector(torch.nn.Conv2d(1, 2, 3))
    weights = det.get_weights_dict()

    weights["weight"][0, 0, 0, 0] = 7.0
    det.load_weights_dict(weights)

    assert float(det._model.model.weight[0, 0, 0, 0]) == 7.0


def test_weights_of_another_architecture_are_refused() -> None:
    # Pesos leidos antes de la fusion: sobran capas (la BN) respecto al modelo.
    before = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, bias=False), torch.nn.BatchNorm2d(2))
    stale = _detector(before).get_weights_dict()
    fused = _detector(torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3)))

    with pytest.raises(ValueError, match="do not match the loaded model"):
        fused.load_weights_dict(stale)


def test_weights_with_another_shape_are_refused() -> None:
    det = _detector(torch.nn.Conv2d(1, 2, 3))

    with pytest.raises(ValueError, match="another shape"):
        det.load_weights_dict({"weight": np.zeros((2, 1, 5, 5), dtype=np.float32)})
