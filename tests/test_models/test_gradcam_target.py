"""Detection-targeted Grad-CAM: anchor selection on synthetic YOLOv8 outputs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.interpretability import select_target_anchor  # noqa: E402


def _raw():
    # (1, 4 + nc, N): xywh + 1 class score for 3 anchors
    raw = torch.zeros((1, 5, 3))
    raw[0, 0, :] = torch.tensor([50.0, 100.0, 300.0])  # cx
    raw[0, 1, :] = torch.tensor([50.0, 100.0, 300.0])  # cy
    raw[0, 4, :] = torch.tensor([0.9, 0.4, 0.8])       # scores
    return raw


def test_picks_highest_score_within_tolerance():
    idx, dist = select_target_anchor(_raw(), (98.0, 102.0), tol_px=20.0)
    assert idx == 1 and dist == pytest.approx(2.828, abs=1e-2)
    # both anchors 0 and 1 inside a big tolerance -> highest score wins
    idx, _ = select_target_anchor(_raw(), (75.0, 75.0), tol_px=60.0)
    assert idx == 0


def test_none_when_nothing_close_or_no_target():
    assert select_target_anchor(_raw(), (200.0, 200.0), tol_px=20.0) == (None, None)
    assert select_target_anchor(_raw(), None, tol_px=20.0) == (None, None)
    assert select_target_anchor(torch.zeros((1, 3, 4)), (0.0, 0.0), tol_px=5.0) == (None, None)


def test_defaults_are_p3_and_targeted_in_production():
    from pathlib import Path

    src = Path("src/models/interpretability.py").read_text()
    assert 'target_layer_name: str = "model.model.15"' in src
    assert src.count("target_xy=centre") == 2  # both production entry points target the chip centre
