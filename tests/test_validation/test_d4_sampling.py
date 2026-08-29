"""GT-anchored D4 sampling and fidelity helpers."""

from __future__ import annotations

import numpy as np

from src.validation.d4_sampling import STRATA, build_pool, chip_window, pointing_game, stratify


def _scene():
    gt = [{"bbox": [100, 100, 106, 106], "vessel_length_m": 40.0},
          {"bbox": [300, 300, 306, 306]},
          {"bbox": [500, 500, 506, 506], "on_land": True},  # excluded (I-DET-2)
          {"bbox": [700, 700, 706, 706]}]                    # missed -> fn
    preds = [{"bbox": [99, 99, 105, 105], "confidence": 0.9, "source": "fused"},
             {"bbox": [302, 302, 308, 308], "confidence": 0.3, "source": "cfar"},
             {"bbox": [900, 900, 904, 904], "confidence": 0.5, "source": "cfar"},   # fp
             {"bbox": [50, 800, 54, 804], "confidence": 0.4, "source": "yolo", "on_land": True}]  # excluded
    return preds, gt


def test_build_pool_classifies_and_skips_land():
    preds, gt = _scene()
    pool = build_pool(preds, gt)
    assert [p["confidence"] for p in pool["tp"]] == [0.9, 0.3]
    assert pool["tp"][0]["gt_bbox"] == [100, 100, 106, 106] and pool["tp"][0]["gt_vessel_length_m"] == 40.0
    assert len(pool["fp"]) == 1 and pool["fp"][0]["bbox"][0] == 900
    assert len(pool["fn"]) == 1 and pool["fn"][0]["gt_bbox"] == [700, 700, 706, 706]


def test_stratify_covers_all_strata_deterministically():
    preds, gt = _scene()
    pool = build_pool(preds, gt)
    a, info = stratify(pool, 2, seed=3)
    b, _ = stratify(pool, 2, seed=3)
    assert [(s["stratum"], s["bbox"]) for s in a] == [(s["stratum"], s["bbox"]) for s in b]
    assert {s["stratum"] for s in a} == set(STRATA)
    assert info["strata"]["tp_high"]["picked"] == 1 and info["strata"]["tp_low"]["picked"] == 1
    assert info["strata"]["fn"]["pool"] == 1


def test_chip_window_clamps_to_raster():
    assert chip_window((10, 10), 64, 1000, 1000) == (0, 0, 64, 64)
    assert chip_window((995, 995), 64, 1000, 1000) == (936, 936, 64, 64)
    assert chip_window((500, 500), 64, 1000, 1000) == (468, 468, 64, 64)
    assert chip_window((10, 10), 64, 32, 32) == (0, 0, 32, 32)


def test_pointing_game_hit_and_mass():
    cam = np.zeros((100, 100))
    cam[50, 50] = 1.0
    cam[10, 90] = 0.25
    inside = pointing_game(cam, [45, 45, 55, 55], tol_px=0)
    assert inside["hit"] is True and inside["mass_in_box"] == 0.8 and inside["peak_distance_px"] == 0.0
    outside = pointing_game(cam, [0, 80, 20, 99], tol_px=0)
    assert outside["hit"] is False and outside["mass_in_box"] == 0.0
    assert pointing_game(np.zeros((4, 4)), [0, 0, 2, 2])["hit"] is False
