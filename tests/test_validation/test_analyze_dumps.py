"""Tests for ``scripts/analyze_prediction_dumps.py`` on synthetic dumps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import analyze_prediction_dumps as apd  # noqa: E402


def _dump(path: Path, scene_id: str) -> None:
    # GT: A (100,100), B (300,300), C (500,500). CFAR hits A and B with tiny
    # 4 px boxes; YOLO hits B and C with 30 px boxes. Fusion by IoU cannot
    # fire on B (4x4 inside 30x30 -> IoU ~0.018) although centres coincide.
    gt = [
        {"bbox": [97, 97, 103, 103], "on_land": False},
        {"bbox": [297, 297, 303, 303], "on_land": False},
        {"bbox": [497, 497, 503, 503], "on_land": False},
    ]
    cfar = [
        {"bbox": [98, 98, 102, 102], "confidence": 0.9, "source": "cfar", "on_land": False},
        {"bbox": [298, 298, 302, 302], "confidence": 0.8, "source": "cfar", "on_land": False},
        {"bbox": [700, 700, 704, 704], "confidence": 0.5, "source": "cfar", "on_land": True},
    ]
    yolo = [
        {"bbox": [285, 285, 315, 315], "confidence": 0.7, "source": "yolo", "on_land": False},
        {"bbox": [485, 485, 515, 515], "confidence": 0.6, "source": "yolo", "on_land": False},
    ]
    path.write_text(
        json.dumps({
            "scene_id": scene_id,
            "ground_truth": gt,
            "sets": {"cfar": cfar, "yolo": yolo, "aidra": cfar + yolo, "fused_only": []},
        })
    )


def test_analyze_reports_geometry_venn_and_pool(tmp_path: Path):
    _dump(tmp_path / "s1.json", "s1")
    _dump(tmp_path / "s2.json", "s2")
    a = apd.analyze(tmp_path, tol_px=20.0, fusion_iou=0.3)

    assert a["num_scenes"] == 2 and a["num_ground_truth"] == 6
    assert a["box_sizes"]["cfar"]["width_px_median"] == 4.0
    assert a["box_sizes"]["yolo"]["width_px_median"] == 30.0

    y = a["yolo_vs_nearest_cfar"]
    assert y["n_yolo"] == 4
    assert y["within_tolerance"] == 2  # the B boxes co-locate with CFAR
    assert y["iou_ge_fusion_threshold"] == 0  # ...but IoU never reaches 0.3
    assert y["iou_max_when_within_tolerance"] < 0.3

    v = a["gt_venn_cfar_yolo"]
    assert v == {"both": 2, "cfar_only": 2, "yolo_only": 2, "neither": 0, "recovered_by_any_rate": 1.0}

    sizes = a["pool_sizes"]
    assert sizes["cfar"] == {"tp": 4, "fp": 2, "fn": 2}
    assert sizes["yolo"] == {"tp": 4, "fp": 0, "fn": 2}
    assert sizes["aidra"]["fn"] == 0
    assert all(p["scene_id"] in {"s1", "s2"} for p in a["pool"]["cfar"]["fp"])
    assert a["pool"]["cfar"]["fp"][0]["on_land"] is True


def test_markdown_and_cli(tmp_path: Path):
    _dump(tmp_path / "s1.json", "s1")
    out = tmp_path / "analysis.json"
    assert apd.main(["--dumps", str(tmp_path), "--output", str(out), "--fusion-iou", "0.3"]) == 0
    assert out.exists() and out.with_suffix(".md").exists()
    md = out.with_suffix(".md").read_text()
    assert "fusión" in md and "Complementariedad" in md
