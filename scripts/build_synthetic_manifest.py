"""
Build a synthetic validation manifest for ``scripts/run_validation.py``.

Writes N synthetic SAR tiles to disk as GeoTIFFs (linear sigma0,
float32) with corresponding ground-truth bounding boxes. Useful for:

  - Smoke-testing the validation harness end-to-end without needing
    xView3-SAR / HRSID downloads.
  - Producing a deterministic baseline reference report tagged
    explicitly as "synthetic" — never confused with a real
    operational metric.

Usage::

    python -m scripts.build_synthetic_manifest \\
        --out data/validation/synthetic \\
        --num-scenes 20 \\
        --tile-size 640 \\
        --vessels-per-scene 5 \\
        --seed 42

Notes
-----
- All ground-truth bounding boxes are in **pixel** coordinates,
  matching the harness contract.
- ``scene_area_km2`` is computed assuming Sentinel-1 GRD IW pixel
  spacing of 10 m × 10 m (a tile of 640×640 px ≈ 40.96 km²).
- The synthetic background uses Rayleigh clutter (standard SAR sea
  surface model) and Gaussian bright points for vessels — same
  distribution used by the L4 reproducibility tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

# Make src importable when invoked as ``python scripts/build_...``.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.pipeline.preprocessing import generate_synthetic_sar_tile  # noqa: E402

logger = logging.getLogger("aidra.build_synthetic")


def _write_geotiff(path: Path, arr: np.ndarray) -> None:
    """Write a single-band GeoTIFF with a benign WGS-84 transform."""
    height, width = arr.shape
    # Place the tile in the open Atlantic far from any real footprint
    # to avoid accidental spatial collisions in tests/dashboards.
    transform = Affine.translation(-30.0, 30.0) * Affine.scale(1e-4, -1e-4)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a synthetic SAR validation manifest."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--vessels-per-scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    # GRD IW spacing: 10 m × 10 m → area in km² = (size_px*10/1000)**2
    area_km2 = (args.tile_size * 10.0 / 1000.0) ** 2

    rng = np.random.default_rng(args.seed)
    for i in range(args.num_scenes):
        scene_seed = int(rng.integers(0, 2**31 - 1))
        # Allow some scenes to have 0 vessels to exercise the FP=0 case.
        n_vessels = int(rng.integers(0, args.vessels_per_scene + 1))
        image, gt = generate_synthetic_sar_tile(
            size=args.tile_size,
            num_vessels=n_vessels,
            seed=scene_seed,
        )
        scene_name = f"synthetic_{i:03d}.tif"
        scene_path = args.out / scene_name
        _write_geotiff(scene_path, image)
        manifest.append({
            "image_id": scene_name,
            "image_path": str(scene_path.resolve()),
            "scene_area_km2": area_km2,
            "ground_truth": [{"bbox": list(map(float, g["bbox"]))} for g in gt],
            "aoi_name": "synthetic-atlantic",
            "seed": scene_seed,
        })

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total_gt = sum(len(e["ground_truth"]) for e in manifest)
    logger.info(
        "Wrote %d scenes (%d ground truths, %.1f km² total) to %s",
        len(manifest),
        total_gt,
        sum(e["scene_area_km2"] for e in manifest),
        args.out,
    )
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
