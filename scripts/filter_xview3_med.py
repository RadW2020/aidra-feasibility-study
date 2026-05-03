"""
Filter the xView3-SAR validation split down to Mediterranean + Suez
scenes for AIDRA D2 evaluation (palanca L17).

Inputs (the user already downloaded these into x-view-us-data/):
  - validation.csv           — labels (~19 k detections, 50 scenes)
  - validation.txt           — aria2 input with signed S3 URLs
  - ESA_xView3_sceneName_mapping.csv — Sentinel-1 product → xView3 ID

Outputs (under data/xview3/):
  - manifest_med.json        — per-scene summary (centroid, n_dets, in_med)
  - validation_med.txt       — aria2 input restricted to Med scenes only
  - report.txt               — human-readable summary

The Mediterranean+Suez bbox is (lat 30..46 N, lon -6..36 E) which
covers the four AIDRA operational zones declared in mvp_oci.md
(Gibraltar, Mar Rojo, Canal de Suez, English Channel … wait —
English Channel is excluded). For the validation scope we keep
strict Mediterranean+Suez and report English Channel separately
when it appears.

Run::

    python -m scripts.filter_xview3_med \\
        --xview-dir x-view-us-data \\
        --out-dir data/xview3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

logger = logging.getLogger("aidra.xview3_filter")

# AIDRA operational zones — Mediterranean + Red Sea + Suez Canal.
DEFAULT_BBOX = (30.0, 46.0, -6.0, 36.0)  # (lat_min, lat_max, lon_min, lon_max)

# Sentinel-1 IW GRD swath ≈ 250 km × 180 km.
S1_IW_PIXEL_SPACING_M = 10.0


@dataclass
class SceneSummary:
    scene_id: str
    n_detections: int
    n_vessels: int
    centroid_lat: float
    centroid_lon: float
    bbox_lat_min: float
    bbox_lat_max: float
    bbox_lon_min: float
    bbox_lon_max: float
    aoi_label: str = ""
    aria2_lines: list[str] = field(default_factory=list)


def _detect_aoi(lat: float, lon: float) -> str:
    """Coarse AOI label so the report makes intuitive sense."""
    # Gibraltar
    if 35 <= lat <= 37 and -7 <= lon <= -3:
        return "gibraltar"
    # Suez canal
    if 28 <= lat <= 33 and 31 <= lon <= 33:
        return "suez-canal"
    # Red Sea
    if 12 <= lat <= 30 and 32 <= lon <= 44:
        return "red-sea"
    # English Channel
    if 49 <= lat <= 51 and -6 <= lon <= 2:
        return "english-channel"
    # Adriatic
    if 39 <= lat <= 46 and 12 <= lon <= 21:
        return "adriatic"
    # Aegean
    if 35 <= lat <= 41 and 22 <= lon <= 30:
        return "aegean"
    # Western Med
    if 30 <= lat <= 44 and -6 <= lon <= 12:
        return "west-med"
    # Eastern Med
    if 30 <= lat <= 38 and 22 <= lon <= 36:
        return "east-med"
    return "other"


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def load_aria2_blocks(validation_txt: Path) -> dict[str, list[str]]:
    """Parse the aria2 input into ``{scene_id: [url_line, checksum_line]}``.

    The xView3 aria2 format pairs each URL with its sha-1 checksum on
    the next line (the second line is indented with a single space —
    aria2's per-resource option syntax). There are no blank
    separators between pairs.
    """
    blocks: dict[str, list[str]] = {}
    current_scene: str | None = None
    pair: list[str] = []
    with validation_txt.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("https://") and "/validation/" in line:
                # Flush previous pair, if any.
                if current_scene and pair:
                    blocks[current_scene] = pair
                tail = line.split("/validation/", 1)[1]
                current_scene = tail.split(".tar.gz", 1)[0]
                pair = [line]
            else:
                pair.append(line)
        if current_scene and pair:
            blocks[current_scene] = pair
    return blocks


def load_detections(validation_csv: Path) -> dict[str, list[dict]]:
    """Group detections by ``scene_id``."""
    out: dict[str, list[dict]] = {}
    with validation_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row["scene_id"]
            out.setdefault(sid, []).append(row)
    return out


def build_summary(
    scene_id: str,
    detections: list[dict],
    aria2_lines: list[str],
) -> SceneSummary:
    lats = [float(d["detect_lat"]) for d in detections]
    lons = [float(d["detect_lon"]) for d in detections]
    n_vessels = sum(1 for d in detections if d.get("is_vessel", "True") == "True")
    centroid_lat = float(median(lats))
    centroid_lon = float(median(lons))
    return SceneSummary(
        scene_id=scene_id,
        n_detections=len(detections),
        n_vessels=n_vessels,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        bbox_lat_min=min(lats),
        bbox_lat_max=max(lats),
        bbox_lon_min=min(lons),
        bbox_lon_max=max(lons),
        aoi_label=_detect_aoi(centroid_lat, centroid_lon),
        aria2_lines=aria2_lines,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xview-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_BBOX[0])
    parser.add_argument("--lat-max", type=float, default=DEFAULT_BBOX[1])
    parser.add_argument("--lon-min", type=float, default=DEFAULT_BBOX[2])
    parser.add_argument("--lon-max", type=float, default=DEFAULT_BBOX[3])
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    val_csv = args.xview_dir / "validation.csv"
    val_txt = args.xview_dir / "validation.txt"
    if not val_csv.exists():
        logger.error("Missing %s", val_csv)
        return 2
    if not val_txt.exists():
        logger.error("Missing %s", val_txt)
        return 2

    aria_blocks = load_aria2_blocks(val_txt)
    detections_by_scene = load_detections(val_csv)
    logger.info(
        "Parsed %d aria2 blocks and %d scenes with detections",
        len(aria_blocks),
        len(detections_by_scene),
    )

    bbox = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    summaries: list[SceneSummary] = []
    for sid, dets in detections_by_scene.items():
        block = aria_blocks.get(sid, [])
        summaries.append(build_summary(sid, dets, block))

    in_med = [s for s in summaries if _in_bbox(s.centroid_lat, s.centroid_lon, bbox)]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Persist summary as JSON (machine-readable).
    manifest = [
        {
            "scene_id": s.scene_id,
            "n_detections": s.n_detections,
            "n_vessels": s.n_vessels,
            "centroid_lat": round(s.centroid_lat, 4),
            "centroid_lon": round(s.centroid_lon, 4),
            "bbox": {
                "lat_min": round(s.bbox_lat_min, 4),
                "lat_max": round(s.bbox_lat_max, 4),
                "lon_min": round(s.bbox_lon_min, 4),
                "lon_max": round(s.bbox_lon_max, 4),
            },
            "aoi_label": s.aoi_label,
            "in_mediterranean": s in in_med,
            "has_aria2_block": bool(s.aria2_lines),
        }
        for s in summaries
    ]
    manifest_path = args.out_dir / "manifest_med.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Aria2 input restricted to Mediterranean.
    med_txt = args.out_dir / "validation_med.txt"
    with med_txt.open("w") as fh:
        for s in in_med:
            for line in s.aria2_lines:
                fh.write(line + "\n")
            fh.write("\n")

    # Human-readable report.
    n_total_dets = sum(s.n_detections for s in summaries)
    n_med_dets = sum(s.n_detections for s in in_med)
    aoi_counter: dict[str, int] = {}
    for s in summaries:
        aoi_counter[s.aoi_label] = aoi_counter.get(s.aoi_label, 0) + 1

    lines: list[str] = []
    lines.append("xView3-SAR validation split — geographic breakdown")
    lines.append("=" * 56)
    lines.append(f"Total scenes:         {len(summaries)}")
    lines.append(f"Total detections:     {n_total_dets}")
    lines.append("")
    lines.append("Distribution by AOI label:")
    for aoi, count in sorted(aoi_counter.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {aoi:<20} {count:>3} scenes")
    lines.append("")
    lines.append(
        f"Mediterranean filter (lat {bbox[0]}..{bbox[1]}, lon {bbox[2]}..{bbox[3]}):"
    )
    lines.append(f"  In-Med scenes:      {len(in_med)}")
    lines.append(f"  In-Med detections:  {n_med_dets}")
    lines.append("")
    lines.append("Mediterranean scenes (centroid sorted by AOI):")
    for s in sorted(in_med, key=lambda x: (x.aoi_label, x.scene_id)):
        lines.append(
            f"  {s.scene_id}  AOI={s.aoi_label:<14} "
            f"centroid=({s.centroid_lat:7.3f}, {s.centroid_lon:7.3f})  "
            f"n_det={s.n_detections}"
        )
    lines.append("")
    lines.append(f"Aria2 input (Med subset) written to: {med_txt}")
    lines.append(f"  → run: aria2c --input-file={med_txt} \\")
    lines.append("           --auto-file-renaming=false --continue=true \\")
    lines.append("           --dir=data/xview3/scenes/")
    lines.append("")
    report = "\n".join(lines)
    (args.out_dir / "report.txt").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
