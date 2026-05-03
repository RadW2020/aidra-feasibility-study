"""
Tests for the EvidenceBundler (D3 evidence package).

Bundler entry point is ``EvidenceBundler.build()``, which talks to the
DB through ``self._db.fetch``. We mock the DB with ``AsyncMock`` so the
suite stays CI-friendly (no PostgreSQL required) while still exercising
the real serialization / hashing / archive code paths.

What we are actually testing:
  * MANIFEST.json contains a SHA256 per file matching a fresh recompute.
  * MANIFEST.sha256 equals the SHA256 of MANIFEST.json on disk
    (the bundle root signature).
  * The tar.gz contains every artefact required by D3.
  * settings.json never leaks ``copernicus_password``.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.config import Settings
from src.traceability.bundler import EvidenceBundler
from src.traceability.hasher import compute_sha256

# ====================================================================
# Helpers
# ====================================================================


def _build_execution_row(mock_execution_record: dict) -> dict:
    """Return a row compatible with _SELECT_EXECUTIONS_BUNDLE (subset of
    columns the bundler reads).  Keys mirror the SELECT in bundler.py.
    """
    return {
        "id": mock_execution_record["id"],
        "created_at": mock_execution_record["created_at"],
        "image_id": mock_execution_record["image_id"],
        "image_title": mock_execution_record["image_title"],
        "image_hash": mock_execution_record["image_hash"],
        "image_bbox_geojson": json.dumps(
            {"type": "Polygon", "coordinates": [[
                [-5.6, 35.9], [-5.0, 35.9],
                [-5.0, 36.3], [-5.6, 36.3],
                [-5.6, 35.9],
            ]]}
        ),
        "image_sensing_date": mock_execution_record["image_sensing_date"],
        "image_size_mb": mock_execution_record["image_size_mb"],
        "search_zone": mock_execution_record["search_zone"],
        "model_name": mock_execution_record["model_name"],
        "model_version": mock_execution_record["model_version"],
        "model_hash": mock_execution_record["model_hash"],
        "model_size_mb": mock_execution_record["model_size_mb"],
        "model_format": mock_execution_record["model_format"],
        "compression_technique": mock_execution_record["compression_technique"],
        "confidence_threshold": mock_execution_record["confidence_threshold"],
        "iou_threshold": mock_execution_record["iou_threshold"],
        "constraint_profile": mock_execution_record["constraint_profile"],
        "tile_size": mock_execution_record["tile_size"],
        "tile_overlap": mock_execution_record["tile_overlap"],
        "num_detections": mock_execution_record["num_detections"],
        "avg_confidence": mock_execution_record["avg_confidence"],
        "total_duration_ms": mock_execution_record["total_duration_ms"],
        "peak_ram_mb": mock_execution_record["peak_ram_mb"],
        "output_hash": mock_execution_record["output_hash"],
        "input_params_hash": mock_execution_record["input_params_hash"],
        "commit_sha": mock_execution_record["commit_sha"],
        "status": mock_execution_record["status"],
        "trigger_type": mock_execution_record["trigger_type"],
        "pipeline_version": mock_execution_record["pipeline_version"],
        "hostname": mock_execution_record["hostname"],
    }


def _build_detection_row(execution_id, image_id: str, model_name: str) -> dict:
    """Return a row compatible with _SELECT_DETECTIONS_BUNDLE."""
    return {
        "detection_id": uuid4(),
        "execution_id": execution_id,
        "created_at": datetime.now(tz=UTC),
        "longitude": -5.4,
        "latitude": 36.05,
        "center_geojson": json.dumps(
            {"type": "Point", "coordinates": [-5.4, 36.05]}
        ),
        "bbox_geojson": json.dumps(
            {"type": "Polygon", "coordinates": [[
                [-5.41, 36.04], [-5.39, 36.04],
                [-5.39, 36.06], [-5.41, 36.06],
                [-5.41, 36.04],
            ]]}
        ),
        "bbox_pixel": [100, 200, 120, 220],
        "confidence": 0.83,
        "source": "fused",
        "cfar_snr": 12.5,
        "yolo_score": 0.83,
        "class_name": "vessel",
        "tile_index": 0,
        "on_land": False,
        "cluster_anomaly": False,
        "thumbnail_path": None,
        "image_id": image_id,
        "model_name": model_name,
        "constraint_profile": "ground",
    }


def _make_bundler_with_mocked_db(
    settings: Settings,
    executions: list[dict],
    detections: list[dict],
    tmp_path: Path,
) -> EvidenceBundler:
    """Construct an EvidenceBundler with an AsyncMock DB returning the
    provided rows. ``models_cards_dir`` points to a fresh tmp dir so the
    real ``models/cards`` of the repo is not consulted.
    """
    db = AsyncMock()

    async def _fetch(query: str, *args):  # noqa: ANN001
        # First call -> executions; second call -> detections.
        # We discriminate by SQL fragment.
        if "FROM execution_log" in query and "FROM detections" not in query:
            return executions
        if "FROM detections" in query:
            return detections
        return []

    db.fetch.side_effect = _fetch

    cards_dir = tmp_path / "model_cards_empty"
    cards_dir.mkdir()
    return EvidenceBundler(db=db, settings=settings, models_cards_dir=cards_dir)


# ====================================================================
# Test 1 — file SHA256 helper / manifest correctness
# ====================================================================


@pytest.mark.asyncio
async def test_compute_manifest_sha256_deterministic(
    tmp_path: Path,
    settings: Settings,
    mock_execution_record: dict,
) -> None:
    """Every entry in MANIFEST.json must equal a fresh SHA256 recompute
    of the corresponding bundle file.

    Bundler exposes no public per-file helper, so we drive it through
    ``build(archive=False)`` and verify each MANIFEST entry against a
    direct recomputation with ``compute_sha256``.
    """
    exec_row = _build_execution_row(mock_execution_record)
    det_row = _build_detection_row(
        exec_row["id"], exec_row["image_id"], exec_row["model_name"]
    )

    bundler = _make_bundler_with_mocked_db(
        settings, [exec_row], [det_row], tmp_path
    )
    bundle_dir = await bundler.build(out_dir=tmp_path / "out", archive=False)

    manifest_path = bundle_dir / "MANIFEST.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"], "MANIFEST.files must not be empty"

    for rel, expected_sha in manifest["files"].items():
        actual = compute_sha256(bundle_dir / rel)
        assert actual == expected_sha, (
            f"SHA mismatch for {rel}: manifest={expected_sha} actual={actual}"
        )
        assert len(expected_sha) == 64


# ====================================================================
# Test 2 — root signature: MANIFEST.sha256 == sha256(MANIFEST.json)
# ====================================================================


@pytest.mark.asyncio
async def test_manifest_root_signature_stable(
    tmp_path: Path,
    settings: Settings,
    mock_execution_record: dict,
) -> None:
    """``MANIFEST.sha256`` is the bundle's root signature.  It must
    equal the SHA256 of MANIFEST.json as written to disk — that single
    line is what an auditor cites to vouch for the whole bundle.
    """
    exec_row = _build_execution_row(mock_execution_record)
    det_row = _build_detection_row(
        exec_row["id"], exec_row["image_id"], exec_row["model_name"]
    )

    bundler = _make_bundler_with_mocked_db(
        settings, [exec_row], [det_row], tmp_path
    )
    bundle_dir = await bundler.build(out_dir=tmp_path / "out", archive=False)

    manifest_path = bundle_dir / "MANIFEST.json"
    sig_path = bundle_dir / "MANIFEST.sha256"
    assert sig_path.exists()

    # Recompute SHA256 of the on-disk MANIFEST.json content.
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    sig_line = sig_path.read_text(encoding="utf-8").strip()
    # Format: "<sha>  MANIFEST.json"
    sig_sha = sig_line.split()[0]

    assert sig_sha == expected, (
        f"MANIFEST.sha256 ({sig_sha}) does not match SHA256 of "
        f"MANIFEST.json on disk ({expected}) — root signature broken"
    )
    assert sig_sha == compute_sha256(manifest_path)


# ====================================================================
# Test 3 — required artefacts present in the tar.gz
# ====================================================================


@pytest.mark.asyncio
async def test_bundle_includes_required_artefacts(
    tmp_path: Path,
    settings: Settings,
    mock_execution_record: dict,
) -> None:
    """The D3 archive must contain every artefact a SatCen auditor
    expects: manifest + signature + executions + detections (csv +
    geojson) + settings snapshot.
    """
    exec_row = _build_execution_row(mock_execution_record)
    det_row = _build_detection_row(
        exec_row["id"], exec_row["image_id"], exec_row["model_name"]
    )

    bundler = _make_bundler_with_mocked_db(
        settings, [exec_row], [det_row], tmp_path
    )
    archive_path = await bundler.build(
        out_dir=tmp_path / "out", archive=True
    )

    assert archive_path.suffix == ".gz"
    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as tar:
        names = {Path(n).name for n in tar.getnames()}

    required = {
        "MANIFEST.json",
        "MANIFEST.sha256",
        "executions.csv",
        "detections.csv",
        "detections.geojson",
        "settings.json",
    }
    missing = required - names
    assert not missing, f"Bundle is missing required artefacts: {missing}"


# ====================================================================
# Test 4 — secrets redacted in settings.json
# ====================================================================


@pytest.mark.asyncio
async def test_bundle_settings_sensitive_fields_redacted(
    tmp_path: Path,
    settings: Settings,
    mock_execution_record: dict,
) -> None:
    """``copernicus_password`` (and friends) must never reach the bundle
    in plaintext. The Settings serializer must mask them as ``***`` or
    omit them entirely.

    Closes the I-EU-1 / data-protection corner of the D3 deliverable.
    """
    # Sanity: the test fixture really has a non-empty password.
    assert settings.copernicus_password == "test_pass"

    exec_row = _build_execution_row(mock_execution_record)
    det_row = _build_detection_row(
        exec_row["id"], exec_row["image_id"], exec_row["model_name"]
    )

    bundler = _make_bundler_with_mocked_db(
        settings, [exec_row], [det_row], tmp_path
    )
    bundle_dir = await bundler.build(out_dir=tmp_path / "out", archive=False)

    settings_path = bundle_dir / "settings.json"
    raw = settings_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    # Real password value must NOT appear anywhere in the file.
    assert "test_pass" not in raw, (
        "Plaintext copernicus_password leaked into settings.json"
    )

    # If the field is present, it must be masked.
    if "copernicus_password" in data:
        assert data["copernicus_password"] == "***", (
            "copernicus_password must be redacted to '***'"
        )
