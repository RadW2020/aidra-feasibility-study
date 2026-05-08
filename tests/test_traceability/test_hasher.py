"""
Tests for the traceability hasher module.

Covers:
- SHA256 file hashing (determinism, different files)
- Numpy array hashing (determinism)
- Detection result hashing (order independence, different data)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.traceability.hasher import (
    compute_array_hash,
    compute_input_params_hash,
    compute_result_hash,
    compute_sha256,
    get_commit_sha,
)

# ====================================================================
# File hashing
# ====================================================================


class TestSHA256File:
    """Tests for compute_sha256 on files."""

    def test_sha256_deterministic(self, tmp_path: Path):
        """The same file must always produce the same SHA256 hash."""
        file = tmp_path / "test.bin"
        file.write_bytes(b"test data for hashing" * 1000)

        hash1 = compute_sha256(file)
        hash2 = compute_sha256(file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest is 64 chars

    def test_sha256_different_files(self, tmp_path: Path):
        """Different file contents must produce different hashes."""
        file_a = tmp_path / "a.bin"
        file_b = tmp_path / "b.bin"
        file_a.write_bytes(b"content A" * 500)
        file_b.write_bytes(b"content B" * 500)

        hash_a = compute_sha256(file_a)
        hash_b = compute_sha256(file_b)

        assert hash_a != hash_b

    def test_sha256_empty_file(self, tmp_path: Path):
        """An empty file must still produce a valid 64-char hash
        (the SHA256 of empty input is well-defined).
        """
        file = tmp_path / "empty.bin"
        file.write_bytes(b"")

        h = compute_sha256(file)
        assert len(h) == 64

    def test_sha256_missing_file_raises(self, tmp_path: Path):
        """Hashing a non-existent file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "nonexistent.bin")


# ====================================================================
# Array hashing
# ====================================================================


class TestArrayHash:
    """Tests for compute_array_hash on numpy arrays."""

    def test_array_hash_deterministic(self):
        """The same array must always produce the same hash."""
        arr = np.arange(1000, dtype=np.float32).reshape(10, 100)

        hash1 = compute_array_hash(arr)
        hash2 = compute_array_hash(arr)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_array_hash_different_data(self):
        """Different arrays must produce different hashes."""
        arr_a = np.zeros((10, 10), dtype=np.float32)
        arr_b = np.ones((10, 10), dtype=np.float32)

        hash_a = compute_array_hash(arr_a)
        hash_b = compute_array_hash(arr_b)

        assert hash_a != hash_b

    def test_array_hash_different_shape(self):
        """Same data in different shapes must produce different hashes
        because shape is included in the hash input.
        """
        data = np.arange(12, dtype=np.float32)
        arr_1d = data.reshape(12)
        arr_2d = data.reshape(3, 4)

        hash_1d = compute_array_hash(arr_1d)
        hash_2d = compute_array_hash(arr_2d)

        assert hash_1d != hash_2d

    def test_array_hash_non_contiguous(self):
        """Non-contiguous arrays (e.g., slices) must be handled
        correctly and produce the same hash as a contiguous copy.
        """
        base = np.arange(100, dtype=np.float32).reshape(10, 10)
        sliced = base[::2, ::2]  # non-contiguous
        copied = np.ascontiguousarray(sliced)

        hash_sliced = compute_array_hash(sliced)
        hash_copied = compute_array_hash(copied)

        assert hash_sliced == hash_copied


# ====================================================================
# Result hashing
# ====================================================================


class TestResultHash:
    """Tests for compute_result_hash on detection lists."""

    def test_result_hash_order_independent(self):
        """The hash must be the same regardless of the order of
        detections in the list, because the function sorts before
        serializing.
        """
        dets = [
            {"longitude": -5.5, "latitude": 36.0, "confidence": 0.8},
            {"longitude": -5.3, "latitude": 36.2, "confidence": 0.6},
            {"longitude": -5.1, "latitude": 35.8, "confidence": 0.9},
        ]

        hash_original = compute_result_hash(dets)
        hash_reversed = compute_result_hash(list(reversed(dets)))

        assert hash_original == hash_reversed
        assert len(hash_original) == 64

    def test_result_hash_different_data(self):
        """Different detection sets must produce different hashes."""
        dets_a = [
            {"longitude": -5.5, "latitude": 36.0, "confidence": 0.8},
        ]
        dets_b = [
            {"longitude": -5.5, "latitude": 36.0, "confidence": 0.9},
        ]

        hash_a = compute_result_hash(dets_a)
        hash_b = compute_result_hash(dets_b)

        assert hash_a != hash_b

    def test_result_hash_deterministic(self):
        """Calling compute_result_hash twice on the same data must
        produce the identical hash.
        """
        dets = [
            {"longitude": 1.0, "latitude": 2.0, "confidence": 0.5},
            {"longitude": 3.0, "latitude": 4.0, "confidence": 0.7},
        ]

        h1 = compute_result_hash(dets)
        h2 = compute_result_hash(dets)

        assert h1 == h2

    def test_result_hash_empty_list(self):
        """An empty detection list must still produce a valid hash."""
        h = compute_result_hash([])
        assert len(h) == 64

    def test_result_hash_with_extra_keys(self):
        """Extra keys in detection dicts (beyond lon/lat/confidence)
        must be included in the hash via sort_keys JSON serialization.
        """
        dets_a = [
            {"longitude": 1.0, "latitude": 2.0, "confidence": 0.5, "source": "cfar"},
        ]
        dets_b = [
            {"longitude": 1.0, "latitude": 2.0, "confidence": 0.5, "source": "yolo"},
        ]

        assert compute_result_hash(dets_a) != compute_result_hash(dets_b)

    def test_result_hash_ignores_per_run_uuid(self):
        """Detection dicts with identical content but different per-run
        UUIDs must produce the same hash.

        This is the production bug audited on 2026-05-08: each Detection
        carries a fresh ``id: uuid.UUID`` (default_factory=uuid.uuid4),
        so ``model_dump()`` injected non-deterministic data into the
        result. The hash now strips ``id`` and other per-execution
        fields (see NON_CONTENT_FIELDS) before serializing.
        """
        common = {
            "longitude": 1.234567,
            "latitude": 2.345678,
            "confidence": 0.91,
            "source": "fused",
            "bbox_pixel": [10.0, 20.0, 30.0, 40.0],
            "tile_index": 5,
        }
        # Two "different" detections that should be the same content-wise.
        det_a = {**common, "id": "11111111-1111-1111-1111-111111111111"}
        det_b = {**common, "id": "22222222-2222-2222-2222-222222222222"}

        assert compute_result_hash([det_a]) == compute_result_hash([det_b])

    def test_result_hash_ignores_thumbnail_path(self):
        """``thumbnail_path`` is derived from execution_id and therefore
        non-deterministic across re-runs of the same input. It must not
        affect the result hash."""
        common = {
            "longitude": 1.0,
            "latitude": 2.0,
            "confidence": 0.5,
            "source": "cfar",
        }
        a = {**common, "thumbnail_path": "/data/thumbnails/exec-A/det-1.png"}
        b = {**common, "thumbnail_path": "/data/thumbnails/exec-B/det-1.png"}
        assert compute_result_hash([a]) == compute_result_hash([b])

    def test_result_hash_uses_real_detection_model_dump(self):
        """End-to-end: two Detection instances with identical content but
        different auto-generated UUIDs and execution-scoped thumbnail
        paths must hash identically when serialized via ``model_dump()``.

        Pins the contract that the engine relies on (engine.py:527).
        """
        from src.pipeline.detection import Detection

        det_a = Detection(
            bbox_pixel=[10.0, 20.0, 30.0, 40.0],
            bbox_geo=[1.0, 2.0, 1.5, 2.5],
            center_geo=[1.25, 2.25],
            confidence=0.9,
            source="fused",
            cfar_snr=12.0,
            yolo_score=0.85,
            tile_index=3,
            thumbnail_path="/data/thumbnails/exec-A/det.png",
        )
        det_b = Detection(
            bbox_pixel=[10.0, 20.0, 30.0, 40.0],
            bbox_geo=[1.0, 2.0, 1.5, 2.5],
            center_geo=[1.25, 2.25],
            confidence=0.9,
            source="fused",
            cfar_snr=12.0,
            yolo_score=0.85,
            tile_index=3,
            thumbnail_path="/data/thumbnails/exec-B/det.png",
        )
        # Sanity check: the underlying objects DO differ (UUID + path)
        assert det_a.id != det_b.id
        assert det_a.thumbnail_path != det_b.thumbnail_path

        h_a = compute_result_hash([det_a.model_dump()])
        h_b = compute_result_hash([det_b.model_dump()])
        assert h_a == h_b


# ====================================================================
# Input params hashing (I-TRACE-4)
# ====================================================================


class TestInputParamsHash:
    """Tests for compute_input_params_hash."""

    def test_deterministic(self):
        params = {"a": 1, "b": "x", "c": [1, 2, 3]}
        assert compute_input_params_hash(params) == compute_input_params_hash(params)

    def test_key_order_irrelevant(self):
        a = {"alpha": 1, "beta": 2}
        b = {"beta": 2, "alpha": 1}
        assert compute_input_params_hash(a) == compute_input_params_hash(b)

    def test_different_values_diff_hash(self):
        a = {"threshold": 0.25}
        b = {"threshold": 0.30}
        assert compute_input_params_hash(a) != compute_input_params_hash(b)

    def test_hex_length(self):
        assert len(compute_input_params_hash({"x": 1})) == 64


# ====================================================================
# Commit SHA capture (I-TRACE-4)
# ====================================================================


class TestCommitSha:
    """Tests for get_commit_sha."""

    def test_returns_string(self):
        sha = get_commit_sha()
        assert isinstance(sha, str)
        assert sha  # non-empty

    def test_aidra_commit_sha_override(self, monkeypatch):
        get_commit_sha.cache_clear()
        monkeypatch.delenv("SOURCE_COMMIT", raising=False)
        monkeypatch.setenv("AIDRA_COMMIT_SHA", "deadbeef" * 5)
        try:
            assert get_commit_sha() == "deadbeef" * 5
        finally:
            get_commit_sha.cache_clear()

    def test_source_commit_takes_precedence_over_aidra_commit_sha(self, monkeypatch):
        """SOURCE_COMMIT (Coolify auto-set per deploy) wins over the legacy
        AIDRA_COMMIT_SHA, which can be hardcoded and become stale."""
        get_commit_sha.cache_clear()
        monkeypatch.setenv("SOURCE_COMMIT", "c0ffee0" + "0" * 33)
        monkeypatch.setenv("AIDRA_COMMIT_SHA", "92b2515")  # stale build-arg
        try:
            assert get_commit_sha() == "c0ffee0" + "0" * 33
        finally:
            get_commit_sha.cache_clear()

    def test_empty_env_falls_through(self, monkeypatch):
        """An env var set to empty/whitespace must not shadow git fallback."""
        get_commit_sha.cache_clear()
        monkeypatch.setenv("SOURCE_COMMIT", "   ")
        monkeypatch.setenv("AIDRA_COMMIT_SHA", "")
        try:
            sha = get_commit_sha()
            # Local dev env will resolve via git; CI without git returns "unknown".
            assert sha == "unknown" or len(sha) >= 7
        finally:
            get_commit_sha.cache_clear()

    def test_env_value_is_stripped(self, monkeypatch):
        get_commit_sha.cache_clear()
        monkeypatch.delenv("SOURCE_COMMIT", raising=False)
        monkeypatch.setenv("AIDRA_COMMIT_SHA", "  abcdef1234  \n")
        try:
            assert get_commit_sha() == "abcdef1234"
        finally:
            get_commit_sha.cache_clear()
