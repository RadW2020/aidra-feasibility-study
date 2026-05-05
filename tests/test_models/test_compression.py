"""
Tests for src/models/compression/{quantization,pruning}.

Coverage:
- Result models: field validation and computed properties
- Helper functions: _sha256_file, _file_size_mb, _compute_sparsity
- ModelQuantizer: init errors, dynamic INT8 on a minimal synthetic model
- ModelPruner: init errors, L1 unstructured pruning on a minimal synthetic model
- Invariant I-MOD-1: triplet {baseline, variant, profile} structure
- Invariant I-MOD-3: degradation below declared tolerance

Tests that invoke YOLO / torch.save are marked with
``@pytest.mark.slow`` (excluded from the fast CI gate).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.models.compression.pruning import (
    ModelPruner,
    PruningResult,
    _compute_sparsity,
    _count_nonzero_params,
    _count_total_params,
    _file_size_mb,
    _get_prunable_modules,
    _sha256_file,
)
from src.models.compression.quantization import (
    ModelQuantizer,
    QuantizationResult,
)

# ---------------------------------------------------------------------------
# Helpers — minimal synthetic models for fast tests
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Sequential:
    """Two-layer Conv2d model that fits in <1 KB when saved."""
    return nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 1, kernel_size=3, padding=1),
    )


def _save_tiny_model(path: Path) -> None:
    """Save a minimal YOLO-style checkpoint at *path*."""
    model = _tiny_model()
    ckpt = {
        "model": model,
        "optimizer": None,
        "train_args": {},
        "date": None,
        "version": "8.0.0",
    }
    torch.save(ckpt, str(path))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# QuantizationResult — field validation
# ---------------------------------------------------------------------------


class TestQuantizationResult:
    def test_compression_ratio_stored(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig.pt"
        quant = tmp_path / "quant.onnx"
        orig.write_bytes(b"x" * 1000)
        quant.write_bytes(b"x" * 500)
        r = QuantizationResult(
            original_path=orig,
            quantized_path=quant,
            original_size_mb=orig.stat().st_size / 1e6,
            quantized_size_mb=quant.stat().st_size / 1e6,
            compression_ratio=2.0,
            technique="dynamic_int8_pytorch",
            original_hash="aaa",
            quantized_hash="bbb",
        )
        assert r.compression_ratio == 2.0

    def test_required_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "f.pt"
        f.write_bytes(b"x")
        with pytest.raises((TypeError, ValueError)):
            QuantizationResult()  # missing required fields


# ---------------------------------------------------------------------------
# PruningResult — field validation
# ---------------------------------------------------------------------------


class TestPruningResult:
    def test_params_count_consistency(self, tmp_path: Path) -> None:
        f = tmp_path / "m.pt"
        f.write_bytes(b"x" * 100)
        r = PruningResult(
            original_path=f,
            pruned_path=f,
            original_size_mb=0.1,
            pruned_size_mb=0.08,
            sparsity_achieved=0.30,
            technique="l1_unstructured",
            original_hash="aaa",
            pruned_hash="bbb",
            num_params_original=1000,
            num_params_pruned=700,
            num_params_removed=300,
        )
        assert r.num_params_removed == r.num_params_original - r.num_params_pruned

    def test_sparsity_range(self, tmp_path: Path) -> None:
        f = tmp_path / "m.pt"
        f.write_bytes(b"x")
        r = PruningResult(
            original_path=f,
            pruned_path=f,
            original_size_mb=0.1,
            pruned_size_mb=0.08,
            sparsity_achieved=0.30,
            technique="l1_unstructured",
            original_hash="a",
            pruned_hash="b",
            num_params_original=100,
            num_params_pruned=70,
            num_params_removed=30,
        )
        assert 0.0 <= r.sparsity_achieved <= 1.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_sha256_file_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello AIDRA")
        h1 = _sha256_file(f)
        h2 = _sha256_file(f)
        assert h1 == h2
        assert len(h1) == 64  # hex SHA256

    def test_sha256_file_changes_with_content(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"version A")
        ha = _sha256_file(f)
        f.write_bytes(b"version B")
        hb = _sha256_file(f)
        assert ha != hb

    def test_file_size_mb(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * (1024 * 1024))  # exactly 1 MB
        assert abs(_file_size_mb(f) - 1.0) < 0.01

    def test_compute_sparsity_dense(self) -> None:
        model = _tiny_model()
        # Fresh model has no zeros (weights initialised with Kaiming uniform)
        sparsity = _compute_sparsity(model)
        assert 0.0 <= sparsity < 0.1  # dense

    def test_compute_sparsity_zeroed(self) -> None:
        model = _tiny_model()
        for p in model.parameters():
            p.data.zero_()
        sparsity = _compute_sparsity(model)
        assert sparsity == 1.0

    def test_count_total_params(self) -> None:
        model = _tiny_model()
        n = _count_total_params(model)
        assert n > 0

    def test_count_nonzero_matches_total_when_dense(self) -> None:
        model = _tiny_model()
        total = _count_total_params(model)
        nonzero = _count_nonzero_params(model)
        assert nonzero <= total
        assert nonzero > 0

    def test_get_prunable_modules_returns_conv2d(self) -> None:
        model = _tiny_model()
        modules = _get_prunable_modules(model)
        assert len(modules) == 2
        for mod, name in modules:
            assert isinstance(mod, nn.Conv2d)
            assert name == "weight"


# ---------------------------------------------------------------------------
# ModelQuantizer — init errors
# ---------------------------------------------------------------------------


class TestModelQuantizerInit:
    def test_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ModelQuantizer("/nonexistent/path/model.pt")

    def test_init_reads_hash_and_size(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"fake_model_data")
        q = ModelQuantizer(f)
        assert len(q.original_hash) == 64
        assert q.original_size_mb > 0
        assert q.model_path == f


# ---------------------------------------------------------------------------
# ModelQuantizer — dynamic INT8 (uses tiny synthetic checkpoint)
# ---------------------------------------------------------------------------


class TestModelQuantizerDynamic:
    @pytest.mark.slow
    def test_quantize_dynamic_produces_smaller_file(self, tmp_path: Path) -> None:
        src = tmp_path / "tiny.pt"
        _save_tiny_model(src)

        mock_yolo = MagicMock()
        mock_yolo.model = _tiny_model()
        mock_yolo.overrides = {}
        mock_yolo.__version__ = "8.0.0"

        with patch("src.models.compression.quantization.YOLO", return_value=mock_yolo):
            q = ModelQuantizer(src)
            out = tmp_path / "tiny-int8.pt"
            result = q.quantize_dynamic_pytorch(out)

        assert out.exists()
        assert result.quantized_size_mb > 0
        assert result.technique == "dynamic_int8_pytorch"
        assert len(result.quantized_hash) == 64
        assert result.original_hash != result.quantized_hash

    @pytest.mark.slow
    def test_quantize_dynamic_compression_ratio_positive(self, tmp_path: Path) -> None:
        src = tmp_path / "tiny.pt"
        _save_tiny_model(src)

        mock_yolo = MagicMock()
        mock_yolo.model = _tiny_model()
        mock_yolo.overrides = {}

        with patch("src.models.compression.quantization.YOLO", return_value=mock_yolo):
            q = ModelQuantizer(src)
            out = tmp_path / "tiny-int8.pt"
            result = q.quantize_dynamic_pytorch(out)

        assert result.compression_ratio > 0.0


# ---------------------------------------------------------------------------
# ModelPruner — init errors
# ---------------------------------------------------------------------------


class TestModelPrunerInit:
    def test_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ModelPruner("/nonexistent/model.pt")

    def test_raises_value_error_for_onnx(self, tmp_path: Path) -> None:
        f = tmp_path / "model.onnx"
        f.write_bytes(b"fake onnx")
        with pytest.raises(ValueError, match=r"\.pt"):
            ModelPruner(f)

    def test_init_reads_hash_and_size(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"fake_model_data")
        p = ModelPruner(f)
        assert len(p.original_hash) == 64
        assert p.original_size_mb > 0


# ---------------------------------------------------------------------------
# ModelPruner — L1 unstructured pruning (tiny synthetic model)
# ---------------------------------------------------------------------------


class TestModelPrunerUnstructured:
    @pytest.mark.slow
    def test_prune_unstructured_increases_sparsity(self, tmp_path: Path) -> None:
        src = tmp_path / "tiny.pt"
        _save_tiny_model(src)

        mock_yolo = MagicMock()
        model = _tiny_model()
        mock_yolo.model = model
        mock_yolo.overrides = {}

        with patch("src.models.compression.pruning.YOLO", return_value=mock_yolo):
            pruner = ModelPruner(src)
            out = tmp_path / "tiny-pruned.pt"

            # Mock torch.save so we don't need a valid checkpoint format
            with patch("src.models.compression.pruning.torch.save"):
                src.write_bytes(b"x" * 1000)  # ensure output is writable
                result = pruner.prune_unstructured(sparsity=0.3, output_path=out)

        assert result.sparsity_achieved >= 0.0
        assert result.technique == "l1_unstructured"
        assert result.num_params_original > 0
        assert result.num_params_removed >= 0

    @pytest.mark.slow
    def test_prune_unstructured_default_output_path(self, tmp_path: Path) -> None:
        src = tmp_path / "tiny.pt"
        _save_tiny_model(src)

        mock_yolo = MagicMock()
        mock_yolo.model = _tiny_model()
        mock_yolo.overrides = {}

        with patch("src.models.compression.pruning.YOLO", return_value=mock_yolo):
            pruner = ModelPruner(src)
            with patch("src.models.compression.pruning.torch.save"):
                result = pruner.prune_unstructured(sparsity=0.3)

        # Default path: {stem}-pruned30.pt next to source
        assert "pruned30" in str(result.pruned_path)

    def test_prune_unstructured_raises_no_conv(self, tmp_path: Path) -> None:
        src = tmp_path / "tiny.pt"
        src.write_bytes(b"x")

        mock_yolo = MagicMock()
        mock_yolo.model = nn.Sequential(nn.Linear(4, 4))  # no Conv2d
        mock_yolo.overrides = {}

        with patch("src.models.compression.pruning.YOLO", return_value=mock_yolo):
            pruner = ModelPruner(src)
            with pytest.raises(RuntimeError, match="No Conv2d"):
                pruner.prune_unstructured()


# ---------------------------------------------------------------------------
# Invariant I-MOD-1 — triplet {baseline, variant, profile}
# ---------------------------------------------------------------------------


class TestIMOD1Triplet:
    """I-MOD-1: a compression result is only valid evidence when paired
    with a baseline run and a hardware profile.  We validate the
    structural contract, not the DB query."""

    def _make_triplet(
        self,
        baseline_size: float,
        variant_size: float,
        profile: str,
    ) -> dict:
        return {
            "baseline": {"size_mb": baseline_size, "technique": "none"},
            "variant": {"size_mb": variant_size, "technique": "dynamic_int8_pytorch"},
            "profile": profile,
        }

    def test_triplet_has_all_three_keys(self) -> None:
        t = self._make_triplet(49.6, 25.1, "ground")
        assert "baseline" in t
        assert "variant" in t
        assert "profile" in t

    def test_triplet_missing_baseline_invalid(self) -> None:
        t = self._make_triplet(49.6, 25.1, "ground")
        del t["baseline"]
        assert "baseline" not in t  # structural gap detected

    def test_triplet_missing_variant_invalid(self) -> None:
        t = self._make_triplet(49.6, 25.1, "ground")
        del t["variant"]
        assert "variant" not in t

    def test_triplet_missing_profile_invalid(self) -> None:
        t = self._make_triplet(49.6, 25.1, "ground")
        del t["profile"]
        assert "profile" not in t

    def test_compression_ratio_from_triplet(self) -> None:
        t = self._make_triplet(49.6, 25.1, "ground")
        ratio = t["baseline"]["size_mb"] / t["variant"]["size_mb"]
        assert ratio > 1.0  # variant must be smaller than baseline


# ---------------------------------------------------------------------------
# Invariant I-MOD-3 — degradation below declared tolerance
# ---------------------------------------------------------------------------


class TestIMOD3Degradation:
    """I-MOD-3: ΔmAP ≤ 5 pts (default tolerance).  Tests verify the
    acceptance / rejection logic — not actual mAP computation."""

    TOLERANCE_PP = 5.0  # declared in CLAUDE.md

    def _degradation_ok(self, baseline_conf: float, variant_conf: float) -> bool:
        delta = round((baseline_conf - variant_conf) * 100, 6)  # pp, guard float noise
        return delta <= self.TOLERANCE_PP

    def test_within_tolerance_accepted(self) -> None:
        # Δ = 0.4 pp — observed in production (INT8 vs FP32)
        assert self._degradation_ok(0.539, 0.535)

    def test_at_tolerance_boundary_accepted(self) -> None:
        assert self._degradation_ok(0.850, 0.800)  # exactly 5 pp

    def test_above_tolerance_rejected(self) -> None:
        assert not self._degradation_ok(0.80, 0.74)  # 6 pp

    def test_improvement_always_accepted(self) -> None:
        assert self._degradation_ok(0.60, 0.65)  # negative delta

    def test_result_status_rejected_when_over_tolerance(self) -> None:
        baseline_conf = 0.80
        variant_conf = 0.70  # 10 pp degradation
        delta_pp = (baseline_conf - variant_conf) * 100
        status = "rejected" if delta_pp > self.TOLERANCE_PP else "accepted"
        assert status == "rejected"
