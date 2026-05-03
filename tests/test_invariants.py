"""
Tests de invariantes AIDRA (I-SAR-*, I-DET-*, I-TRACE-*, I-AIA-*).

Estos tests son ``@pytest.mark.invariant`` y se ejecutan via:

    pytest -k invariant -x

Cada uno fija una propiedad declarada en CLAUDE.md y debe ser
``barato`` — no descarga datos ni requiere BD: usan fixtures
sinteticas (numpy arrays, SAFE de juguete, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# =====================================================================
# I-SAR-1 — preprocess_full flag quality=invalid si falta paso critico
# =====================================================================


@pytest.mark.invariant
class TestISAR1QualityGate:
    """I-SAR-1: escena que llega a detection ha pasado todos los pasos."""

    def test_invalid_when_missing_calibration(self):
        from src.pipeline.preprocessing import _evaluate_scene_quality

        quality, reasons = _evaluate_scene_quality(
            cal_row=None,
            gcps=[("dummy",)],
            geo_transform=(0, 1, 0, 0, 0, 1),
            valid_footprint={"type": "Polygon", "coordinates": []},
            num_tiles=4,
        )
        assert quality == "invalid"
        assert "missing_calibration_lut" in reasons

    def test_invalid_when_missing_gcps(self):
        from src.pipeline.preprocessing import _evaluate_scene_quality

        quality, reasons = _evaluate_scene_quality(
            cal_row=np.ones(10),
            gcps=None,
            geo_transform=(0, 1, 0, 0, 0, 1),
            valid_footprint={"type": "Polygon", "coordinates": []},
            num_tiles=4,
        )
        assert quality == "invalid"
        assert "missing_gcps" in reasons

    def test_invalid_when_no_tiles(self):
        from src.pipeline.preprocessing import _evaluate_scene_quality

        quality, reasons = _evaluate_scene_quality(
            cal_row=np.ones(10),
            gcps=[("dummy",)],
            geo_transform=(0, 1, 0, 0, 0, 1),
            valid_footprint={"type": "Polygon", "coordinates": []},
            num_tiles=0,
        )
        assert quality == "invalid"
        assert "no_tiles_generated" in reasons

    def test_valid_with_full_chain(self):
        from src.pipeline.preprocessing import _evaluate_scene_quality

        quality, reasons = _evaluate_scene_quality(
            cal_row=np.ones(10),
            gcps=[("dummy",)],
            geo_transform=(0, 1, 0, 0, 0, 1),
            valid_footprint={"type": "Polygon", "coordinates": [[]]},
            num_tiles=4,
        )
        assert quality == "valid"
        assert reasons == []


# =====================================================================
# I-SAR-3 — global-land-mask deshabilitado / no importable en pipeline
# =====================================================================


@pytest.mark.invariant
class TestISAR3LandMaskIsInformationalOnly:
    """I-SAR-3 + I-DET-2: footprint clipping filtra swath; global-land-mask
    se permite SOLO como flag on_land informativo (nunca filtra)."""

    def test_engine_does_not_skip_on_land_detections(self):
        """global-land-mask no debe usarse para `continue`/`skip` en _save_detections.

        Busca patrones tipo:
            if not _globe.is_ocean(...): continue
            if has_land_mask and ... not _globe.is_ocean(...): continue
        que indiquen filtrado decisorio (prohibido por I-DET-2).
        """
        src = Path("src/pipeline/engine.py").read_text()
        # Localiza el cuerpo de _save_detections.
        marker = "_save_detections"
        assert marker in src, "_save_detections missing"
        body = src[src.index(marker):]

        # Heuristica: cualquier `not _globe.is_ocean(...)` seguido en un
        # rango cercano por `continue` o `skipped_land` denota filtrado
        # decisorio. La unica via permitida es popular `on_land` y seguir.
        for idx in _find_all(body, "is_ocean"):
            window = body[idx : idx + 220]
            assert "skipped_land" not in window, (
                "I-DET-2 violado: global-land-mask usado para skipear "
                "(debe ser informativo: solo poblar on_land)"
            )
            assert "continue" not in window, (
                "I-DET-2 violado: global-land-mask usado para continue "
                "(debe ser informativo: solo poblar on_land)"
            )


def _find_all(haystack: str, needle: str) -> list[int]:
    out: list[int] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return out
        out.append(i)
        start = i + len(needle)


# =====================================================================
# I-SAR / I-DET — flags on_land y cluster_anomaly no se pierden
# =====================================================================


@pytest.mark.invariant
class TestIDET2And3Flags:
    """I-DET-2 / I-DET-3: flags conservados, no descartados."""

    def test_detection_model_has_flags(self):
        from src.pipeline.detection import Detection

        det = Detection(
            bbox_pixel=[0, 0, 10, 10],
            confidence=0.5,
            source="cfar",
        )
        assert hasattr(det, "on_land")
        assert hasattr(det, "cluster_anomaly")
        assert det.on_land is False
        assert det.cluster_anomaly is False

    def test_flag_cluster_anomaly_marks_dense_cluster(self):
        from src.pipeline.detection import Detection
        from src.pipeline.postprocessing import flag_cluster_anomaly

        # 12 detections within a tight 0.005-deg cluster → all flagged.
        dets = [
            Detection(
                bbox_pixel=[0, 0, 10, 10],
                confidence=0.5,
                source="cfar",
                center_geo=[0.0 + i * 0.0005, 0.0],
            )
            for i in range(12)
        ]
        flagged = flag_cluster_anomaly(dets, radius_deg=0.01, min_neighbours=8)
        assert flagged == 12
        assert all(d.cluster_anomaly for d in dets)

    def test_flag_cluster_anomaly_skips_sparse(self):
        from src.pipeline.detection import Detection
        from src.pipeline.postprocessing import flag_cluster_anomaly

        # 3 isolated detections far apart → none flagged.
        dets = [
            Detection(
                bbox_pixel=[0, 0, 10, 10],
                confidence=0.5,
                source="cfar",
                center_geo=[float(i), float(i)],
            )
            for i in range(3)
        ]
        flagged = flag_cluster_anomaly(dets, radius_deg=0.01, min_neighbours=8)
        assert flagged == 0
        assert not any(d.cluster_anomaly for d in dets)


# =====================================================================
# I-TRACE-4 — input_params_hash + commit_sha en cada run
# =====================================================================


@pytest.mark.invariant
class TestITRACE4InputHashAndCommit:
    """I-TRACE-4: trazabilidad reforzada con commit_sha + params hash."""

    def test_input_params_hash_deterministic_and_order_invariant(self):
        from src.traceability.hasher import compute_input_params_hash

        a = {"alpha": 1, "beta": [1, 2, 3], "gamma": "x"}
        b = {"gamma": "x", "beta": [1, 2, 3], "alpha": 1}
        assert compute_input_params_hash(a) == compute_input_params_hash(b)
        assert len(compute_input_params_hash(a)) == 64

    def test_get_commit_sha_returns_value(self, monkeypatch):
        from src.traceability.hasher import get_commit_sha

        get_commit_sha.cache_clear()
        monkeypatch.setenv("AIDRA_COMMIT_SHA", "f" * 40)
        try:
            assert get_commit_sha() == "f" * 40
        finally:
            get_commit_sha.cache_clear()

    def test_execution_record_has_commit_sha_and_input_hash_fields(self):
        from src.db.models import ExecutionRecord

        fields = ExecutionRecord.model_fields
        assert "commit_sha" in fields
        assert "input_params_hash" in fields


# =====================================================================
# I-TRACE-1 — SHA256 disponible para artefactos
# =====================================================================


@pytest.mark.invariant
class TestITRACE1Hashing:
    def test_compute_sha256_file(self, tmp_path):
        from src.traceability.hasher import compute_sha256

        p = tmp_path / "x.bin"
        p.write_bytes(b"abc" * 1024)
        h = compute_sha256(p)
        assert len(h) == 64
        assert compute_sha256(p) == h

    def test_compute_result_hash_order_invariant(self):
        from src.traceability.hasher import compute_result_hash

        a = [
            {"longitude": 1.0, "latitude": 2.0, "confidence": 0.9},
            {"longitude": 3.0, "latitude": 4.0, "confidence": 0.5},
        ]
        b = list(reversed(a))
        assert compute_result_hash(a) == compute_result_hash(b)


# =====================================================================
# I-AIA-1 — gate ai-act-card en _register_model
# =====================================================================


@pytest.mark.invariant
class TestIAIA1AICardGate:
    """I-AIA-1: ningun modelo se registra sin MODEL_CARD.md."""

    def test_require_model_card_raises_when_missing(self, tmp_path):
        from src.models.manager import ModelManager

        # Build a manager without invoking __init__ (no DB).
        mgr = ModelManager.__new__(ModelManager)
        # Point gate to an empty cards dir.
        mgr._MODEL_CARDS_DIR = tmp_path
        with pytest.raises(FileNotFoundError, match="MODEL_CARD"):
            mgr._require_model_card("missing-model", tmp_path / "missing.pt")

    def test_require_model_card_passes_when_present(self, tmp_path):
        from src.models.manager import ModelManager

        card = tmp_path / "okmodel.MODEL_CARD.md"
        card.write_text("# ok")
        mgr = ModelManager.__new__(ModelManager)
        mgr._MODEL_CARDS_DIR = tmp_path
        mgr._require_model_card("okmodel", tmp_path / "okmodel.pt")

    def test_no_silent_fallback_method_left(self):
        from src.models.manager import ModelManager

        assert not hasattr(ModelManager, "_find_fallback_model"), (
            "I-AIA-1 violado: _find_fallback_model fue reintroducido"
        )


# =====================================================================
# Synthetic SAFE fixture for SAR metadata parser smoke test
# =====================================================================


@pytest.fixture
def synthetic_safe(tmp_path: Path) -> Path:
    """Create a minimal Sentinel-1 SAFE-like directory."""
    safe = tmp_path / "S1A_IW_GRDH_1SDV_TEST.SAFE"
    safe.mkdir()
    (safe / "annotation").mkdir()

    # manifest.safe — just enough for our parser
    manifest_xml = """<?xml version='1.0' encoding='UTF-8'?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"
           xmlns:s1="http://www.esa.int/safe/sentinel-1.0/sentinel-1">
  <metadataSection>
    <metadataObject>
      <metadataWrap>
        <xmlData>
          <s1:standAloneProductInformation>
            <s1:productType>GRD</s1:productType>
            <s1:transmitterReceiverPolarisation>VV</s1:transmitterReceiverPolarisation>
            <s1:transmitterReceiverPolarisation>VH</s1:transmitterReceiverPolarisation>
          </s1:standAloneProductInformation>
          <s1:orbitReference>
            <s1:relativeOrbitNumber>74</s1:relativeOrbitNumber>
            <s1:pass>DESCENDING</s1:pass>
          </s1:orbitReference>
        </xmlData>
      </metadataWrap>
    </metadataObject>
  </metadataSection>
</xfdu:XFDU>
"""
    (safe / "manifest.safe").write_text(manifest_xml.strip())

    ann_xml = """<?xml version='1.0' encoding='UTF-8'?>
<product>
  <geolocationGrid>
    <geolocationGridPointList>
      <geolocationGridPoint><incidenceAngle>30.0</incidenceAngle></geolocationGridPoint>
      <geolocationGridPoint><incidenceAngle>40.0</incidenceAngle></geolocationGridPoint>
    </geolocationGridPointList>
  </geolocationGrid>
</product>
"""
    (safe / "annotation" / "vv.xml").write_text(ann_xml.strip())
    return safe


@pytest.mark.invariant
class TestSARMetadataParser:
    """Q3 GEOINT: metadata SAR persistible desde manifest SAFE."""

    def test_parse_sar_metadata_from_synthetic_safe(self, synthetic_safe: Path):
        from src.pipeline.preprocessing import parse_sar_metadata

        meta = parse_sar_metadata(synthetic_safe)
        assert meta.get("product_type") == "GRD"
        assert meta.get("orbit_direction") == "DESCENDING"
        assert meta.get("relative_orbit") == 74
        # Polarisation aggregates VV + VH alphabetically.
        assert meta.get("polarisation") in {"VH+VV", "VV+VH"}
        assert meta.get("incidence_angle") == pytest.approx(35.0)
