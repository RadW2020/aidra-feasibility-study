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
# I-DET-4 — thresholds proceden de Settings, nunca de literales
# =====================================================================


@pytest.mark.invariant
class TestIDET4ThresholdsFromSettings:
    """I-DET-4: ``confidence_threshold`` / ``iou_threshold`` salen de
    ``Settings`` o de una request explicita. Ningun literal 0.25/0.45
    duplicado fuera de ``src/config.py`` decide una inferencia."""

    def test_pipeline_request_leaves_thresholds_unset(self):
        from src.pipeline.engine import PipelineRequest

        req = PipelineRequest()
        assert req.confidence_threshold is None
        assert req.iou_threshold is None

    def test_engine_resolves_thresholds_from_settings(self):
        from src.config import Settings
        from src.pipeline.engine import PipelineEngine, PipelineRequest

        settings = Settings(
            _env_file=None, confidence_threshold=0.9, iou_threshold=0.1
        )
        engine = object.__new__(PipelineEngine)
        engine.config = settings

        resolved = engine._apply_settings_defaults(PipelineRequest(zone="gibraltar"))
        assert resolved.confidence_threshold == 0.9
        assert resolved.iou_threshold == 0.1

        explicit = engine._apply_settings_defaults(
            PipelineRequest(zone="gibraltar", confidence_threshold=0.5, iou_threshold=0.2)
        )
        assert explicit.confidence_threshold == 0.5
        assert explicit.iou_threshold == 0.2

    def test_validate_request_rejects_unresolved_thresholds(self):
        from src.pipeline.engine import PipelineEngine, PipelineError, PipelineRequest

        engine = object.__new__(PipelineEngine)
        with pytest.raises(PipelineError, match="I-DET-4"):
            engine._validate_request(PipelineRequest(zone="gibraltar"))

    def test_background_callers_do_not_pin_thresholds(self):
        """Cron y Tip & Cue construyen PipelineRequest sin thresholds:
        heredan Settings via _apply_settings_defaults."""
        for path in ("src/pipeline/scheduler_jobs.py", "src/tipcue/scheduler.py"):
            src = Path(path).read_text()
            assert "confidence_threshold=" not in src, path
            assert "iou_threshold=" not in src, path

    def test_no_duplicated_threshold_literals_outside_settings(self):
        import re

        pattern = re.compile(r"(confidence_threshold|iou_threshold)\W{0,25}0\.(25|45)\b")
        offenders: list[str] = []
        for path in Path("src").rglob("*.py"):
            if path.as_posix() == "src/config.py":
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
        assert not offenders, "I-DET-4 violado (literal duplicado):\n" + "\n".join(offenders)

    def test_fusion_weights_come_from_settings(self):
        from src.config import Settings
        from src.pipeline.detection import DetectionEngine

        engine = DetectionEngine(
            fusion_iou_threshold=0.3,
            cfar_min_cluster_size=5,
            cfar_cluster_eps=1.5,
            cfar_min_mean_snr=2.0,
            fusion_yolo_weight=Settings(_env_file=None, fusion_yolo_weight=0.8).fusion_yolo_weight,
        )
        assert engine.fusion_yolo_weight == 0.8
        src = Path("src/pipeline/detection.py").read_text()
        assert "0.5 * y_det" not in src, "pesos de fusion hardcodeados"

    def test_recorder_refuses_implicit_thresholds(self):
        import asyncio

        from src.traceability.recorder import ExecutionRecorder

        recorder = ExecutionRecorder(db=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="I-DET-4|explicit"):
            asyncio.run(
                recorder.create_pending(
                    image_id="x",
                    image_hash="h",
                    model_name="m",
                    model_version="v",
                    model_hash="mh",
                    model_size_mb=1.0,
                )
            )


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


# =====================================================================
# I-MOD-3: variantes degradadas se marcan rejected, nunca se borran
# =====================================================================


@pytest.mark.invariant
class TestIMOD3RejectedVariants:
    """I-MOD-3: el esquema debe poder marcar variantes rechazadas.

    Hasta la migracion 015 models_registry no tenia columna de estado,
    haciendo el invariante incumplible: la variante int8-dynamic que
    produjo x4.5 detecciones sobre el baseline quedo registrada sin
    marca alguna.
    """

    MIGRATION = Path("src/db/migrations/015_model_status.sql")

    def test_status_migration_exists_with_constraints(self):
        sql = self.MIGRATION.read_text()
        assert "ADD COLUMN IF NOT EXISTS status" in sql
        assert "rejection_reason" in sql
        # Un rejected sin justificacion violaria I-MOD-3.
        assert "status <> 'rejected' OR rejection_reason IS NOT NULL" in sql

    def test_upsert_does_not_clobber_status(self):
        """Un rescan de modelos (UPSERT) no debe resucitar un rejected."""
        from src.db.queries import UPSERT_MODEL

        _, update_clause = UPSERT_MODEL.split("DO UPDATE SET", 1)
        assert "status" not in update_clause, (
            "I-MOD-3 violado: el UPSERT sobreescribe status y un rescan "
            "borraria la marca rejected"
        )

    def test_rejected_variant_keeps_evidence(self):
        """Marcar, no borrar: la migracion 015 usa UPDATE, nunca DELETE."""
        sql = self.MIGRATION.read_text().upper()
        assert "DELETE" not in sql
        assert "DROP TABLE" not in sql


# =====================================================================
# Gate de paneles: agregados de execution_log controlan regimen de
# procesado (auditoria 2026-08-16)
# =====================================================================


@pytest.mark.invariant
class TestDashboardRegimeControl:
    """Los agregados sobre execution_log deben controlar el regimen.

    Un run 'cue' procesa un recorte de AOI (9-56 tiles); un 'scheduled'
    o 'manual' procesa la escena completa (~1350 tiles). Mezclarlos en un
    AVG/PERCENTILE invierte conclusiones: el benchmark de compresion
    llego a mostrar el int8 2x MAS LENTO que el baseline cuando en igual
    regimen es un 23% mas rapido. Este gate parsea los dashboards y exige
    que toda agregacion sobre metricas por-run de execution_log filtre o
    desglose por trigger_type / num_tiles.
    """

    DASHBOARDS = Path("grafana/dashboards")
    # Metricas por-run cuya media/percentil depende del tamano del AOI.
    REGIME_SENSITIVE = (
        "inference_ms",
        "num_detections",
        "peak_ram_mb",
        "cpu_usage_pct",
        "total_duration_ms",
        "image_size_mb",
    )
    AGGREGATES = ("AVG(", "PERCENTILE_CONT", "STDDEV")

    @staticmethod
    def _iter_sql(dashboard: dict):
        def walk(panel):
            for target in panel.get("targets", []) or []:
                sql = target.get("rawSql")
                if sql:
                    yield panel.get("title", "?"), sql
            for sub in panel.get("panels", []) or []:
                yield from walk(sub)

        for panel in dashboard.get("panels", []):
            yield from walk(panel)

    def test_aggregates_over_execution_log_control_regime(self):
        import json as _json

        offenders = []
        for path in sorted(self.DASHBOARDS.glob("*.json")):
            dash = _json.loads(path.read_text())
            for title, sql in self._iter_sql(dash):
                if "execution_log" not in sql:
                    continue
                upper = sql.upper()
                aggregated = any(
                    f"{agg.upper()}" in upper.replace(" ", "")
                    or agg.upper() in upper
                    for agg in self.AGGREGATES
                ) and any(col in sql for col in self.REGIME_SENSITIVE)
                if not aggregated:
                    continue
                controls_regime = "trigger_type" in sql or "num_tiles" in sql
                if not controls_regime:
                    offenders.append(f"{path.name} :: {title}")
        assert not offenders, (
            "Paneles que agregan metricas por-run sin controlar el regimen "
            "crop/full (anadir trigger_type <> 'cue' o desglose): "
            + "; ".join(offenders)
        )


# =====================================================================
# I-DET-1 / GEOINT: la ruta multi-perfil persiste los metadatos SAR
# =====================================================================


@pytest.mark.invariant
class TestRunAllProfilesSarMetadata:
    """run_all_profiles debe persistir sar_meta y fijar _current_metadata.

    Hasta 2026-08-16 solo engine.run() lo hacia: cada fila de
    trigger-all-profiles quedaba sin incidence_angle/pixel_spacing (los
    NULL de la auditoria) y _run_detection clippaba el footprint contra
    el metadata rancio de la ULTIMA escena procesada por run().
    """

    @staticmethod
    def _source_of_run_all_profiles() -> str:
        import ast
        import inspect

        from src.pipeline.engine import PipelineEngine

        tree = ast.parse(inspect.getsource(PipelineEngine))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "run_all_profiles"
            ):
                return ast.unparse(node)
        raise AssertionError("run_all_profiles no encontrado")

    def test_persists_sar_metadata_per_profile(self):
        src = self._source_of_run_all_profiles()
        assert "update_sar_metadata" in src, (
            "run_all_profiles dejo de persistir metadatos SAR: las filas "
            "multi-perfil vuelven a quedar sin incidence_angle/pixel_spacing"
        )

    def test_sets_current_metadata_for_footprint_clipping(self):
        src = self._source_of_run_all_profiles()
        assert "self._current_metadata" in src, (
            "run_all_profiles no fija _current_metadata: el footprint "
            "clipping usaria la escena del run() anterior"
        )


@pytest.mark.invariant
class TestPreAffineMaskMarking:
    """Migracion 016: la era pre-mascara-afin se marca, nunca se borra (§8)."""

    MIGRATION = Path("src/db/migrations/016_mark_pre_affine_mask.sql")

    def test_migration_marks_and_never_deletes(self):
        sql = self.MIGRATION.read_text()
        assert "methodology:pre-affine-mask" in sql
        upper = sql.upper()
        assert "DELETE" not in upper and "DROP" not in upper and "TRUNCATE" not in upper
        # Idempotente: no re-marca filas ya marcadas.
        assert "NOT LIKE '%methodology:pre-affine-mask%'" in sql

    def test_panels_04_06_exclude_marked_era(self):
        import json as _json

        for stem in ("04-constraint-profiles", "06-obdp-value"):
            dash = _json.loads(Path(f"grafana/dashboards/{stem}.json").read_text())
            found = _json.dumps(dash).count("pre-affine-mask")
            assert found >= 2, (
                f"{stem}: los agregados dejaron de excluir la era "
                "pre-affine-mask (recuentos no comparables)"
            )
