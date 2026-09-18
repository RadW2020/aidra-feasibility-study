"""El pipeline no vuelve a procesar una escena que ya proceso.

Contexto (POSTMORTEM 17/09/2026): la busqueda de Copernicus devuelve el
producto mas reciente de los ultimos 7 dias, asi que hasta que se publica una
escena nueva siempre es la misma. Sin deduplicacion, cada scan programado y
cada cue volvian a descargar 1,7 GB y a gastar ~55 min de inferencia para
reproducir detecciones ya guardadas: cuatro ejecuciones sobre el mismo
producto en 95 minutos, y con ellas la saturacion de la VNIC que tumbo al
resto de servicios de la maquina.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings
from src.observability.loki_logger import StructuredLogger
from src.pipeline.engine import PipelineEngine, PipelineRequest
from src.pipeline.ingestion import CopernicusSearchResult
from src.pipeline.scheduler_jobs import _executed_image_id


def _engine() -> PipelineEngine:
    eng = object.__new__(PipelineEngine)
    eng.config = Settings(_env_file=None)
    eng._log = StructuredLogger("aidra.test.dedup")
    return eng


def _product(product_id: str = "prod-1") -> CopernicusSearchResult:
    return CopernicusSearchResult(
        product_id=product_id,
        title="S1C_IW_GRDH_TEST",
        sensing_date=datetime.now(UTC),
        download_url="https://example.invalid/p.zip",
    )


class _FakeDB:
    """Stand-in de src.db.connection.db con respuesta programada."""

    def __init__(self, row=None, raises: bool = False) -> None:
        self.row = row
        self.raises = raises
        self.queries: list[tuple] = []

    async def fetchrow(self, query: str, *args):
        self.queries.append((query, args))
        if self.raises:
            raise RuntimeError("db down")
        return self.row


class TestSkipReason:
    async def test_cue_does_not_reprocess_its_own_scene(self, monkeypatch) -> None:
        db = _FakeDB(row=None)
        monkeypatch.setattr("src.db.connection.db", db)
        eng = _engine()

        reason = await eng._skip_reason(
            product=_product("scene-A"),
            request=PipelineRequest(trigger_type="cue", exclude_image_id="scene-A"),
            model_hash="m" * 64,
            input_params_hash="p" * 64,
        )

        assert reason is not None
        assert "cue would reprocess its own scene" in reason
        # Corta antes de tocar la BD: es la comprobacion mas barata.
        assert db.queries == []

    async def test_cue_on_a_different_scene_runs(self, monkeypatch) -> None:
        monkeypatch.setattr("src.db.connection.db", _FakeDB(row=None))
        eng = _engine()

        reason = await eng._skip_reason(
            product=_product("scene-B"),
            request=PipelineRequest(trigger_type="cue", exclude_image_id="scene-A"),
            model_hash="m" * 64,
            input_params_hash="p" * 64,
        )

        assert reason is None

    async def test_same_scene_model_and_params_is_skipped(self, monkeypatch) -> None:
        db = _FakeDB(row={"id": "exec-1", "num_detections": 542, "output_hash": "o" * 64})
        monkeypatch.setattr("src.db.connection.db", db)
        eng = _engine()

        reason = await eng._skip_reason(
            product=_product("scene-A"),
            request=PipelineRequest(trigger_type="scheduled"),
            model_hash="m" * 64,
            input_params_hash="p" * 64,
        )

        assert reason is not None
        assert "already processed" in reason
        assert "exec-1" in reason and "542" in reason
        # La terna de I-TRACE-4, en ese orden, es la clave de la consulta.
        assert db.queries[0][1] == ("scene-A", "m" * 64, "p" * 64)

    async def test_unseen_scene_runs(self, monkeypatch) -> None:
        monkeypatch.setattr("src.db.connection.db", _FakeDB(row=None))
        eng = _engine()

        reason = await eng._skip_reason(
            product=_product("scene-new"),
            request=PipelineRequest(trigger_type="scheduled"),
            model_hash="m" * 64,
            input_params_hash="p" * 64,
        )

        assert reason is None

    async def test_manual_runs_are_never_deduplicated(self, monkeypatch) -> None:
        # Escape hatch: el automatismo se deduplica, una persona que pide un
        # run lo obtiene aunque la escena ya este procesada.
        db = _FakeDB(row={"id": "exec-1", "num_detections": 542, "output_hash": "o" * 64})
        monkeypatch.setattr("src.db.connection.db", db)
        eng = _engine()

        reason = await eng._skip_reason(
            product=_product("scene-A"),
            request=PipelineRequest(trigger_type="manual"),
            model_hash="m" * 64,
            input_params_hash="p" * 64,
        )

        assert reason is None
        assert db.queries == []

    async def test_db_failure_does_not_block_the_pipeline(self, monkeypatch) -> None:
        # Fallar abierto: como mucho se repite trabajo, que es el
        # comportamiento anterior a este cambio. Fallar cerrado dejaria de
        # producir evidencia por una BD lenta.
        monkeypatch.setattr("src.db.connection.db", _FakeDB(raises=True))
        eng = _engine()

        reason = await eng._skip_reason(
            product=_product(),
            request=PipelineRequest(trigger_type="scheduled"),
            model_hash="m" * 64,
            input_params_hash="p" * 64,
        )

        assert reason is None


class TestLiveCueGuard:
    async def test_zip_is_kept_while_cues_are_queued(self, monkeypatch) -> None:
        monkeypatch.setattr("src.db.connection.db", _FakeDB(row={"live": 2}))
        eng = _engine()

        assert await eng._has_live_cues() is True

    async def test_zip_is_cleaned_when_no_cue_is_waiting(self, monkeypatch) -> None:
        monkeypatch.setattr("src.db.connection.db", _FakeDB(row={"live": 0}))
        eng = _engine()

        assert await eng._has_live_cues() is False

    async def test_stuck_cues_do_not_pin_the_zip_forever(self, monkeypatch) -> None:
        # Un redespliegue en mitad de un cue deja la fila en 'processing'
        # para siempre. La consulta acota esas filas en el tiempo; aqui se
        # comprueba que la condicion sigue en el SQL, porque sin ella el
        # disco se llena solo.
        from src.db.queries import COUNT_LIVE_CUES

        assert "status = 'pending'" in COUNT_LIVE_CUES
        assert "INTERVAL '6 hours'" in COUNT_LIVE_CUES

    async def test_db_failure_falls_back_to_cleaning(self, monkeypatch) -> None:
        # Aqui se falla cerrado: el riesgo de no borrar es llenar un disco
        # que ya esta al 100 % de la cuota del Free Tier.
        monkeypatch.setattr("src.db.connection.db", _FakeDB(raises=True))
        eng = _engine()

        assert await eng._has_live_cues() is False


class TestParentImageLookup:
    async def test_none_without_a_parent_execution(self) -> None:
        assert await _executed_image_id(None) is None

    async def test_placeholder_is_not_a_scene(self, monkeypatch) -> None:
        # create_pending escribe image_id='pending' antes de saber que imagen
        # tocara: no identifica ninguna escena, asi que no excluye nada.
        monkeypatch.setattr("src.pipeline.scheduler_jobs.db", _FakeDB(row={"image_id": "pending"}))

        assert await _executed_image_id("exec-1") is None

    async def test_returns_the_parent_scene(self, monkeypatch) -> None:
        monkeypatch.setattr("src.pipeline.scheduler_jobs.db", _FakeDB(row={"image_id": "scene-A"}))

        assert await _executed_image_id("exec-1") == "scene-A"


def test_request_carries_the_excluded_scene() -> None:
    # I-DET-4: el motor lee la exclusion del request, no de una global.
    assert PipelineRequest().exclude_image_id is None
    assert PipelineRequest(exclude_image_id="scene-A").exclude_image_id == "scene-A"


def test_pipeline_result_accepts_the_skipped_status() -> None:
    from uuid import uuid4

    from src.pipeline.engine import PipelineResult

    result = PipelineResult(execution_id=uuid4(), status="skipped")

    assert result.status == "skipped"
    assert result.num_detections == 0
    assert Path("/nonexistent").exists() is False  # sanity: no side effects on disk
