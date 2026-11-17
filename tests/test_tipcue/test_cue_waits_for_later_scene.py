"""Un cue solo se cierra cuando se re-observa con una escena posterior a T0.

Contexto (revision de plataforma 2026-11-17): desde la deduplicacion de
escenas, la busqueda devolvia al cue la misma imagen que lo genero, el run
terminaba en ``skipped`` y el cue se cerraba como ``completed``. En 30 dias,
29 cues "completados" y ninguna re-observacion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.config import Settings
from src.db.queries import (
    SELECT_EXECUTION_IMAGE_ID,
    SELECT_PENDING_CUES,
    UPDATE_CUE_AWAITING_SCENE,
    UPDATE_CUE_STATUS,
)
from src.pipeline import scheduler_jobs
from src.pipeline.engine import PipelineResult

T0 = datetime(2026, 11, 14, 6, 18, tzinfo=UTC)


class _FakeDB:
    def __init__(self, cue: dict, awaiting_status: str = "pending") -> None:
        self.cue = cue
        self.awaiting_status = awaiting_status
        self.calls: list[tuple] = []

    async def fetch(self, query: str, *args):
        assert query == SELECT_PENDING_CUES
        return [self.cue]

    async def fetchrow(self, query: str, *args):
        assert query == SELECT_EXECUTION_IMAGE_ID
        return {"image_id": "scene-A", "image_sensing_date": T0}

    async def fetchval(self, query: str, *args):
        self.calls.append((query, args))
        return self.awaiting_status

    async def execute(self, query: str, *args):
        self.calls.append((query, args))


def _cue() -> dict:
    return {
        "id": uuid4(),
        "target_zone": "gibraltar_strait",
        "triggered_by": uuid4(),
        "priority": 1,
        "target_bbox_geojson": None,
    }


def _engine(result: PipelineResult) -> MagicMock:
    engine = MagicMock()
    engine.config = Settings(_env_file=None)
    engine.run = AsyncMock(return_value=result)
    return engine


async def test_cue_run_is_restricted_to_scenes_after_its_parent(monkeypatch) -> None:
    db = _FakeDB(_cue())
    monkeypatch.setattr(scheduler_jobs, "db", db)
    engine = _engine(PipelineResult(execution_id=uuid4(), status="success", num_detections=3))

    await scheduler_jobs.process_pending_cues(engine)

    request = engine.run.await_args.args[0]
    assert request.trigger_type == "cue"
    assert request.exclude_image_id == "scene-A"
    assert request.sensed_after == T0


async def test_skipped_cue_goes_back_to_the_queue(monkeypatch) -> None:
    cue = _cue()
    db = _FakeDB(cue)
    monkeypatch.setattr(scheduler_jobs, "db", db)
    execution_id = uuid4()
    engine = _engine(PipelineResult(execution_id=execution_id, status="skipped"))

    await scheduler_jobs.process_pending_cues(engine)

    assert len(db.calls) == 1
    query, args = db.calls[0]
    # No se cierra como 'completed': espera, con los plazos de Settings.
    assert query == UPDATE_CUE_AWAITING_SCENE
    assert args == (
        cue["id"],
        execution_id,
        engine.config.tipcue_recheck_minutes,
        engine.config.tipcue_max_wait_hours,
    )


async def test_cue_run_on_a_later_scene_completes_the_cue(monkeypatch) -> None:
    cue = _cue()
    db = _FakeDB(cue)
    monkeypatch.setattr(scheduler_jobs, "db", db)
    execution_id = uuid4()
    engine = _engine(PipelineResult(execution_id=execution_id, status="success", num_detections=7))

    await scheduler_jobs.process_pending_cues(engine)

    assert db.calls == [(UPDATE_CUE_STATUS, (cue["id"], "completed", execution_id, "success", 7))]


def test_waiting_cue_keeps_its_attempts_and_expires() -> None:
    # La espera no es un reintento fallido, y no dura para siempre.
    sql = " ".join(UPDATE_CUE_AWAITING_SCENE.split())
    assert "attempts = GREATEST(attempts - 1, 0)" in sql
    assert "THEN 'expired' ELSE 'pending'" in sql
    assert "'completed'" not in sql
