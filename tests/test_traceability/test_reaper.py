"""Tests for the orphan-execution reaper.

The reaper transitions executions stuck in ``pending``/``running`` past
a configurable threshold to ``failed``. Without it, container kills and
OOMs leave rows in flight indefinitely, which contaminates dashboards
and reconciliation queries.

Two layers are exercised:

* ``ExecutionRecorder.reap_orphans`` — verifies it forwards the threshold
  to the parameterised SQL and projects ``asyncpg.Record`` rows to the
  ``{id, prior_status, created_at}`` shape the scheduler logs.
* ``reap_orphan_executions`` — verifies the scheduler job swallows DB
  failures and logs each reaped row.

The DB layer is mocked at the method level following the project's
existing pattern in ``tests/test_api/conftest.py`` — a real PostGIS
fixture would test the SQL itself, but the SQL is asserted by inspection
and a follow-up integration test in CI is the right place for that.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.db.queries import REAP_ORPHAN_EXECUTIONS
from src.traceability.recorder import ExecutionRecorder

# ====================================================================
# ExecutionRecorder.reap_orphans
# ====================================================================


class TestReapOrphans:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_orphans(self):
        """No stuck rows → empty list, single fetch call with threshold."""
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[])

        recorder = ExecutionRecorder(db=db)
        result = await recorder.reap_orphans(threshold_minutes=60)

        assert result == []
        db.fetch.assert_awaited_once_with(REAP_ORPHAN_EXECUTIONS, 60)

    @pytest.mark.asyncio
    async def test_projects_rows_to_dicts(self):
        """asyncpg Records are projected to plain dicts the scheduler can log."""
        now = datetime.now(UTC)
        stuck_id = uuid4()

        # Records expose dict() coercion via __iter__/__getitem__; emulate
        # with a tiny stub so the test does not need a live asyncpg.
        class _FakeRecord:
            def __init__(self, mapping: dict) -> None:
                self._m = mapping

            def keys(self):
                return self._m.keys()

            def __getitem__(self, k):
                return self._m[k]

            def __iter__(self):
                return iter(self._m.items())

        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[
            _FakeRecord({
                "id": stuck_id,
                "prior_status": "running",
                "created_at": now - timedelta(hours=3),
            })
        ])

        recorder = ExecutionRecorder(db=db)
        result = await recorder.reap_orphans(threshold_minutes=60)

        assert len(result) == 1
        assert result[0]["id"] == stuck_id
        assert result[0]["prior_status"] == "running"
        assert result[0]["created_at"] < now

    @pytest.mark.asyncio
    async def test_threshold_is_coerced_to_int(self):
        """Float thresholds (rare misconfig) become ints before SQL."""
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[])

        recorder = ExecutionRecorder(db=db)
        await recorder.reap_orphans(threshold_minutes=90.7)  # type: ignore[arg-type]

        db.fetch.assert_awaited_once_with(REAP_ORPHAN_EXECUTIONS, 90)


# ====================================================================
# scheduler_jobs.reap_orphan_executions
# ====================================================================


class TestReaperJob:
    @pytest.mark.asyncio
    async def test_swallows_db_errors(self, monkeypatch, caplog):
        """Reaper must not propagate DB failures (would poison scheduler)."""
        from src.pipeline import scheduler_jobs

        async def _boom(*args, **kwargs):
            raise RuntimeError("DB unreachable")

        # Patch the singleton's fetch so ExecutionRecorder.reap_orphans
        # raises when invoked.
        monkeypatch.setattr(scheduler_jobs.db, "fetch", _boom)

        # Should not raise.
        await scheduler_jobs.reap_orphan_executions(threshold_minutes=60)

    @pytest.mark.asyncio
    async def test_logs_each_reaped_row(self, monkeypatch, caplog):
        """When rows are reaped, one warning per row + one summary INFO."""
        import logging

        from src.pipeline import scheduler_jobs

        reaped_id = uuid4()
        created = datetime.now(UTC) - timedelta(hours=2)

        async def _fake_fetch(query, threshold):
            assert query == REAP_ORPHAN_EXECUTIONS
            assert threshold == 30
            return [{
                "id": reaped_id,
                "prior_status": "pending",
                "created_at": created,
            }]

        monkeypatch.setattr(scheduler_jobs.db, "fetch", _fake_fetch)

        with caplog.at_level(logging.INFO, logger="aidra.scheduler"):
            await scheduler_jobs.reap_orphan_executions(threshold_minutes=30)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        infos = [r for r in caplog.records if r.levelname == "INFO"]
        assert any(str(reaped_id) in r.getMessage() for r in warnings)
        assert any("marked 1 executions" in r.getMessage() for r in infos)
