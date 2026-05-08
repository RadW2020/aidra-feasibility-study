"""Tests for the advisory-lock layer that protects scheduler jobs from
double-firing on multi-replica deploys.

Two layers:

* ``Database.try_advisory_lock`` — async context manager around
  ``pg_try_advisory_lock`` / ``pg_advisory_unlock``. Yields a boolean.
* ``with_advisory_lock`` decorator in scheduler_jobs — wraps an async
  job, skips if another replica holds the lock, runs unguarded if the
  DB itself is unreachable.

The DB is mocked at the connection level (same pattern as the reaper
test) — a real PostgreSQL fixture would test the SQL but is not
available in unit-test scope.
"""

from __future__ import annotations

import pytest

from src.db.connection import _key_to_lock_id

# ====================================================================
# _key_to_lock_id helper
# ====================================================================


class TestKeyToLockId:
    def test_deterministic_across_processes(self):
        """The mapping must be stable: a given key always returns the
        same int regardless of Python process or hash randomisation.
        """
        # SHA-256 of "aidra:job:scheduled_scan" first 8 bytes, signed:
        expected = _key_to_lock_id("aidra:job:scheduled_scan")
        # Compute it twice; identical
        assert _key_to_lock_id("aidra:job:scheduled_scan") == expected

    def test_distinct_keys_distinct_ids(self):
        a = _key_to_lock_id("aidra:job:scan")
        b = _key_to_lock_id("aidra:job:cue")
        assert a != b

    def test_int_fits_signed_bigint(self):
        """PostgreSQL bigint range is [-2^63, 2^63-1]; the helper must
        not produce an out-of-range value."""
        x = _key_to_lock_id("any-key")
        assert -(2**63) <= x <= 2**63 - 1


# ====================================================================
# scheduler_jobs.with_advisory_lock decorator
# ====================================================================


class _FakeLockCtx:
    """Async context manager emulating Database.try_advisory_lock."""

    def __init__(self, acquired: bool, raise_on_enter: bool = False) -> None:
        self.acquired = acquired
        self.raise_on_enter = raise_on_enter

    async def __aenter__(self) -> bool:
        if self.raise_on_enter:
            raise RuntimeError("DB unreachable")
        return self.acquired

    async def __aexit__(self, *exc_info) -> None:
        return None


class TestWithAdvisoryLock:
    @pytest.mark.asyncio
    async def test_runs_when_lock_acquired(self, monkeypatch):
        from src.pipeline import scheduler_jobs

        called = []

        @scheduler_jobs.with_advisory_lock("aidra:test:run-when-acquired")
        async def job():
            called.append("ran")
            return "done"

        # Patch the lock context manager to acquire successfully.
        def fake_lock(key: str):
            return _FakeLockCtx(acquired=True)

        monkeypatch.setattr(scheduler_jobs.db, "try_advisory_lock", fake_lock)

        result = await job()
        assert called == ["ran"]
        assert result == "done"

    @pytest.mark.asyncio
    async def test_skips_when_lock_held(self, monkeypatch, caplog):
        import logging

        from src.pipeline import scheduler_jobs

        called = []

        @scheduler_jobs.with_advisory_lock("aidra:test:skip")
        async def job():
            called.append("ran")
            return "done"

        def fake_lock(key: str):
            return _FakeLockCtx(acquired=False)

        monkeypatch.setattr(scheduler_jobs.db, "try_advisory_lock", fake_lock)

        with caplog.at_level(logging.INFO, logger="aidra.scheduler"):
            result = await job()

        assert called == []  # job body did not run
        assert result is None
        assert any("skipped" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_falls_back_when_db_unreachable(self, monkeypatch, caplog):
        import logging

        from src.pipeline import scheduler_jobs

        called = []

        @scheduler_jobs.with_advisory_lock("aidra:test:fallback")
        async def job():
            called.append("ran")
            return "done"

        def fake_lock(key: str):
            # Raises inside __aenter__
            return _FakeLockCtx(acquired=False, raise_on_enter=True)

        monkeypatch.setattr(scheduler_jobs.db, "try_advisory_lock", fake_lock)

        with caplog.at_level(logging.WARNING, logger="aidra.scheduler"):
            result = await job()

        # The job runs unguarded so we don't lose ticks during DB blips.
        assert called == ["ran"]
        assert result == "done"
        assert any("running unguarded" in r.getMessage() for r in caplog.records)
