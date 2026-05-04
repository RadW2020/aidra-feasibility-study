"""Tests for the soft memory budget behaviour of ProfileManager.

Background: ``setrlimit(RLIMIT_AS, ...)`` was removed because it
killed the host FastAPI process the moment a sat-* run started — the
PyTorch + CUDA shared libraries map ~8-12 GB of virtual memory
regardless of resident set, and ``RLIMIT_AS`` guards virtual address
space, not RSS.  The new contract is:

- The manager NEVER calls ``resource.setrlimit``.
- After the pipeline returns, peak RSS is compared against
  ``profile.memory_limit_mb``; a breach is recorded in
  :attr:`ProfiledResult.notes` and the run is still marked
  ``success`` so the detections produced are not lost.
"""

from __future__ import annotations

import resource
from unittest.mock import patch

import pytest

from src.profiles.definitions import ConstraintProfile
from src.profiles.manager import ProfileManager
from src.profiles.metrics_collector import ResourceMetrics


# Convenience: build a profile with arbitrary memory budget and the
# loosest CPU limit so CPU enforcement stays out of the way.
def _profile(name: str, memory_limit_mb: int) -> ConstraintProfile:
    return ConstraintProfile(
        name=name,
        display_name=name,
        cpu_limit=4.0,
        memory_limit_mb=memory_limit_mb,
        docker_cpus="4",
        docker_memory=f"{memory_limit_mb}m",
        description=f"test profile {name}",
        simulates="unit test",
    )


@pytest.mark.invariant
class TestNoRlimitAS:
    """The manager must not touch RLIMIT_AS — that is the regression."""

    @pytest.mark.asyncio
    async def test_setrlimit_never_called(self) -> None:
        tiny = _profile("tiny", 64)
        manager = ProfileManager(profiles={"tiny": tiny})

        def noop_pipeline(*_args, **_kwargs):
            return {"detections": [], "metrics": ResourceMetrics()}

        with (
            patch("src.profiles.manager.get_profile", return_value=tiny),
            patch.object(resource, "setrlimit") as setrlimit_spy,
        ):
            await manager.run_with_profile("tiny", noop_pipeline)

        assert setrlimit_spy.call_count == 0, (
            "ProfileManager must not call setrlimit; RLIMIT_AS killed "
            "the host process when ML libraries mapped >budget VM."
        )


class TestBudgetExceedanceNote:
    """When peak RSS > profile budget, notes describe the breach but
    the run is still considered successful."""

    @pytest.mark.asyncio
    async def test_breach_records_note_and_keeps_success(self) -> None:
        tiny = _profile("tiny", 100)
        manager = ProfileManager(profiles={"tiny": tiny})

        def heavy_pipeline(*_args, **_kwargs):
            return {"detections": []}

        # Force the collector to report a peak RSS above the budget.
        with (
            patch("src.profiles.manager.get_profile", return_value=tiny),
            patch(
                "src.profiles.manager.ResourceCollector.stop",
                return_value=ResourceMetrics(
                    peak_ram_mb=300.0, avg_cpu_percent=10.0
                ),
            ),
        ):
            result = await manager.run_with_profile("tiny", heavy_pipeline)

        assert result.success is True
        assert result.notes is not None
        assert "200" in result.notes  # excess MB above the 100 MB budget
        assert "OOM" in result.notes  # message must mention what would happen

    @pytest.mark.asyncio
    async def test_within_budget_no_note(self) -> None:
        roomy = _profile("roomy", 4096)
        manager = ProfileManager(profiles={"roomy": roomy})

        def light_pipeline(*_args, **_kwargs):
            return {"detections": []}

        with (
            patch("src.profiles.manager.get_profile", return_value=roomy),
            patch(
                "src.profiles.manager.ResourceCollector.stop",
                return_value=ResourceMetrics(
                    peak_ram_mb=512.0, avg_cpu_percent=5.0
                ),
            ),
        ):
            result = await manager.run_with_profile("roomy", light_pipeline)

        assert result.success is True
        assert result.notes is None
