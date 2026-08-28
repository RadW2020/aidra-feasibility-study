"""Soft memory budget enforcement for constraint profiles.

``RLIMIT_AS`` killed the whole FastAPI process (PyTorch maps 8-12 GB of
virtual address space regardless of RSS), so the budget is enforced on
*measured* peak RSS instead: :class:`MemoryGuard` is injected into the
detection loop and raises :class:`MemoryBudgetExceeded` (a ``MemoryError``)
at the first per-tile check after the ``ResourceCollector`` has sampled an
RSS above the profile budget. Granularity is one tile (~100 ms sampling);
on real on-board hardware the kernel would have OOM-killed the process at
the same point.
"""

from __future__ import annotations

from typing import Any


class MemoryBudgetExceeded(MemoryError):
    """Peak RSS exceeded the constraint profile's memory budget."""


class MemoryGuard:
    """Raise once the collector observes RSS above ``limit_mb``."""

    def __init__(self, limit_mb: int | float, collector: Any, profile_name: str = "") -> None:
        self.limit_mb = float(limit_mb)
        self._collector = collector
        self.profile_name = profile_name
        self.checks = 0
        self.tripped_at_mb: float | None = None

    def check(self) -> None:
        self.checks += 1
        peak = float(self._collector.peak_ram_mb_so_far())
        if peak > self.limit_mb:
            self.tripped_at_mb = peak
            raise MemoryBudgetExceeded(
                f"memory budget exceeded: peak RSS {peak:.0f} MB > "
                f"{self.limit_mb:.0f} MB budget of profile '{self.profile_name}' "
                f"(check #{self.checks}); on real on-board hardware this run "
                "would have been OOM-killed here"
            )
