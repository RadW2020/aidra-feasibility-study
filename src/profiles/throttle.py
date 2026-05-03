"""Soft CPU enforcement via wall-clock duty cycling.

Provides :class:`CPUThrottle`, a tiny throttler used by sub-core
constraint profiles (sat-low at 0.5 OCPU, sat-extreme at 0.25 OCPU)
to emulate fractional-core hardware limits when real cgroup
partitioning is not available — for example when AIDRA runs inside a
shared Coolify-managed container without delegated cgroup write
access.

The throttle is intentionally simple: after each work unit (e.g. a
tile inference) the caller invokes :meth:`tick`, which compares the
CPU time used so far against the wall-clock time and sleeps just long
enough to keep the average ``cpu_time / wall_time`` close to the
target fraction.

This is a SOFT limit:
- It only converges over many ticks; a single tile burst is not
  preempted.
- It uses :func:`time.process_time`, which on Linux returns the sum
  of CPU time across all threads of the process.  AIDRA's inference
  loop is single-threaded today (see TECHNICAL_SPEC §detection), so
  this matches the intended semantics.  If the loop is parallelised
  later, the throttle should be re-evaluated.

The trade-off is documented in the relevant ``MODEL_CARD.md`` so the
D3 / D4 evidence is honest about the enforcement mechanism.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


# Per-tick sleep cap.  Keeps a single oversized tile from translating
# into a huge stall that would skew p95 latency reporting.  At 0.5 s
# the throttle still converges within a few ticks for typical S1 GRD
# tiles (CFAR ~50 ms, YOLO ~1-3 s on ARM CPU).
_MAX_SLEEP_PER_TICK_S = 0.5

# Lower bound on the target fraction to avoid division by zero and
# avalanche-style sleep amplification.  0.05 = 5% utilisation,
# already below the lowest profile we ship (sat-extreme at 0.25).
_MIN_TARGET = 0.05


class CPUThrottle:
    """Maintain an average CPU utilisation around ``target_fraction``.

    Args:
        target_fraction: Desired ratio of CPU time to wall-clock time,
            in ``(0, 1]``.  A value ``>= 1.0`` makes :meth:`tick` a
            no-op (full speed).  Values below :data:`_MIN_TARGET` are
            clamped up.

    Example:
        >>> throttle = CPUThrottle(0.25)
        >>> for tile in tiles:
        ...     run_inference(tile)
        ...     throttle.tick()
    """

    def __init__(self, target_fraction: float) -> None:
        if target_fraction is None:
            target_fraction = 1.0
        clamped = max(_MIN_TARGET, min(1.0, float(target_fraction)))
        self.target: float = clamped
        self._enabled: bool = clamped < 1.0
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-anchor the throttle at the current instant.

        Useful when the caller wants to start a fresh measurement
        window — for example between the CFAR pass and the YOLO pass.
        """
        self._wall_start = time.monotonic()
        self._cpu_start = time.process_time()
        self._total_sleep_s: float = 0.0

    def tick(self) -> None:
        """Sleep, if needed, to keep the cumulative duty cycle on target.

        No-op when ``target_fraction >= 1.0`` (i.e. ground / sat-high).
        """
        if not self._enabled:
            return

        wall_elapsed = time.monotonic() - self._wall_start
        cpu_elapsed = time.process_time() - self._cpu_start

        # Desired wall time so far: cpu_elapsed / target.
        # If actual wall is shorter, we've been "too busy" — sleep
        # the gap (capped per-tick).
        target_wall = cpu_elapsed / self.target
        sleep_for = target_wall - wall_elapsed
        if sleep_for <= 0:
            return

        sleep_for = min(sleep_for, _MAX_SLEEP_PER_TICK_S)
        time.sleep(sleep_for)
        self._total_sleep_s += sleep_for

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the throttle actually inserts sleeps."""
        return self._enabled

    @property
    def total_sleep_s(self) -> float:
        """Cumulative sleep time injected since the last :meth:`reset`."""
        return self._total_sleep_s

    def observed_fraction(self) -> float:
        """Current ``cpu_time / wall_time`` ratio since the last reset.

        Returns ``0.0`` if no wall time has elapsed yet (e.g. the
        throttle was just constructed and never ticked).
        """
        wall_elapsed = time.monotonic() - self._wall_start
        if wall_elapsed <= 0:
            return 0.0
        cpu_elapsed = time.process_time() - self._cpu_start
        return cpu_elapsed / wall_elapsed
