"""Soft CPU enforcement via wall-clock duty cycling.

Provides :class:`CPUThrottle`, used by sub-core constraint profiles
(sat-low at 0.5 OCPU, sat-extreme at 0.25 OCPU) to emulate
fractional-core hardware limits when real cgroup partitioning is not
available — for example when AIDRA runs inside a shared
Coolify-managed container without delegated cgroup write access.

The throttle is intentionally simple: each work unit (e.g. a tile
inference) is bracketed by a :meth:`tick`, which measures the wall
time consumed by the work and inserts a proportional sleep so the
average ``work_time / total_time`` ratio converges to the target
fraction.

Why wall-time and not CPU time
------------------------------
An earlier implementation drove the throttle from
:func:`time.process_time`.  That measures CPU time **summed across
all threads** of the process.  PyTorch's OpenMP-parallelised
operators dispatch work to several worker threads, so a tile that
took 1.5 s of wall time appeared to consume 3-5 s of CPU.  The
throttle then computed an over-large sleep — for ``target=0.25`` the
factor ballooned to ~7.5x the expected slowdown (sat-low actually
took 250 min on run #3 vs the expected 66 min).

Wall-time work measurement removes that ambiguity entirely: whatever
the work *actually* took on the host, the throttle adds a
proportional pause.  In steady state the achievable duty cycle equals
the target exactly:

    target = work / (work + sleep)
    sleep  = work * (1 / target - 1)

The trade-off is documented in the relevant ``MODEL_CARD.md`` so the
D3 / D4 evidence is honest about the enforcement mechanism.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


# Per-tick sleep cap.  Acts as a runaway safety net.  10 s is generous
# even for sat-extreme (target 0.25) with 2-3 s tiles, where the
# expected sleep is 6-9 s; a runaway tile of 30 s would otherwise
# request 90 s.
_MAX_SLEEP_PER_TICK_S = 10.0

# Lower bound on the target fraction to avoid division by zero and
# avalanche-style sleep amplification.  0.05 = 5% utilisation,
# already below the lowest profile we ship (sat-extreme at 0.25).
_MIN_TARGET = 0.05


class CPUThrottle:
    """Maintain an average ``work_time / wall_time`` close to ``target_fraction``.

    Args:
        target_fraction: Desired ratio of work time to total wall
            time, in ``(0, 1]``.  ``>= 1.0`` makes :meth:`tick` a
            no-op (full speed).  Below :data:`_MIN_TARGET` is clamped
            up.

    Usage::

        throttle = CPUThrottle(0.25)
        throttle.reset()
        for tile in tiles:
            run_inference(tile)
            throttle.tick()
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

        Useful between phases (e.g. between the CFAR and YOLO passes)
        so that any one-off delay before the first tile is not counted
        as work for the next phase.
        """
        now = time.monotonic()
        self._wall_start: float = now
        self._last_tick: float = now
        self._total_work_s: float = 0.0
        self._total_sleep_s: float = 0.0

    def tick(self) -> None:
        """Measure the work just completed and sleep proportionally.

        The work duration is the wall time elapsed since the previous
        :meth:`tick` (or :meth:`reset`).  If the throttle is disabled
        (``target_fraction >= 1.0``) the call is a no-op except for
        bookkeeping.
        """
        now = time.monotonic()
        work_s = now - self._last_tick
        if work_s < 0:  # clock skew safety
            work_s = 0.0
        self._total_work_s += work_s

        if not self._enabled:
            self._last_tick = now
            return

        # In steady state: work / (work + sleep) = target  =>
        # sleep = work * (1/target - 1)
        sleep_for = work_s * (1.0 / self.target - 1.0)
        if sleep_for <= 0:
            self._last_tick = now
            return

        sleep_for = min(sleep_for, _MAX_SLEEP_PER_TICK_S)
        time.sleep(sleep_for)
        self._total_sleep_s += sleep_for
        self._last_tick = time.monotonic()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether :meth:`tick` actually inserts sleeps."""
        return self._enabled

    @property
    def total_sleep_s(self) -> float:
        """Cumulative sleep injected since the last :meth:`reset`."""
        return self._total_sleep_s

    @property
    def total_work_s(self) -> float:
        """Cumulative wall-time work observed since the last :meth:`reset`."""
        return self._total_work_s

    def observed_fraction(self) -> float:
        """Current ``work_time / wall_time`` ratio since the last reset.

        Returns ``0.0`` when no wall time has elapsed yet (e.g. the
        throttle was just reset).  In steady state this should be
        close to :attr:`target`.
        """
        wall_elapsed = time.monotonic() - self._wall_start
        if wall_elapsed <= 0:
            return 0.0
        return self._total_work_s / wall_elapsed
