"""
Recolector de metricas de recursos durante la inferencia.

Usa psutil para monitorizar el proceso del pipeline en un thread
separado, muestreando cada 100ms.  Captura uso de CPU y RAM para
generar series temporales y estadisticas agregadas.

Usage:
    from src.profiles.metrics_collector import ResourceCollector

    collector = ResourceCollector(sample_interval_ms=100)
    collector.start()
    # ... ejecutar pipeline ...
    metrics = collector.stop()
    print(metrics.peak_ram_mb, metrics.avg_cpu_percent)
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (no external dep on numpy here).

    ``values`` is unsorted on purpose; we sort a copy so callers keep
    their timeline ordering. Returns ``0.0`` for empty input so the
    metric stays serialisable.
    """
    if not values:
        return 0.0
    p = max(0.0, min(100.0, p))
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac)


@dataclass
class ResourceMetrics:
    """Metricas de recursos recolectadas durante una ejecucion.

    Attributes
    ----------
    duration_ms:
        Duracion total de la recoleccion en milisegundos.
    peak_ram_mb:
        Pico de uso de RAM (RSS) en megabytes.
    avg_ram_mb:
        Uso medio de RAM (RSS) en megabytes.
    peak_cpu_percent:
        Pico de uso de CPU en porcentaje (0-100 por core).
    avg_cpu_percent:
        Uso medio de CPU en porcentaje.
    samples:
        Numero de muestras tomadas.
    ram_timeline:
        Serie temporal de uso de RAM (MB) por muestra.
    cpu_timeline:
        Serie temporal de uso de CPU (%) por muestra.
    latency_p95_ms:
        Percentil 95 del intervalo entre muestras consecutivas. Sirve
        como proxy del jitter de inferencia: cuando la CPU se satura
        bajo perfil restringido, los muestreos se retrasan y este
        valor crece. ``0.0`` cuando no se han recogido muestras.
    energy_estimated_j:
        Energia estimada (Joules) consumida durante la recoleccion.
        Calculada como ``avg_cpu_fraction * tdp_watts * duration_s``,
        donde ``avg_cpu_fraction = avg_cpu_percent / (100 * cpus)``.
        Solo se rellena via :meth:`ResourceCollector.attach_energy_estimate`
        cuando el perfil declara ``tdp_watts``. Su intencion es
        comparar perfiles, no medir consumo absoluto de un satelite.
    energy_method:
        Cadena con el metodo usado para estimar energia
        (``"tdp_x_cpu_fraction"`` o ``"unavailable"``).
    """

    duration_ms: float = 0.0
    peak_ram_mb: float = 0.0
    avg_ram_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    samples: int = 0
    ram_timeline: list[float] = field(default_factory=list)
    cpu_timeline: list[float] = field(default_factory=list)
    latency_p95_ms: float = 0.0
    energy_estimated_j: float | None = None
    energy_method: str = "unavailable"


class ResourceCollector:
    """Recolector de metricas de recursos basado en psutil.

    Lanza un thread de muestreo que captura uso de CPU y RAM
    del proceso objetivo a intervalos regulares.

    Parameters
    ----------
    sample_interval_ms:
        Intervalo de muestreo en milisegundos (default: 100).
    pid:
        PID del proceso a monitorizar.  Si ``None``, usa el
        proceso actual.
    """

    def __init__(
        self,
        sample_interval_ms: int = 100,
        pid: int | None = None,
    ) -> None:
        self._interval_s: float = sample_interval_ms / 1000.0
        self._pid: int = pid if pid is not None else os.getpid()
        self._process: psutil.Process | None = None

        self._ram_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._sample_intervals_ms: list[float] = []

        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia la recoleccion de metricas en un thread separado.

        Raises
        ------
        RuntimeError
            Si la recoleccion ya esta en curso.
        """
        if self._running:
            raise RuntimeError("ResourceCollector is already running")

        try:
            self._process = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            logger.error("Process with PID %d not found", self._pid)
            raise

        # Reset state
        self._ram_samples = []
        self._cpu_samples = []
        self._sample_intervals_ms = []
        self._stop_event.clear()

        # Prime cpu_percent — the first call always returns 0.0
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            self._process.cpu_percent(interval=None)

        self._running = True
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="resource-collector",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "ResourceCollector started (pid=%d, interval=%dms)",
            self._pid,
            int(self._interval_s * 1000),
        )

    def stop(self) -> ResourceMetrics:
        """Detiene la recoleccion y retorna las metricas agregadas.

        Returns
        -------
        ResourceMetrics
            Metricas recolectadas durante el periodo de muestreo.

        Raises
        ------
        RuntimeError
            Si la recoleccion no esta en curso.
        """
        if not self._running:
            raise RuntimeError("ResourceCollector is not running")

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._end_time = time.perf_counter()
        self._running = False

        return self._compute_metrics()

    @property
    def is_running(self) -> bool:
        """``True`` si la recoleccion esta en curso."""
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Loop de muestreo ejecutado en el thread separado."""
        while not self._stop_event.is_set():
            t_sample_start = time.perf_counter()
            try:
                if self._process is not None and self._process.is_running():
                    mem_info = self._process.memory_info()
                    ram_mb = mem_info.rss / (1024 * 1024)
                    cpu_pct = self._process.cpu_percent(interval=None)

                    self._ram_samples.append(ram_mb)
                    self._cpu_samples.append(cpu_pct)
                else:
                    logger.warning("Monitored process (pid=%d) is no longer running", self._pid)
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                logger.warning("Cannot sample process %d: %s", self._pid, exc)
                break

            self._stop_event.wait(timeout=self._interval_s)
            # Record actual elapsed time per sample — its p95 is a
            # proxy of inference jitter under a constrained profile.
            t_sample_end = time.perf_counter()
            self._sample_intervals_ms.append(
                (t_sample_end - t_sample_start) * 1000.0
            )

    def _compute_metrics(self) -> ResourceMetrics:
        """Calcula las metricas agregadas a partir de las muestras."""
        duration_ms = (self._end_time - self._start_time) * 1000.0
        n_samples = len(self._ram_samples)

        if n_samples == 0:
            logger.warning("No samples collected — returning empty metrics")
            return ResourceMetrics(duration_ms=duration_ms)

        peak_ram = max(self._ram_samples)
        avg_ram = sum(self._ram_samples) / n_samples
        peak_cpu = max(self._cpu_samples)
        avg_cpu = sum(self._cpu_samples) / n_samples
        latency_p95 = _percentile(self._sample_intervals_ms, 95.0)

        return ResourceMetrics(
            duration_ms=duration_ms,
            peak_ram_mb=round(peak_ram, 2),
            avg_ram_mb=round(avg_ram, 2),
            peak_cpu_percent=round(peak_cpu, 2),
            avg_cpu_percent=round(avg_cpu, 2),
            samples=n_samples,
            ram_timeline=list(self._ram_samples),
            cpu_timeline=list(self._cpu_samples),
            latency_p95_ms=round(latency_p95, 3),
        )

    # ------------------------------------------------------------------
    # Energy estimate (Q3 simulation)
    # ------------------------------------------------------------------

    @staticmethod
    def attach_energy_estimate(
        metrics: ResourceMetrics,
        profile_name: str,
    ) -> ResourceMetrics:
        """Populate ``metrics.energy_estimated_j`` from the profile TDP.

        Computed as
        ``avg_cpu_fraction * tdp_watts * duration_seconds``,
        where ``avg_cpu_fraction = avg_cpu_percent / (100 * cpu_limit)``.

        ``cpu_limit`` is read from
        :data:`src.profiles.definitions.PROFILES`. Profiles without a
        declared ``tdp_watts`` leave the estimate as ``None`` and tag
        ``energy_method = "unavailable"`` so audits do not mistake a
        zero for "no consumption".
        """
        # Lazy import to avoid a circular import at module load.
        from src.profiles.definitions import PROFILES

        profile = PROFILES.get(profile_name)
        if profile is None or profile.tdp_watts is None:
            metrics.energy_estimated_j = None
            metrics.energy_method = "unavailable"
            return metrics

        cpu_cap = max(profile.cpu_limit, 1e-3)
        avg_cpu_fraction = max(0.0, min(1.0, metrics.avg_cpu_percent / (100.0 * cpu_cap)))
        duration_s = metrics.duration_ms / 1000.0
        joules = avg_cpu_fraction * profile.tdp_watts * duration_s
        metrics.energy_estimated_j = round(joules, 3)
        metrics.energy_method = "tdp_x_cpu_fraction"
        return metrics
