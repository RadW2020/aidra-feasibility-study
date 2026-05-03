"""
Gestion de ejecucion del pipeline con perfiles de restriccion.

Ejecuta una funcion de pipeline bajo restricciones de CPU y RAM
controladas, recopilando metricas de rendimiento.  Permite ejecutar
el mismo escenario bajo todos los perfiles y generar un informe
comparativo que identifica el punto de inflexion de degradacion.

Metodo de restriccion: ``resource.setrlimit`` para memoria + monitoreo
``psutil`` en thread separado.  No requiere Docker-in-Docker.

Usage:
    from src.profiles.manager import ProfileManager

    manager = ProfileManager()
    result = await manager.run_with_profile("sat-mid", pipeline_fn, image_id="abc")
    all_results = await manager.run_all_profiles(pipeline_fn, image_id="abc")
    report = manager.compare_profiles(all_results)
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.profiles.definitions import (
    PROFILE_ORDER,
    PROFILES,
    ConstraintProfile,
    get_profile,
)
from src.profiles.metrics_collector import ResourceCollector, ResourceMetrics

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass
class ProfiledResult:
    """Resultado de ejecutar el pipeline bajo un perfil de restriccion.

    Attributes
    ----------
    profile:
        Perfil bajo el cual se ejecuto.
    success:
        ``True`` si la ejecucion completo sin errores.
    error:
        Mensaje de error si fallo (e.g. ``"OOM"``, ``"timeout"``).
    metrics:
        Metricas de recursos recolectadas por ``ResourceCollector``.
    inference_ms:
        Tiempo de inferencia en milisegundos.
    peak_ram_mb:
        Pico de uso de RAM en MB (shortcut de ``metrics.peak_ram_mb``).
    cpu_percent:
        Uso medio de CPU en porcentaje.
    num_detections:
        Numero de detecciones producidas.
    avg_confidence:
        Confianza media de las detecciones.
    detections:
        Lista de detecciones del pipeline.
    raw_result:
        Resultado crudo devuelto por la funcion del pipeline.
    """

    profile: ConstraintProfile
    success: bool = False
    error: str | None = None
    metrics: ResourceMetrics | None = None
    inference_ms: float | None = None
    peak_ram_mb: float | None = None
    cpu_percent: float | None = None
    num_detections: int | None = None
    avg_confidence: float | None = None
    detections: list[Any] | None = None
    raw_result: Any = None


@dataclass
class ComparisonReport:
    """Informe comparativo de ejecucion bajo multiples perfiles.

    Attributes
    ----------
    rows:
        Lista de diccionarios con metricas por perfil.
    inflection_point:
        Nombre del perfil donde se detecta degradacion significativa
        (precision cae >20% respecto a ground o el pipeline falla).
    all_passed:
        ``True`` si todos los perfiles completaron exitosamente.
    summary:
        Texto resumen legible del informe.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    inflection_point: str | None = None
    all_passed: bool = True
    summary: str = ""


# ------------------------------------------------------------------
# ProfileManager
# ------------------------------------------------------------------


class ProfileManager:
    """Ejecuta el pipeline de deteccion bajo perfiles de restriccion.

    Parameters
    ----------
    profiles:
        Diccionario de perfiles disponibles.  Por defecto usa
        ``PROFILES`` de ``definitions.py``.
    """

    def __init__(
        self,
        profiles: dict[str, ConstraintProfile] | None = None,
    ) -> None:
        self.profiles: dict[str, ConstraintProfile] = profiles or PROFILES
        self._warn_if_platform_lacks_enforcement()

    @staticmethod
    def _warn_if_platform_lacks_enforcement() -> None:
        """Log a clear warning when the host cannot enforce profiles.

        Constraint enforcement (RAM via RLIMIT_AS, CPU via
        sched_setaffinity, fractional CPU via cgroups v2) only works
        reliably on Linux. On macOS / Windows the Python wrappers are
        no-ops, so sat-low / sat-mid / sat-extreme runs will produce
        IDENTICAL telemetry to a `ground` baseline. This is a known
        limitation; the warning makes it visible to operators.
        """
        import platform
        system = platform.system()
        if system != "Linux":
            logger.warning(
                "ProfileManager: constraint enforcement is a NO-OP on "
                "%s — sat-* profiles will not actually limit RAM or "
                "CPU. Telemetry from sat-low / sat-mid / sat-extreme "
                "will be indistinguishable from `ground`. Run on a "
                "Linux container (Docker is fine) for real benchmarks.",
                system,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_with_profile(
        self,
        profile_name: str,
        pipeline_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> ProfiledResult:
        """Ejecuta ``pipeline_fn`` bajo las restricciones de un perfil.

        Parameters
        ----------
        profile_name:
            Nombre del perfil (e.g. ``"sat-mid"``).
        pipeline_fn:
            Funcion (sync o async) que ejecuta el pipeline.
        *args, **kwargs:
            Argumentos pasados a ``pipeline_fn``.

        Returns
        -------
        ProfiledResult
            Resultado de la ejecucion con metricas de recursos.
        """
        profile = get_profile(profile_name)
        logger.info(
            "Running pipeline with profile '%s' (cpu=%.2f, ram=%dMB)",
            profile.name,
            profile.cpu_limit,
            profile.memory_limit_mb,
        )

        result = ProfiledResult(profile=profile)
        collector = ResourceCollector(sample_interval_ms=100)
        original_limits = self._get_memory_limits()

        original_affinity = self._get_cpu_affinity()

        try:
            # 1. Apply resource limits (memory + CPU)
            self._apply_memory_limit(profile.memory_limit_mb)
            self._apply_cpu_limit(profile.cpu_limit)

            # 2. Start resource monitoring
            collector.start()

            # 3. Execute pipeline function
            start_time = time.perf_counter()

            if asyncio.iscoroutinefunction(pipeline_fn):
                raw = await pipeline_fn(*args, **kwargs)
            else:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pipeline_fn(*args, **kwargs)
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # 4. Stop monitoring and collect metrics
            metrics = collector.stop()

            # 5. Extract detection info from result
            result.success = True
            result.raw_result = raw
            result.metrics = metrics
            result.inference_ms = elapsed_ms
            result.peak_ram_mb = metrics.peak_ram_mb
            result.cpu_percent = metrics.avg_cpu_percent

            self._extract_detection_info(result, raw)

            logger.info(
                "Profile '%s' completed: %.0fms, peak_ram=%.1fMB, detections=%s",
                profile.name,
                elapsed_ms,
                metrics.peak_ram_mb,
                result.num_detections,
            )

        except MemoryError:
            if collector.is_running:
                metrics = collector.stop()
                result.metrics = metrics
            result.error = "OOM"
            logger.warning("Profile '%s' failed: OOM", profile.name)

        except TimeoutError:
            if collector.is_running:
                metrics = collector.stop()
                result.metrics = metrics
            result.error = "timeout"
            logger.warning("Profile '%s' failed: timeout", profile.name)

        except Exception as exc:
            if collector.is_running:
                metrics = collector.stop()
                result.metrics = metrics
            result.error = f"error: {exc}"
            logger.exception("Profile '%s' failed with error", profile.name)

        finally:
            # Restore original resource limits
            self._restore_memory_limits(original_limits)
            self._restore_cpu_affinity(original_affinity)

        return result

    async def run_all_profiles(
        self,
        pipeline_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, ProfiledResult]:
        """Ejecuta el pipeline con todos los perfiles secuencialmente.

        Los perfiles se ejecutan en orden de mas a menos recursos
        (ground -> sat-extreme).

        Parameters
        ----------
        pipeline_fn:
            Funcion del pipeline a ejecutar.
        *args, **kwargs:
            Argumentos pasados a ``pipeline_fn``.

        Returns
        -------
        dict[str, ProfiledResult]
            Diccionario con resultados indexados por nombre de perfil.
        """
        results: dict[str, ProfiledResult] = {}

        for profile_name in PROFILE_ORDER:
            if profile_name not in self.profiles:
                continue
            logger.info("--- Profile %s ---", profile_name)
            result = await self.run_with_profile(
                profile_name, pipeline_fn, *args, **kwargs
            )
            results[profile_name] = result

        return results

    def compare_profiles(
        self,
        results: dict[str, ProfiledResult],
    ) -> ComparisonReport:
        """Genera informe comparativo de multiples perfiles.

        Identifica el punto de inflexion donde el pipeline comienza
        a degradarse significativamente o falla.

        Parameters
        ----------
        results:
            Resultados de ``run_all_profiles()``.

        Returns
        -------
        ComparisonReport
            Informe con tabla comparativa y analisis.
        """
        report = ComparisonReport()
        ground_result = results.get("ground")
        ground_detections = (
            ground_result.num_detections
            if ground_result and ground_result.num_detections
            else 0
        )

        for profile_name in PROFILE_ORDER:
            if profile_name not in results:
                continue

            r = results[profile_name]
            row: dict[str, Any] = {
                "profile": profile_name,
                "display_name": r.profile.display_name,
                "cpu_limit": r.profile.cpu_limit,
                "memory_limit_mb": r.profile.memory_limit_mb,
                "success": r.success,
                "error": r.error,
                "inference_ms": r.inference_ms,
                "peak_ram_mb": r.peak_ram_mb,
                "cpu_percent": r.cpu_percent,
                "num_detections": r.num_detections,
                "avg_confidence": r.avg_confidence,
            }

            # Calculate degradation vs ground
            if ground_detections > 0 and r.num_detections is not None:
                detection_ratio = r.num_detections / ground_detections
                row["detection_retention_pct"] = round(detection_ratio * 100, 1)
            else:
                row["detection_retention_pct"] = None

            report.rows.append(row)

            # Detect inflection point
            if not r.success:
                report.all_passed = False
                if report.inflection_point is None:
                    report.inflection_point = profile_name
            elif (
                report.inflection_point is None
                and ground_detections > 0
                and r.num_detections is not None
            ):
                detection_ratio = r.num_detections / ground_detections
                if detection_ratio < 0.80:
                    report.inflection_point = profile_name

        # Build summary
        report.summary = self._build_summary(report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_detection_info(result: ProfiledResult, raw: Any) -> None:
        """Extrae informacion de detecciones del resultado crudo.

        Soporta resultados como dict con claves ``detections``,
        ``num_detections``, ``avg_confidence``, o como lista directa.
        """
        if raw is None:
            return

        if isinstance(raw, dict):
            detections = raw.get("detections", [])
            result.detections = detections
            result.num_detections = raw.get("num_detections", len(detections))
            result.avg_confidence = raw.get("avg_confidence")

            if result.avg_confidence is None and detections:
                confidences = [
                    d.get("confidence", 0.0)
                    for d in detections
                    if isinstance(d, dict)
                ]
                if confidences:
                    result.avg_confidence = sum(confidences) / len(confidences)

        elif isinstance(raw, list):
            result.detections = raw
            result.num_detections = len(raw)
            confidences = [
                d.get("confidence", 0.0)
                for d in raw
                if isinstance(d, dict)
            ]
            if confidences:
                result.avg_confidence = sum(confidences) / len(confidences)

    @staticmethod
    def _get_memory_limits() -> tuple[int, int]:
        """Obtiene los limites de memoria actuales (soft, hard).

        Returns
        -------
        tuple[int, int]
            Limites (soft, hard) de ``RLIMIT_AS`` o ``(-1, -1)`` si
            la plataforma no soporta ``setrlimit``.
        """
        if platform.system() == "Windows":
            return (-1, -1)
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            return (soft, hard)
        except (ValueError, OSError):
            return (-1, -1)

    @staticmethod
    def _apply_memory_limit(memory_limit_mb: int) -> None:
        """Aplica limite de memoria virtual via ``resource.setrlimit``.

        Parameters
        ----------
        memory_limit_mb:
            Limite de memoria en megabytes.
        """
        if platform.system() == "Windows":
            logger.debug("setrlimit not available on Windows — skipping")
            return

        limit_bytes = memory_limit_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            logger.debug("Memory limit set to %dMB via RLIMIT_AS", memory_limit_mb)
        except (ValueError, OSError) as exc:
            logger.warning(
                "Could not set memory limit to %dMB: %s", memory_limit_mb, exc
            )

    @staticmethod
    def _restore_memory_limits(original: tuple[int, int]) -> None:
        """Restaura los limites de memoria originales.

        Parameters
        ----------
        original:
            Tupla (soft, hard) obtenida de ``_get_memory_limits()``.
        """
        if platform.system() == "Windows" or original == (-1, -1):
            return
        try:
            resource.setrlimit(resource.RLIMIT_AS, original)
            logger.debug("Memory limits restored")
        except (ValueError, OSError) as exc:
            logger.warning("Could not restore memory limits: %s", exc)

    @staticmethod
    def _build_summary(report: ComparisonReport) -> str:
        """Construye un resumen textual del informe comparativo.

        Parameters
        ----------
        report:
            Informe con las filas ya calculadas.

        Returns
        -------
        str
            Resumen legible con las conclusiones principales.
        """
        lines: list[str] = ["=== Profile Comparison Report ===", ""]

        # Header
        lines.append(
            f"{'Profile':<15} {'CPU':>5} {'RAM(MB)':>8} {'OK':>4} "
            f"{'Time(ms)':>10} {'Peak RAM':>10} {'Detections':>11} "
            f"{'Confidence':>11} {'Retention':>10}"
        )
        lines.append("-" * 95)

        for row in report.rows:
            inf_ms = f"{row['inference_ms']:.0f}" if row["inference_ms"] else "N/A"
            peak = f"{row['peak_ram_mb']:.1f}" if row["peak_ram_mb"] else "N/A"
            dets = str(row["num_detections"]) if row["num_detections"] is not None else "N/A"
            conf = f"{row['avg_confidence']:.3f}" if row["avg_confidence"] else "N/A"
            ret = (
                f"{row['detection_retention_pct']:.1f}%"
                if row["detection_retention_pct"] is not None
                else "N/A"
            )
            ok = "Y" if row["success"] else f"N ({row['error']})"

            lines.append(
                f"{row['profile']:<15} {row['cpu_limit']:>5.2f} "
                f"{row['memory_limit_mb']:>8} {ok:>4} "
                f"{inf_ms:>10} {peak:>10} {dets:>11} "
                f"{conf:>11} {ret:>10}"
            )

        lines.append("")

        if report.inflection_point:
            lines.append(
                f"Inflection point: '{report.inflection_point}' — "
                f"significant degradation or failure begins at this profile."
            )
        else:
            lines.append("No inflection point detected — all profiles perform adequately.")

        if not report.all_passed:
            failed = [r["profile"] for r in report.rows if not r["success"]]
            lines.append(f"Failed profiles: {', '.join(failed)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CPU affinity limits
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cpu_affinity() -> list[int] | None:
        """Get current CPU affinity (list of allowed CPU IDs)."""
        try:
            return list(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            # Not available on macOS or Windows
            return None

    @staticmethod
    def _apply_cpu_limit(cpu_limit: float) -> None:
        """Limit CPU cores via os.sched_setaffinity.

        For fractional limits (e.g. 0.5), restricts to 1 core.
        The fractional part is a "soft" limit — the process uses
        fewer cycles but the OS doesn't enforce sub-core allocation.
        For real sub-core limiting, Docker/cgroups would be needed.

        Parameters
        ----------
        cpu_limit:
            Number of CPU cores to allow (can be fractional).
        """
        try:
            available = list(os.sched_getaffinity(0))
            num_cores = max(1, int(cpu_limit))
            if num_cores < len(available):
                restricted = available[:num_cores]
                os.sched_setaffinity(0, restricted)
                logger.debug(
                    "CPU affinity set to %d cores: %s", num_cores, restricted
                )
            else:
                logger.debug(
                    "CPU limit %.1f >= available %d cores — no restriction applied",
                    cpu_limit,
                    len(available),
                )
        except (AttributeError, OSError) as exc:
            # sched_setaffinity not available on macOS
            logger.debug("CPU affinity not available on this platform: %s", exc)

    @staticmethod
    def _restore_cpu_affinity(original: list[int] | None) -> None:
        """Restore original CPU affinity."""
        if original is None:
            return
        try:
            os.sched_setaffinity(0, original)
            logger.debug("CPU affinity restored to %d cores", len(original))
        except (AttributeError, OSError) as exc:
            logger.debug("Could not restore CPU affinity: %s", exc)
