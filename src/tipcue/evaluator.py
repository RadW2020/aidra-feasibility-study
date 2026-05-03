"""
Evaluador de Tips: decide si las detecciones de un pipeline generan un Cue.

Dada una lista de detecciones de una ejecucion del pipeline, evalua
reglas basadas en zona de interes, confianza minima, numero de
detecciones y cooldown para decidir si se debe generar un cue de
re-observacion automatica.

Usage:
    from src.tipcue.evaluator import TipEvaluator
    from src.tipcue.zones import DEFAULT_ZONES

    evaluator = TipEvaluator(zones_of_interest=DEFAULT_ZONES)
    tips = evaluator.evaluate(detections, execution_id=exec_id)
    for tip in tips:
        if tip.should_cue:
            await scheduler.create_cue(...)
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from src.tipcue.zones import Zone, get_active_zones

logger = logging.getLogger(__name__)


@dataclass
class TipResult:
    """Resultado de la evaluacion de un tip para una zona.

    Attributes
    ----------
    should_cue:
        ``True`` si se debe generar un cue de re-observacion.
    reason:
        Razon textual de la decision (e.g.
        ``"high_confidence_in_interest_zone"``).
    target_bbox:
        Subzona recomendada para el cue
        ``[lon_min, lat_min, lon_max, lat_max]``.
    priority:
        Prioridad del cue: 0=normal, 1=alta, 2=urgente.
    triggering_detections:
        IDs de las detecciones que generaron el tip.
    execution_id:
        ID de la ejecucion original que contiene las detecciones.
    zone_id:
        ID de la zona de interes en la que se encontraron las
        detecciones.
    """

    should_cue: bool = False
    reason: str = ""
    target_bbox: list[float] = field(default_factory=list)
    priority: int = 0
    triggering_detections: list[UUID] = field(default_factory=list)
    execution_id: UUID | None = None
    zone_id: str | None = None


class TipEvaluator:
    """Evalua detecciones contra zonas de interes para generar tips.

    Parameters
    ----------
    zones_of_interest:
        Lista de zonas de interes.  Si ``None``, usa las zonas
        activas de ``DEFAULT_ZONES``.
    min_confidence:
        Confianza minima requerida para que una deteccion cuente
        como relevante.
    min_detections:
        Numero minimo de detecciones relevantes en una zona para
        generar un tip.
    cooldown_minutes:
        Tiempo minimo entre tips para la misma zona (evita
        generar tips duplicados).
    """

    def __init__(
        self,
        zones_of_interest: list[Zone] | None = None,
        min_confidence: float = 0.7,
        min_detections: int = 2,
        cooldown_minutes: int = 60,
    ) -> None:
        self.zones: list[Zone] = zones_of_interest or get_active_zones()
        self.min_confidence: float = min_confidence
        self.min_detections: int = min_detections
        self.cooldown_minutes: int = cooldown_minutes

        # Registro interno de ultimo tip por zona para cooldown
        self._last_tip_time: dict[str, float] = {}

    def evaluate(
        self,
        detections: list[dict[str, Any]],
        execution_id: UUID,
    ) -> list[TipResult]:
        """Evalua detecciones contra las zonas de interes.

        Cada deteccion debe ser un dict con al menos las claves:
        - ``longitude`` (float)
        - ``latitude`` (float)
        - ``confidence`` (float)
        - ``id`` (str o UUID, opcional)

        Parameters
        ----------
        detections:
            Lista de detecciones del pipeline.
        execution_id:
            ID de la ejecucion del pipeline.

        Returns
        -------
        list[TipResult]
            Lista de resultados — uno por cada zona donde se
            cumplen las reglas.  Puede estar vacia si ninguna
            zona califica.
        """
        results: list[TipResult] = []
        now = time.time()

        for zone in self.zones:
            if not zone.active:
                continue

            # Check cooldown for this zone
            if self._is_in_cooldown(zone.id, now):
                logger.debug(
                    "Zone '%s' is in cooldown — skipping evaluation",
                    zone.id,
                )
                continue

            # Filter detections in this zone with sufficient confidence
            matching: list[dict[str, Any]] = []
            for det in detections:
                lon = det.get("longitude", 0.0)
                lat = det.get("latitude", 0.0)
                conf = det.get("confidence", 0.0)

                if conf >= self.min_confidence and zone.contains_point(lon, lat):
                    matching.append(det)

            # Check minimum detections threshold
            if len(matching) < self.min_detections:
                logger.debug(
                    "Zone '%s': %d detections (need >= %d) — no tip",
                    zone.id,
                    len(matching),
                    self.min_detections,
                )
                continue

            # Generate tip
            tip = self._create_tip(zone, matching, execution_id)
            results.append(tip)

            # Record tip time for cooldown
            self._last_tip_time[zone.id] = now

            logger.info(
                "Tip generated for zone '%s': %d detections, priority=%d, reason='%s'",
                zone.id,
                len(matching),
                tip.priority,
                tip.reason,
            )

        return results

    def reset_cooldowns(self) -> None:
        """Reinicia todos los cooldowns de zona.

        Util para testing o para forzar re-evaluacion inmediata.
        """
        self._last_tip_time.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, zone_id: str, now: float) -> bool:
        """Comprueba si la zona esta en periodo de cooldown.

        Parameters
        ----------
        zone_id:
            Identificador de la zona.
        now:
            Timestamp actual (``time.time()``).

        Returns
        -------
        bool
            ``True`` si han pasado menos de ``cooldown_minutes``
            desde el ultimo tip en esta zona.
        """
        last = self._last_tip_time.get(zone_id)
        if last is None:
            return False
        elapsed_minutes = (now - last) / 60.0
        return elapsed_minutes < self.cooldown_minutes

    def _create_tip(
        self,
        zone: Zone,
        matching_detections: list[dict[str, Any]],
        execution_id: UUID,
    ) -> TipResult:
        """Crea un TipResult a partir de las detecciones filtradas.

        Calcula un sub-bbox centrado en las detecciones con un margen
        y determina la prioridad basandose en la confianza media y
        el numero de detecciones.

        Parameters
        ----------
        zone:
            Zona de interes donde se encontraron las detecciones.
        matching_detections:
            Detecciones que pasan los filtros de confianza y ubicacion.
        execution_id:
            ID de la ejecucion original.

        Returns
        -------
        TipResult
        """
        # Compute sub-bbox around detections with margin
        target_bbox = self._compute_target_bbox(matching_detections)

        # Determine priority
        avg_conf = sum(d.get("confidence", 0.0) for d in matching_detections) / len(
            matching_detections
        )
        priority = self._compute_priority(
            num_detections=len(matching_detections),
            avg_confidence=avg_conf,
            zone_priority=zone.priority,
        )

        # Build reason
        reason = self._build_reason(
            num_detections=len(matching_detections),
            avg_confidence=avg_conf,
            zone=zone,
        )

        # Collect triggering detection IDs
        triggering_ids: list[UUID] = []
        for det in matching_detections:
            det_id = det.get("id")
            if det_id is not None:
                if isinstance(det_id, UUID):
                    triggering_ids.append(det_id)
                else:
                    with contextlib.suppress(ValueError):
                        triggering_ids.append(UUID(str(det_id)))

        return TipResult(
            should_cue=True,
            reason=reason,
            target_bbox=target_bbox,
            priority=priority,
            triggering_detections=triggering_ids,
            execution_id=execution_id,
            zone_id=zone.id,
        )

    @staticmethod
    def _compute_target_bbox(
        detections: list[dict[str, Any]],
        margin_deg: float = 0.05,
    ) -> list[float]:
        """Calcula un bounding box ajustado a las detecciones.

        Parameters
        ----------
        detections:
            Detecciones con claves ``longitude`` y ``latitude``.
        margin_deg:
            Margen en grados alrededor de las detecciones.

        Returns
        -------
        list[float]
            ``[lon_min, lat_min, lon_max, lat_max]`` con margen.
        """
        lons = [d.get("longitude", 0.0) for d in detections]
        lats = [d.get("latitude", 0.0) for d in detections]

        return [
            min(lons) - margin_deg,
            min(lats) - margin_deg,
            max(lons) + margin_deg,
            max(lats) + margin_deg,
        ]

    @staticmethod
    def _compute_priority(
        num_detections: int,
        avg_confidence: float,
        zone_priority: int,
    ) -> int:
        """Determina la prioridad del cue.

        Combina la prioridad intrinseca de la zona con la cantidad
        y confianza de las detecciones.

        Parameters
        ----------
        num_detections:
            Numero de detecciones relevantes.
        avg_confidence:
            Confianza media de las detecciones.
        zone_priority:
            Prioridad intrinseca de la zona.

        Returns
        -------
        int
            Prioridad: 0=normal, 1=alta, 2=urgente.
        """
        priority = zone_priority

        # Muchas detecciones con alta confianza = urgente
        if num_detections >= 5 and avg_confidence >= 0.85:
            priority = max(priority, 2)
        elif num_detections >= 3 or avg_confidence >= 0.80:
            priority = max(priority, 1)

        return min(priority, 2)

    @staticmethod
    def _build_reason(
        num_detections: int,
        avg_confidence: float,
        zone: Zone,
    ) -> str:
        """Construye la razon textual del tip.

        Parameters
        ----------
        num_detections:
            Numero de detecciones relevantes.
        avg_confidence:
            Confianza media.
        zone:
            Zona de interes.

        Returns
        -------
        str
        """
        parts: list[str] = []

        if avg_confidence >= 0.85:
            parts.append("high_confidence")
        else:
            parts.append("moderate_confidence")

        parts.append(f"in_{zone.id}")

        if num_detections >= 5:
            parts.append("cluster_detected")

        return "_".join(parts) + f" ({num_detections} detections, avg_conf={avg_confidence:.2f})"
