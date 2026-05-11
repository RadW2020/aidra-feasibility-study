"""
CueScheduler: gestiona la cola de cues y su ejecucion.

Este modulo es complementario al APScheduler de la aplicacion.
El CueScheduler se encarga de la logica de negocio de los cues
(consultar pendientes, ejecutar pipeline, actualizar estado),
mientras que APScheduler se encarga de la programacion temporal.

Usage:
    from src.tipcue.scheduler import CueScheduler

    scheduler = CueScheduler(db=db)
    scheduler.set_engine(engine)     # Inyectar despues de crear engine
    results = await scheduler.process_pending(max_cues=5)
    cue_id = await scheduler.create_cue(
        triggered_by=exec_id,
        target_bbox=[-5.5, 35.9, -5.3, 36.1],
        priority=1,
    )
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from src.db.connection import Database
from src.db.models import TaskingEntry
from src.db.queries import (
    INSERT_CUE,
    SELECT_PENDING_CUES,
    UPDATE_CUE_AFTER_ERROR,
    UPDATE_CUE_STATUS,
)
from src.pipeline.engine import PipelineRequest, PipelineResult
from src.tipcue.zones import resolve_search_zone

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Protocol for PipelineEngine (avoid circular imports)
# ------------------------------------------------------------------


class PipelineEngineProtocol(Protocol):
    """Protocolo minimo que debe cumplir el engine del pipeline.

    Definido como Protocol para evitar importaciones circulares.
    El engine real se inyecta en runtime via ``set_engine()``.
    """

    async def run(
        self,
        request: PipelineRequest,
    ) -> PipelineResult:
        """Ejecuta el pipeline de deteccion."""
        ...


# ------------------------------------------------------------------
# CueResult
# ------------------------------------------------------------------


class CueResult:
    """Resultado de procesar un cue individual.

    Attributes
    ----------
    cue_id:
        ID del cue en ``tasking_queue``.
    execution_id:
        ID de la ejecucion generada, o ``None`` si no se ejecuto.
    status:
        Estado final: ``"completed"``, ``"no_image_found"``, ``"error"``.
    confirmed_detections:
        Numero de detecciones confirmadas en la re-observacion.
    original_detections:
        Numero de detecciones del tip original (si disponible).
    error:
        Mensaje de error si el procesamiento fallo.
    """

    def __init__(
        self,
        cue_id: UUID,
        execution_id: UUID | None = None,
        status: str = "pending",
        confirmed_detections: int | None = None,
        original_detections: int | None = None,
        error: str | None = None,
    ) -> None:
        self.cue_id = cue_id
        self.execution_id = execution_id
        self.status = status
        self.confirmed_detections = confirmed_detections
        self.original_detections = original_detections
        self.error = error

    def __repr__(self) -> str:
        return (
            f"CueResult(cue_id={self.cue_id}, status='{self.status}', "
            f"confirmed={self.confirmed_detections})"
        )


# ------------------------------------------------------------------
# CueScheduler
# ------------------------------------------------------------------

# Additional queries used only by the scheduler
SELECT_CUES_BY_STATUS = """
    SELECT *, ST_AsGeoJSON(target_bbox) AS target_bbox_geojson
    FROM tasking_queue
    WHERE ($1::text IS NULL OR status = $1)
    ORDER BY priority DESC, created_at
    LIMIT $2
"""

CHECK_RECENT_CUE = """
    SELECT COUNT(*) FROM tasking_queue
    WHERE target_zone = $1
      AND status IN ('pending', 'processing')
      AND created_at > $2
"""


class CueScheduler:
    """Gestor de la cola de cues (Tip & Cue).

    Parameters
    ----------
    db:
        Instancia de ``Database`` para acceso a PostgreSQL.
    engine:
        Motor de pipeline.  Puede inyectarse despues via
        ``set_engine()`` para evitar dependencias circulares.
    cooldown_minutes:
        Tiempo minimo entre cues para la misma zona.
    """

    def __init__(
        self,
        db: Database,
        engine: PipelineEngineProtocol | None = None,
        cooldown_minutes: int = 60,
    ) -> None:
        self._db: Database = db
        self._engine: PipelineEngineProtocol | None = engine
        self._cooldown_minutes: int = cooldown_minutes

    def set_engine(self, engine: PipelineEngineProtocol) -> None:
        """Inyecta el motor del pipeline (evita dependencia circular).

        Parameters
        ----------
        engine:
            Instancia que implementa ``PipelineEngineProtocol``.
        """
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_pending(self, max_cues: int = 5) -> list[CueResult]:
        """Procesa cues pendientes de la cola.

        1. Consulta ``tasking_queue`` con ``status='pending'``
        2. Para cada cue, ejecuta el pipeline con los parametros del tip
        3. Actualiza el estado del cue con el resultado

        Parameters
        ----------
        max_cues:
            Numero maximo de cues a procesar en esta invocacion.

        Returns
        -------
        list[CueResult]
            Resultados del procesamiento de cada cue.

        Raises
        ------
        RuntimeError
            Si el engine no ha sido inyectado.
        """
        if self._engine is None:
            raise RuntimeError(
                "PipelineEngine not set. Call set_engine() before process_pending()."
            )

        rows = await self._db.fetch(SELECT_PENDING_CUES, max_cues)
        if not rows:
            logger.debug("No pending cues in the queue")
            return []

        logger.info("Processing %d pending cue(s)", len(rows))
        results: list[CueResult] = []

        for row in rows:
            row["id"]
            result = await self._process_single_cue(row)
            results.append(result)

        return results

    async def create_cue(
        self,
        triggered_by: UUID,
        target_bbox: list[float],
        priority: int = 1,
        reason: str = "auto",
        zone: str | None = None,
    ) -> UUID:
        """Inserta un cue en ``tasking_queue``.

        Verifica cooldown: no crea el cue si ya existe uno pendiente
        en la misma zona dentro del periodo de cooldown.

        Parameters
        ----------
        triggered_by:
            UUID de la ejecucion que origino el tip.
        target_bbox:
            Bounding box de la subzona objetivo
            ``[lon_min, lat_min, lon_max, lat_max]``.
        priority:
            Prioridad del cue (0=normal, 1=alta, 2=urgente).
        reason:
            Razon textual de la creacion del cue.
        zone:
            Nombre de la zona de interes (para control de cooldown).

        Returns
        -------
        UUID
            ID del cue creado.

        Raises
        ------
        ValueError
            Si la zona esta en cooldown.
        """
        # Check cooldown
        if zone is not None:
            cooldown_cutoff = datetime.now(tz=UTC) - timedelta(minutes=self._cooldown_minutes)
            count = await self._db.fetchval(CHECK_RECENT_CUE, zone, cooldown_cutoff)
            if count and count > 0:
                raise ValueError(
                    f"Zone '{zone}' has a pending cue within the last "
                    f"{self._cooldown_minutes} minutes (cooldown active)"
                )

        # Build GeoJSON for the bbox
        bbox_geojson = self._bbox_to_geojson(target_bbox)

        # Insert cue
        cue_id = await self._db.fetchval(
            INSERT_CUE,
            triggered_by,
            None,  # triggering_detections (optional JSONB)
            bbox_geojson,
            zone,
            priority,
            reason,
        )

        logger.info(
            "Cue created: id=%s, zone=%s, priority=%d, reason='%s'",
            cue_id,
            zone,
            priority,
            reason,
        )
        return cue_id

    async def get_queue(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TaskingEntry]:
        """Lista cues de la cola, opcionalmente filtrados por estado.

        Parameters
        ----------
        status:
            Filtrar por estado (``"pending"``, ``"completed"``,
            ``"error"``, etc.).  ``None`` para todos.
        limit:
            Numero maximo de resultados.

        Returns
        -------
        list[TaskingEntry]
            Lista de entradas de la cola.
        """
        rows = await self._db.fetch(SELECT_CUES_BY_STATUS, status, limit)
        entries: list[TaskingEntry] = []

        for row in rows:
            entry = self._row_to_tasking_entry(row)
            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_single_cue(self, row: Any) -> CueResult:
        """Procesa un cue individual.

        Parameters
        ----------
        row:
            Registro de ``asyncpg.Record`` de ``tasking_queue``.

        Returns
        -------
        CueResult
        """
        cue_id: UUID = row["id"]
        triggered_by: UUID | None = row.get("triggered_by")
        target_zone: str | None = row.get("target_zone")
        target_bbox_geojson: str | None = row.get("target_bbox_geojson")

        logger.info(
            "Processing cue %s (zone=%s, priority=%d)",
            cue_id,
            target_zone,
            row.get("priority", 0),
        )

        try:
            # Parse bbox from GeoJSON for the pipeline
            bbox: list[float] | None = None
            if target_bbox_geojson:
                bbox = self._geojson_to_bbox(target_bbox_geojson)

            # Execute pipeline
            pipeline_request = PipelineRequest(
                zone=resolve_search_zone(target_zone),
                trigger_type="cue",
                triggered_by=triggered_by,
                aoi_bbox=bbox,
            )
            pipeline_result = await self._engine.run(  # type: ignore[union-attr]
                pipeline_request,
            )

            # Extract results
            execution_id = pipeline_result.execution_id
            num_detections = pipeline_result.num_detections
            result_status = pipeline_result.status

            # Parse execution_id to UUID if string
            exec_uuid: UUID | None = None
            if execution_id is not None:
                exec_uuid = (
                    execution_id if isinstance(execution_id, UUID) else UUID(str(execution_id))
                )

            # Update cue in database
            await self._db.execute(
                UPDATE_CUE_STATUS,
                cue_id,
                "completed",
                exec_uuid,
                result_status,
                num_detections,
            )

            return CueResult(
                cue_id=cue_id,
                execution_id=exec_uuid,
                status="completed",
                confirmed_detections=num_detections,
            )

        except Exception as exc:
            logger.exception("Error processing cue %s", cue_id)

            # Update cue with error status
            await self._db.execute(
                UPDATE_CUE_AFTER_ERROR,
                cue_id,
                None,
                "error",
                0,
                str(exc),
            )

            return CueResult(
                cue_id=cue_id,
                status="error",
                error=str(exc),
            )

    @staticmethod
    def _bbox_to_geojson(bbox: list[float]) -> str:
        """Convierte un bbox ``[lon_min, lat_min, lon_max, lat_max]`` a GeoJSON.

        Parameters
        ----------
        bbox:
            Bounding box en formato ``[lon_min, lat_min, lon_max, lat_max]``.

        Returns
        -------
        str
            GeoJSON Polygon como string JSON.
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
            ],
        }
        return json.dumps(geojson)

    @staticmethod
    def _geojson_to_bbox(geojson_str: str) -> list[float]:
        """Extrae un bbox de un GeoJSON Polygon string.

        Parameters
        ----------
        geojson_str:
            GeoJSON Polygon serializado como string.

        Returns
        -------
        list[float]
            ``[lon_min, lat_min, lon_max, lat_max]``.
        """
        geojson = json.loads(geojson_str)
        coords = geojson.get("coordinates", [[]])[0]
        if not coords:
            return [0.0, 0.0, 0.0, 0.0]

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]

    @staticmethod
    def _row_to_tasking_entry(row: Any) -> TaskingEntry:
        """Convierte un registro de base de datos a ``TaskingEntry``.

        Parameters
        ----------
        row:
            Registro ``asyncpg.Record`` de ``tasking_queue``.

        Returns
        -------
        TaskingEntry
        """
        bbox_geojson_str = row.get("target_bbox_geojson")
        bbox_geojson: dict[str, Any] | None = None
        if bbox_geojson_str:
            bbox_geojson = json.loads(bbox_geojson_str)

        return TaskingEntry(
            id=row["id"],
            created_at=row["created_at"],
            trigger_type=row.get("trigger_type", "auto"),
            triggered_by=row.get("triggered_by"),
            target_bbox_geojson=bbox_geojson,
            target_zone=row.get("target_zone"),
            priority=row.get("priority", 0),
            reason=row.get("reason"),
            status=row.get("status", "pending"),
            execution_id=row.get("execution_id"),
            result_status=row.get("result_status"),
            confirmed_detections=row.get("confirmed_detections"),
            attempts=row.get("attempts", 0),
        )
