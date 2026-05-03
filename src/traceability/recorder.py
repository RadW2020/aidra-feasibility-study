"""
Grabacion de registros de proveniencia en la tabla execution_log.

Cada ejecucion del pipeline genera un registro inmutable que documenta
todos los parametros de entrada, el modelo usado, las metricas de
rendimiento y el hash del resultado.  El registro es la base del
sistema de traceability de AIDRA.

Usage:
    from src.traceability.recorder import ExecutionRecorder
    from src.db.connection import db

    recorder = ExecutionRecorder(db=db)
    exec_id = await recorder.create_pending(
        image_id="S1A_IW_...",
        model_name="yolov8n-sar",
        profile="ground",
    )
    await recorder.update_status(exec_id, "success")
    record = await recorder.get(exec_id)
"""

from __future__ import annotations

import json
import logging
import socket
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from src.db.connection import Database
from src.db.models import ExecutionRecord
from src.db.queries import (
    INSERT_EXECUTION,
    SELECT_EXECUTION_BY_ID,
    SELECT_EXECUTIONS,
)

logger = logging.getLogger(__name__)


# Additional queries for the recorder
UPDATE_EXECUTION_STATUS = """
    UPDATE execution_log
    SET status = $2,
        error_message = $3
    WHERE id = $1
"""


UPDATE_EXECUTION_SAR_METADATA = """
    UPDATE execution_log
    SET incidence_angle = COALESCE($2, incidence_angle),
        polarisation    = COALESCE($3, polarisation),
        orbit_direction = COALESCE($4, orbit_direction),
        relative_orbit  = COALESCE($5, relative_orbit),
        product_type    = COALESCE($6, product_type),
        pixel_spacing   = COALESCE($7, pixel_spacing)
    WHERE id = $1
"""

UPDATE_EXECUTION_FIELDS = """
    UPDATE execution_log
    SET num_detections = COALESCE($2, num_detections),
        avg_confidence = COALESCE($3, avg_confidence),
        max_confidence = COALESCE($4, max_confidence),
        min_confidence = COALESCE($5, min_confidence),
        total_duration_ms = COALESCE($6, total_duration_ms),
        download_ms = COALESCE($7, download_ms),
        preprocessing_ms = COALESCE($8, preprocessing_ms),
        inference_ms = COALESCE($9, inference_ms),
        postprocessing_ms = COALESCE($10, postprocessing_ms),
        peak_ram_mb = COALESCE($11, peak_ram_mb),
        avg_ram_mb = COALESCE($12, avg_ram_mb),
        cpu_usage_pct = COALESCE($13, cpu_usage_pct),
        num_tiles = COALESCE($14, num_tiles),
        output_hash = COALESCE($15, output_hash),
        status = COALESCE($16, status),
        error_message = COALESCE($17, error_message)
    WHERE id = $1
"""


class ExecutionRecorder:
    """Grabador de registros de proveniencia del pipeline.

    Gestiona el ciclo de vida de un registro de ejecucion:
    ``create_pending`` -> ``update`` -> ``update_status``.
    Tambien proporciona acceso de lectura a registros existentes.

    Parameters
    ----------
    db:
        Instancia de ``Database`` para acceso a PostgreSQL.
    """

    def __init__(self, db: Database) -> None:
        self._db: Database = db

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_pending(
        self,
        image_id: str,
        image_hash: str,
        model_name: str,
        model_version: str,
        model_hash: str,
        model_size_mb: float,
        output_hash: str = "",
        image_title: str | None = None,
        image_bbox_geojson: str | None = None,
        image_sensing_date: datetime | None = None,
        image_size_mb: float | None = None,
        search_zone: str | None = None,
        model_format: str = "pytorch",
        compression_technique: str = "none",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        constraint_profile: str = "ground",
        cpu_limit: float | None = None,
        memory_limit_mb: int | None = None,
        tile_size: int = 640,
        tile_overlap: int = 64,
        trigger_type: str = "manual",
        triggered_by: UUID | None = None,
        pipeline_version: str = "1.0.0",
        input_params_hash: str | None = None,
        commit_sha: str | None = None,
        execution_id: UUID | None = None,
    ) -> UUID:
        """Crea un registro de ejecucion con status='pending'.

        Se invoca al inicio del pipeline antes de comenzar el
        procesamiento.  Los campos de resultado (detections, timing,
        metrics) se actualizan luego via ``update()``.

        Parameters
        ----------
        image_id:
            Identificador de la imagen (e.g. ID de Copernicus).
        image_hash:
            SHA256 de la imagen de entrada.
        model_name:
            Nombre del modelo (e.g. ``"yolov8n-sar"``).
        model_version:
            Version del modelo (e.g. ``"1.0.0"``).
        model_hash:
            SHA256 de los pesos del modelo.
        model_size_mb:
            Tamano del modelo en MB.
        output_hash:
            Hash del resultado (se actualiza al finalizar).
        image_title:
            Titulo de la imagen.
        image_bbox_geojson:
            Bounding box de la imagen como GeoJSON string.
        image_sensing_date:
            Fecha de captura de la imagen.
        image_size_mb:
            Tamano de la imagen en MB.
        search_zone:
            Zona de busqueda.
        model_format:
            Formato del modelo (``"pytorch"``, ``"onnx"``).
        compression_technique:
            Tecnica de compresion aplicada.
        confidence_threshold:
            Umbral de confianza usado.
        iou_threshold:
            Umbral IoU para NMS.
        constraint_profile:
            Perfil de restriccion bajo el que se ejecuta.
        cpu_limit:
            Limite de CPU del perfil.
        memory_limit_mb:
            Limite de RAM del perfil.
        tile_size:
            Tamano de tile en pixeles.
        tile_overlap:
            Solapamiento de tiles en pixeles.
        trigger_type:
            Tipo de trigger (``"manual"``, ``"scheduled"``, ``"cue"``).
        triggered_by:
            UUID del cue o ejecucion que desencadeno esta.
        pipeline_version:
            Version del pipeline.
        input_params_hash:
            Hash de los parametros de entrada.

        Returns
        -------
        UUID
            ID del registro creado.
        """
        if execution_id is None:
            execution_id = uuid4()
        hostname = socket.gethostname()

        await self._db.execute(
            INSERT_EXECUTION,
            execution_id,                   # $1  id
            image_id,                       # $2  image_id
            image_title,                    # $3  image_title
            image_hash,                     # $4  image_hash
            image_bbox_geojson,             # $5  image_bbox (GeoJSON)
            image_sensing_date,             # $6  image_sensing_date
            image_size_mb,                  # $7  image_size_mb
            search_zone,                    # $8  search_zone
            model_name,                     # $9  model_name
            model_version,                  # $10 model_version
            model_hash,                     # $11 model_hash
            model_size_mb,                  # $12 model_size_mb
            model_format,                   # $13 model_format
            compression_technique,          # $14 compression_technique
            confidence_threshold,           # $15 confidence_threshold
            iou_threshold,                  # $16 iou_threshold
            constraint_profile,             # $17 constraint_profile
            cpu_limit,                      # $18 cpu_limit
            memory_limit_mb,                # $19 memory_limit_mb
            tile_size,                      # $20 tile_size
            tile_overlap,                   # $21 tile_overlap
            0,                              # $22 num_detections
            None,                           # $23 avg_confidence
            None,                           # $24 max_confidence
            None,                           # $25 min_confidence
            None,                           # $26 total_duration_ms
            None,                           # $27 download_ms
            None,                           # $28 preprocessing_ms
            None,                           # $29 inference_ms
            None,                           # $30 postprocessing_ms
            None,                           # $31 peak_ram_mb
            None,                           # $32 avg_ram_mb
            None,                           # $33 cpu_usage_pct
            None,                           # $34 num_tiles
            output_hash,                    # $35 output_hash
            input_params_hash,              # $36 input_params_hash
            "pending",                      # $37 status
            None,                           # $38 error_message
            trigger_type,                   # $39 trigger_type
            triggered_by,                   # $40 triggered_by
            pipeline_version,               # $41 pipeline_version
            hostname,                       # $42 hostname
            commit_sha,                     # $43 commit_sha
        )

        logger.info(
            "Execution record created: id=%s, image=%s, model=%s, profile=%s",
            execution_id,
            image_id,
            model_name,
            constraint_profile,
        )
        return execution_id

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update(
        self,
        execution_id: UUID,
        num_detections: int | None = None,
        avg_confidence: float | None = None,
        max_confidence: float | None = None,
        min_confidence: float | None = None,
        total_duration_ms: float | None = None,
        download_ms: float | None = None,
        preprocessing_ms: float | None = None,
        inference_ms: float | None = None,
        postprocessing_ms: float | None = None,
        peak_ram_mb: float | None = None,
        avg_ram_mb: float | None = None,
        cpu_usage_pct: float | None = None,
        num_tiles: int | None = None,
        output_hash: str | None = None,
        status: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Actualiza campos de resultado de una ejecucion existente.

        Usa ``COALESCE`` en SQL para solo actualizar los campos que
        se proporcionan (los ``None`` mantienen el valor anterior).

        Parameters
        ----------
        execution_id:
            ID del registro a actualizar.
        num_detections:
            Numero total de detecciones.
        avg_confidence:
            Confianza media.
        max_confidence:
            Confianza maxima.
        min_confidence:
            Confianza minima.
        total_duration_ms:
            Duracion total en ms.
        download_ms:
            Tiempo de descarga en ms.
        preprocessing_ms:
            Tiempo de preproceso en ms.
        inference_ms:
            Tiempo de inferencia en ms.
        postprocessing_ms:
            Tiempo de postproceso en ms.
        peak_ram_mb:
            Pico de RAM en MB.
        avg_ram_mb:
            RAM media en MB.
        cpu_usage_pct:
            Uso de CPU en porcentaje.
        num_tiles:
            Numero de tiles procesados.
        output_hash:
            Hash SHA256 del resultado.
        status:
            Estado de la ejecucion.
        error_message:
            Mensaje de error (si aplica).
        """
        await self._db.execute(
            UPDATE_EXECUTION_FIELDS,
            execution_id,       # $1
            num_detections,     # $2
            avg_confidence,     # $3
            max_confidence,     # $4
            min_confidence,     # $5
            total_duration_ms,  # $6
            download_ms,        # $7
            preprocessing_ms,   # $8
            inference_ms,       # $9
            postprocessing_ms,  # $10
            peak_ram_mb,        # $11
            avg_ram_mb,         # $12
            cpu_usage_pct,      # $13
            num_tiles,          # $14
            output_hash,        # $15
            status,             # $16
            error_message,      # $17
        )

        logger.debug("Execution %s updated", execution_id)

    async def update_sar_metadata(
        self,
        execution_id: UUID,
        incidence_angle: float | None = None,
        polarisation: str | None = None,
        orbit_direction: str | None = None,
        relative_orbit: int | None = None,
        product_type: str | None = None,
        pixel_spacing: float | None = None,
    ) -> None:
        """Adjunta metadatos SAR (Sentinel-1) al execution_log.

        Se invoca tras ``preprocess_full`` con el dict ``metadata['sar']``
        cuando esta disponible. Cierra criterio Q3 GEOINT (metadata
        rica por escena para auditoria + filtrado).
        """
        await self._db.execute(
            UPDATE_EXECUTION_SAR_METADATA,
            execution_id,
            incidence_angle,
            polarisation,
            orbit_direction,
            relative_orbit,
            product_type,
            pixel_spacing,
        )

    async def update_status(
        self,
        execution_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Actualiza solo el estado de una ejecucion.

        Parameters
        ----------
        execution_id:
            ID del registro.
        status:
            Nuevo estado (``"pending"``, ``"running"``, ``"success"``,
            ``"error"``).
        error_message:
            Mensaje de error si ``status`` es ``"error"``.
        """
        await self._db.execute(
            UPDATE_EXECUTION_STATUS,
            execution_id,
            status,
            error_message,
        )
        logger.info("Execution %s status -> %s", execution_id, status)

    # ------------------------------------------------------------------
    # Record (convenience: create + update in one call)
    # ------------------------------------------------------------------

    async def record(self, execution: ExecutionRecord) -> UUID:
        """Inserta un registro completo de ejecucion.

        Metodo de conveniencia que inserta todos los campos de una
        vez, incluyendo los de resultado.  Util cuando todos los
        datos estan disponibles al finalizar el pipeline.

        Parameters
        ----------
        execution:
            Registro completo de la ejecucion.

        Returns
        -------
        UUID
            ID del registro insertado.
        """
        # Build GeoJSON string from bbox if available
        image_bbox_geojson: str | None = None
        if hasattr(execution, "image_bbox_geojson") and execution.image_bbox_geojson:
            if isinstance(execution.image_bbox_geojson, dict):
                image_bbox_geojson = json.dumps(execution.image_bbox_geojson)
            else:
                image_bbox_geojson = str(execution.image_bbox_geojson)

        hostname = execution.hostname or socket.gethostname()

        await self._db.execute(
            INSERT_EXECUTION,
            execution.id,                       # $1
            execution.image_id,                 # $2
            execution.image_title,              # $3
            execution.image_hash,               # $4
            image_bbox_geojson,                 # $5
            execution.image_sensing_date,       # $6
            execution.image_size_mb,            # $7
            execution.search_zone,              # $8
            execution.model_name,               # $9
            execution.model_version,            # $10
            execution.model_hash,               # $11
            execution.model_size_mb,            # $12
            execution.model_format,             # $13
            execution.compression_technique,    # $14
            execution.confidence_threshold,     # $15
            execution.iou_threshold,            # $16
            execution.constraint_profile,       # $17
            execution.cpu_limit,                # $18
            execution.memory_limit_mb,          # $19
            execution.tile_size,                # $20
            execution.tile_overlap,             # $21
            execution.num_detections,           # $22
            execution.avg_confidence,           # $23
            execution.max_confidence,           # $24
            execution.min_confidence,           # $25
            execution.total_duration_ms,        # $26
            execution.download_ms,              # $27
            execution.preprocessing_ms,         # $28
            execution.inference_ms,             # $29
            execution.postprocessing_ms,        # $30
            execution.peak_ram_mb,              # $31
            execution.avg_ram_mb,               # $32
            execution.cpu_usage_pct,            # $33
            execution.num_tiles,                # $34
            execution.output_hash,              # $35
            execution.input_params_hash,        # $36
            execution.status,                   # $37
            execution.error_message,            # $38
            execution.trigger_type,             # $39
            execution.triggered_by,             # $40
            execution.pipeline_version,         # $41
            hostname,                           # $42
            execution.commit_sha,               # $43
        )

        logger.info(
            "Full execution record inserted: id=%s, status=%s",
            execution.id,
            execution.status,
        )
        return execution.id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, execution_id: UUID) -> ExecutionRecord | None:
        """Recupera un registro de ejecucion por ID.

        Parameters
        ----------
        execution_id:
            UUID del registro.

        Returns
        -------
        ExecutionRecord | None
            El registro o ``None`` si no existe.
        """
        row = await self._db.fetchrow(SELECT_EXECUTION_BY_ID, execution_id)
        if row is None:
            return None
        return self._row_to_record(row)

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        profile: str | None = None,
        model_name: str | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ExecutionRecord]:
        """Lista registros de ejecucion con filtros opcionales.

        Parameters
        ----------
        limit:
            Numero maximo de resultados (default: 50).
        offset:
            Desplazamiento para paginacion.
        profile:
            Filtrar por perfil de restriccion.
        model_name:
            Filtrar por nombre de modelo.
        status:
            Filtrar por estado.
        date_from:
            Filtrar desde esta fecha.
        date_to:
            Filtrar hasta esta fecha.

        Returns
        -------
        list[ExecutionRecord]
            Lista de registros que cumplen los filtros.
        """
        rows = await self._db.fetch(
            SELECT_EXECUTIONS,
            profile,
            model_name,
            status,
            date_from,
            date_to,
            limit,
            offset,
        )
        return [self._row_to_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: Any) -> ExecutionRecord:
        """Convierte un registro de base de datos a ``ExecutionRecord``.

        Parameters
        ----------
        row:
            Registro ``asyncpg.Record`` de ``execution_log``.

        Returns
        -------
        ExecutionRecord
        """
        return ExecutionRecord(
            id=row["id"],
            created_at=row["created_at"],
            image_id=row["image_id"],
            image_title=row.get("image_title"),
            image_hash=row["image_hash"],
            image_sensing_date=row.get("image_sensing_date"),
            image_size_mb=row.get("image_size_mb"),
            search_zone=row.get("search_zone"),
            model_name=row["model_name"],
            model_version=row["model_version"],
            model_hash=row["model_hash"],
            model_size_mb=row["model_size_mb"],
            model_format=row.get("model_format", "pytorch"),
            compression_technique=row.get("compression_technique", "none"),
            confidence_threshold=row.get("confidence_threshold", 0.25),
            iou_threshold=row.get("iou_threshold", 0.45),
            constraint_profile=row.get("constraint_profile", "ground"),
            cpu_limit=row.get("cpu_limit"),
            memory_limit_mb=row.get("memory_limit_mb"),
            tile_size=row.get("tile_size", 640),
            tile_overlap=row.get("tile_overlap", 64),
            num_detections=row.get("num_detections", 0),
            avg_confidence=row.get("avg_confidence"),
            max_confidence=row.get("max_confidence"),
            min_confidence=row.get("min_confidence"),
            total_duration_ms=row.get("total_duration_ms"),
            download_ms=row.get("download_ms"),
            preprocessing_ms=row.get("preprocessing_ms"),
            inference_ms=row.get("inference_ms"),
            postprocessing_ms=row.get("postprocessing_ms"),
            peak_ram_mb=row.get("peak_ram_mb"),
            avg_ram_mb=row.get("avg_ram_mb"),
            cpu_usage_pct=row.get("cpu_usage_pct"),
            num_tiles=row.get("num_tiles"),
            output_hash=row.get("output_hash", ""),
            input_params_hash=row.get("input_params_hash"),
            commit_sha=row.get("commit_sha"),
            incidence_angle=row.get("incidence_angle"),
            polarisation=row.get("polarisation"),
            orbit_direction=row.get("orbit_direction"),
            relative_orbit=row.get("relative_orbit"),
            product_type=row.get("product_type"),
            pixel_spacing=row.get("pixel_spacing"),
            status=row.get("status", "pending"),
            error_message=row.get("error_message"),
            trigger_type=row.get("trigger_type", "manual"),
            triggered_by=row.get("triggered_by"),
            pipeline_version=row.get("pipeline_version", "1.0.0"),
            hostname=row.get("hostname"),
        )
