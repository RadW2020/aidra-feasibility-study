"""
Orquestador del pipeline completo de deteccion de barcos.

Flujo de ejecucion (15 pasos):
1.  Validar parametros de entrada
2.  Autenticar con Copernicus (si no hay token vigente)
3.  Buscar imagen en Copernicus (o usar image_id proporcionado)
4.  Descargar imagen
5.  Calcular hash SHA256 de la imagen
6.  Extraer producto descargado
7.  Preprocesar (calibrar, filtrar, tilear)
8.  Ejecutar deteccion (CFAR + YOLO) bajo perfil de restriccion
9.  Calcular hash SHA256 del resultado
10. Registrar en execution_log (create_pending + update)
11. Guardar detecciones en PostGIS
12. Evaluar Tip & Cue (si habilitado)
13. Emitir metricas Prometheus
14. Limpiar archivos temporales
15. Retornar resultado

Manejo de errores:
- Cada paso tiene timeout individual
- Si un paso falla, se registra en execution_log con status="error"
- La imagen descargada se limpia incluso si el pipeline falla
- Los errores se loguean en Loki con contexto completo
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import logging
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from src.models.manager import ModelManager

from pydantic import BaseModel, Field

from src.config import Settings
from src.observability.loki_logger import StructuredLogger
from src.observability.prometheus_metrics import (
    CPU_USAGE_PERCENT,
    DETECTIONS_TOTAL,
    DOWNLOAD_DURATION,
    INFERENCE_DURATION,
    MODEL_SIZE_MB,
    PEAK_RAM_MB,
    PIPELINE_RUNS_TOTAL,
    SCENES_PROCESSED_TOTAL,
    TIPS_GENERATED_TOTAL,
)
from src.pipeline.detection import Detection, DetectionEngine, DetectionMetrics, DetectionResult
from src.pipeline.ingestion import SEARCH_ZONES, CopernicusSearchResult, ImageIngester
from src.pipeline.postprocessing import compute_detection_stats
from src.pipeline.preprocessing import preprocess_full
from src.profiles.manager import ProfileManager
from src.tipcue.evaluator import TipEvaluator, TipResult
from src.traceability.hasher import (
    compute_input_params_hash,
    compute_result_hash,
    compute_sha256,
    get_commit_sha,
)
from src.traceability.recorder import ExecutionRecorder

logger = logging.getLogger("aidra.pipeline")


# ====================================================================
# Exception hierarchy
# ====================================================================


class PipelineError(Exception):
    """Error generico del pipeline."""


class IngestionError(PipelineError):
    """Error durante la descarga/busqueda de imagenes."""


class AuthenticationError(IngestionError):
    """Error de autenticacion con Copernicus."""


class PreprocessingError(PipelineError):
    """Error durante el preprocesamiento SAR."""


class DetectionError(PipelineError):
    """Error durante la inferencia."""


class ProfileError(PipelineError):
    """Error al ejecutar bajo perfil de restriccion."""


class OOMError(ProfileError):
    """Out of memory bajo perfil de restriccion."""


class TimeoutError(ProfileError):
    """Timeout bajo perfil de restriccion."""


# ====================================================================
# Timeouts and retry configuration
# ====================================================================

TIMEOUTS: dict[str, int] = {
    "copernicus_auth": 30,
    "copernicus_search": 60,
    "copernicus_download": 1800,
    "preprocessing": 300,
    "inference_per_tile": 60,
    "inference_total": 3600,
    "pipeline_total": 1800,
}

RETRY_CONFIG: dict[str, dict[str, Any]] = {
    "copernicus_auth": {"max_retries": 3, "backoff_seconds": [2, 5, 10]},
    "copernicus_search": {"max_retries": 2, "backoff_seconds": [3, 10]},
    "copernicus_download": {"max_retries": 2, "backoff_seconds": [5, 15]},
    "db_write": {"max_retries": 3, "backoff_seconds": [1, 2, 5]},
}


# ====================================================================
# Pydantic request / result models
# ====================================================================


class PipelineRequest(BaseModel):
    """Solicitud de ejecucion del pipeline.

    Attributes
    ----------
    zone:
        Zona de busqueda de imagenes (clave de ``SEARCH_ZONES``).
    model:
        Variante de modelo a usar (e.g. ``"yolov8n-sar"``).
    profile:
        Perfil de restriccion de recursos (e.g. ``"ground"``).
    image_id:
        ID especifico de imagen Copernicus.  Cuando es ``None`` se
        busca la imagen mas reciente en la zona.
    aoi_bbox:
        Sub-area de interes ``[lon_min, lat_min, lon_max, lat_max]``.
    confidence_threshold:
        Umbral de confianza minimo para detecciones.
    iou_threshold:
        Umbral IoU para NMS.
    date_from:
        Inicio del rango de busqueda.
    date_to:
        Fin del rango de busqueda.
    trigger_type:
        Tipo de trigger (``"manual"``, ``"scheduled"``, ``"cue"``).
    triggered_by:
        UUID de la ejecucion que genero un cue que disparo esta.
    """

    zone: str = "gibraltar"
    model: str = "yolov8n-sar"
    profile: str = "ground"
    sensor: str = "s1"  # "s1" for Sentinel-1 SAR, "s2" for Sentinel-2 optical
    image_id: str | None = None
    aoi_bbox: list[float] | None = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    date_from: datetime | None = None
    date_to: datetime | None = None
    trigger_type: str = "manual"
    triggered_by: UUID | None = None


class PipelineResult(BaseModel):
    """Resultado de una ejecucion del pipeline.

    Attributes
    ----------
    execution_id:
        UUID del registro en ``execution_log``.
    status:
        Estado final (``"success"`` o ``"error"``).
    detections:
        Lista de detecciones producidas.
    metrics:
        Metricas de rendimiento de la deteccion.
    num_detections:
        Numero total de detecciones.
    output_hash:
        SHA256 de las detecciones serializadas.
    total_duration_ms:
        Duracion total del pipeline en milisegundos.
    error:
        Mensaje de error si ``status`` es ``"error"``.
    """

    execution_id: UUID
    status: str
    detections: list[Detection] = Field(default_factory=list)
    metrics: DetectionMetrics = Field(default_factory=DetectionMetrics)
    num_detections: int = 0
    output_hash: str = ""
    total_duration_ms: float = 0.0
    error: str | None = None


# ====================================================================
# Pipeline engine
# ====================================================================


class PipelineEngine:
    """Orquestador central del pipeline de deteccion de barcos.

    Coordina todos los modulos del sistema (ingesta, preprocesamiento,
    deteccion, postprocesamiento, traceability, tip & cue, metricas)
    en una secuencia de 15 pasos con manejo completo de errores.

    Parameters
    ----------
    ingester:
        Modulo de ingesta de imagenes Copernicus.
    detector:
        Motor de deteccion (CFAR + YOLO).
    recorder:
        Grabador de registros de proveniencia.
    profile_manager:
        Gestor de perfiles de restriccion de recursos.
    tip_evaluator:
        Evaluador de Tip & Cue (puede ser ``None`` si deshabilitado).
    config:
        Configuracion centralizada del sistema.
    """

    def __init__(
        self,
        ingester: ImageIngester,
        detector_engine: DetectionEngine,
        model_manager: ModelManager,
        recorder: ExecutionRecorder,
        profile_manager: ProfileManager,
        tip_evaluator: TipEvaluator | None,
        config: Settings,
    ) -> None:
        self.ingester = ingester
        self.detector_engine = detector_engine
        self.model_manager = model_manager
        self.recorder = recorder
        self.profile_manager = profile_manager
        self.tip_evaluator = tip_evaluator
        self.config = config
        self._log = StructuredLogger("aidra.pipeline.engine")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: PipelineRequest, execution_id: UUID | None = None) -> PipelineResult:
        """Ejecuta el pipeline completo de deteccion.

        Pasos:
        1. Validar parametros de entrada.
        2. Buscar imagen en Copernicus (o por ID directo).
        3. Descargar imagen.
        4. Calcular hash SHA256 de la imagen.
        5. Extraer el producto descargado.
        6. Preprocesar (calibrar, filtrar, tilear).
        7. Ejecutar deteccion bajo perfil de restriccion.
        8. Calcular hash SHA256 del resultado.
        9. Registrar en execution_log.
        10. Guardar detecciones en PostGIS.
        11. Evaluar Tip & Cue.
        12. Emitir metricas Prometheus.
        13. Limpiar archivos temporales.

        Parameters
        ----------
        request:
            Parametros de la solicitud de pipeline.

        Returns
        -------
        PipelineResult
            Resultado con detecciones, metricas y metadatos de la
            ejecucion.

        Raises
        ------
        PipelineError
            Si alguno de los pasos criticos falla de forma
            irrecuperable.
        """
        image_path: Path | None = None
        extract_path: Path | None = None
        start_time = time.monotonic()

        try:
            # ---- Step 1: Validate request ----
            self._validate_request(request)

            # ---- Step 2: Load Model Dynamically ----
            model_name = request.model or self.config.default_model
            detector = await self.model_manager.get_model(
                name=model_name,
                confidence_threshold=request.confidence_threshold,
                iou_threshold=request.iou_threshold,
            )
            model_info = detector.get_model_info()

            # ---- Step 2b: Create pending execution record ----
            self._log.info(
                "Pipeline starting",
                extra={
                    "zone": request.zone,
                    "profile": request.profile,
                    "model": model_name,
                    "trigger_type": request.trigger_type,
                },
            )

            input_params_hash = compute_input_params_hash(
                self._build_input_params(request, model_info)
            )
            commit_sha = get_commit_sha()

            execution_id = await self.recorder.create_pending(
                image_id=request.image_id or "pending",
                image_hash="pending",
                model_name=model_info["name"],
                model_version=model_info.get("version", "1.0.0"),
                model_hash=model_info["hash"],
                model_size_mb=model_info["size_mb"],
                search_zone=request.zone,
                model_format=model_info["format"],
                confidence_threshold=request.confidence_threshold,
                iou_threshold=request.iou_threshold,
                constraint_profile=request.profile,
                tile_size=self.config.tile_size,
                tile_overlap=self.config.tile_overlap,
                trigger_type=request.trigger_type,
                triggered_by=request.triggered_by,
                input_params_hash=input_params_hash,
                commit_sha=commit_sha,
                execution_id=execution_id,
            )
            await self.recorder.update_status(execution_id, "running")

            # ---- Step 3: Search for image ----
            product = await self._search_image(request)

            # ---- Step 4: Download image ----
            download_start = time.monotonic()
            image_path = await self._download_with_retry(product)
            download_ms = (time.monotonic() - download_start) * 1000.0

            DOWNLOAD_DURATION.observe(download_ms / 1000.0)

            # ---- Step 5: Hash the downloaded image ----
            image_hash = compute_sha256(image_path)

            # ---- Step 6: Extract the product ----
            extract_path = await self.ingester.extract(image_path)

            # ---- Step 6b: Update execution record with real image metadata ----
            await self.recorder.update(
                execution_id=execution_id,
                status="running",
            )
            # Use a direct DB update via recorder for image fields
            # that were placeholders at creation time
            from src.db.connection import db as _db

            footprint_geojson = (
                json.dumps(product.footprint)
                if product.footprint
                else None
            )

            await _db.execute(
                """
                UPDATE execution_log
                SET image_id = $2,
                    image_hash = $3,
                    image_title = $4,
                    image_sensing_date = $5,
                    image_size_mb = $6,
                    image_bbox = ST_GeomFromGeoJSON($7)
                WHERE id = $1
                """,
                execution_id,
                product.product_id,
                image_hash,
                product.title,
                product.sensing_date,
                product.size_mb,
                footprint_geojson,
            )

            # ---- Step 7: Preprocess (calibrate, filter, tile) ----
            preprocess_start = time.monotonic()
            try:
                preprocessed = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: preprocess_full(
                            product_dir=extract_path,
                            aoi_bbox=request.aoi_bbox,
                            tile_size=self.config.tile_size,
                            overlap=self.config.tile_overlap,
                        ),
                    ),
                    timeout=TIMEOUTS["preprocessing"],
                )
            except builtins.TimeoutError as exc:
                raise PreprocessingError(
                    f"Preprocessing timed out after {TIMEOUTS['preprocessing']}s"
                ) from exc
            except Exception as exc:
                raise PreprocessingError(f"Preprocessing failed: {exc}") from exc
            preprocessing_ms = (time.monotonic() - preprocess_start) * 1000.0

            tiles = preprocessed["tiles"]
            num_tiles = preprocessed["metadata"]["num_tiles"]

            # Store metadata for footprint clipping in _save_detections
            self._current_metadata = preprocessed.get("metadata", {})

            # ---- Step 7a: Quality gate (I-SAR-1) ----
            scene_quality = self._current_metadata.get("quality", "valid")
            with contextlib.suppress(Exception):
                SCENES_PROCESSED_TOTAL.labels(quality=scene_quality).inc()
            if scene_quality == "invalid":
                quality_reasons = self._current_metadata.get("quality_reasons", [])
                self._log.warning(
                    "Skipping detection: scene quality=invalid",
                    extra={
                        "execution_id": str(execution_id),
                        "reasons": quality_reasons,
                    },
                )
                await self.recorder.update_status(
                    execution_id,
                    "invalid",
                    error_message=f"quality_invalid: {','.join(quality_reasons)}",
                )
                return PipelineResult(
                    execution_id=execution_id,
                    status="invalid",
                    num_detections=0,
                    detections=[],
                    output_hash="",
                    error=f"quality_invalid: {','.join(quality_reasons)}",
                )

            # ---- Step 7b: Persist SAR metadata (Q3 GEOINT) ----
            sar_meta = self._current_metadata.get("sar") or {}
            if sar_meta:
                try:
                    await self.recorder.update_sar_metadata(
                        execution_id=execution_id,
                        incidence_angle=sar_meta.get("incidence_angle"),
                        polarisation=sar_meta.get("polarisation"),
                        orbit_direction=sar_meta.get("orbit_direction"),
                        relative_orbit=sar_meta.get("relative_orbit"),
                        product_type=sar_meta.get("product_type"),
                        pixel_spacing=sar_meta.get("pixel_spacing"),
                    )
                except Exception as exc:
                    self._log.warning(
                        "Could not persist SAR metadata",
                        extra={"execution_id": str(execution_id), "error": str(exc)},
                    )

            # ---- Step 8: Detect (under profile constraints) ----
            detection_result = await self._run_detection(
                tiles=tiles,
                profile=request.profile,
                detector=detector,
                sensor=request.sensor,
            )

            inference_ms = detection_result.metrics.total_inference_ms

            # ---- Step 9: Hash the result ----
            output_hash = compute_result_hash(
                [d.model_dump() for d in detection_result.detections]
            )

            # ---- Step 10: Compute stats and update execution record ----
            total_ms = (time.monotonic() - start_time) * 1000.0

            det_stats = compute_detection_stats(
                [d.model_dump() for d in detection_result.detections]
            )

            await self.recorder.update(
                execution_id=execution_id,
                num_detections=len(detection_result.detections),
                avg_confidence=det_stats.get("avg_confidence"),
                max_confidence=det_stats.get("max_confidence"),
                min_confidence=det_stats.get("min_confidence"),
                total_duration_ms=total_ms,
                download_ms=download_ms,
                preprocessing_ms=preprocessing_ms,
                inference_ms=inference_ms,
                peak_ram_mb=detection_result.metrics.peak_ram_mb,
                cpu_usage_pct=detection_result.metrics.cpu_percent,
                num_tiles=num_tiles,
                output_hash=output_hash,
                status="success",
                notes=detection_result.notes,
            )

            # ---- Step 11: Flag anomalies + thumbnails + save to PostGIS ----
            from src.pipeline.postprocessing import flag_cluster_anomaly
            from src.pipeline.thumbnails import generate_thumbnails

            flag_cluster_anomaly(detection_result.detections)

            # Wow effect #1: SAR crop per detection. Skipea detecciones
            # filtradas por anomaly/land si configurado, para no inflar
            # el almacenamiento.
            try:
                thumbnails_root = Path(
                    getattr(self.config, "thumbnails_dir", "/data/thumbnails")
                )
                generate_thumbnails(
                    detections=detection_result.detections,
                    tiles=tiles,
                    execution_id=execution_id,
                    out_root=thumbnails_root,
                )
            except Exception as exc:
                self._log.warning(
                    "Thumbnail generation failed (continuing)",
                    extra={"execution_id": str(execution_id), "error": str(exc)},
                )

            await self._save_detections(execution_id, detection_result.detections)

            # ---- Step 12: Evaluate Tip & Cue ----
            if self.tip_evaluator and self.config.tipcue_enabled:
                detection_dicts = [d.model_dump() for d in detection_result.detections]
                # Enrich with lon/lat from center_geo for evaluator
                for dd in detection_dicts:
                    center = dd.get("center_geo", [])
                    if len(center) == 2:
                        dd["longitude"] = center[0]
                        dd["latitude"] = center[1]
                tips = self.tip_evaluator.evaluate(
                    detections=detection_dicts,
                    execution_id=execution_id,
                )
                for tip in tips:
                    if tip.should_cue:
                        await self._create_cue(tip)

            # ---- Step 13: Emit Prometheus metrics ----
            self._emit_metrics(
                profile=request.profile,
                model=request.model,
                detection_result=detection_result,
                status="success",
                execution_id=execution_id,
            )

            self._log.info(
                "Pipeline completed successfully",
                extra={
                    "execution_id": str(execution_id),
                    "profile": request.profile,
                    "num_detections": len(detection_result.detections),
                    "total_ms": round(total_ms, 1),
                },
            )

            return PipelineResult(
                execution_id=execution_id,
                status="success",
                detections=detection_result.detections,
                metrics=detection_result.metrics,
                num_detections=len(detection_result.detections),
                output_hash=output_hash,
                total_duration_ms=total_ms,
            )

        except Exception as exc:
            # ---- Error handling: always record status ----
            error_msg = f"{type(exc).__name__}: {exc}"

            if execution_id is not None:
                await self._safe_update_status(execution_id, "error", error_msg)

            self._log.error(
                "Pipeline failed",
                extra={
                    "execution_id": str(execution_id) if execution_id else None,
                    "error": error_msg,
                    "profile": request.profile,
                },
                exc_info=True,
            )

            # Emit error metric — attach run_id exemplar so a failure
            # spike in Grafana links straight to the matching log
            # bundle in Loki and to the row in execution_log.
            _err_exemplar = (
                {"trace_id": str(execution_id)}
                if execution_id is not None
                else None
            )
            _err_metric = PIPELINE_RUNS_TOTAL.labels(
                profile=request.profile,
                model_variant=request.model,
                status=self._classify_error(exc),
            )
            if _err_exemplar is not None:
                _err_metric.inc(1, exemplar=_err_exemplar)
            else:
                _err_metric.inc()

            if isinstance(exc, PipelineError):
                raise
            raise PipelineError(error_msg) from exc

        finally:
            # ---- Step 14: Cleanup temporary files ----
            if extract_path and extract_path.exists():
                await self._cleanup(extract_path)
            if image_path and image_path.exists():
                await self._cleanup(image_path)

    async def run_all_profiles(
        self, request: PipelineRequest,
        execution_ids: dict[str, UUID] | None = None,
    ) -> dict[str, PipelineResult]:
        """Ejecuta el pipeline con todos los perfiles sobre la misma imagen.

        Descarga y preprocesa la imagen una sola vez, luego ejecuta
        la deteccion con cada perfil de restriccion disponible.

        Parameters
        ----------
        request:
            Parametros base del pipeline.  El campo ``profile`` se
            sobreescribe para cada perfil.

        Returns
        -------
        dict[str, PipelineResult]
            Diccionario con resultados indexados por nombre de perfil.
        """
        results: dict[str, PipelineResult] = {}
        image_path: Path | None = None
        extract_path: Path | None = None

        try:
            # Search and download once
            product = await self._search_image(request)

            download_start = time.monotonic()
            image_path = await self._download_with_retry(product)
            download_ms = (time.monotonic() - download_start) * 1000.0

            image_hash = compute_sha256(image_path)
            extract_path = await self.ingester.extract(image_path)

            # Preprocess once
            preprocessed = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: preprocess_full(
                    product_dir=extract_path,
                    aoi_bbox=request.aoi_bbox,
                    tile_size=self.config.tile_size,
                    overlap=self.config.tile_overlap,
                ),
            )

            tiles = preprocessed["tiles"]
            num_tiles = preprocessed["metadata"]["num_tiles"]

            # Run detection for each profile
            from src.profiles.definitions import PROFILE_ORDER

            for profile_name in PROFILE_ORDER:
                if profile_name not in self.profile_manager.profiles:
                    continue

                self._log.info(
                    "Running detection with profile",
                    extra={"profile": profile_name},
                )

                start_time = time.monotonic()

                try:
                    # Create execution record for this profile
                    pre_id = (execution_ids or {}).get(profile_name)
                    footprint_geojson_str = (
                        json.dumps(product.footprint)
                        if product.footprint
                        else None
                    )
                    # Load detector for this model
                    det = await self.model_manager.get_model(request.model)
                    det_info = det.get_model_info()
                    execution_id = await self.recorder.create_pending(
                        image_id=product.product_id,
                        image_hash=image_hash,
                        model_name=det_info.get("name", request.model),
                        model_version=det_info.get("version", "1.0.0"),
                        model_hash=det_info.get("hash", "unknown"),
                        model_size_mb=det_info.get("size_mb", 0.0),
                        image_title=product.title,
                        image_bbox_geojson=footprint_geojson_str,
                        image_sensing_date=product.sensing_date,
                        image_size_mb=product.size_mb,
                        search_zone=request.zone,
                        model_format=det_info.get("format", "pytorch"),
                        confidence_threshold=request.confidence_threshold,
                        iou_threshold=request.iou_threshold,
                        constraint_profile=profile_name,
                        tile_size=self.config.tile_size,
                        tile_overlap=self.config.tile_overlap,
                        trigger_type=request.trigger_type,
                        triggered_by=request.triggered_by,
                        execution_id=pre_id,
                    )
                    await self.recorder.update_status(execution_id, "running")

                    # Run detection under this profile
                    detection_result = await self._run_detection(
                        tiles=tiles,
                        profile=profile_name,
                        detector=det,
                        sensor=request.sensor,
                    )

                    output_hash = compute_result_hash(
                        [d.model_dump() for d in detection_result.detections]
                    )
                    total_ms = (time.monotonic() - start_time) * 1000.0

                    det_stats = compute_detection_stats(
                        [d.model_dump() for d in detection_result.detections]
                    )

                    await self.recorder.update(
                        execution_id=execution_id,
                        num_detections=len(detection_result.detections),
                        avg_confidence=det_stats.get("avg_confidence"),
                        max_confidence=det_stats.get("max_confidence"),
                        min_confidence=det_stats.get("min_confidence"),
                        total_duration_ms=total_ms,
                        download_ms=download_ms,
                        inference_ms=detection_result.metrics.total_inference_ms,
                        peak_ram_mb=detection_result.metrics.peak_ram_mb,
                        cpu_usage_pct=detection_result.metrics.cpu_percent,
                        num_tiles=num_tiles,
                        output_hash=output_hash,
                        status="success",
                        notes=detection_result.notes,
                    )

                    await self._save_detections(
                        execution_id, detection_result.detections
                    )

                    self._emit_metrics(
                        profile=profile_name,
                        model=request.model,
                        detection_result=detection_result,
                        status="success",
                        execution_id=execution_id,
                    )

                    results[profile_name] = PipelineResult(
                        execution_id=execution_id,
                        status="success",
                        detections=detection_result.detections,
                        metrics=detection_result.metrics,
                        num_detections=len(detection_result.detections),
                        output_hash=output_hash,
                        total_duration_ms=total_ms,
                    )

                except MemoryError:
                    total_ms = (time.monotonic() - start_time) * 1000.0
                    await self._safe_update_status(
                        execution_id, "error", "OOM under profile"
                    )
                    PIPELINE_RUNS_TOTAL.labels(
                        profile=profile_name,
                        model_variant=request.model,
                        status="oom",
                    ).inc(1, exemplar={"trace_id": str(execution_id)})
                    results[profile_name] = PipelineResult(
                        execution_id=execution_id,
                        status="error",
                        error="OOM",
                        total_duration_ms=total_ms,
                    )

                except Exception as exc:
                    total_ms = (time.monotonic() - start_time) * 1000.0
                    error_msg = f"{type(exc).__name__}: {exc}"
                    await self._safe_update_status(
                        execution_id, "error", error_msg
                    )
                    PIPELINE_RUNS_TOTAL.labels(
                        profile=profile_name,
                        model_variant=request.model,
                        status="error",
                    ).inc(1, exemplar={"trace_id": str(execution_id)})
                    results[profile_name] = PipelineResult(
                        execution_id=execution_id,
                        status="error",
                        error=error_msg,
                        total_duration_ms=total_ms,
                    )

        finally:
            if extract_path and extract_path.exists():
                await self._cleanup(extract_path)
            if image_path and image_path.exists():
                await self._cleanup(image_path)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_input_params(
        self, request: PipelineRequest, model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Construye el diccionario que alimenta input_params_hash.

        Recoge los parametros que afectan al resultado del run: request
        (zona, AOI, fechas, thresholds, modelo) + Settings relevantes
        (tile size/overlap, perfil por defecto). El hash de este dict
        cierra I-TRACE-4 junto con commit_sha.
        """
        return {
            "request": {
                "zone": request.zone,
                "model": request.model,
                "profile": request.profile,
                "image_id": request.image_id,
                "aoi_bbox": request.aoi_bbox,
                "confidence_threshold": request.confidence_threshold,
                "iou_threshold": request.iou_threshold,
                "date_from": request.date_from.isoformat()
                if request.date_from
                else None,
                "date_to": request.date_to.isoformat()
                if request.date_to
                else None,
                "trigger_type": request.trigger_type,
            },
            "settings": {
                "tile_size": self.config.tile_size,
                "tile_overlap": self.config.tile_overlap,
                "default_model": self.config.default_model,
                "default_profile": self.config.default_profile,
                "cfar_guard_size": self.config.cfar_guard_size,
                "cfar_training_size": self.config.cfar_training_size,
                "cfar_pfa": self.config.cfar_pfa,
                "pipeline_version": "1.0.0",
            },
            "model": {
                "name": model_info.get("name"),
                "version": model_info.get("version"),
                "hash": model_info.get("hash"),
                "format": model_info.get("format"),
            },
        }

    def _validate_request(self, request: PipelineRequest) -> None:
        """Validate that request parameters are sane.

        Parameters
        ----------
        request:
            Pipeline request to validate.

        Raises
        ------
        PipelineError
            If the zone is unknown or bbox is malformed.
        """
        if request.image_id is None and request.zone not in SEARCH_ZONES:
            raise PipelineError(
                f"Unknown search zone '{request.zone}'. "
                f"Available zones: {list(SEARCH_ZONES.keys())}"
            )

        if request.aoi_bbox is not None and len(request.aoi_bbox) != 4:
            raise PipelineError(
                "aoi_bbox must have exactly 4 elements "
                "[lon_min, lat_min, lon_max, lat_max]"
            )

        if not 0.0 <= request.confidence_threshold <= 1.0:
            raise PipelineError(
                f"confidence_threshold must be in [0, 1], got {request.confidence_threshold}"
            )

    async def _search_image(
        self, request: PipelineRequest
    ) -> CopernicusSearchResult:
        """Search Copernicus for the target image.

        If ``request.image_id`` is provided, the product is fetched
        directly by ID instead of searching by zone/dates.

        Parameters
        ----------
        request:
            Pipeline request containing zone and date filters, or a
            specific ``image_id``.

        Returns
        -------
        CopernicusSearchResult
            The best-matching product.

        Raises
        ------
        IngestionError
            If no products are found or the ID lookup fails.
        """
        # --- Direct lookup by product ID ---
        if request.image_id:
            try:
                product = await asyncio.wait_for(
                    self.ingester.search_by_id(request.image_id),
                    timeout=TIMEOUTS["copernicus_search"],
                )
                return product
            except builtins.TimeoutError as exc:
                raise IngestionError(
                    f"Copernicus ID lookup timed out after {TIMEOUTS['copernicus_search']}s"
                ) from exc
            except Exception as exc:
                raise IngestionError(
                    f"Copernicus ID lookup failed for '{request.image_id}': {exc}"
                ) from exc

        # --- Search by zone and date range ---
        try:
            zone_info = SEARCH_ZONES.get(request.zone, SEARCH_ZONES["gibraltar"])
            now = datetime.now(tz=UTC)

            date_from = request.date_from or (now - timedelta(days=7))
            date_to = request.date_to or now

            search_results = await asyncio.wait_for(
                self.ingester.search(
                    bbox=zone_info["bbox"],
                    start_date=date_from.strftime("%Y-%m-%d"),
                    end_date=date_to.strftime("%Y-%m-%d"),
                    max_results=1,
                    sensor=request.sensor,
                ),
                timeout=TIMEOUTS["copernicus_search"],
            )
        except builtins.TimeoutError as exc:
            raise IngestionError(
                f"Copernicus search timed out after {TIMEOUTS['copernicus_search']}s"
            ) from exc
        except Exception as exc:
            raise IngestionError(f"Copernicus search failed: {exc}") from exc

        if not search_results:
            raise IngestionError(
                f"No images found for zone '{request.zone}' "
                f"in the specified date range"
            )

        return search_results[0]

    async def _download_with_retry(
        self, product: CopernicusSearchResult
    ) -> Path:
        """Download a product with retry logic.

        Parameters
        ----------
        product:
            Copernicus product to download.

        Returns
        -------
        Path
            Local path to the downloaded zip file.

        Raises
        ------
        IngestionError
            If all download attempts fail.
        """
        retry_cfg = RETRY_CONFIG["copernicus_download"]
        last_error: Exception | None = None

        for attempt in range(retry_cfg["max_retries"] + 1):
            try:
                path = await asyncio.wait_for(
                    self.ingester.download(product),
                    timeout=TIMEOUTS["copernicus_download"],
                )
                return path
            except builtins.TimeoutError:
                last_error = IngestionError(
                    f"Download timed out after {TIMEOUTS['copernicus_download']}s"
                )
            except Exception as exc:
                last_error = exc

            if attempt < retry_cfg["max_retries"]:
                backoff = retry_cfg["backoff_seconds"][
                    min(attempt, len(retry_cfg["backoff_seconds"]) - 1)
                ]
                self._log.warning(
                    "Download attempt failed, retrying",
                    extra={
                        "attempt": attempt + 1,
                        "backoff_seconds": backoff,
                        "error": str(last_error),
                    },
                )
                await asyncio.sleep(backoff)

        raise IngestionError(
            f"Download failed after {retry_cfg['max_retries'] + 1} attempts: "
            f"{last_error}"
        )

    async def _run_detection(
        self,
        tiles: list[dict[str, Any]],
        profile: str,
        detector: Any = None,
        sensor: str = "s1",
    ) -> DetectionResult:
        """Execute detection, optionally under resource constraints.

        Parameters
        ----------
        tiles:
            Preprocessed tiles from the SAR image.
        profile:
            Name of the constraint profile.

        Returns
        -------
        DetectionResult
            Detection results with metrics.

        Raises
        ------
        DetectionError
            If detection fails.
        OOMError
            If the process runs out of memory under a profile.
        TimeoutError
            If detection exceeds the configured timeout.
        """
        # Prepare tiles in the format expected by DetectionEngine
        # The preprocessing module outputs tiles with 'array' key;
        # DetectionEngine expects 'data' key.
        # The preprocessing module now stores the GLOBAL pixel→lon/lat
        # affine on each S1 tile (rotation-aware).  Prefer that — fall
        # back to a per-tile axis-aligned approximation for S2 (whose
        # geo_transform is in UTM, not WGS-84).
        is_s1 = sensor.lower() == "s1"
        formatted_tiles: list[dict[str, Any]] = []
        for idx, tile in enumerate(tiles):
            formatted_tile: dict[str, Any] = {
                "data": tile.get("array", tile.get("data")),
                "tile_index": idx,
                "row_offset": tile.get("row_offset", 0),
                "col_offset": tile.get("col_offset", 0),
            }
            global_gt = tile.get("geo_transform")
            if is_s1 and global_gt is not None:
                # Use the rotation-aware global affine directly; detection
                # adds (col_offset, row_offset) to the bbox before applying.
                formatted_tile["geo_transform"] = tuple(global_gt)
            elif "geo_bounds" in tile:
                gb = tile["geo_bounds"]
                if gb.get("lon_min") is not None and gb.get("lat_max") is not None:
                    tile_size = tile.get("array", tile.get("data")).shape[0]
                    px_x = (gb["lon_max"] - gb["lon_min"]) / tile_size if tile_size else 1.0
                    px_y = (gb["lat_min"] - gb["lat_max"]) / tile_size if tile_size else -1.0
                    # Per-tile axis-aligned fallback: row_offset/col_offset
                    # are absorbed into the per-tile origin (lon_min/lat_max),
                    # so detection geocoding must NOT add them again.
                    formatted_tile["geo_transform"] = (
                        gb["lon_min"], px_x, 0.0,
                        gb["lat_max"], 0.0, px_y,
                    )
                    formatted_tile["row_offset"] = 0
                    formatted_tile["col_offset"] = 0
            formatted_tiles.append(formatted_tile)

        # Run CFAR for SAR sensors regardless of which YOLO weight loaded.
        # Tying this to the sensor (not the model name) keeps the SAR
        # detection path alive when only a generic YOLO fallback is on disk.
        cfar_detector = None
        model_name = str(detector.get_model_info().get("name", "")).lower()
        wants_cfar = sensor.lower() == "s1" or "sar" in model_name
        if wants_cfar:
            try:
                cfar_detector = await self.model_manager.get_model("cfar-default")
            except Exception:
                self._log.warning("CFAR detector requested but not available")

        try:
            if profile != "ground":
                profiled_result = await self.profile_manager.run_with_profile(
                    profile_name=profile,
                    pipeline_fn=self.detector_engine.run,
                    tiles=formatted_tiles,
                    detector=detector,
                    cfar=cfar_detector,
                    constraint_profile=profile,
                )

                if not profiled_result.success:
                    if profiled_result.error == "OOM":
                        raise OOMError(
                            f"Out of memory under profile '{profile}'"
                        )
                    if profiled_result.error == "timeout":
                        raise TimeoutError(
                            f"Timeout under profile '{profile}'"
                        )
                    raise DetectionError(
                        f"Detection failed under profile '{profile}': "
                        f"{profiled_result.error}"
                    )

                # Extract DetectionResult from raw result and carry
                # any profile-level notes (e.g. memory budget breach)
                # forward so the recorder can persist them.
                raw = profiled_result.raw_result
                if isinstance(raw, DetectionResult):
                    if profiled_result.notes:
                        raw.notes = profiled_result.notes
                    return raw
                # The profiled run wraps the result; build a DetectionResult
                return DetectionResult(
                    detections=getattr(raw, "detections", []),
                    metrics=getattr(raw, "metrics", DetectionMetrics()),
                    notes=profiled_result.notes,
                )

            else:
                detection_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.detector_engine.run(
                            tiles=formatted_tiles,
                            detector=detector,
                            cfar=cfar_detector,
                            constraint_profile="ground",
                        ),
                    ),
                    timeout=TIMEOUTS["inference_total"],
                )
                return detection_result

        except (OOMError, TimeoutError):
            raise
        except MemoryError as exc:
            raise OOMError(f"Out of memory during detection: {exc}") from exc
        except builtins.TimeoutError as exc:
            raise TimeoutError(
                f"Detection timed out after {TIMEOUTS['inference_total']}s"
            ) from exc
        except Exception as exc:
            if isinstance(exc, (DetectionError, ProfileError)):
                raise
            raise DetectionError(f"Detection failed: {exc}") from exc

    async def _save_detections(
        self,
        execution_id: UUID,
        detections: list[Detection],
    ) -> None:
        """Persist detections to the PostGIS database.

        Each detection is inserted individually using the
        ``INSERT_DETECTION`` query from ``src.db.queries``.

        Parameters
        ----------
        execution_id:
            ID of the parent execution record.
        detections:
            List of detections to save.
        """
        from src.db.connection import db
        from src.db.queries import INSERT_DETECTION

        # I-SAR-3: footprint clipping geometrico filtra borde de swath
        # (no tierra). I-DET-2: global-land-mask se mantiene como flag
        # INFORMATIVO unicamente — pobla on_land=True pero nunca descarta
        # (la deteccion se conserva, simplemente queda excluida de
        # metricas de mar y filtrable en API/dashboard).
        try:
            from global_land_mask import globe as _globe
            _has_land_mask = True
        except ImportError:
            _globe = None
            _has_land_mask = False
            self._log.info(
                "global-land-mask not installed — on_land flag stays False"
            )

        # Professional Edge Filtering: Spatial Clipping
        # Use the valid footprint from preprocessing if available
        valid_area_poly = None
        Point = None
        if hasattr(self, '_current_metadata') and 'valid_footprint' in self._current_metadata:
            try:
                from shapely.geometry import Point as _Point
                from shapely.geometry import shape
                Point = _Point
                valid_area_poly = shape(self._current_metadata['valid_footprint'])
                self._log.info("Using footprint clipping")
            except ImportError:
                pass

        # Fallback to columnar density if spatial clipping is not available
        edge_lons: set[float] = set()
        if valid_area_poly is None and len(detections) > 15:
            from collections import Counter
            lon_counts: Counter = Counter()
            for det in detections:
                if det.center_geo and len(det.center_geo) == 2:
                    lon_counts[round(det.center_geo[0], 3)] += 1
            for lon_val, cnt in lon_counts.items():
                if cnt >= 8:
                    edge_lons.add(lon_val)

        saved = 0
        skipped_no_geo = 0
        skipped_edge = 0
        for det in detections:
            try:
                # 1. Basic Geolocation Check
                if not det.center_geo or len(det.center_geo) != 2:
                    skipped_no_geo += 1
                    continue
                lon, lat = det.center_geo

                # 2. Spatial Clipping to valid swath (I-SAR-3)
                if valid_area_poly is not None and Point is not None:
                    if not valid_area_poly.contains(Point(lon, lat)):
                        skipped_edge += 1
                        continue

                # 3. Fallback Heuristic Filtering (only if no footprint)
                elif round(lon, 3) in edge_lons:
                    skipped_edge += 1
                    continue

                # Build GeoJSON for bbox_geo
                bbox_geo_geojson: str | None = None
                if det.bbox_geo and len(det.bbox_geo) == 4:
                    lon_min, lat_min, lon_max, lat_max = det.bbox_geo
                    bbox_geo_geojson = json.dumps({
                        "type": "Polygon",
                        "coordinates": [[
                            [lon_min, lat_min],
                            [lon_max, lat_min],
                            [lon_max, lat_max],
                            [lon_min, lat_max],
                            [lon_min, lat_min],
                        ]],
                    })

                # I-DET-2: pobla on_land via global-land-mask (informativo).
                on_land = bool(getattr(det, "on_land", False))
                if not on_land and _has_land_mask and _globe is not None:
                    try:
                        on_land = bool(not _globe.is_ocean(lat, lon))
                    except Exception:
                        on_land = False
                cluster_anomaly = bool(getattr(det, "cluster_anomaly", False))
                thumbnail_path = getattr(det, "thumbnail_path", None) or None

                await db.execute(
                    INSERT_DETECTION,
                    det.id,
                    execution_id,
                    lon,
                    lat,
                    bbox_geo_geojson,
                    det.bbox_pixel,
                    det.confidence,
                    det.source,
                    det.cfar_snr,
                    det.yolo_score,
                    det.class_name,
                    det.tile_index,
                    0,  # tile_row_offset
                    0,  # tile_col_offset
                    on_land,           # I-DET-2
                    cluster_anomaly,   # I-DET-3
                    thumbnail_path,    # wow #1
                )
                saved += 1
            except Exception as save_exc:
                self._log.warning(
                    "Failed to save detection",
                    extra={
                        "detection_id": str(det.id),
                        "execution_id": str(execution_id),
                        "error": str(save_exc)[:200],
                    },
                )

        if skipped_no_geo > 0:
            self._log.warning(
                "Skipped detections with missing geolocation",
                extra={
                    "execution_id": str(execution_id),
                    "skipped": skipped_no_geo,
                },
            )

        self._log.info(
            "Detections saved to PostGIS",
            extra={
                "execution_id": str(execution_id),
                "saved": saved,
                "total": len(detections),
                "skipped_no_geo": skipped_no_geo,
                "skipped_edge": skipped_edge,
            },
        )

    async def _create_cue(self, tip: TipResult) -> None:
        """Create a cue entry in the tasking_queue from a tip.

        Parameters
        ----------
        tip:
            TipResult that triggered the cue generation.
        """
        from src.db.connection import db
        from src.db.queries import INSERT_CUE

        try:
            target_bbox_geojson: str | None = None
            if tip.target_bbox and len(tip.target_bbox) == 4:
                lon_min, lat_min, lon_max, lat_max = tip.target_bbox
                target_bbox_geojson = json.dumps({
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_min, lat_min],
                        [lon_max, lat_min],
                        [lon_max, lat_max],
                        [lon_min, lat_max],
                        [lon_min, lat_min],
                    ]],
                })

            triggering_ids = [str(uid) for uid in tip.triggering_detections]

            await db.execute(
                INSERT_CUE,
                tip.execution_id,
                triggering_ids,
                target_bbox_geojson,
                tip.zone_id,
                tip.priority,
                tip.reason,
            )

            TIPS_GENERATED_TOTAL.inc()
            self._log.info(
                "Cue created from tip",
                extra={
                    "zone_id": tip.zone_id,
                    "priority": tip.priority,
                    "reason": tip.reason,
                    "execution_id": str(tip.execution_id),
                },
            )
        except Exception:
            self._log.error(
                "Failed to create cue",
                extra={
                    "zone_id": tip.zone_id,
                    "execution_id": str(tip.execution_id),
                },
                exc_info=True,
            )

    def _emit_metrics(
        self,
        profile: str,
        model: str,
        detection_result: DetectionResult,
        status: str,
        execution_id: UUID | None = None,
    ) -> None:
        """Emit Prometheus metrics for this pipeline execution.

        When ``execution_id`` is supplied, it is attached as an
        OpenMetrics ``trace_id`` exemplar to the run-level metrics.
        This closes invariant **I-TRACE-3** — the ``run_id`` reaches
        Loki (via :class:`StructuredLogger`) and now also reaches
        Prometheus exposition (via exemplar) without inflating
        cardinality with a high-uniqueness label.

        Parameters
        ----------
        profile:
            Constraint profile name.
        model:
            Model variant name.
        detection_result:
            Detection result containing metrics.
        status:
            Execution status (``"success"`` or ``"error"``).
        execution_id:
            Optional UUID of the pipeline run; emitted as a
            ``trace_id`` exemplar.
        """
        exemplar = (
            {"trace_id": str(execution_id)} if execution_id is not None else None
        )
        if exemplar is not None:
            PIPELINE_RUNS_TOTAL.labels(
                profile=profile,
                model_variant=model,
                status=status,
            ).inc(1, exemplar=exemplar)
            INFERENCE_DURATION.labels(
                profile=profile,
                model_variant=model,
            ).observe(
                detection_result.metrics.total_inference_ms / 1000.0,
                exemplar=exemplar,
            )
        else:
            PIPELINE_RUNS_TOTAL.labels(
                profile=profile,
                model_variant=model,
                status=status,
            ).inc()
            INFERENCE_DURATION.labels(
                profile=profile,
                model_variant=model,
            ).observe(detection_result.metrics.total_inference_ms / 1000.0)

        PEAK_RAM_MB.labels(
            profile=profile,
            model_variant=model,
        ).set(detection_result.metrics.peak_ram_mb)

        CPU_USAGE_PERCENT.labels(
            profile=profile,
            model_variant=model,
        ).set(detection_result.metrics.cpu_percent)

        MODEL_SIZE_MB.labels(
            model_variant=model,
        ).set(0.0)  # Set from detection result if available

        # Count detections by source
        for det in detection_result.detections:
            DETECTIONS_TOTAL.labels(
                source=det.source,
                profile=profile,
            ).inc()

    async def _cleanup(self, path: Path) -> None:
        """Remove a file or directory, logging but not raising on failure.

        Parameters
        ----------
        path:
            File or directory to remove.
        """
        try:
            if path.is_dir():
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.rmtree, path
                )
            elif path.is_file():
                path.unlink()
            self._log.debug(
                "Cleaned up temporary file",
                extra={"path": str(path)},
            )
        except OSError:
            self._log.warning(
                "Failed to clean up temporary file",
                extra={"path": str(path)},
            )

    async def _safe_update_status(
        self,
        execution_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update execution status, swallowing exceptions.

        This is used in error-handling paths where we must not let a
        secondary DB failure mask the original error.

        Parameters
        ----------
        execution_id:
            Execution record to update.
        status:
            New status value.
        error_message:
            Optional error message.
        """
        try:
            await self.recorder.update_status(
                execution_id, status, error_message
            )
        except Exception:
            self._log.error(
                "Failed to update execution status in error handler",
                extra={
                    "execution_id": str(execution_id),
                    "target_status": status,
                },
                exc_info=True,
            )

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Map an exception to a Prometheus status label.

        Parameters
        ----------
        exc:
            The exception to classify.

        Returns
        -------
        str
            One of ``"oom"``, ``"timeout"``, or ``"error"``.
        """
        if isinstance(exc, (OOMError, MemoryError)):
            return "oom"
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return "timeout"
        return "error"
