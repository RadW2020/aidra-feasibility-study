"""
Pydantic models para la API. Mapean los datos de la DB a objetos Python tipados.
Estos modelos los usan TODOS los agentes como contratos de interfaz.

Pydantic v2 (BaseModel from pydantic).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ====================================================================
# Database record models
# ====================================================================


class ExecutionRecord(BaseModel):
    """Registro completo de una ejecucion del pipeline."""

    id: UUID
    created_at: datetime
    image_id: str
    image_title: str | None = None
    image_hash: str
    image_sensing_date: datetime | None = None
    image_size_mb: float | None = None
    search_zone: str | None = None
    model_name: str
    model_version: str
    model_hash: str
    model_size_mb: float
    model_format: str = "pytorch"
    compression_technique: str = "none"
    # Read model: None only when a partial SELECT omitted the column.
    # Inserts always carry explicit values (I-DET-4, see recorder).
    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    constraint_profile: str = "ground"
    cpu_limit: float | None = None
    memory_limit_mb: int | None = None
    tile_size: int = 640
    tile_overlap: int = 64
    num_detections: int = 0
    # I-DET-2 / R11: subset of num_detections with quality_verdict =
    # 'valid_sea_target'. None for rows persisted before migration 017
    # that had no detections to backfill from.
    num_valid_targets: int | None = None
    # I-MOD-2 (migration 019): per-tile inference latency percentiles.
    inference_p50_ms: float | None = None
    inference_p95_ms: float | None = None
    avg_confidence: float | None = None
    max_confidence: float | None = None
    min_confidence: float | None = None
    total_duration_ms: float | None = None
    download_ms: float | None = None
    preprocessing_ms: float | None = None
    inference_ms: float | None = None
    postprocessing_ms: float | None = None
    peak_ram_mb: float | None = None
    avg_ram_mb: float | None = None
    cpu_usage_pct: float | None = None
    num_tiles: int | None = None
    output_hash: str
    input_params_hash: str | None = None
    commit_sha: str | None = None
    incidence_angle: float | None = None
    polarisation: str | None = None
    orbit_direction: str | None = None
    relative_orbit: int | None = None
    product_type: str | None = None
    pixel_spacing: float | None = None
    status: str = "pending"
    error_message: str | None = None
    trigger_type: str = "manual"
    triggered_by: UUID | None = None
    pipeline_version: str = "1.0.0"
    hostname: str | None = None
    notes: str | None = None


class DetectionRecord(BaseModel):
    """Deteccion individual de un barco."""

    id: UUID
    execution_id: UUID
    created_at: datetime
    longitude: float
    latitude: float
    bbox_pixel: list[float]
    confidence: float
    source: str  # "cfar", "yolo", "fused"
    cfar_snr: float | None = None
    yolo_score: float | None = None
    class_name: str = "vessel"
    tile_index: int
    # Joined fields
    constraint_profile: str | None = None
    model_name: str | None = None
    image_id: str | None = None
    on_land: bool = False
    cluster_anomaly: bool = False
    quality_verdict: str = "candidate"
    thumbnail_path: str | None = None
    has_thumbnail: bool = False
    # URL of the SAR crop served by GET /api/detections/{id}/thumbnail.png;
    # thumbnail_path is the server-side file and only kept for old clients.
    thumbnail_url: str | None = None
    tier: str | None = None


class ModelInfo(BaseModel):
    """Informacion de un modelo registrado."""

    id: UUID
    name: str
    version: str
    format: str
    file_hash: str
    size_mb: float
    base_model: str | None = None
    compression_technique: str = "none"
    num_params: int | None = None
    input_size: list[int] = Field(default=[640, 640])
    classes: list[str] = Field(default=["vessel"])
    # Registry status (I-MOD-3): active | candidate | rejected | retired, and
    # whether the SAR gate and the AI Act card gate (I-AIA-1) would accept
    # the model. Without these a client cannot tell a rejected variant or a
    # COCO detector from an evaluated one.
    status: str = "active"
    rejection_reason: str | None = None
    sar_compatible: bool | None = None
    has_model_card: bool | None = None


class TaskingEntry(BaseModel):
    """Entrada de la cola de Tip & Cue."""

    id: UUID
    created_at: datetime
    trigger_type: str
    triggered_by: UUID | None = None
    target_bbox_geojson: dict[str, Any] | None = None
    target_zone: str | None = None
    priority: int = 0
    reason: str | None = None
    status: str = "pending"
    execution_id: UUID | None = None
    result_status: str | None = None
    confirmed_detections: int | None = None
    attempts: int = 0
    max_attempts: int = 3
    last_error: str | None = None


class BenchmarkResult(BaseModel):
    """Resultado agregado de benchmark por modelo/perfil."""

    model_name: str
    model_version: str
    model_size_mb: float
    compression_technique: str
    constraint_profile: str
    runs: int
    avg_inference_ms: float
    p50_inference_ms: float | None = None
    p95_inference_ms: float | None = None
    avg_peak_ram_mb: float
    avg_cpu_pct: float
    avg_detections: float
    avg_confidence: float | None = None


# ====================================================================
# Request / Response models para la API
# ====================================================================


class PaginatedResponse(BaseModel):
    """Respuesta paginada generica."""

    items: list[Any]
    total: int
    limit: int
    offset: int
    # Offset of the next page, or None on the last one; and the effective
    # filters, so a client can state exactly what a count means.
    next_offset: int | None = None
    filters: dict[str, Any] | None = None


class PipelineTriggerRequest(BaseModel):
    """Solicitud para iniciar una ejecucion del pipeline."""

    zone: str = "gibraltar"
    model: str | None = None
    model_version: str | None = None
    profile: str = "ground"
    sensor: str = "s1"  # "s1" for Sentinel-1 SAR, "s2" for Sentinel-2 optical
    image_id: str | None = None
    aoi_bbox: list[float] | None = None
    # None -> Settings.confidence_threshold (I-DET-4).
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PipelineTriggerResponse(BaseModel):
    """Respuesta tras iniciar el pipeline.

    execution_id is created upfront by the API handler and returned
    immediately so callers can track the execution via
    GET /pipeline/status or GET /traceability/{execution_id}.
    """

    execution_id: UUID | None = None
    status: str = "started"
    # What will actually run (model version and thresholds resolved from
    # Settings, I-DET-4), non-blocking preflight warnings, and how to follow
    # the run. Additive: older clients only read execution_id/status.
    resolved_request: dict[str, Any] | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    poll: dict[str, Any] | None = None


class PipelineStatusResponse(BaseModel):
    """Estado actual del pipeline."""

    running: bool
    current_profile: str | None = None
    progress: float | None = None
    current_execution_id: UUID | None = None
    # ``running`` above only reflects runs started through this API. The
    # fields below come from execution_log and the scheduler, so scheduled
    # and cue runs are visible too; ``busy`` is what a client should check.
    busy: bool = False
    in_flight: list[dict[str, Any]] = Field(default_factory=list)
    engine_available: bool | None = None
    scheduler: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: str
    db: str
    models_loaded: int
    model_files_count: int | None = None
    registered_models_count: int | None = None
    scheduler: str
    version: str = "1.0.0"
    uptime_seconds: float | None = None


class CueCreateRequest(BaseModel):
    """Solicitud para crear una entrada Tip & Cue."""

    bbox: list[float] = Field(description="[lon_min, lat_min, lon_max, lat_max] in WGS-84 degrees.")
    priority: int = Field(1, description="0-10; the cue processor serves higher priorities first.")
    reason: str = Field("manual", max_length=500, description="Why the area needs another look.")
    zone: str | None = Field(
        None,
        description="Tip & Cue zone id or pipeline search zone; decides which scenes the processor searches.",
    )
    allow_duplicate: bool = Field(
        False, description="Queue even if an identical cue is already pending or processing.",
    )


class ComparisonRequest(BaseModel):
    """Solicitud de comparacion entre modelos/perfiles."""

    models: list[str] | None = None
    profiles: list[str] | None = None
    image_id: str | None = None
