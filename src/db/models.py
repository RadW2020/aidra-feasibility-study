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
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    constraint_profile: str = "ground"
    cpu_limit: float | None = None
    memory_limit_mb: int | None = None
    tile_size: int = 640
    tile_overlap: int = 64
    num_detections: int = 0
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
    thumbnail_path: str | None = None
    has_thumbnail: bool = False


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


class PipelineTriggerRequest(BaseModel):
    """Solicitud para iniciar una ejecucion del pipeline."""

    zone: str = "gibraltar"
    model: str = "yolov8n-sar"
    profile: str = "ground"
    sensor: str = "s1"  # "s1" for Sentinel-1 SAR, "s2" for Sentinel-2 optical
    image_id: str | None = None
    aoi_bbox: list[float] | None = None
    confidence_threshold: float = 0.25


class PipelineTriggerResponse(BaseModel):
    """Respuesta tras iniciar el pipeline.

    execution_id is created upfront by the API handler and returned
    immediately so callers can track the execution via
    GET /pipeline/status or GET /traceability/{execution_id}.
    """

    execution_id: UUID | None = None
    status: str = "started"


class PipelineStatusResponse(BaseModel):
    """Estado actual del pipeline."""

    running: bool
    current_profile: str | None = None
    progress: float | None = None
    current_execution_id: UUID | None = None


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: str
    db: str
    models_loaded: int
    scheduler: str
    version: str = "1.0.0"
    uptime_seconds: float | None = None


class CueCreateRequest(BaseModel):
    """Solicitud para crear una entrada Tip & Cue."""

    bbox: list[float]
    priority: int = 1
    reason: str = "manual"
    zone: str | None = None


class ComparisonRequest(BaseModel):
    """Solicitud de comparacion entre modelos/perfiles."""

    models: list[str] | None = None
    profiles: list[str] | None = None
    image_id: str | None = None
