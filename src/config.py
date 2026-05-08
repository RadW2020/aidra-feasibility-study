"""
Configuracion centralizada con pydantic-settings.

Lee variables de entorno y archivo .env.  Cada campo tiene un valor por
defecto sensible para desarrollo local; en produccion los valores se
sobreescriben mediante variables de entorno o el archivo ``.env``.

Usage:
    from src.config import Settings

    settings = Settings()                    # lee .env + env vars
    settings = Settings(_env_file=".env")    # forzar archivo concreto
    print(settings.database_url)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration.

    Every attribute maps 1-to-1 to an environment variable with the same name
    (case-insensitive).  For example ``database_url`` reads ``DATABASE_URL``.
    """

    # ---- Base de datos ----
    database_url: str = (
        "postgresql+asyncpg://aidra:changeme@localhost:5432/aidra"
    )

    # ---- Copernicus Data Space credentials ----
    copernicus_user: str = ""
    copernicus_password: str = ""

    # ---- Directorios ----
    models_dir: str = "/app/models"
    images_dir: str = "/data/images"
    thumbnails_dir: str = "/data/thumbnails"

    # ---- Pipeline defaults ----
    default_zone: str = "gibraltar"
    default_model: str = "yolov8n-sar"
    default_profile: str = "ground"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # ---- CFAR defaults ----
    cfar_guard_size: int = 3
    cfar_training_size: int = 15
    cfar_pfa: float = 1e-5

    # ---- Tile defaults ----
    tile_size: int = 640
    tile_overlap: int = 64

    # ---- Edge swath filter (I-SAR-2) ----
    # Drops detections whose pixel center lies within ``edge_buffer_px``
    # of any scene edge. Sentinel-1 GRD swath borders concentrate
    # speckle / ambiguity ghosts; SAR vessel-detection literature uses
    # 16–64 px buffers (≈160–640 m at 10 m GRD pixel spacing).
    edge_buffer_px: int = 32

    # ---- Tip & Cue ----
    tipcue_enabled: bool = True
    tipcue_min_confidence: float = 0.7
    tipcue_min_detections: int = 2
    tipcue_cooldown_minutes: int = 60

    # ---- Scheduler ----
    scheduler_enabled: bool = True
    scheduler_interval_hours: int = 6

    # ---- Observabilidad ----
    prometheus_enabled: bool = True
    loki_url: str = "http://aidra-loki:3100"
    log_level: str = "INFO"

    # ---- API protection ----
    # Optional bearer token for state-changing endpoints. When empty,
    # local development and tests keep the unauthenticated behavior.
    aidra_api_token: str = ""

    # ---- CORS ----
    # Comma-separated list of origins allowed to call the API. Default
    # covers local development; production must override via
    # AIDRA_CORS_ORIGINS so dashboards/clients hosted on other domains
    # can reach the service without modifying source.
    aidra_cors_origins: str = (
        "http://localhost,http://localhost:3000,http://localhost:8000"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS origins into a list, stripping
        whitespace and dropping empty entries.
        """
        return [
            o.strip()
            for o in self.aidra_cors_origins.split(",")
            if o.strip()
        ]

    # ---- Limites ----
    max_image_size_gb: float = 2.0
    max_concurrent_pipelines: int = 1
    pipeline_timeout_seconds: int = 600

    # ---- Orphan reaper ----
    # Executions stuck in pending/running longer than this threshold are
    # auto-marked 'failed' by a scheduled job. Default 60 min comfortably
    # exceeds pipeline_timeout_seconds (10 min) plus download/preprocess
    # retries so a legitimately slow run is never killed by the reaper.
    orphan_reaper_threshold_minutes: int = 60
    orphan_reaper_interval_minutes: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
