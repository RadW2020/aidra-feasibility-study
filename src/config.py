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
    database_url: str = "postgresql+asyncpg://aidra:changeme@localhost:5432/aidra"

    # ---- Copernicus Data Space credentials ----
    copernicus_user: str = ""
    copernicus_password: str = ""

    # ---- Directorios ----
    models_dir: str = "/app/models"
    # Cards copied by the Dockerfile outside the models volume; synced into
    # ``models_dir/cards`` at startup (src/models/cards_sync.py). Empty or
    # missing -> no sync (local dev runs from the repo).
    model_cards_dist_dir: str = "/app/models_dist/cards"
    images_dir: str = "/data/images"
    thumbnails_dir: str = "/data/thumbnails"

    # ---- Pipeline defaults ----
    default_zone: str = "gibraltar"
    default_model: str = "vesseltracker-sar-yolov8"
    default_model_version: str = "v1.0"
    default_profile: str = "ground"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # ---- CFAR defaults ----
    # Window geometry of the production CA-CFAR. Until 2026-08-29
    # ModelManager instantiated ``CFARDetector()`` with the class defaults
    # (guard 8 / training 20) and silently ignored these fields, which said
    # 3 / 15. The fields now drive the detector (single source of truth,
    # I-DET-4) and their defaults are set to what actually ran, so every
    # historical run and the xView3 validation of 2026-08-28 remain
    # comparable. Changing them is a benchmarked decision, not a default.
    cfar_guard_size: int = 8
    cfar_training_size: int = 20
    cfar_pfa: float = 1e-5
    # Per-tile DBSCAN clustering of raw CFAR pixel hits. Defaults match
    # the previously hardcoded values in detection.py: a single bright
    # speckle pixel is not a vessel (min_cluster_size=5), pixels within
    # 1.5 px of each other belong to the same vessel (eps), and the
    # cluster's mean SNR has to exceed 2.0 (~3 dB above clutter) to
    # gate sea-state false positives.
    cfar_min_cluster_size: int = 5
    cfar_cluster_eps: float = 1.5
    cfar_min_mean_snr: float = 2.0

    # ---- Detection fusion / postprocessing ----
    # IoU threshold above which a CFAR detection is matched to a YOLO
    # detection in the same tile and fused into a single Detection
    # with source='fused'. Below the threshold both survive separately.
    fusion_iou_threshold: float = 0.3
    # How a CFAR cluster is matched to a YOLO box before fusing (R14).
    # "center": Euclidean distance between bbox centres <=
    # ``fusion_center_tolerance_px`` (xView3-style; 20 px = 200 m at 10 m
    # GRD spacing). "iou": legacy IoU >= ``fusion_iou_threshold`` — measured
    # to never fire on xView3 (CFAR clusters ~4 px vs YOLO boxes ~40 px,
    # median IoU 0.07): 0 fused detections over 1 997 vessels.
    fusion_mode: str = "center"
    fusion_center_tolerance_px: float = 20.0
    # Weight of the YOLO score in a fused detection's confidence; the
    # CFAR SNR-derived confidence gets ``1 - fusion_yolo_weight``.
    # I-DET-4: lives here so input_params_hash captures it.
    fusion_yolo_weight: float = 0.5
    # What YOLO sees (R15). "unfiltered": the calibrated linear sigma0
    # BEFORE the Lee filter, converted with the standard dB stretch — the
    # Lee filter is right for CFAR's multiplicative-noise model but cost
    # vesseltracker-sar-yolov8 35 % of its recall on xView3 (Pd 0.143 ->
    # 0.093, 11/11 scenes). "filtered": legacy, YOLO sees the Lee output.
    yolo_input: str = "unfiltered"
    # I-DET-3: cluster_anomaly heuristic. A detection is flagged if at
    # least ``cluster_anomaly_min_neighbours`` other detections sit
    # within ``cluster_anomaly_radius_deg`` (great-circle approximation
    # over lon/lat). High densities are typically swath edge artefacts,
    # speckle bursts or unfiltered land returns.
    cluster_anomaly_radius_deg: float = 0.01
    cluster_anomaly_min_neighbours: int = 8

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
    # Direct HTTP push to Loki. Production has no Promtail (mounting
    # docker.sock on a shared host was rejected), so this is Loki's only
    # ingestion path — turning it off leaves I-TRACE-3 unsatisfied.
    loki_enabled: bool = True
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
    aidra_cors_origins: str = "http://localhost,http://localhost:3000,http://localhost:8000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS origins into a list, stripping
        whitespace and dropping empty entries.
        """
        return [o.strip() for o in self.aidra_cors_origins.split(",") if o.strip()]

    # ---- Limites ----
    max_image_size_gb: float = 2.0
    max_concurrent_pipelines: int = 1
    pipeline_timeout_seconds: int = 600

    # ---- Orphan reaper ----
    # Executions stuck in pending/running longer than this threshold are
    # auto-marked 'failed' by a scheduled job. Default 60 min comfortably
    # exceeds pipeline_timeout_seconds (10 min) plus download/preprocess
    # retries so a legitimately slow run is never killed by the reaper.
    # Debe superar la ejecucion legitima mas lenta: sat-extreme tarda
    # ~150 min en escena completa. Con 60 min el reaper marco failed un
    # run sat-extreme vivo y sano (2026-08-16, exec e52ad118).
    orphan_reaper_threshold_minutes: int = 240
    orphan_reaper_interval_minutes: int = 15

    # ---- Constraint profiles: memory budget ----
    # "abort": a run whose peak RSS exceeds the profile's memory_limit_mb is
    # aborted (MemoryError -> execution_log status=error, error_message with
    # the breach) at the next per-tile check, emulating the kernel OOM-kill
    # of real on-board hardware. "measure": legacy behaviour — the breach is
    # only recorded in notes and the run completes. RLIMIT_AS is not used
    # because PyTorch maps 8-12 GB of virtual memory regardless of RSS.
    profile_memory_enforcement: str = "abort"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
