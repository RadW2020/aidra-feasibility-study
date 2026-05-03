-- ====================================================
-- AIDRA Database Schema
-- PostgreSQL 16 + PostGIS 3.4
-- Migration: 001_init
-- ====================================================

-- Extensiones
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Para busquedas de texto

-- ====================================================
-- Tabla: execution_log
-- Registro inmutable de cada ejecucion del pipeline.
-- Nucleo de la trazabilidad AIDRA.
-- ====================================================
CREATE TABLE execution_log (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Input
    image_id            TEXT NOT NULL,
    image_title         TEXT,
    image_hash          TEXT NOT NULL,
    image_bbox          GEOMETRY(POLYGON, 4326),
    image_sensing_date  TIMESTAMPTZ,
    image_size_mb       REAL,
    search_zone         TEXT,

    -- Modelo
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    model_hash          TEXT NOT NULL,
    model_size_mb       REAL NOT NULL,
    model_format        TEXT NOT NULL DEFAULT 'pytorch',
    compression_technique TEXT DEFAULT 'none',

    -- Parametros
    confidence_threshold REAL NOT NULL DEFAULT 0.25,
    iou_threshold       REAL NOT NULL DEFAULT 0.45,
    constraint_profile  TEXT NOT NULL DEFAULT 'ground',
    cpu_limit           REAL,
    memory_limit_mb     INTEGER,
    tile_size           INTEGER DEFAULT 640,
    tile_overlap        INTEGER DEFAULT 64,

    -- Resultados
    num_detections      INTEGER NOT NULL DEFAULT 0,
    avg_confidence      REAL,
    max_confidence      REAL,
    min_confidence      REAL,

    -- Metricas de rendimiento
    total_duration_ms   REAL,
    download_ms         REAL,
    preprocessing_ms    REAL,
    inference_ms        REAL,
    postprocessing_ms   REAL,
    peak_ram_mb         REAL,
    avg_ram_mb          REAL,
    cpu_usage_pct       REAL,
    num_tiles           INTEGER,

    -- Trazabilidad
    output_hash         TEXT NOT NULL,
    input_params_hash   TEXT,

    -- Estado
    status              TEXT NOT NULL DEFAULT 'pending',
    error_message       TEXT,
    trigger_type        TEXT NOT NULL DEFAULT 'manual',
    triggered_by        UUID REFERENCES execution_log(id),

    -- Metadatos
    pipeline_version    TEXT DEFAULT '1.0.0',
    hostname            TEXT,
    notes               TEXT
);

-- ====================================================
-- Tabla: detections
-- Detecciones individuales de barcos.
-- Cada deteccion pertenece a una ejecucion del pipeline.
-- ====================================================
CREATE TABLE detections (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id        UUID NOT NULL REFERENCES execution_log(id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Ubicacion
    center_geo          GEOMETRY(POINT, 4326) NOT NULL,
    bbox_geo            GEOMETRY(POLYGON, 4326),
    bbox_pixel          REAL[] NOT NULL,

    -- Deteccion
    confidence          REAL NOT NULL,
    source              TEXT NOT NULL,
    cfar_snr            REAL,
    yolo_score          REAL,
    class_name          TEXT DEFAULT 'vessel',

    -- Tile
    tile_index          INTEGER NOT NULL,
    tile_row_offset     INTEGER,
    tile_col_offset     INTEGER
);

-- ====================================================
-- Tabla: models_registry
-- Registro de variantes de modelo disponibles.
-- ====================================================
CREATE TABLE models_registry (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    name                TEXT NOT NULL,
    version             TEXT NOT NULL,
    format              TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    file_hash           TEXT NOT NULL UNIQUE,
    size_mb             REAL NOT NULL,

    base_model          TEXT,
    compression_technique TEXT DEFAULT 'none',
    compression_params  JSONB,

    num_params          BIGINT,
    num_layers          INTEGER,
    input_size          INTEGER[] DEFAULT '{640,640}',
    classes             TEXT[] DEFAULT '{"vessel"}',

    metadata            JSONB,

    UNIQUE(name, version)
);
