-- ====================================================
-- AIDRA Traceability Hardening
-- PostgreSQL 16 + PostGIS 3.4
-- Migration: 004_traceability
--
-- Cierra invariantes:
--   I-TRACE-4: cada run referencia commit SHA + hash de Settings.
--   I-DET-2 / I-DET-3: flags on_land y cluster_anomaly materializados.
-- ====================================================

-- ----------------------------------------------------
-- execution_log: trazabilidad reforzada
-- ----------------------------------------------------
ALTER TABLE execution_log
    ADD COLUMN IF NOT EXISTS commit_sha TEXT;

COMMENT ON COLUMN execution_log.commit_sha IS
    'Git commit SHA del codigo que ejecuto el run (HEAD del repo).';
COMMENT ON COLUMN execution_log.input_params_hash IS
    'SHA256 del Settings + parametros del request (determinista, ordenado).';

-- ----------------------------------------------------
-- execution_log: metadata SAR (Sentinel-1 manifest)
-- Extraido de manifest.safe + annotation/*.xml en preprocesado.
-- ----------------------------------------------------
ALTER TABLE execution_log
    ADD COLUMN IF NOT EXISTS incidence_angle  REAL,
    ADD COLUMN IF NOT EXISTS polarisation     TEXT,
    ADD COLUMN IF NOT EXISTS orbit_direction  TEXT,
    ADD COLUMN IF NOT EXISTS relative_orbit   INTEGER,
    ADD COLUMN IF NOT EXISTS product_type     TEXT,
    ADD COLUMN IF NOT EXISTS pixel_spacing    REAL;

COMMENT ON COLUMN execution_log.incidence_angle IS 'Angulo de incidencia medio (grados)';
COMMENT ON COLUMN execution_log.polarisation IS 'Polarizacion(es) procesadas (e.g. VV, VV+VH)';
COMMENT ON COLUMN execution_log.orbit_direction IS 'ASCENDING o DESCENDING';
COMMENT ON COLUMN execution_log.relative_orbit IS 'Numero de orbita relativa Sentinel-1';
COMMENT ON COLUMN execution_log.product_type IS 'Tipo de producto: GRD, SLC, OCN';
COMMENT ON COLUMN execution_log.pixel_spacing IS 'Pixel spacing en metros (azimuth)';

-- ----------------------------------------------------
-- detections: flags on_land + cluster_anomaly
-- ----------------------------------------------------
ALTER TABLE detections
    ADD COLUMN IF NOT EXISTS on_land BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS cluster_anomaly BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN detections.on_land IS
    'True si la deteccion cae sobre tierra segun footprint mask. Excluida de metricas de mar (I-DET-2).';
COMMENT ON COLUMN detections.cluster_anomaly IS
    'True si forma parte de un cluster con densidad > umbral por km2 (I-DET-3). Probable artefacto.';

CREATE INDEX IF NOT EXISTS idx_detections_on_land ON detections(on_land);
CREATE INDEX IF NOT EXISTS idx_detections_cluster_anomaly ON detections(cluster_anomaly);
