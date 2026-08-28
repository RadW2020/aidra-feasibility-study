-- 018: anclas de trazabilidad en validation_runs (I-TRACE-4, R13).
--
-- Los reportes D2 llevaban solo model_name y configuracion del matcher.
-- Sin commit_sha ni hashes de imagen un evaluador no puede reproducir la
-- cifra. Se anaden columnas explicitas para lo que se filtra en paneles
-- (commit_sha, pipeline_path) y JSONB para el bloque completo de
-- provenance (model_hash, settings_hash, seed, image_hashes, versiones)
-- y de params (tile size, thresholds, steps ejercitados).
--
-- Filas antiguas conservan NULL: son evidencia de la era sin anclas, no
-- se borran ni se rellenan a posteriori.

ALTER TABLE validation_runs
    ADD COLUMN IF NOT EXISTS commit_sha TEXT,
    ADD COLUMN IF NOT EXISTS pipeline_path TEXT,
    ADD COLUMN IF NOT EXISTS provenance_json JSONB,
    ADD COLUMN IF NOT EXISTS params_json JSONB;

COMMENT ON COLUMN validation_runs.commit_sha IS
    'Git commit del codigo que produjo el reporte (I-TRACE-4). NULL = importado antes de 018.';
COMMENT ON COLUMN validation_runs.pipeline_path IS
    'detector | full | synthetic. "full" = DetectionEngine de produccion (Lee, sea mask, fusion, edge filter).';
COMMENT ON COLUMN validation_runs.provenance_json IS
    'model_hash, settings_hash, seed, image_hashes por escena, versiones de librerias, host.';
COMMENT ON COLUMN validation_runs.params_json IS
    'tile_size, tile_overlap, thresholds, steps ejercitados/no ejercitados, sea_only.';

CREATE INDEX IF NOT EXISTS idx_validation_runs_pipeline_path
    ON validation_runs(pipeline_path, created_at DESC);
