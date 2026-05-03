-- ====================================================
-- AIDRA Detection Thumbnails (Wow Effect #1)
-- PostgreSQL 16 + PostGIS 3.4
-- Migration: 005_thumbnails
--
-- Cierra criterio:
--   Q3 GEOINT: cada deteccion lleva evidencia visual auditable
--              (crop SAR ±32 px alrededor del bbox_pixel).
--   AI Act / D4: explainability — ver el barco junto al numero.
-- ====================================================

ALTER TABLE detections
    ADD COLUMN IF NOT EXISTS thumbnail_path TEXT;

COMMENT ON COLUMN detections.thumbnail_path IS
    'Ruta relativa al PNG con el crop SAR de la deteccion, dentro de /data/thumbnails/. NULL si no se genero.';

CREATE INDEX IF NOT EXISTS idx_detections_has_thumbnail
    ON detections((thumbnail_path IS NOT NULL));
