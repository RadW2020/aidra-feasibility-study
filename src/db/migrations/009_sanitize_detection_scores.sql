-- ====================================================
-- AIDRA Detection score sanitization
-- PostgreSQL 16
-- Migration: 009_sanitize_detection_scores
--
-- Historical CFAR rows may contain Infinity/NaN when background noise
-- estimation divides by zero. New inserts sanitize these values in the
-- pipeline; this migration normalizes already-persisted rows.
-- ====================================================

UPDATE detections
SET cfar_snr = NULL
WHERE cfar_snr IS NOT NULL
  AND NOT isfinite(cfar_snr::double precision);

UPDATE detections
SET yolo_score = NULL
WHERE yolo_score IS NOT NULL
  AND NOT isfinite(yolo_score::double precision);
