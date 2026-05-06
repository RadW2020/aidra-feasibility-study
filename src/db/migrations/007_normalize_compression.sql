-- ====================================================
-- AIDRA Compression-technique name normalization
-- PostgreSQL 16
-- Migration: 007_normalize_compression
--
-- Why this migration exists:
--   Older code paths labelled the dynamic-int8 compression as
--   'int8_dynamic'; the current naming (src/models/manager.py and
--   src/models/compression/quantization.py) is 'dynamic_int8'.
--   Historical execution_log rows therefore split a single physical
--   model variant into two compression buckets, which inflated every
--   bench panel that grouped by compression_technique on the
--   compression-bench dashboard (one row became two duplicate bars).
--   This migration collapses any legacy 'int8_dynamic' rows to the
--   canonical 'dynamic_int8' name.  Idempotent: matches zero rows on
--   a freshly-seeded database.
-- ====================================================

UPDATE execution_log
SET compression_technique = 'dynamic_int8'
WHERE compression_technique = 'int8_dynamic';
