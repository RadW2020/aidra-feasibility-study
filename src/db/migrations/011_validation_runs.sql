-- ====================================================
-- AIDRA Validation runs (mAP / Pd / FAR persistence)
-- PostgreSQL 16 + PostGIS
-- Migration: 011_validation_runs
--
-- Closes the C1 finding from the 2026-05-08 audit: dashboards
-- 03-compression-bench and 10-evaluator-evidence flagged
-- 'NEEDS_DB_METRIC: mAP/Pd/FAR' as a hardcoded literal because the
-- validation harness in scripts/run_validation.py wrote JSON files
-- to disk but never landed in the DB. Without a table the panels
-- could not show real numbers; without real numbers the SatCen Q3
-- compression decision rests on confidence + latency proxies only.
--
-- A validation row is a snapshot of a single (model, dataset)
-- evaluation: the matcher, IoU/center thresholds, the four
-- confusion-matrix counts, the four headline metrics, and a link
-- to the execution_log row that produced the predictions when
-- the validation was run end-to-end. ``execution_id`` is nullable
-- so a model-only validation (synthetic GT, dataset-only sweep)
-- can also persist.
-- ====================================================

CREATE TABLE IF NOT EXISTS validation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    execution_id UUID REFERENCES execution_log(id) ON DELETE SET NULL,

    -- Model under evaluation. Captured even when execution_id is NULL
    -- (model-only validation against a static dataset).
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'unknown',
    model_hash TEXT,
    compression_technique TEXT NOT NULL DEFAULT 'none',

    -- Dataset / source label so we can stack multiple validation
    -- backends (xView3, HRSID, OpenSARShip, synthetic-seed-42, ...).
    dataset TEXT NOT NULL,
    dataset_split TEXT,

    -- Matcher configuration. Audited so a panel can compare runs
    -- evaluated under identical thresholds only.
    match_mode TEXT NOT NULL DEFAULT 'iou',
    iou_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    center_tolerance_px DOUBLE PRECISION NOT NULL DEFAULT 20.0,
    confidence_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    -- Confusion matrix.
    num_scenes INTEGER NOT NULL DEFAULT 0,
    num_ground_truth INTEGER NOT NULL DEFAULT 0,
    num_predictions INTEGER NOT NULL DEFAULT 0,
    true_positives INTEGER NOT NULL DEFAULT 0,
    false_positives INTEGER NOT NULL DEFAULT 0,
    false_negatives INTEGER NOT NULL DEFAULT 0,
    total_area_km2 DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    -- Headline metrics.
    map_at_iou DOUBLE PRECISION,
    pd_recall DOUBLE PRECISION,
    far_per_km2 DOUBLE PRECISION,
    precision DOUBLE PRECISION,

    -- Optional artefacts.
    pr_curve_json JSONB,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_validation_runs_model
    ON validation_runs(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_validation_runs_execution
    ON validation_runs(execution_id);

CREATE INDEX IF NOT EXISTS idx_validation_runs_compression_dataset
    ON validation_runs(compression_technique, dataset, created_at DESC);

COMMENT ON TABLE validation_runs IS
    'Persisted validation metrics (mAP, Pd, FAR/km², precision) per (model, dataset) evaluation. Closes audit finding C1 from 2026-05-08.';
