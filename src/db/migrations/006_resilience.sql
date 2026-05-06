-- ====================================================
-- AIDRA Orbital Resilience persistence
-- PostgreSQL 16
-- Migration: 006_resilience
--
-- Why this migration exists:
--   Dashboard 08 (orbital-resilience) queried only Prometheus counters,
--   which reset on every redeploy. The dashboard therefore showed
--   "No data" in the demo environment after each Coolify rebuild.
--   We now persist resilience-simulation outputs so the dashboard
--   reflects the historical record rather than a transient gauge.
-- ====================================================

-- One row per (sweep call, num_flips) data point.  Multiple rows
-- with the same sweep_id form a single bit-flip sweep.
CREATE TABLE IF NOT EXISTS bitflip_runs (
    id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sweep_id             UUID            NOT NULL,
    created_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    model_variant        TEXT            NOT NULL,
    model_size_bytes     BIGINT,
    num_flips            INTEGER         NOT NULL,
    avg_detections       DOUBLE PRECISION,
    avg_confidence       DOUBLE PRECISION,
    std_detections       DOUBLE PRECISION,
    degradation_pct      DOUBLE PRECISION NOT NULL,
    baseline_detections  INTEGER         NOT NULL,
    baseline_confidence  DOUBLE PRECISION NOT NULL,
    critical_threshold   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bitflip_runs_sweep
    ON bitflip_runs (sweep_id);
CREATE INDEX IF NOT EXISTS idx_bitflip_runs_created_at
    ON bitflip_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bitflip_runs_variant
    ON bitflip_runs (model_variant);

COMMENT ON TABLE bitflip_runs IS
    'Bit-flip resilience sweep results.  One row per (sweep, num_flips) point.';

-- One row per orbit simulation invocation.
CREATE TABLE IF NOT EXISTS orbit_sim_runs (
    id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    satellite            TEXT            NOT NULL,
    total_images         INTEGER         NOT NULL,
    processed_images     INTEGER         NOT NULL,
    skipped_images       INTEGER         NOT NULL,
    cfar_fallback_count  INTEGER         NOT NULL,
    process_count        INTEGER         NOT NULL DEFAULT 0,
    fallback_cfar_count  INTEGER         NOT NULL DEFAULT 0,
    skip_count           INTEGER         NOT NULL DEFAULT 0,
    models_used          JSONB,
    battery_timeline     DOUBLE PRECISION[],
    final_battery_wh     DOUBLE PRECISION,
    energy_efficiency    DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_orbit_sim_runs_created_at
    ON orbit_sim_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orbit_sim_runs_satellite
    ON orbit_sim_runs (satellite);

COMMENT ON TABLE orbit_sim_runs IS
    'Orbit-simulation aggregates with battery timeline and decision counters.';

-- One row per drift check (whether or not drift was detected).
CREATE TABLE IF NOT EXISTS drift_alerts (
    id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    is_drifting      BOOLEAN         NOT NULL,
    metric           TEXT            NOT NULL,
    z_score          DOUBLE PRECISION,
    recent_mean      DOUBLE PRECISION,
    historical_mean  DOUBLE PRECISION,
    recommendation   TEXT,
    window_size      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_created_at
    ON drift_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_metric
    ON drift_alerts (metric);

COMMENT ON TABLE drift_alerts IS
    'Drift-detection results from /api/orbital/resilience/drift calls.';
