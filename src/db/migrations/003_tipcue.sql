-- ====================================================
-- AIDRA Tip & Cue: Tasking Queue
-- PostgreSQL 16 + PostGIS 3.4
-- Migration: 003_tipcue
-- ====================================================

-- ====================================================
-- Tabla: tasking_queue
-- Cola de Tip & Cue.
-- ====================================================
CREATE TABLE tasking_queue (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    trigger_type        TEXT NOT NULL DEFAULT 'cue',
    triggered_by        UUID REFERENCES execution_log(id),
    triggering_detections UUID[],

    target_bbox         GEOMETRY(POLYGON, 4326) NOT NULL,
    target_zone         TEXT,
    priority            INTEGER NOT NULL DEFAULT 0,
    reason              TEXT,

    status              TEXT NOT NULL DEFAULT 'pending',
    scheduled_at        TIMESTAMPTZ,
    executed_at         TIMESTAMPTZ,
    execution_id        UUID REFERENCES execution_log(id),

    result_status       TEXT,
    confirmed_detections INTEGER,

    cooldown_until      TIMESTAMPTZ,
    attempts            INTEGER DEFAULT 0,
    max_attempts        INTEGER DEFAULT 3,
    last_error          TEXT
);

-- Indices
CREATE INDEX idx_tasking_status ON tasking_queue(status);
CREATE INDEX idx_tasking_priority ON tasking_queue(priority DESC, created_at);
CREATE INDEX idx_tasking_bbox ON tasking_queue USING GIST(target_bbox);
CREATE INDEX idx_tasking_triggered_by ON tasking_queue(triggered_by);
