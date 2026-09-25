-- ====================================================
-- Migration 021: audit log and idempotency keys for API clients
--
-- execution_log records WHAT ran, with full provenance. Nothing recorded WHO
-- asked for it: mutating API calls left at most an INFO log line. With agents
-- as first-class clients the question "what did the agent do, and why was it
-- refused?" needs a durable answer, so every non-GET /api call writes one row
-- here (src/api/audit.py). Rows are append-only: nothing in the code updates
-- or deletes them.
--
-- api_idempotency_keys backs the Idempotency-Key header on
-- POST /api/pipeline/trigger and POST /api/tasking/cue: a client that retries
-- after a timeout gets the original response back instead of a second run or
-- a duplicate cue (src/api/idempotency.py).
-- ====================================================

CREATE TABLE IF NOT EXISTS api_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id      TEXT NOT NULL,
    actor           TEXT NOT NULL,
    actor_scope     TEXT,
    authenticated   BOOLEAN NOT NULL DEFAULT FALSE,
    client          TEXT,
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    operation       TEXT NOT NULL,
    status_code     INTEGER NOT NULL,
    outcome         TEXT NOT NULL,
    error_code      TEXT,
    duration_ms     REAL,
    resource_type   TEXT,
    resource_id     TEXT,
    idempotency_key TEXT,
    idempotent_replay BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT api_audit_log_outcome_check
        CHECK (outcome IN ('success', 'rejected', 'error'))
);

COMMENT ON TABLE api_audit_log IS
    'One row per mutating /api call (who, what, on which resource, outcome). Append-only.';
COMMENT ON COLUMN api_audit_log.actor IS
    'Principal name from the bearer token (AIDRA_API_TOKENS), operator for the legacy token, anonymous without one.';
COMMENT ON COLUMN api_audit_log.client IS
    'Self-declared X-AIDRA-Client header (e.g. aidra-mcp/1.0 via claude-code). Informational, not an identity.';
COMMENT ON COLUMN api_audit_log.outcome IS
    'success (2xx/3xx), rejected (4xx: the caller must change something), error (5xx).';

CREATE INDEX IF NOT EXISTS idx_api_audit_created ON api_audit_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_audit_actor ON api_audit_log (actor, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_audit_resource ON api_audit_log (resource_type, resource_id);

CREATE TABLE IF NOT EXISTS api_idempotency_keys (
    actor           TEXT NOT NULL,
    key             TEXT NOT NULL,
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    request_hash    TEXT NOT NULL,
    status_code     INTEGER,
    response_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    PRIMARY KEY (actor, key)
);

COMMENT ON TABLE api_idempotency_keys IS
    'Idempotency-Key store: same actor + key + body replays the stored response for 24 h.';
