-- Migration: 014_clear_pending_tipcue_mapping_errors
-- Purpose: clear pending retries that already captured the old
-- "Unknown search zone" error before the Tip & Cue zone mapping fix deployed.
-- Migration 013 handled exhausted failed rows; this covers non-exhausted
-- pending rows so the next scheduler tick starts from clean retry state.

UPDATE tasking_queue
SET status = 'pending',
    scheduled_at = NULL,
    executed_at = NULL,
    execution_id = NULL,
    result_status = NULL,
    confirmed_detections = NULL,
    cooldown_until = NULL,
    attempts = 0,
    last_error = NULL
WHERE status = 'pending'
  AND target_zone IN ('gibraltar_strait', 'algeciras_port', 'med_patrol')
  AND last_error ILIKE 'Unknown search zone%';
