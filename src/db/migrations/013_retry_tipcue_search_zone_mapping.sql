-- Migration: 013_retry_tipcue_search_zone_mapping
-- Purpose: retry cues that failed only because Tip & Cue operational zone IDs
-- were passed directly to the Sentinel-1 search layer before the zone mapping
-- fix. Historical failures without this exact error remain untouched.

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
WHERE status = 'failed'
  AND target_zone IN ('gibraltar_strait', 'algeciras_port', 'med_patrol')
  AND last_error ILIKE 'Unknown search zone%';
