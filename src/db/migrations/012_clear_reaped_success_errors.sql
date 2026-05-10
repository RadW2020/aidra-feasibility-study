-- Migration: 012_clear_reaped_success_errors
-- Purpose: remove stale reaper messages from executions that later finished.
-- The runtime fix clears error_message on late success, but rows already
-- persisted before that fix remain contradictory in dashboards/audits.

UPDATE execution_log
SET error_message = NULL
WHERE status = 'success'
  AND error_message ILIKE 'reaped:%';

